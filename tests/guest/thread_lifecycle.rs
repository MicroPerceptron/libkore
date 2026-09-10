//! Injected into Kore's founded `hello` by check-thread-guest.py.
//! Calls the actual foreign adapter from native workers, not a mock controller.
use core::{
    ffi::c_void,
    mem::MaybeUninit,
    ptr,
    sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering},
};
use kore_ffi::*;
#[global_allocator]
static HEAP: kore_rt::heap::Heap<kore_rt::heap::HeapSupplySource> =
    kore_rt::heap::Heap::new_in(kore_rt::heap::HeapSupplySource);
static STAGE: AtomicU32 = AtomicU32::new(0);
pub fn stage() -> u8 {
    STAGE.load(Ordering::Acquire) as u8
}
static KEY: AtomicU64 = AtomicU64::new(0);
static TRACE: AtomicU64 = AtomicU64::new(0);
static READY: AtomicU32 = AtomicU32::new(0);
static GO: AtomicU32 = AtomicU32::new(0);
static WORD: AtomicU32 = AtomicU32::new(0);
static TARGET: AtomicPtr<KoreThread> = AtomicPtr::new(ptr::null_mut());
static DONE: AtomicU32 = AtomicU32::new(0);
fn pointer(n: usize) -> *mut c_void {
    ptr::without_provenance_mut(n)
}
unsafe extern "C" fn record(value: *mut c_void) {
    TRACE
        .try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
            Some(old * 10 + value.addr() as u64)
        })
        .unwrap();
    if value.addr() == 3 {
        DONE.store(1, Ordering::Release);
    }
}
fn wait(word: &AtomicU32, value: u32) -> bool {
    let Ok(start) = kore_rt::clock::uptime() else {
        return false;
    };
    loop {
        let observed = word.load(Ordering::Acquire);
        if observed == value {
            return true;
        }
        let Ok(now) = kore_rt::clock::uptime() else {
            return false;
        };
        if now.as_nanos().saturating_sub(start.as_nanos()) > 5_000_000_000 {
            return false;
        }
        let _ = kore_rt::futex::wait(word, observed, Some(core::time::Duration::from_millis(1)));
    }
}
unsafe extern "C" fn worker(mode: *mut c_void) -> *mut c_void {
    let mut first = MaybeUninit::uninit();
    let mut second = MaybeUninit::uninit();
    unsafe {
        if kore_thread_cleanup_push(first.as_mut_ptr(), Some(record), pointer(1)) != 0
            || kore_thread_cleanup_push(second.as_mut_ptr(), Some(record), pointer(9)) != 0
            || kore_thread_setspecific(KEY.load(Ordering::Acquire), pointer(3)) != 0
            || __cxa_thread_atexit_impl(Some(record), pointer(2), ptr::null_mut()) != 0
            || __cxa_thread_atexit(Some(record), pointer(4), ptr::null_mut()) != 0
        {
            return pointer(99);
        }
        match mode.addr() {
            1 => pointer(42),
            2 => kore_thread_exit(pointer(43)),
            3 => {
                READY.store(1, Ordering::Release);
                loop {
                    // Every foreign wait entry checks cancellation. Independent
                    // controller workers have no doorbell: preserve Unsupported
                    // and park via the native deadline helper between retries.
                    // This witnesses cancellation delivery, not futex wake routing.
                    if kore_futex_wait(WORD.as_ptr(), 0, KORE_FUTEX_INFINITE)
                        == KoreFutexOutcome::Unsupported
                    {
                        let _ = kore_rt::pace::cede();
                    }
                }
            }
            4 => {
                if kore_thread_setcancelstate(KORE_THREAD_CANCEL_DISABLE, ptr::null_mut()) != 0 {
                    return pointer(99);
                }
                READY.store(1, Ordering::Release);
                if !wait(&GO, 1) {
                    return pointer(99);
                }
                kore_thread_testcancel(); // must preserve the pending request
                let mut old = -1;
                if kore_thread_setcancelstate(KORE_THREAD_CANCEL_ENABLE, &mut old) != 0
                    || old != KORE_THREAD_CANCEL_DISABLE
                {
                    return pointer(99);
                }
                kore_thread_testcancel();
                pointer(99)
            }
            _ => pointer(99),
        }
    }
}
unsafe extern "C" fn target(_: *mut c_void) -> *mut c_void {
    if wait(&GO, 1) {
        pointer(44)
    } else {
        pointer(99)
    }
}
unsafe extern "C" fn joiner(_: *mut c_void) -> *mut c_void {
    READY.store(1, Ordering::Release);
    unsafe {
        kore_thread_join(TARGET.load(Ordering::Acquire), ptr::null_mut());
    }
    pointer(99)
}
unsafe fn create(
    call: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    argument: usize,
) -> *mut KoreThread {
    let mut thread = ptr::null_mut();
    let result =
        unsafe { kore_thread_create(&mut thread, ptr::null(), Some(call), pointer(argument)) };
    if result != 0 {
        STAGE.store(100 + (result as u32).min(50), Ordering::Release);
        ptr::null_mut()
    } else {
        thread
    }
}
unsafe fn joined(thread: *mut KoreThread, expected: *mut c_void) -> bool {
    let mut result = pointer(98);
    let status = unsafe { kore_thread_join(thread, &mut result) };
    if status != 0 {
        STAGE.store(80 + (status as u32).min(50), Ordering::Release);
        return false;
    }
    if result != expected {
        STAGE.store(60, Ordering::Release);
        return false;
    }
    true
}
pub fn run() -> bool {
    unsafe {
        let mut key = 0;
        if kore_thread_key_create(&mut key, Some(record)) != 0 {
            return false;
        }
        KEY.store(key, Ordering::Release);
        // Normal return skips cleanup stack; explicit exit/cancel runs it LIFO.
        for mode in 1..=4 {
            STAGE.store(mode as u32, Ordering::Release);
            TRACE.store(0, Ordering::Release);
            READY.store(0, Ordering::Release);
            GO.store(0, Ordering::Release);
            DONE.store(0, Ordering::Release);
            let thread = create(worker, mode);
            if thread.is_null() {
                return false;
            }
            if mode >= 3 {
                if !wait(&READY, 1) || kore_thread_cancel(thread) != 0 {
                    return false;
                }
                if mode == 4 {
                    if DONE.load(Ordering::Acquire) != 0 {
                        return false;
                    }
                    GO.store(1, Ordering::Release);
                }
            }
            let expected = match mode {
                1 => pointer(42),
                2 => pointer(43),
                _ => kore_thread_canceled(),
            };
            if !joined(thread, expected) {
                return false;
            }
            if TRACE.load(Ordering::Acquire) != if mode == 1 { 423 } else { 91423 } {
                STAGE.store(61, Ordering::Release);
                return false;
            }
        }
        STAGE.store(5, Ordering::Release);
        // Canceling a blocked joiner cannot consume/detach its target.
        GO.store(0, Ordering::Release);
        READY.store(0, Ordering::Release);
        let child = create(target, 0);
        if child.is_null() {
            return false;
        }
        TARGET.store(child, Ordering::Release);
        let joining = create(joiner, 0);
        if joining.is_null()
            || !wait(&READY, 1)
            || kore_thread_cancel(joining) != 0
            || !joined(joining, kore_thread_canceled())
        {
            return false;
        }
        GO.store(1, Ordering::Release);
        if !joined(child, pointer(44)) {
            return false;
        }
        STAGE.store(6, Ordering::Release);
        // More than one full controller generation of canceled detached workers.
        for _ in 0..40 {
            TRACE.store(0, Ordering::Release);
            READY.store(0, Ordering::Release);
            DONE.store(0, Ordering::Release);
            let thread = create(worker, 3);
            if thread.is_null()
                || !wait(&READY, 1)
                || kore_thread_cancel(thread) != 0
                || kore_thread_detach(thread) != 0
                || !wait(&DONE, 1)
            {
                return false;
            }
            if TRACE.load(Ordering::Acquire) != 91423 {
                return false;
            }
        }
        kore_thread_key_delete(key) == 0
    }
}
