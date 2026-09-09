//! Caller-allocated, opaque hash tables over hashbrown's public HashTable API.
//! All entries remain caller-owned. No global heap, locks, Rust layout,
//! destructor callbacks, or Rust unwinding crosses the C boundary.

use core::{
    alloc::{AllocError, Allocator as RustAllocator, Layout},
    ffi::c_void,
    ptr::{self, NonNull},
};
use kore::collections::HashTable;

/// Opaque, uniquely owned table. Never construct, copy, or inspect its layout.
pub struct KoreHashTable {
    _private: [u8; 0],
}

/// Allocation callbacks must obey Rust's `core::alloc::Allocator` contract. Their context
/// must remain live until table destruction; allocation returns null on OOM.
/// Allocate receives (context, size, alignment) and returns disjoint storage
/// valid for that layout. Deallocate receives the same pointer and layout.
/// Callbacks may not unwind or reenter the table. Changing an application-wide
/// default must not invalidate callbacks or allocations retained by a live table.
#[repr(C)]
#[derive(Clone)]
pub struct KoreHashAllocator {
    pub context: *mut c_void,
    pub allocate: Option<unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void>,
    pub deallocate: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize)>,
}

/// Return nonzero when an existing entry equals the caller's query.
pub type KoreHashEqual =
    Option<unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void) -> i32>;
/// Return nonzero to continue visiting; zero stops traversal.
pub type KoreHashVisit = Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32>;

pub const KORE_HASH_OK: i32 = 0;
pub const KORE_HASH_MISSING: i32 = 1;
pub const KORE_HASH_REPLACED: i32 = 2;
pub const KORE_HASH_INVALID: i32 = -1;
pub const KORE_HASH_NO_MEMORY: i32 = -2;

#[derive(Clone)]
struct CallbackAllocator(KoreHashAllocator);

// SAFETY: create's unsafe caller contract requires both callbacks to implement
// Allocator, including disjoint, aligned storage and matching deallocation.
unsafe impl RustAllocator for CallbackAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            let ptr = NonNull::new(layout.align() as *mut u8).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(ptr, 0));
        }
        let callback = self.0.allocate.ok_or(AllocError)?;
        // SAFETY: callback lifetime and contract are guaranteed by create.
        let raw = unsafe { callback(self.0.context, layout.size(), layout.align()) };
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or(AllocError)?;
        if raw.addr() % layout.align() != 0 {
            // SAFETY: release exactly the callback's returned allocation.
            unsafe {
                (self.0.deallocate.unwrap())(self.0.context, raw, layout.size(), layout.align())
            };
            return Err(AllocError);
        }
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: forwarded allocation and exact layout; callback is live.
            unsafe {
                (self.0.deallocate.unwrap())(
                    self.0.context,
                    ptr.as_ptr().cast(),
                    layout.size(),
                    layout.align(),
                )
            };
        }
    }
}

struct Entry {
    hash: u64,
    value: *mut c_void,
}
struct OwnedTable {
    entries: HashTable<Entry, CallbackAllocator>,
}

/// Create a table, returning null for invalid callbacks or allocation failure.
/// # Safety
/// Callbacks obey KoreHashAllocator and never unwind; their context outlives the table.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_create(
    allocator: KoreHashAllocator,
    capacity: usize,
) -> *mut KoreHashTable {
    if allocator.allocate.is_none() || allocator.deallocate.is_none() {
        return ptr::null_mut();
    }
    let mut entries = HashTable::new_in(CallbackAllocator(allocator));
    if entries.try_reserve(capacity, |e: &Entry| e.hash).is_err() {
        return ptr::null_mut();
    }
    let Ok(storage) = entries.allocator().allocate(Layout::new::<OwnedTable>()) else {
        return ptr::null_mut();
    };
    let table = storage.cast::<OwnedTable>().as_ptr();
    // SAFETY: fresh, sufficiently sized and aligned allocation, uniquely owned.
    unsafe { table.write(OwnedTable { entries }) };
    table.cast()
}

/// Destroy table storage, leaving all entries owned by the caller. Null is OK.
/// # Safety
/// Non-null table is live, uniquely owned, and is destroyed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_destroy(table: *mut KoreHashTable) {
    if table.is_null() {
        return;
    }
    let owned = table.cast::<OwnedTable>();
    // SAFETY: caller provides the live, unique handle.
    let allocator = unsafe { (*owned).entries.allocator().clone() };
    unsafe {
        ptr::drop_in_place(owned);
        allocator.deallocate(
            NonNull::new_unchecked(owned.cast()),
            Layout::new::<OwnedTable>(),
        );
    }
}

/// Return the entry count (zero for null).
/// # Safety
/// Non-null table is live and not concurrently mutated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_len(table: *const KoreHashTable) -> usize {
    unsafe { table.cast::<OwnedTable>().as_ref() }.map_or(0, |t| t.entries.len())
}

/// Reserve additional entries, preserving contents on failure.
/// # Safety
/// Non-null table is live and exclusively borrowed for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_reserve(
    table: *mut KoreHashTable,
    additional: usize,
) -> i32 {
    let Some(t) = (unsafe { table.cast::<OwnedTable>().as_mut() }) else {
        return KORE_HASH_INVALID;
    };
    match t.entries.try_reserve(additional, |e| e.hash) {
        Ok(()) => KORE_HASH_OK,
        Err(_) => KORE_HASH_NO_MEMORY,
    }
}

/// Insert or replace an equal entry; return the replaced pointer through old.
/// # Safety
/// KoreHashTable is live/exclusive; entry stays live and its key/hash immutable until
/// removed. KoreHashEqual never mutates/reenters this table or unwinds; equal keys have
/// equal hashes. Old is writable, aligned and disjoint from table/entry storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_insert(
    table: *mut KoreHashTable,
    hash: u64,
    entry: *mut c_void,
    equal: KoreHashEqual,
    context: *mut c_void,
    old: *mut *mut c_void,
) -> i32 {
    if old.is_null() {
        return KORE_HASH_INVALID;
    }
    unsafe { old.write(ptr::null_mut()) };
    let (Some(t), Some(equal)) = ((unsafe { table.cast::<OwnedTable>().as_mut() }), equal) else {
        return KORE_HASH_INVALID;
    };
    if entry.is_null() {
        return KORE_HASH_INVALID;
    }
    if let Some(existing) = t
        .entries
        .find_mut(hash, |e| unsafe { equal(e.value, entry, context) != 0 })
    {
        unsafe { old.write(core::mem::replace(&mut existing.value, entry)) };
        return KORE_HASH_REPLACED;
    }
    if t.entries.try_reserve(1, |e| e.hash).is_err() {
        return KORE_HASH_NO_MEMORY;
    }
    t.entries
        .insert_unique(hash, Entry { hash, value: entry }, |e| e.hash);
    KORE_HASH_OK
}

/// Find an entry, returning MISSING and a null output when absent.
/// # Safety
/// KoreHashTable is live/not concurrently mutated; query/context satisfy the equality
/// callback, which never mutates/reenters this table or unwinds. Out is writable,
/// aligned and disjoint from table and callback input storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_find(
    table: *const KoreHashTable,
    hash: u64,
    query: *const c_void,
    equal: KoreHashEqual,
    context: *mut c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return KORE_HASH_INVALID;
    }
    unsafe { out.write(ptr::null_mut()) };
    let (Some(t), Some(equal)) = ((unsafe { table.cast::<OwnedTable>().as_ref() }), equal) else {
        return KORE_HASH_INVALID;
    };
    match t
        .entries
        .find(hash, |e| unsafe { equal(e.value, query, context) != 0 })
    {
        Some(e) => {
            unsafe { out.write(e.value) };
            KORE_HASH_OK
        }
        None => KORE_HASH_MISSING,
    }
}

/// Remove an entry and return its caller-owned pointer.
/// # Safety
/// Same contract as find, with exclusive access to the table.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_remove(
    table: *mut KoreHashTable,
    hash: u64,
    query: *const c_void,
    equal: KoreHashEqual,
    context: *mut c_void,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return KORE_HASH_INVALID;
    }
    unsafe { out.write(ptr::null_mut()) };
    let (Some(t), Some(equal)) = ((unsafe { table.cast::<OwnedTable>().as_mut() }), equal) else {
        return KORE_HASH_INVALID;
    };
    match t
        .entries
        .find_entry(hash, |e| unsafe { equal(e.value, query, context) != 0 })
    {
        Ok(e) => {
            unsafe { out.write(e.remove().0.value) };
            KORE_HASH_OK
        }
        Err(_) => KORE_HASH_MISSING,
    }
}

/// Remove all entries without destroying them or releasing table capacity.
/// # Safety
/// Non-null table is live and exclusively borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_clear(table: *mut KoreHashTable) {
    if let Some(t) = unsafe { table.cast::<OwnedTable>().as_mut() } {
        t.entries.clear();
    }
}

/// KoreHashVisit entries in unspecified order, stopping if visit returns zero.
/// # Safety
/// KoreHashTable is live/not concurrently mutated. KoreHashVisit/context remain valid and
/// never mutate/reenter this table or unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_hash_table_visit(
    table: *const KoreHashTable,
    visit: KoreHashVisit,
    context: *mut c_void,
) -> i32 {
    let (Some(t), Some(visit)) = ((unsafe { table.cast::<OwnedTable>().as_ref() }), visit) else {
        return KORE_HASH_INVALID;
    };
    for e in t.entries.iter() {
        if unsafe { visit(e.value, context) } == 0 {
            break;
        }
    }
    KORE_HASH_OK
}

#[cfg(all(test, feature = "test-foreign", not(miri), not(target_os = "none")))]
#[test]
fn compiled_c_hash_contract() {
    unsafe extern "C" {
        fn c_check_hash();
        fn zig_check_hash() -> u32;
    }
    // SAFETY: fixture owns live storage and obeys every callback contract.
    unsafe {
        c_check_hash();
        assert_eq!(zig_check_hash(), 0);
    };
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    struct Budget {
        remaining: usize,
        live: usize,
    }
    unsafe extern "C" fn allocate(ctx: *mut c_void, size: usize, align: usize) -> *mut c_void {
        let budget = unsafe { &mut *ctx.cast::<Budget>() };
        if budget.remaining == 0 {
            return ptr::null_mut();
        }
        let p = unsafe { alloc::alloc::alloc(Layout::from_size_align(size, align).unwrap()) };
        if !p.is_null() {
            budget.remaining -= 1;
            budget.live += 1;
        }
        p.cast()
    }
    unsafe extern "C" fn deallocate(ctx: *mut c_void, p: *mut c_void, size: usize, align: usize) {
        unsafe {
            (*ctx.cast::<Budget>()).live -= 1;
        }
        unsafe { alloc::alloc::dealloc(p.cast(), Layout::from_size_align(size, align).unwrap()) };
    }
    unsafe extern "C" fn equal(a: *const c_void, b: *const c_void, _: *mut c_void) -> i32 {
        i32::from(unsafe { *a.cast::<u64>() == *b.cast::<u64>() })
    }
    #[test]
    fn collisions_oom_and_callback_ownership() {
        let mut budget = Budget {
            remaining: 100,
            live: 0,
        };
        let budget_ptr = &raw mut budget;
        let allocator = KoreHashAllocator {
            context: budget_ptr.cast(),
            allocate: Some(allocate),
            deallocate: Some(deallocate),
        };
        let mut values = [0u64; 64];
        let mut out = ptr::null_mut();
        // SAFETY: allocations and callback context outlive the table; entries
        // remain at stable addresses and are never changed while stored.
        unsafe {
            let table = kore_hash_table_create(allocator.clone(), 0);
            assert!(!table.is_null());
            for (i, v) in values.iter_mut().enumerate() {
                *v = i as u64;
                assert_eq!(
                    kore_hash_table_insert(
                        table,
                        0,
                        (v as *mut u64).cast(),
                        Some(equal),
                        ptr::null_mut(),
                        &mut out
                    ),
                    KORE_HASH_OK
                );
            }
            (*budget_ptr).remaining = 0;
            assert_eq!(kore_hash_table_reserve(table, 10000), KORE_HASH_NO_MEMORY);
            let query = 42u64;
            assert_eq!(
                kore_hash_table_find(
                    table,
                    0,
                    (&raw const query).cast(),
                    Some(equal),
                    ptr::null_mut(),
                    &mut out
                ),
                KORE_HASH_OK
            );
            assert_eq!(*out.cast::<u64>(), 42);
            assert_eq!(
                kore_hash_table_remove(
                    table,
                    0,
                    (&raw const query).cast(),
                    Some(equal),
                    ptr::null_mut(),
                    &mut out
                ),
                KORE_HASH_OK
            );
            assert_eq!(kore_hash_table_len(table), 63);
            kore_hash_table_destroy(table);
            assert_eq!((*budget_ptr).live, 0);
            (*budget_ptr).remaining = 1;
            assert!(kore_hash_table_create(allocator, 8).is_null());
            assert_eq!((*budget_ptr).live, 0);
        }
    }
}
