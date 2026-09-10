// Generated from kore.h. Signatures come from Zig translate-c at build time.
pub const abi = @import("kore-c");
pub const allocator = @import("allocator.zig").allocator;
pub const alloc = struct {
    pub const aligned = abi.kore_alloc_aligned;
    pub const calloc = abi.kore_alloc_calloc;
    pub const free = abi.kore_alloc_free;
    pub const hash_allocator = abi.kore_alloc_hash_allocator;
    pub const malloc = abi.kore_alloc_malloc;
    pub const realloc = abi.kore_alloc_realloc;
};
pub const dag = struct {
    pub const detach = abi.kore_dag_detach;
    pub const link = abi.kore_dag_link;
};
pub const fs = struct {
    pub const close = abi.kore_fs_close;
    pub const create = abi.kore_fs_create;
    pub const entry = abi.kore_fs_entry;
    pub const list = abi.kore_fs_list;
    pub const open = abi.kore_fs_open;
    pub const read = abi.kore_fs_read;
    pub const read_data = abi.kore_fs_read_data;
    pub const resize = abi.kore_fs_resize;
    pub const stat = abi.kore_fs_stat;
    pub const sync = abi.kore_fs_sync;
    pub const unlink = abi.kore_fs_unlink;
    pub const write = abi.kore_fs_write;
};
pub const futex = struct {
    pub const available = abi.kore_futex_available;
    pub const wait = abi.kore_futex_wait;
    pub const wake = abi.kore_futex_wake;
    pub const wake_all = abi.kore_futex_wake_all;
};
pub const hash = struct {
    pub const table_clear = abi.kore_hash_table_clear;
    pub const table_create = abi.kore_hash_table_create;
    pub const table_create_default = abi.kore_hash_table_create_default;
    pub const table_destroy = abi.kore_hash_table_destroy;
    pub const table_find = abi.kore_hash_table_find;
    pub const table_insert = abi.kore_hash_table_insert;
    pub const table_len = abi.kore_hash_table_len;
    pub const table_remove = abi.kore_hash_table_remove;
    pub const table_reserve = abi.kore_hash_table_reserve;
    pub const table_visit = abi.kore_hash_table_visit;
};
pub const random = struct {
    pub const fill = abi.kore_random_fill;
    pub const fill_from = abi.kore_random_fill_from;
    pub const range = abi.kore_random_range;
    pub const @"u64" = abi.kore_random_u64;
};
pub const spmd = struct {
    pub const dispatch = abi.kore_spmd_dispatch;
};
pub const task = struct {
    pub const poll = abi.kore_task_poll;
    pub const prepare = abi.kore_task_prepare;
    pub const release = abi.kore_task_release;
};
pub const thread = struct {
    pub const atexit = abi.kore_thread_atexit;
    pub const cancel = abi.kore_thread_cancel;
    pub const canceled = abi.kore_thread_canceled;
    pub const cleanup_pop = abi.kore_thread_cleanup_pop;
    pub const cleanup_push = abi.kore_thread_cleanup_push;
    pub const create = abi.kore_thread_create;
    pub const detach = abi.kore_thread_detach;
    pub const exit = abi.kore_thread_exit;
    pub const getspecific = abi.kore_thread_getspecific;
    pub const join = abi.kore_thread_join;
    pub const key_create = abi.kore_thread_key_create;
    pub const key_delete = abi.kore_thread_key_delete;
    pub const run = abi.kore_thread_run;
    pub const setcancelstate = abi.kore_thread_setcancelstate;
    pub const setcanceltype = abi.kore_thread_setcanceltype;
    pub const setspecific = abi.kore_thread_setspecific;
    pub const testcancel = abi.kore_thread_testcancel;
};
pub const time = struct {
    pub const sleep_for = abi.kore_time_sleep_for;
    pub const sleep_until = abi.kore_time_sleep_until;
    pub const uptime = abi.kore_time_uptime;
    pub const wall = abi.kore_time_wall;
    pub const yield = abi.kore_time_yield;
};
