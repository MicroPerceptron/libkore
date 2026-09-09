// Generated from kore.h. Signatures come from Zig translate-c at build time.
pub const abi = @import("kore-c");
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
    pub const run = abi.kore_thread_run;
};
pub const time = struct {
    pub const sleep_for = abi.kore_time_sleep_for;
    pub const sleep_until = abi.kore_time_sleep_until;
    pub const uptime = abi.kore_time_uptime;
    pub const wall = abi.kore_time_wall;
    pub const yield = abi.kore_time_yield;
};
