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
pub const time = struct {
    pub const uptime = abi.kore_time_uptime;
    pub const wall = abi.kore_time_wall;
};
