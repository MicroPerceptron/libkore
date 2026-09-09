//! std.mem.Allocator over the Rust application's installed global heap.
//! The host must install its Kore runtime heap before using this allocator.
const std = @import("std");
const c = @import("kore-c");

pub const allocator: std.mem.Allocator = .{
    .ptr = undefined,
    .vtable = &.{
        .alloc = allocate,
        .resize = std.mem.Allocator.noResize,
        .remap = remap,
        .free = free,
    },
};

fn allocate(_: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
    return @ptrCast(c.kore_alloc_aligned(len, alignment.toByteUnits()) orelse return null);
}

fn remap(_: *anyopaque, memory: []u8, _: std.mem.Alignment, new_len: usize, _: usize) ?[*]u8 {
    return @ptrCast(c.kore_alloc_realloc(memory.ptr, new_len) orelse return null);
}

fn free(_: *anyopaque, memory: []u8, _: std.mem.Alignment, _: usize) void {
    c.kore_alloc_free(memory.ptr);
}
