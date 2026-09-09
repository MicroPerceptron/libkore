const kore = @import("kore");
const c = kore.abi;

comptime {
    if (@sizeOf(c.KoreFsHandle) != 16) @compileError("handle ABI");
    if (@sizeOf(c.KoreFsCompletion) != 80) @compileError("completion ABI");
    if (@offsetOf(c.KoreFsCompletion, "count") != 32) @compileError("completion ABI");
}

export fn zig_submit_create(ctx: ?*c.KoreFsContext, parent: c.KoreFsHandle) callconv(.c) u32 {
    const name = "zig-file";
    if (kore.fs.create(ctx, parent, name.ptr, name.len, c.KORE_FS_FILE) != c.KORE_STATUS_QUEUED) return 1;
    if (kore.fs.close(ctx, parent) != c.KORE_STATUS_BUSY) return 2;
    return 0;
}

export fn zig_task_step(_: ?*anyopaque) callconv(.c) c.KoreTaskAction {
    return .{ .operation = c.KORE_TASK_DONE, .reserved = 0, .nanoseconds = 0 };
}
export fn zig_prepare_task(batch: ?*c.KoreTaskBatch, handle: *c.KoreTaskHandle) callconv(.c) u32 {
    return kore.task.prepare(batch, zig_task_step, null, handle);
}
export fn zig_check_random() callconv(.c) u32 {
    var output: u64 = 88;
    if (kore.random.fill(null, 0) != c.KORE_RANDOM_STATUS_OK) return 1;
    if (kore.random.range(1, 1, &output) != c.KORE_RANDOM_STATUS_INVALID_RANGE) return 2;
    return if (output == 88) 0 else 3;
}

const HashArena = struct {
    bytes: [4096]u8 align(64) = undefined,
    used: usize = 0,
    live: usize = 0,
};
fn hashAlloc(ctx: ?*anyopaque, size: usize, alignment: usize) callconv(.c) ?*anyopaque {
    const arena: *HashArena = @ptrCast(@alignCast(ctx.?));
    const base = @intFromPtr(&arena.bytes);
    const address = (base + arena.used + alignment - 1) & ~(alignment - 1);
    const offset = address - base;
    if (offset > arena.bytes.len or size > arena.bytes.len - offset) return null;
    arena.used = offset + size;
    arena.live += 1;
    return @ptrCast(&arena.bytes[offset]);
}
fn hashFree(ctx: ?*anyopaque, _: ?*anyopaque, _: usize, _: usize) callconv(.c) void {
    const arena: *HashArena = @ptrCast(@alignCast(ctx.?));
    arena.live -= 1;
}
fn hashEqual(a: ?*const anyopaque, b: ?*const anyopaque, _: ?*anyopaque) callconv(.c) i32 {
    const left: *const u64 = @ptrCast(@alignCast(a.?));
    const right: *const u64 = @ptrCast(@alignCast(b.?));
    return @intFromBool(left.* == right.*);
}
export fn zig_check_hash() callconv(.c) u32 {
    var arena = HashArena{};
    const table = kore.hash.table_create(.{ .context = &arena, .allocate = hashAlloc, .deallocate = hashFree }, 2) orelse return 1;
    var value: u64 = 42;
    var out: ?*anyopaque = null;
    if (kore.hash.table_insert(table, 7, &value, hashEqual, null, &out) != c.KORE_HASH_OK) return 2;
    if (kore.hash.table_find(table, 7, &value, hashEqual, null, &out) != c.KORE_HASH_OK) return 3;
    if (out != @as(?*anyopaque, @ptrCast(&value))) return 4;
    kore.hash.table_destroy(table);
    return if (arena.live == 0) 0 else 5;
}
