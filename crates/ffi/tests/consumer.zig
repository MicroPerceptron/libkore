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
