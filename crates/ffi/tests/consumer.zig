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
