const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const c = b.addTranslateC(.{
        .root_source_file = b.path("extern/c/include/kore.h"),
        .target = target,
        .optimize = optimize,
    });
    const kore = b.addModule("kore", .{
        .root_source_file = b.path("extern/zig/src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "kore-c", .module = c.createModule() }},
    });
    const consumer = b.addObject(.{
        .name = "kore-zig-consumer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("crates/ffi/tests/consumer.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "kore", .module = kore }},
        }),
    });
    b.step("check", "Compile the translated module and its consumer").dependOn(&consumer.step);
}
