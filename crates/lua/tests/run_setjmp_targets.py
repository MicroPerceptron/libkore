#!/usr/bin/env python3
"""Boot the vendored AArch64 and RV64 setjmp seams under full-system QEMU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


SUPPORTED_ZIG_LINE = (0, 16)
SUCCESS = "setjmp-runtime: ok"


def zig_release_line(version: str) -> tuple[int, int] | None:
    release = version.split("-", 1)[0].split("+", 1)[0]
    parts = release.split(".")
    if len(parts) != 3:
        return None
    try:
        major, minor, _patch = (int(part) for part in parts)
    except ValueError:
        return None
    return major, minor


def resolve_tool(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise SystemExit(f"required tool not found: {value}")
    return resolved


def execute(
    command: list[str], *, environment: dict[str, str] | None = None
) -> str:
    print("+", shlex.join(command), flush=True)
    try:
        result = subprocess.run(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, end="")
        raise SystemExit(f"command timed out: {shlex.join(command)}") from error
    print(result.stdout, end="")
    if result.returncode != 0:
        raise SystemExit(
            f"command exited with status {result.returncode}: "
            f"{shlex.join(command)}"
        )
    return result.stdout


def compile_probe(
    *,
    zig: str,
    lua_dir: Path,
    output_dir: Path,
    architecture: str,
) -> Path:
    probe_dir = lua_dir / "tests" / "setjmp-runtime"
    if architecture == "aarch64":
        target = "aarch64-linux-musl"
        start = probe_dir / "aarch64-start.s"
        linker = probe_dir / "aarch64.ld"
        assembly_dir = lua_dir / "vendor" / "musl-setjmp" / "aarch64"
        assembly = [assembly_dir / "setjmp.s", assembly_dir / "longjmp.s"]
        target_flags: list[str] = []
    else:
        target = "riscv64-linux-musl"
        start = probe_dir / "riscv64-start.S"
        linker = probe_dir / "riscv64.ld"
        assembly_dir = lua_dir / "vendor" / "musl-setjmp" / "riscv64"
        assembly = [assembly_dir / "setjmp.S", assembly_dir / "longjmp.S"]
        target_flags = ["-mabi=lp64d", "-mcmodel=medany"]

    output = output_dir / f"setjmp-{architecture}.elf"
    harness = output_dir / f"setjmp-harness-{architecture}.o"
    environment = os.environ.copy()
    environment["ZIG_LOCAL_CACHE_DIR"] = str(output_dir / f"zig-cache-{architecture}")
    environment["ZIG_GLOBAL_CACHE_DIR"] = str(
        output_dir / f"zig-global-cache-{architecture}"
    )
    common = [
        zig,
        "cc",
        "-target",
        target,
        *target_flags,
        "-fPIC",
    ]
    compile_harness = [
        *common,
        "-O2",
        "-ffreestanding",
        "-fno-stack-protector",
        "-fno-builtin",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-command-line-argument",
        "-c",
        str(probe_dir / "harness.c"),
        "-o",
        str(harness),
    ]
    execute(compile_harness, environment=environment)

    link_probe = [
        *common,
        "-nostdlib",
        "-static",
        "-Wno-unused-command-line-argument",
        "-T",
        str(linker),
        str(start),
        str(harness),
        *(str(path) for path in assembly),
        "-o",
        str(output),
    ]
    execute(link_probe, environment=environment)
    return output


def boot_aarch64(qemu: str, image: Path) -> None:
    output = execute(
        [
            qemu,
            "-M",
            "virt",
            "-cpu",
            "max",
            "-accel",
            "tcg",
            "-m",
            "128M",
            "-display",
            "none",
            "-monitor",
            "none",
            "-serial",
            "stdio",
            "-semihosting-config",
            "enable=on,target=native",
            "-device",
            f"loader,file={image},cpu-num=0",
        ]
    )
    if SUCCESS not in output:
        raise SystemExit("AArch64 probe exited without its success marker")


def boot_riscv64(qemu: str, image: Path) -> None:
    output = execute(
        [
            qemu,
            "-M",
            "virt",
            "-cpu",
            "rv64",
            "-accel",
            "tcg",
            "-m",
            "128M",
            "-display",
            "none",
            "-monitor",
            "none",
            "-serial",
            "stdio",
            "-bios",
            "none",
            "-device",
            f"loader,file={image},cpu-num=0",
        ]
    )
    if SUCCESS not in output:
        raise SystemExit("RV64 probe exited without its success marker")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", default=os.environ.get("KORE_ZIG", "zig"))
    parser.add_argument("--qemu-aarch64", default="qemu-system-aarch64")
    parser.add_argument("--qemu-riscv64", default="qemu-system-riscv64")
    arguments = parser.parse_args()

    zig = resolve_tool(arguments.zig)
    qemu_aarch64 = resolve_tool(arguments.qemu_aarch64)
    qemu_riscv64 = resolve_tool(arguments.qemu_riscv64)
    version = execute([zig, "version"]).strip()
    if zig_release_line(version) != SUPPORTED_ZIG_LINE:
        raise SystemExit(
            f"expected Zig {SUPPORTED_ZIG_LINE[0]}.{SUPPORTED_ZIG_LINE[1]}.x, "
            f"found {version or '<empty>'}"
        )

    lua_dir = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="kore-lua-setjmp-") as temporary:
        output_dir = Path(temporary)
        aarch64 = compile_probe(
            zig=zig,
            lua_dir=lua_dir,
            output_dir=output_dir,
            architecture="aarch64",
        )
        riscv64 = compile_probe(
            zig=zig,
            lua_dir=lua_dir,
            output_dir=output_dir,
            architecture="riscv64",
        )
        boot_aarch64(qemu_aarch64, aarch64)
        boot_riscv64(qemu_riscv64, riscv64)

    print("Lua target setjmp runtime probes: ok")


if __name__ == "__main__":
    main()
