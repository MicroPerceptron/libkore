#!/usr/bin/env python3
"""Check real renamed-dependency executables in each selectable facade mode.

Three modes, because each is a different final link and only a real executable
proves it:

  runtime   default-features = false        -> no allocator required
  fs        default-features = false + fs   -> allocator required (see below)
  full      default features                -> collections, allocator required
  exec      default-features = false + exec -> invocation surface, allocator required

The dependency is renamed (`sdk = { package = "kore-std", .. }`) on purpose: a
consumer that reaches the library under its own chosen name cannot be relying on
the crate name leaking through a macro or a path.

`runtime` must link with no global allocator at all. That is the property the
allocation-free consumer needs and the one this check exists to defend.

`full` selects `collections`, which pulls `hashbrown`, so that fixture supplies
its own allocator — proving the heap requirement is the *consumer's* to satisfy
and is not forced by the facade on someone who never asked for collections.

`fs` also requires one today, and that is a measured fact rather than an
intended design: selecting `fs` pulls `kore-dagfs/runtime`, which allocates, so
an fs-only consumer cannot link without a global allocator. Making a filesystem
consumer allocation-free is a change to `kore-dagfs` — a Kore owner — and under
`docs/integration.md` it has to be made there and re-exported, never patched
into the exported copy. The fixture therefore supplies an allocator for this
mode and the check pins the *current* requirement, so the day `kore-dagfs` stops
allocating this file is what has to be updated to notice.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


MODES = {
    # name: (cargo feature spec, needs a global allocator)
    "runtime": ("default-features=false", False),
    "fs": ('default-features=false, features=["fs"]', True),
    "full": ("", True),
    # libkore #18's surface. It pulls `fs` (the request vocabulary names
    # filesystem rights), so it inherits that mode's allocator requirement.
    "exec": ('default-features=false, features=["exec"]', True),
}

ALLOCATOR = """
extern crate alloc;
struct Bump;
// SAFETY: the fixture never allocates; this satisfies the final link only.
unsafe impl core::alloc::GlobalAlloc for Bump {
    unsafe fn alloc(&self, _: core::alloc::Layout) -> *mut u8 { core::ptr::null_mut() }
    unsafe fn dealloc(&self, _: *mut u8, _: core::alloc::Layout) {}
}
#[global_allocator]
static ALLOC: Bump = Bump;
"""

BODY = {
    "runtime": "let _ = (sdk::time::uptime(), sdk::time::wall());",
    "fs": "let _ = (sdk::time::uptime(), core::mem::size_of::<sdk::fs::Fs>());",
    "full": "let _ = (sdk::time::uptime(), core::mem::size_of::<sdk::collections::HashSet<u32>>());",
    # A real consumer names an executable it holds and asks the protocol how
    # many grant channels it must pre-pin. Neither is authority; both are the
    # surface libkore #18 exists to expose.
    "exec": "let ns = sdk::exec::CapabilityNamespace::<u32, 2>::new();\n    let _ = (ns.metadata().count(), sdk::exec::EXEC_REPLY_GRANTS);",
}


def check(arch: str, mode: str) -> None:
    spec, needs_allocator = MODES[mode]
    with tempfile.TemporaryDirectory(prefix="libkore-heapless-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        dependency = 'sdk = { package="kore-std", path=' + json.dumps(str(ROOT))
        if spec:
            dependency += ", " + spec
        dependency += " }\n"
        (work / "Cargo.toml").write_text(
            '[package]\nname="libkore-' + mode + '-consumer"\nversion="0.0.0"\nedition="2024"\n'
            '[workspace]\n[dependencies]\n' + dependency
        )
        (work / "src/main.rs").write_text('#![no_std]\n#![no_main]\n'
            + (ALLOCATOR if needs_allocator else "")
            + '''
#[unsafe(no_mangle)]
pub extern "C" fn _start(_: usize) -> ! {
    ''' + BODY[mode] + '''
    loop { core::hint::spin_loop(); }
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop { core::hint::spin_loop(); }
}
''')
        toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
        env = dict(os.environ, RUSTUP_TOOLCHAIN=toolchain)
        env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target/heapless-consumer"))
        subprocess.run([
            "cargo", "check", "--target", str(ROOT / f"targets/{arch}-unknown-kore-user.json"),
            "-Z", "json-target-spec", "-Z", "build-std=core,alloc,compiler_builtins",
            "-Z", "build-std-features=compiler-builtins-mem",
        ], cwd=work, env=env, check=True)
        allocator = "with a consumer allocator" if needs_allocator else "allocation-free"
        print(f"{mode} executable ({allocator}): {arch}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["x86_64", "aarch64", "riscv64gc"])
    parser.add_argument("--mode", choices=sorted(MODES))
    args = parser.parse_args()
    for arch in ([args.arch] if args.arch else ["x86_64", "aarch64", "riscv64gc"]):
        for mode in ([args.mode] if args.mode else ["runtime", "fs", "full", "exec"]):
            check(arch, mode)
