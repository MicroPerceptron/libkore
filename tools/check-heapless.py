#!/usr/bin/env python3
"""Check a real no-allocator executable, separately from heap-using consumers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check(arch: str) -> None:
    with tempfile.TemporaryDirectory(prefix="libkore-heapless-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        (work / "Cargo.toml").write_text(
            '[package]\nname="libkore-heapless-consumer"\nversion="0.0.0"\nedition="2024"\n'
            '[workspace]\n[dependencies]\nsdk = { package="kore-std", path='
            + json.dumps(str(ROOT)) + ', default-features=false }\n'
        )
        (work / "src/main.rs").write_text('''#![no_std]
#![no_main]
#[unsafe(no_mangle)]
pub extern "C" fn _start(_: usize) -> ! {
    let _ = (sdk::time::uptime(), sdk::time::wall());
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
            "-Z", "json-target-spec", "-Z", "build-std=core,compiler_builtins",
            "-Z", "build-std-features=compiler-builtins-mem",
        ], cwd=work, env=env, check=True)
        print(f"allocation-free executable: {arch}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["x86_64", "aarch64", "riscv64gc"])
    args = parser.parse_args()
    for arch in ([args.arch] if args.arch else ["x86_64", "aarch64", "riscv64gc"]):
        check(arch)
