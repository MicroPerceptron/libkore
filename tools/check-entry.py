#!/usr/bin/env python3
"""Link ordinary Rust main through the public facade on every Kore user ISA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ARCHES = ("x86_64", "aarch64", "riscv64gc")

GOOD = {
    "unit": "fn main() {}",
    "typed": "fn main() -> std::ExitCode { std::ExitCode::APPLICATION_15 }",
    "result": "fn main() -> Result<(), std::ExitCode> { Err(std::ExitCode::LINK_FAILED) }",
    "never": "fn main() -> ! { loop { core::hint::spin_loop(); } }",
    "custom": """
struct Outcome;
impl std::Termination for Outcome {
    fn report(self) -> std::ExitCode { std::ExitCode::OK }
}
fn main() -> Outcome { Outcome }
""",
    "heap": """
fn main() -> Result<(), std::ExitCode> {
    let mut v: Vec<String> = vec![String::from("kore")];
    v.push("main".to_string());
    let boxed = Box::new(v);
    let text = std::format!("{}", boxed.len());
    if text == "2" { Ok(()) } else { Err(std::ExitCode::INTERNAL) }
}
""",
}
BAD = {
    "integer": "fn main() -> i32 { 0 }",
    "integer_error": "fn main() -> Result<(), i32> { Err(256) }",
    "integer_report": """
struct Outcome;
impl std::Termination for Outcome { fn report(self) -> i32 { 0 } }
fn main() -> Outcome { Outcome }
""",
}


def main(arches):
    channel = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    env = dict(os.environ, RUSTUP_TOOLCHAIN=channel,
               CARGO_TARGET_DIR=str(ROOT / "target/entry-consumer"), CARGO_INCREMENTAL="0")
    with tempfile.TemporaryDirectory(prefix="libkore-entry-") as temporary:
        work = Path(temporary)
        (work / "src/bin").mkdir(parents=True)
        (work / "Cargo.toml").write_text(
            '[package]\nname="entry-consumer"\nversion="0.0.0"\nedition="2024"\n'
            '[workspace]\n[dependencies.std]\npackage="kore-entry"\npath='
            + json.dumps(str(ROOT / "crates/entry")) + '\n'
        )
        # Reuse the reviewed dependency versions while Cargo adds this fixture.
        (work / "Cargo.lock").write_bytes((ROOT / "Cargo.lock").read_bytes())
        for name, source in GOOD.items():
            (work / f"src/bin/{name}.rs").write_text(source)
        for arch in arches:
            command = ["cargo", "build", "--release", "-Zjson-target-spec",
                       "--target", str(ROOT / f"targets/{arch}-unknown-kore-user.json"),
                       "-Zbuild-std=core,alloc,compiler_builtins",
                       "-Zbuild-std-features=compiler-builtins-mem"]
            subprocess.run(command + ["--bins"], cwd=work, env=env, check=True)
            for name, source in BAD.items():
                path = work / f"src/bin/{name}.rs"
                path.write_text(source)
                result = subprocess.run(command + ["--bin", name], cwd=work, env=env,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                path.unlink()
                if result.returncode == 0 or not any(
                    marker in result.stdout for marker in ("E0277", "E0053")
                ):
                    raise RuntimeError(f"{arch}/{name}: expected a return-contract rejection\n{result.stdout}")
            print(f"entry: {arch}: {len(GOOD)} linked programs; {len(BAD)} invalid return types rejected", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHES, help="check one CI target instead of all targets")
    args = parser.parse_args()
    main((args.arch,) if args.arch else ARCHES)
