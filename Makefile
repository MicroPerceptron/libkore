ZIG ?= zig
export ZIG

.PHONY: test check fmt docs owners cross miri native-check

test:
	cargo test --locked --workspace --all-features
	cargo test --locked --manifest-path native/Cargo.toml -p kore-rt -p kore-dagfs --features kore-dagfs/runtime
	python3 -m unittest discover -s tools -p 'test_*.py'

check:
	cargo check --locked -p kore-std --no-default-features
	cargo clippy --locked --workspace --all-targets --all-features --no-deps -- -D warnings
	python3 tools/check-native.py

fmt:
	cargo fmt -p kore-std -p libkore-consumer -p kore-ffi -p kore-entry -p kore-wasi --check
	$(ZIG) fmt --check build.zig build.zig.zon crates/ffi/tests/consumer.zig extern/zig/src/root.zig extern/zig/src/allocator.zig

docs:
	cargo doc --locked --workspace --all-features
	RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --all-features --no-deps

owners:
	@mkdir -p target
	cargo metadata --locked --all-features --format-version 1 > target/owners.json
	python3 tools/check-owners.py < target/owners.json

# Actual Kore user targets, including its hardware-float x86 ABI. A stock
# x86_64-unknown-none check alone cannot certify this interface. `kore-wasi`
# is absent: Wasmtime on a Kore target needs Kore's vendored patches
# (target-lexicon's triple, the no_std compiler split, the fiber switch), so
# its Kore build is certified in the consumer that carries them — Kore's
# userland workspace, whose `keytest-wasm-x86` runs the fixture on Kore.
cross:
	@set -eu; for arch in x86_64 aarch64 riscv64gc; do \
	  cargo check --locked -p kore-std -p kore-ffi -p kore-entry --all-features \
	    --target "targets/$$arch-unknown-kore-user.json" \
	    -Z build-std=core,alloc,compiler_builtins \
	    -Z build-std-features=compiler-builtins-mem; \
	done
	python3 tools/check-alloc-symbols.py
	python3 tools/check-entry.py

miri:
	cargo miri test --locked --workspace --all-features --exclude kore-wasi
	cargo miri test --locked --manifest-path native/Cargo.toml -p kore-rt erased_poll_and_drop

native-check:
	python3 tools/check-native.py

.PHONY: bindings bindings-check
bindings:
	python3 tools/generate-bindings.py

bindings-check:
	python3 tools/check-bindings.py
