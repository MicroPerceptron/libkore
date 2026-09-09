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
	cargo fmt -p kore-std -p libkore-consumer --check

docs:
	cargo doc --locked --workspace --all-features
	RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --all-features --no-deps

owners:
	@mkdir -p target
	cargo metadata --locked --all-features --format-version 1 > target/owners.json
	python3 tools/check-owners.py < target/owners.json

# Actual Kore user targets, including its hardware-float x86 ABI. A stock
# x86_64-unknown-none check alone cannot certify this interface.
cross:
	@set -eu; for arch in x86_64 aarch64 riscv64gc; do \
	  cargo check --locked -p kore-std --all-features \
	    --target "targets/$$arch-unknown-kore-user.json" \
	    -Z build-std=core,alloc,compiler_builtins \
	    -Z build-std-features=compiler-builtins-mem; \
	done

miri:
	cargo miri test --locked --workspace --all-features
	cargo miri test --locked --manifest-path native/Cargo.toml -p kore-rt erased_poll_and_drop

native-check:
	python3 tools/check-native.py
