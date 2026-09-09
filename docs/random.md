# Random values without manual seed management

```rust
# use kore_std as kore;
use kore::random::{random, SystemRng, RandomError};
fn example() -> Result<(), RandomError> {
    let id: u128 = random()?;
    let token: [u8; 32] = random()?;
    let mut rng = SystemRng::new();
    let sample: f64 = rng.random()?;
    let index = rng.random_range(0..100)?;
    let mut bytes = [0; 128];
    rng.fill(&mut bytes)?;
    # let _ = (id, token, sample, index);
    Ok(())
}
```

`random::<T>()`, `fill` and `random_range` seed automatically. Keep `SystemRng`
for repeated draws: it retains a private ChaCha20 generator, with no global lock,
TLS, allocation or shared stream. The API is inspired by standard-library typed
random generation; it does not bind callers to Rust's still-experimental
[`std::random`](https://doc.rust-lang.org/std/random/index.html) traits. Kore keeps
fallible operations because an unsupported or unavailable entropy source must
remain a reported refusal, not a fabricated seed or an implicit panic.

The implementation uses maintained RustCrypto
[`ChaCha20Rng`](https://docs.rs/chacha20/0.10.2/chacha20/struct.ChaCha20Rng.html)
with its `rng` and `zeroize` features; no cipher implementation is copied here.
All integers, bool, f32, f64, and fixed byte arrays are supported. Floats lie in
[0, 1) at their mantissa precision. The u64 half-open range API uses rejection
sampling, with a bounded attempt count and explicit InvalidRange or
SamplingExhausted results. It never substitutes biased modulo output.

The generator lazily obtains 32 bytes from the existing native entropy source.
Before a request would exceed its 64 KiB output budget, it discards the old stream
and obtains a fresh seed. A single larger fill completes under one fresh seed
and forces reseeding before the next nonempty call. Entropy is never requested
partway through publication: **any fill error leaves the entire output buffer
unchanged**. Empty fills need no entropy. A failed reseed does not fall back to
an old stream. Seed temporaries and retained generator state zeroize on drop;
this is not a claim to erase every compiler/register copy or to survive process
memory disclosure. The wrapper offers no Clone or state serialization. After
snapshot/restore, discard an existing owner and construct a fresh one.

`random::available()` reports native source support, not a guarantee that the next
hardware draw succeeds. x86_64 uses feature-checked RDRAND, aarch64 requires the
kernel-stamped CPU entropy feature, and unsupported sources (currently including
RISC-V userland) still refuse. Hardware draw retries remain bounded. The raw
source remains available explicitly as `random::entropy`; unlike the high-level
fill, its low-level fill can modify a prefix before failure.

C/Zig use `kore_random_fill`, `kore_random_u64`, and `kore_random_range` with typed
status results and unchanged output on failure. For repeated fills, the Rust host
can provide a private `KoreRandom::new()` owner to `kore_random_fill_from`; this
avoids seeding each free-function call. Calls require exclusive access to that
owner and disjoint caller buffers. There is no implicit per-thread registry or
opaque allocation to free. Zig's generated namespace is `kore.random`.

Validation includes the standard zero-key ChaCha known-answer prefix, seeded
primitive/range checks, bounded rejection, failed-seed/reseed atomic output, empty
and large-fill policy, linked C/Zig error paths, Miri, and actual Kore target
compilation. These test generator/API contracts; they do not certify hardware
entropy quality or add an entropy source to unsupported machines.
