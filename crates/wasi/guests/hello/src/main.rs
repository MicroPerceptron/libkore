//! A stock `wasm32-wasip2` program: ordinary `std`, nothing kore-specific.
//!
//! Every line it prints is *computed* from an import — arguments, the clocks,
//! entropy, stdout itself — so a host that stubs an interface prints a wrong
//! line, not a missing one. It succeeds exactly when the host delivered the
//! arguments its witness passes, which is how the exit mapping is witnessed:
//! WASI 0.2's `exit` carries success or failure only, and `std` maps every
//! nonzero status to failure.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!("wasi-hello: args {args:?}");

    let started = Instant::now();
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("wasi-hello: wall clock past 2026: {}", unix > 1_767_225_600);

    // Entropy through `wasi:random` (std's hash-map keys draw from it).
    let seeded = std::collections::hash_map::RandomState::new();
    let _ = std::hash::BuildHasher::hash_one(&seeded, 97u32);

    println!("wasi-hello: monotonic clock advances: {}", started.elapsed().as_nanos() > 0);
    eprintln!("wasi-hello: this line is stderr");

    // A status the host must carry out: success only for the witness's argv.
    std::process::exit(if args == ["kore", "wasi"] { 0 } else { 1 });
}
