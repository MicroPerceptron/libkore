//! A stock `wasm32-wasip2` program using `std::fs`: ordinary `std`, nothing
//! kore-specific.
//!
//! It works in its preopen, `.`: creates `wasi-files.txt`, writes a line,
//! reads it back, finds it in the directory listing, checks its size,
//! removes it, and checks it is gone. Every line it prints is *computed*
//! from what the filesystem answered, and it succeeds exactly when every
//! step did — so a host that stubs an operation prints a wrong line and
//! fails, rather than printing nothing.

use std::fs;
use std::io::Write;

const NAME: &str = "wasi-files.txt";

fn main() {
    let line = format!("kore wasi {}\n", 40 + 57);

    let written = fs::File::create(NAME).and_then(|mut file| file.write_all(line.as_bytes()));
    println!("wasi-files: wrote {}", written.is_ok());

    let read = fs::read_to_string(NAME);
    let round_trip = read.as_deref().ok() == Some(line.as_str());
    println!("wasi-files: read back the line {round_trip}");

    let listed = fs::read_dir(".")
        .map(|entries| entries.flatten().any(|entry| entry.file_name() == NAME))
        .unwrap_or(false);
    println!("wasi-files: listed in . {listed}");

    let size = fs::metadata(NAME).map(|meta| meta.len()).unwrap_or(0);
    println!("wasi-files: size {size}");

    let removed = fs::remove_file(NAME).is_ok();
    let gone = fs::metadata(NAME).is_err();
    println!("wasi-files: removed {removed}, gone {gone}");

    let ok = written.is_ok()
        && round_trip
        && listed
        && size == line.len() as u64
        && removed
        && gone;
    std::process::exit(if ok { 0 } else { 1 });
}
