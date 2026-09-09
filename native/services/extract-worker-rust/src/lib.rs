//! Pure-Rust format worker adapters.
//!
//! The broker contract remains in `kore-svc-extract`; this crate is the
//! parser-bearing address-space payload and owns all parser dependencies.

#![no_std]

#[cfg(test)]
extern crate std;

pub use kore_svc_extract::{registry, schema};

mod coverage;
pub mod formats;

pub use coverage::production_coverage;
pub use formats::{
    AdapterError, parse_bmp, parse_csv, parse_epub, parse_json_rust, parse_plain_text, parse_tar,
    parse_ttf, parse_xml,
};
