//! Executable-only facade. Name this dependency `std` for an ordinary `fn main()`.
//! Reusable libraries depend on `kore-std`; services keep their existing entry.
#![no_std]

pub use core::*;
pub use kore::*;
// These names overlap core's glob; select the application-facing APIs.
pub use kore::{process, random, task, time};
pub use kore_start as startup;

extern crate alloc as rust_alloc;
#[cfg(target_os = "none")]
#[global_allocator]
static PROCESS_HEAP: kore::runtime::heap::Heap<kore::runtime::heap::HeapSupplySource> =
    kore::runtime::heap::Heap::new_in(kore::runtime::heap::HeapSupplySource);

pub use rust_alloc::{alloc, boxed, format, string, vec};

/// Rust selects the edition-specific prelude when it injects `extern crate std`.
/// This supplies only the APIs implemented by Kore and core/alloc.
pub mod prelude {
    pub use kore::prelude::*;
    pub mod v1 {
        pub use core::prelude::v1::*;
        pub use rust_alloc::{
            boxed::Box,
            format,
            string::{String, ToString},
            vec,
            vec::Vec,
        };
    }
    pub mod rust_2015 {
        pub use super::v1::*;
    }
    pub mod rust_2018 {
        pub use super::v1::*;
    }
    pub mod rust_2021 {
        pub use super::v1::*;
        pub use core::prelude::rust_2021::*;
    }
    pub mod rust_2024 {
        pub use super::v1::*;
        pub use core::prelude::rust_2024::*;
    }
}
