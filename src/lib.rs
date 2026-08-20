#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(slice_ptr_get)]

pub mod driver;
pub mod error;
pub mod linker;
pub mod platform;
pub mod prelude;
pub mod ring;
pub mod runtime;
pub mod service;
pub mod system;
pub mod utils;
