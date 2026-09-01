#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(slice_ptr_get)]

pub mod error;
pub mod platform;
pub mod prelude;
pub mod runtime;
pub mod service;
pub mod system;
pub mod utils;
