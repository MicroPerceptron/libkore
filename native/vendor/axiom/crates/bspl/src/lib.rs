#![no_std]

#[cfg(test)]
extern crate alloc;
#[cfg(test)]
extern crate std;

mod constants;
mod error;
mod format;
mod lz4;
mod model;
mod parse;
mod read;

#[cfg(test)]
mod tests;

pub use error::Error;
pub use format::PixelFormat;
pub use lz4::decompress_sparse_lz4_payload;
pub use model::{Animation, AnimationFrame, AnimationFrameData, AnimationFrames, Encoding, Header};
pub use parse::{dirty_tile_required_words, parse};
