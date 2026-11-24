use crate::error::Error;
use crate::format::PixelFormat;
use crate::parse::parse_next_frame;

/// Encoding format: how frame pixel data is stored.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Encoding {
    /// Full pixel data per frame.
    Dense = 0,
    /// Only dirty tile payloads per frame.
    Sparse = 1,
    /// LZ4-compressed dirty tile payloads.
    SparseLz4 = 2,
}

impl Encoding {
    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Dense),
            1 => Some(Self::Sparse),
            2 => Some(Self::SparseLz4),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Header {
    pub encoding: Encoding,
    pub pixel_format: PixelFormat,
    pub frame_count: u16,
    pub img_w: u32,
    pub img_h: u32,
    pub tile_w: u16,
    pub tile_h: u16,
}

impl Header {
    pub const fn tiles_x(self) -> u32 {
        self.img_w.div_ceil(self.tile_w as u32)
    }

    pub const fn tiles_y(self) -> u32 {
        self.img_h.div_ceil(self.tile_h as u32)
    }

    /// Number of u64 words needed for the dirty tile bitmask.
    pub const fn dirty_word_count(self) -> usize {
        let tile_count = self.tiles_x() as usize * self.tiles_y() as usize;
        (tile_count + 63) / 64
    }
}

/// A parsed bspl asset — always treated as an animation.
///
/// Static images are single-frame animations. Callers always iterate
/// frames regardless of source type.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Animation<'a> {
    header: Header,
    frames: &'a [u8],
}

impl<'a> Animation<'a> {
    pub const fn header(self) -> Header {
        self.header
    }

    pub(crate) const fn new(header: Header, frames: &'a [u8]) -> Self {
        Self { header, frames }
    }

    pub fn frames(self) -> AnimationFrames<'a> {
        AnimationFrames {
            header: self.header,
            bytes: self.frames,
            cursor: 0,
            remaining: self.header.frame_count as usize,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AnimationFrame<'a> {
    pub duration_us: u32,
    pub dirty_words_le: &'a [u8],
    pub data: AnimationFrameData<'a>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AnimationFrameData<'a> {
    Dense {
        pixels: &'a [u8],
    },
    Sparse {
        payload: &'a [u8],
    },
    SparseLz4 {
        compressed_payload: &'a [u8],
        raw_payload_len: usize,
    },
}

pub struct AnimationFrames<'a> {
    pub(crate) header: Header,
    pub(crate) bytes: &'a [u8],
    pub(crate) cursor: usize,
    pub(crate) remaining: usize,
}

impl<'a> Iterator for AnimationFrames<'a> {
    type Item = Result<AnimationFrame<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match parse_next_frame(self.header, self.bytes, &mut self.cursor) {
            Ok(frame) => {
                self.remaining -= 1;
                Some(Ok(frame))
            }
            Err(error) => {
                self.remaining = 0;
                Some(Err(error))
            }
        }
    }
}
