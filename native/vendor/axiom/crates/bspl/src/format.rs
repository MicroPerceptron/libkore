#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    Rgba8 = 0,
    Rgb888 = 1,
    Bgr888 = 2,
    Rgb565 = 3,
    Bgr565 = 4,
    Xrgb8888 = 5,
}

impl PixelFormat {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Rgba8),
            1 => Some(Self::Rgb888),
            2 => Some(Self::Bgr888),
            3 => Some(Self::Rgb565),
            4 => Some(Self::Bgr565),
            5 => Some(Self::Xrgb8888),
            _ => None,
        }
    }

    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 | Self::Xrgb8888 => 4,
            Self::Rgb888 | Self::Bgr888 => 3,
            Self::Rgb565 | Self::Bgr565 => 2,
        }
    }

    #[inline]
    pub fn decode_rgb(self, pixel: &[u8]) -> (u8, u8, u8) {
        match self {
            Self::Rgba8 | Self::Rgb888 => (pixel[0], pixel[1], pixel[2]),
            Self::Bgr888 => (pixel[2], pixel[1], pixel[0]),
            Self::Rgb565 => {
                let value = u16::from_le_bytes([pixel[0], pixel[1]]);
                let r5 = (value >> 11) as u8;
                let g6 = ((value >> 5) & 0x3F) as u8;
                let b5 = (value & 0x1F) as u8;
                (r5 << 3 | r5 >> 2, g6 << 2 | g6 >> 4, b5 << 3 | b5 >> 2)
            }
            Self::Bgr565 => {
                let value = u16::from_le_bytes([pixel[0], pixel[1]]);
                let b5 = (value >> 11) as u8;
                let g6 = ((value >> 5) & 0x3F) as u8;
                let r5 = (value & 0x1F) as u8;
                (r5 << 3 | r5 >> 2, g6 << 2 | g6 >> 4, b5 << 3 | b5 >> 2)
            }
            Self::Xrgb8888 => (pixel[2], pixel[1], pixel[0]),
        }
    }
}
