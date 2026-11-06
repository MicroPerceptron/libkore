//! Settings-only desktop controls. File paths remain in the Settings process.
use crate::appearance::Density;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    Light = 0,
    #[default]
    Dark = 1,
    System = 2,
}
impl Mode {
    pub const fn from_code(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Light),
            1 => Some(Self::Dark),
            2 => Some(Self::System),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DesktopRequest {
    Query,
    /// Startup restoration finished, including fallback after unreadable preferences.
    RestoreComplete,
    Appearance {
        mode: Mode,
        density: Density,
    },
    /// One temporary shared upload, independent of the app's window size.
    CreateWallpaper {
        width: u16,
        height: u16,
    },
    /// Snapshot the owned upload into the compositor's retained background.
    ApplyWallpaper {
        surface: u64,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DesktopState {
    pub mode: Mode,
    pub density: Density,
    pub width: u16,
    pub height: u16,
}
