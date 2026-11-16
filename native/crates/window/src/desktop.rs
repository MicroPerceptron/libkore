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
/// How the pointer follows a mouse: Settings ▸ Keyboard & Pointer. A virtual
/// machine's emulated mouse follows its host's pointer instead, unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointerPreference {
    /// Faster motion moves the pointer further (libinput's adaptive profile).
    /// Off, the pointer moves in proportion to the mouse at every speed.
    pub acceleration: bool,
    pub speed: PointerSpeed,
}
impl Default for PointerPreference {
    fn default() -> Self {
        Self {
            acceleration: true,
            speed: PointerSpeed::DEFAULT,
        }
    }
}
/// The pointer speed setting, in five steps from slowest to fastest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PointerSpeed(i8);
impl PointerSpeed {
    pub const SLOWEST: Self = Self(-2);
    pub const DEFAULT: Self = Self(0);
    pub const FASTEST: Self = Self(2);
    pub const ALL: [Self; 5] = [Self(-2), Self(-1), Self(0), Self(1), Self(2)];
    /// A step from -2 (slowest) through 0 (the default) to 2 (fastest).
    pub const fn new(step: i8) -> Option<Self> {
        if step < Self::SLOWEST.0 || step > Self::FASTEST.0 {
            None
        } else {
            Some(Self(step))
        }
    }
    pub const fn step(self) -> i8 {
        self.0
    }
    /// libinput's speed adjustment for this step, in -1..=1.
    pub fn adjustment(self) -> f64 {
        f64::from(self.0) / 2.0
    }
}
impl Default for PointerSpeed {
    fn default() -> Self {
        Self::DEFAULT
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
    /// The seat's pointer preference, applied from every mouse's next report.
    Pointer(PointerPreference),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DesktopState {
    pub mode: Mode,
    pub density: Density,
    pub width: u16,
    pub height: u16,
}
