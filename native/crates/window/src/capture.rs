//! The screenshot vocabulary (`res/SCREENSHOTS.md`): what a capture asks
//! for, what a frozen frame is, and why one was refused. Data only; the
//! authority is the read-only lend on the capture writer's own pair.

use crate::appearance::Density;

/// Where screenshots are written, from the home's root. The writer
/// requires it to exist; nothing creates it for the user.
pub const FOLDER: &str = "pictures/screenshots";

/// The writer's one launch argument, `--app <name>`: the program a window
/// capture's window runs, as its controller named it, for the file's text
/// and its line. Inert words; the frame is the lend's.
pub const APP_FLAG: &str = "--app";

/// What a capture asks for: a PrintScreen chord (`res/KEYBOARD.md` §5) or
/// the `screenshot` command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CaptureMode {
    /// The whole screen the seat is on.
    Screen = 1,
    /// One window's own content, covered or minimized: the seat's focused
    /// window, or the one the request names.
    Window = 2,
    /// A region the user selects over the frozen screen.
    Region = 3,
}

impl CaptureMode {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Screen),
            2 => Some(Self::Window),
            3 => Some(Self::Region),
            _ => None,
        }
    }

    /// The word a file name and its metadata use.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Screen => "screen",
            Self::Window => "window",
            Self::Region => "region",
        }
    }
}

/// One frozen frame, as the compositor lends it to the capture writer.
///
/// The lent view holds `height` rows of `width` XRGB8888 pixels, packed
/// (`width * 4` bytes a row), at the head of `bytes`. The high byte of each
/// pixel is not part of the image: the compositor's painters leave it
/// `0xff` in places and zero in others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureFrame {
    pub width: u16,
    pub height: u16,
    pub mode: CaptureMode,
    /// The screen it was frozen from (`0` is the primary).
    pub screen: u8,
    /// The UI density the screen was drawn at.
    pub density: Density,
    /// Nanoseconds since the UNIX epoch at the freeze, or `None` while the
    /// clock has no wall anchor.
    pub wall_ns: Option<u64>,
    /// Nanoseconds since boot at the freeze.
    pub uptime_ns: u64,
    /// What the freeze cost the compositor's pass: the frame's allocation
    /// and copy, in nanoseconds. Every shot carries the measurement of its
    /// own (`res/SCREENSHOTS.md`, Budgets).
    pub freeze_ns: u64,
    /// Where the image's top-left pixel sits on its screen: the selected
    /// region's origin, or the window's client aperture's (its last
    /// placement while minimized). `(0, 0)` for a whole screen.
    pub x: u16,
    pub y: u16,
    /// Whether the pixels are a frame presented for this capture. A window
    /// that presents nothing within the bound is copied as its backing
    /// stood, which may be mid-paint: `false`, and the file says so. A
    /// screen or region is the compositor's own complete frame: `true`.
    pub synced: bool,
}

impl CaptureFrame {
    /// The bytes the pixels occupy: `width * height * 4`.
    pub const fn pixel_bytes(&self) -> u64 {
        self.width as u64 * self.height as u64 * 4
    }

    /// The lend's length: the pixels in whole pages, so the frozen
    /// allocation it maps is page-exclusive.
    pub const fn bytes(&self) -> u64 {
        self.pixel_bytes().next_multiple_of(4096)
    }

    /// A frame with pixels to lend.
    pub const fn is_valid(&self) -> bool {
        self.width != 0 && self.height != 0
    }
}

/// Why a capture did not happen. The controller says so on the seat's
/// terminal; a capture is never dropped silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CaptureRefusal {
    /// This seat's previous capture is still being written.
    Busy = 1,
    /// The compositor had no room for the frozen frame.
    OutOfMemory = 2,
    /// The seat has no screen to capture.
    Unavailable = 3,
    // 4 was `Unsupported`, a mode not yet built, until window and region
    // capture landed; it decodes to nothing.
    /// A window capture found no window of this seat to take: nothing is
    /// focused, the named window is gone or another seat's, or it has never
    /// presented a frame.
    NoWindow = 5,
    /// The user cancelled the selection.
    Cancelled = 6,
}

impl CaptureRefusal {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Busy),
            2 => Some(Self::OutOfMemory),
            3 => Some(Self::Unavailable),
            5 => Some(Self::NoWindow),
            6 => Some(Self::Cancelled),
            _ => None,
        }
    }

    /// The reason, as the seat's terminal shows it.
    pub const fn describe(self) -> &'static str {
        match self {
            Self::Busy => "the previous screenshot is still being saved",
            Self::OutOfMemory => "no memory for the frozen screen",
            Self::Unavailable => "this seat has no screen",
            Self::NoWindow => "there is no window to capture",
            Self::Cancelled => "cancelled",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_and_zero_is_none() {
        for mode in [CaptureMode::Screen, CaptureMode::Window, CaptureMode::Region] {
            assert_eq!(CaptureMode::from_code(mode.code()), Some(mode));
        }
        for refusal in [
            CaptureRefusal::Busy,
            CaptureRefusal::OutOfMemory,
            CaptureRefusal::Unavailable,
            CaptureRefusal::NoWindow,
            CaptureRefusal::Cancelled,
        ] {
            assert_eq!(CaptureRefusal::from_code(refusal.code()), Some(refusal));
            assert!(!refusal.describe().is_empty());
        }
        for code in [0, 4, 0xff] {
            assert_eq!(CaptureMode::from_code(code), None);
        }
        for code in [0, 4, 7, 0xff] {
            assert_eq!(CaptureRefusal::from_code(code), None);
        }
    }

    #[test]
    fn a_frame_lends_its_pixels_in_whole_pages() {
        let frame = CaptureFrame {
            width: 1280,
            height: 800,
            mode: CaptureMode::Screen,
            screen: 0,
            density: Density::Native,
            wall_ns: None,
            uptime_ns: 1,
            freeze_ns: 0,
            x: 0,
            y: 0,
            synced: true,
        };
        assert!(frame.is_valid());
        assert_eq!(frame.bytes(), 1280 * 800 * 4, "already whole pages");
        let odd = CaptureFrame { width: 33, height: 7, ..frame };
        assert_eq!((odd.pixel_bytes(), odd.bytes()), (33 * 7 * 4, 4096));
        assert!(!CaptureFrame { width: 0, ..frame }.is_valid());
        assert!(!CaptureFrame { height: 0, ..frame }.is_valid());
    }
}
