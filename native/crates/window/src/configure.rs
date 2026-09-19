//! Portable layout requests. These describe a window, never grant authority.

/// A content rectangle in physical display pixels, excluding future chrome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bounds {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

impl Bounds {
    pub const fn is_valid(self) -> bool {
        self.width != 0 && self.height != 0
    }

    pub(crate) const fn word(self) -> u64 {
        self.x as u64
            | (self.y as u64) << 16
            | (self.width as u64) << 32
            | (self.height as u64) << 48
    }

    pub(crate) const fn from_word(word: u64) -> Self {
        Self {
            x: word as u16,
            y: (word >> 16) as u16,
            width: (word >> 32) as u16,
            height: (word >> 48) as u16,
        }
    }
}

/// A size expressed as a fraction of the layout domain, per axis, in
/// permille (1..=1000 — the tree's fixed-point fraction, as in
/// [`crate`]-adjacent `kore_ui::metrics::blend`).
///
/// A window states *proportion*, never pixels: the same request means the
/// same thing on a 1280x800 panel and a 4K one, and it survives a screen
/// resize or a move between panels without the client re-asking. Absolute
/// limits belong in [`LayoutHints`](crate::control::LayoutHints), whose
/// `min`/`max` are pixel bounds the WM clamps the resolved rect against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fraction {
    width: u16,
    height: u16,
}

impl Fraction {
    /// Whole domain on both axes.
    pub const FULL: Self = Self {
        width: 1000,
        height: 1000,
    };

    /// `None` unless each axis is a real fraction (1..=1000). Zero would
    /// name an empty window and >1000 would name more than the domain.
    pub const fn permille(width: u16, height: u16) -> Option<Self> {
        if width == 0 || width > 1000 || height == 0 || height > 1000 {
            return None;
        }
        Some(Self { width, height })
    }

    pub const fn width_permille(self) -> u16 {
        self.width
    }

    pub const fn height_permille(self) -> u16 {
        self.height
    }

    /// This fraction of `(width, height)`, rounded to nearest, never zero.
    /// The caller supplies the domain; this type never learns a screen size.
    pub const fn resolve(self, width: u32, height: u32) -> (u32, u32) {
        (
            resolve_axis(self.width, width),
            resolve_axis(self.height, height),
        )
    }

    const fn word(self) -> u64 {
        self.width as u64 | (self.height as u64) << 16
    }

    const fn from_word(word: u64) -> Option<Self> {
        if word >> 32 != 0 {
            return None;
        }
        Self::permille(word as u16, (word >> 16) as u16)
    }
}

/// One axis of [`Fraction::resolve`]: `span * permille / 1000`, rounded to
/// nearest and floored at one pixel — a window asked for in good faith is
/// never resolved out of existence by a small domain.
const fn resolve_axis(permille: u16, span: u32) -> u32 {
    let scaled = ((span as u64 * permille as u64 + 500) / 1000) as u32;
    if scaled == 0 { 1 } else { scaled }
}

/// Compositor policy may refuse requests that cannot fit its display.
///
/// Every variant is *relative*: a preference is a proportion of whatever
/// domain the WM hands this window, never a pixel rectangle. Pixel limits
/// are [`LayoutHints`](crate::control::LayoutHints)' job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preference {
    #[default]
    Tile,
    FullHeightAspect {
        width: u16,
        height: u16,
    },
    /// Float at this fraction of the domain, centred in it.
    Floating(Fraction),
}

impl Preference {
    pub const fn is_valid(self) -> bool {
        match self {
            Self::Tile => true,
            Self::FullHeightAspect { width, height } => width != 0 && height != 0,
            // A `Fraction` cannot be built invalid.
            Self::Floating(_) => true,
        }
    }

    pub(crate) const fn words(self) -> (u64, u64) {
        match self {
            Self::Tile => (0, 0),
            Self::FullHeightAspect { width, height } => (1, width as u64 | (height as u64) << 16),
            Self::Floating(extent) => (2, extent.word()),
        }
    }

    pub(crate) const fn from_words(kind: u64, value: u64) -> Option<Self> {
        let result = match kind {
            0 if value == 0 => Self::Tile,
            1 if value >> 32 == 0 => Self::FullHeightAspect {
                width: value as u16,
                height: (value >> 16) as u16,
            },
            2 => match Fraction::from_word(value) {
                Some(extent) => Self::Floating(extent),
                None => return None,
            },
            _ => return None,
        };
        if result.is_valid() {
            Some(result)
        } else {
            None
        }
    }
}

/// Only a controller can assign its ordered output stream to a terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Content {
    Terminal,
    Application,
}

/// A serial identifies the client aperture's size contract. A client must
/// acknowledge it with a matching backing before committing that configuration.
/// The compositor may move the aperture without changing the serial: screen
/// coordinates are compositor-owned and never require a replacement surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Configure {
    pub window: u64,
    pub serial: u64,
    pub bounds: Bounds,
    pub focused: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fraction_is_a_real_proportion_and_survives_its_wire_word() {
        assert_eq!(Fraction::permille(0, 500), None, "an empty axis");
        assert_eq!(Fraction::permille(500, 0), None);
        assert_eq!(Fraction::permille(1001, 500), None, "more than the domain");
        assert_eq!(Fraction::permille(500, 1001), None);
        let half = Fraction::permille(500, 500).unwrap();
        assert_eq!((half.width_permille(), half.height_permille()), (500, 500));
        assert_eq!(Fraction::from_word(half.word()), Some(half));
        assert_eq!(
            Fraction::from_word(Fraction::FULL.word()),
            Some(Fraction::FULL)
        );
        // Reserved high bits and out-of-range axes are refused, never guessed.
        assert_eq!(Fraction::from_word(1 << 32), None);
        assert_eq!(Fraction::from_word(0), None);
    }

    #[test]
    fn a_fraction_resolves_against_the_domain_and_never_vanishes() {
        let half = Fraction::permille(500, 500).unwrap();
        assert_eq!(half.resolve(1280, 800), (640, 400));
        // The same request on a bigger panel is bigger — the whole point.
        assert_eq!(half.resolve(3840, 2160), (1920, 1080));
        assert_eq!(Fraction::FULL.resolve(1280, 800), (1280, 800));
        // Rounds to nearest, and a tiny domain still yields a real window.
        assert_eq!(Fraction::permille(1, 1).unwrap().resolve(100, 100), (1, 1));
        assert_eq!(
            Fraction::permille(333, 333).unwrap().resolve(10, 10),
            (3, 3)
        );
    }

    #[test]
    fn the_floating_preference_round_trips_relative_through_the_wire() {
        let preference = Preference::Floating(Fraction::permille(618, 382).unwrap());
        let (kind, value) = preference.words();
        assert_eq!(kind, 2);
        assert_eq!(Preference::from_words(kind, value), Some(preference));
        // Every variant still validates, and an unknown kind is refused.
        assert!(preference.is_valid());
        assert_eq!(Preference::from_words(2, 0), None, "an empty fraction");
        assert_eq!(Preference::from_words(3, 0), None, "an unknown kind");
    }
}
