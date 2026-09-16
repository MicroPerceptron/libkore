//! Resolved presentation preference, never authority to modify desktop settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorScheme {
    Light,
    #[default]
    Dark,
}

/// Native backing pixels per UI unit, in exact quarter steps. Wire values
/// are quarters (2..=10); zero is reserved for legacy native/double flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Density {
    Half = 2,
    ThreeQuarter = 3,
    #[default]
    Native = 4,
    OneQuarter = 5,
    OneHalf = 6,
    OneThreeQuarter = 7,
    Double = 8,
    TwoQuarter = 9,
    TwoHalf = 10,
}
impl Density {
    pub const ALL: [Self; 9] = [
        Self::Half,
        Self::ThreeQuarter,
        Self::Native,
        Self::OneQuarter,
        Self::OneHalf,
        Self::OneThreeQuarter,
        Self::Double,
        Self::TwoQuarter,
        Self::TwoHalf,
    ];
    pub const MAX: Self = Self::TwoHalf;
    pub const fn from_quarters(value: u8) -> Option<Self> {
        match value {
            2 => Some(Self::Half),
            3 => Some(Self::ThreeQuarter),
            4 => Some(Self::Native),
            5 => Some(Self::OneQuarter),
            6 => Some(Self::OneHalf),
            7 => Some(Self::OneThreeQuarter),
            8 => Some(Self::Double),
            9 => Some(Self::TwoQuarter),
            10 => Some(Self::TwoHalf),
            _ => None,
        }
    }
    pub const fn quarters(self) -> u8 {
        self as u8
    }
    pub const fn percent(self) -> u16 {
        self as u16 * 25
    }
    pub const fn as_f32(self) -> f32 {
        self as u8 as f32 / 4.0
    }
    /// Round half up using widened arithmetic; never wrap a pixel extent.
    pub const fn pixels(self, units: u32) -> u32 {
        let value = (units as u64 * self as u64 + 2) / 4;
        if value > u32::MAX as u64 {
            u32::MAX
        } else {
            value as u32
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Self::Half => "Display scale: 50%",
            Self::ThreeQuarter => "Display scale: 75%",
            Self::Native => "Display scale: 100%",
            Self::OneQuarter => "Display scale: 125%",
            Self::OneHalf => "Display scale: 150%",
            Self::OneThreeQuarter => "Display scale: 175%",
            Self::Double => "Display scale: 200%",
            Self::TwoQuarter => "Display scale: 225%",
            Self::TwoHalf => "Display scale: 250%",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_quarter_tier_has_exact_geometry_and_bounded_wire_admission() {
        for (i, density) in Density::ALL.into_iter().enumerate() {
            assert_eq!(density.percent(), 50 + i as u16 * 25);
            assert_eq!(Density::from_quarters(density.quarters()), Some(density));
            assert_eq!(density.pixels(12), 6 + i as u32 * 3);
            assert_eq!(density.pixels(24), 12 + i as u32 * 6);
        }
        for bad in [0, 1, 11, 255] {
            assert!(Density::from_quarters(bad).is_none());
        }
        assert_eq!(Density::MAX.pixels(u32::MAX), u32::MAX);
    }
}
