/// Child boot-block coordinates of exec's canonical private filesystem face.
///
/// The coordinates are observable but not constructible by applications: one
/// canonical value owns the ABI layout, while launch presets merely compose it
/// beside other service faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilesystemFace {
    request_kick: u16,
    control_ring: u16,
    data_window: u16,
    return_send: u16,
    return_land: u16,
}

impl FilesystemFace {
    pub const CANONICAL: Self = Self {
        request_kick: 0,
        control_ring: 0,
        data_window: 1,
        return_send: 0,
        return_land: 1,
    };

    pub const fn request_kick(self) -> u16 {
        self.request_kick
    }

    pub const fn control_ring(self) -> u16 {
        self.control_ring
    }

    pub const fn data_window(self) -> u16 {
        self.data_window
    }

    pub const fn return_send(self) -> u16 {
        self.return_send
    }

    pub const fn return_land(self) -> u16 {
        self.return_land
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_coordinates_remain_exact() {
        let face = FilesystemFace::CANONICAL;
        assert_eq!(face.request_kick(), 0);
        assert_eq!(face.control_ring(), 0);
        assert_eq!(face.data_window(), 1);
        assert_eq!(face.return_send(), 0);
        assert_eq!(face.return_land(), 1);
    }
}
