/// Child boot-block coordinates for one window client face.
///
/// Construction is closed to the two standardized layouts. The coordinates
/// are inert lookup metadata, but allowing applications to assemble arbitrary
/// mixtures would let one service client consume another face's boot entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowFace {
    ring: u16,
    request_kick: u16,
    grant_recv: u16,
}

impl WindowFace {
    /// The original standalone window face.
    pub const STANDALONE: Self = Self {
        ring: 0,
        request_kick: 0,
        grant_recv: 0,
    };

    /// A window face composed beside exec's canonical filesystem face.
    ///
    /// FS owns `SVC_RING@0..=1` and `SVC_SEND@0`; the window pair therefore
    /// begins at ring 2 and uses request kick 1. Grant receives are a separate
    /// role namespace, so index 0 remains canonical.
    pub const WITH_FILESYSTEM: Self = Self {
        ring: 2,
        request_kick: 1,
        grant_recv: 0,
    };

    /// `SVC_RING` index carrying the shared window pair.
    pub const fn ring(self) -> u16 {
        self.ring
    }

    /// `SVC_SEND` index carrying the compositor request kick.
    pub const fn request_kick(self) -> u16 {
        self.request_kick
    }

    /// `SVC_GRANT` index carrying the compositor grant receiver.
    pub const fn grant_recv(self) -> u16 {
        self.grant_recv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardized_faces_keep_their_exact_coordinates() {
        assert_eq!(
            (
                WindowFace::STANDALONE.ring(),
                WindowFace::STANDALONE.request_kick(),
                WindowFace::STANDALONE.grant_recv(),
            ),
            (0, 0, 0)
        );
        assert_eq!(
            (
                WindowFace::WITH_FILESYSTEM.ring(),
                WindowFace::WITH_FILESYSTEM.request_kick(),
                WindowFace::WITH_FILESYSTEM.grant_recv(),
            ),
            (2, 1, 0)
        );
    }
}
