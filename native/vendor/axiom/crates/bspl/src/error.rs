#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    AssetTooSmall,
    BadMagic,
    UnsupportedVersion,
    InvalidGeometry,
    Truncated,
    ScratchTooSmall,
    DecompressFailed,
}

impl Error {
    pub const fn label(self) -> &'static str {
        match self {
            Self::AssetTooSmall => "asset-too-small",
            Self::BadMagic => "asset-bad-magic",
            Self::UnsupportedVersion => "asset-unsupported-version",
            Self::InvalidGeometry => "asset-invalid-geometry",
            Self::Truncated => "asset-truncated",
            Self::ScratchTooSmall => "asset-scratch-too-small",
            Self::DecompressFailed => "asset-decompress-failed",
        }
    }
}
