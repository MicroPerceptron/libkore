//! Cross-platform graphics submission and surface-lifecycle contract.
//!
//! [`GraphicsBackend`] consumes allocator-typed [`axui_core::FrameView`] values
//! without taking allocation or presentation authority away from the host.
//! Native surfaces, memory-domain tokens, device handles, and mappings remain
//! behind the backend's associated `Target` type.

#![no_std]
#![feature(allocator_api)]
#![forbid(unsafe_code)]

#[cfg(test)]
extern crate std;

use core::{alloc::Allocator, num::NonZeroU64};

use axinput::PointerPosition;
use axui_core::{Color, Command, FrameView, Paint, Quad, Rect};

pub mod viewport;

/// Maximum number of planes in one portable surface description.
pub const MAX_SURFACE_PLANES: usize = 4;

/// A two-dimensional physical-pixel extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent2d {
    width: u32,
    height: u32,
}

impl Extent2d {
    /// Creates a nonempty extent.
    pub const fn new(width: u32, height: u32) -> Result<Self, ContractError> {
        if width == 0 || height == 0 {
            return Err(ContractError::ZeroExtent);
        }
        Ok(Self { width, height })
    }

    /// Returns the width.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the height.
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// A nonempty physical-pixel rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl PixelRect {
    /// Creates a nonempty rectangle whose end coordinates do not overflow.
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Result<Self, ContractError> {
        if width == 0 || height == 0 {
            return Err(ContractError::ZeroExtent);
        }
        if x.checked_add(width).is_none() || y.checked_add(height).is_none() {
            return Err(ContractError::RangeOverflow);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Covers a complete extent.
    pub const fn full(extent: Extent2d) -> Self {
        Self {
            x: 0,
            y: 0,
            width: extent.width,
            height: extent.height,
        }
    }

    /// Returns the horizontal origin.
    pub const fn x(self) -> u32 {
        self.x
    }

    /// Returns the vertical origin.
    pub const fn y(self) -> u32 {
        self.y
    }

    /// Returns the width.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the height.
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Whether the rectangle is entirely inside an extent.
    pub const fn fits(self, extent: Extent2d) -> bool {
        self.x + self.width <= extent.width && self.y + self.height <= extent.height
    }
}

/// An exact nonempty subrange of an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    offset: u64,
    bytes: NonZeroU64,
}

impl ByteRange {
    /// Creates a range whose exclusive end is representable.
    pub const fn new(offset: u64, bytes: u64) -> Result<Self, ContractError> {
        let Some(bytes) = NonZeroU64::new(bytes) else {
            return Err(ContractError::ZeroRange);
        };
        if offset.checked_add(bytes.get()).is_none() {
            return Err(ContractError::RangeOverflow);
        }
        Ok(Self { offset, bytes })
    }

    /// Returns the allocation-relative byte offset.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the range length.
    pub const fn bytes(self) -> u64 {
        self.bytes.get()
    }

    /// Returns the exclusive allocation-relative end.
    pub const fn end(self) -> u64 {
        self.offset + self.bytes.get()
    }
}

/// Extensible pixel-format identifier.
///
/// Unknown identifiers remain representable for backend-specific capability
/// probing. A portable format gives its channel order, transfer function, and
/// component precision in the constant name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PixelFormat(u32);

impl PixelFormat {
    /// One little-endian `0x00RRGGBB` word per pixel; X is ignored and RGB is
    /// normalized sRGB. This maps directly to Kore's `XRGB8888` surface.
    pub const XRGB8888_SRGB: Self = Self(1);
    /// Three independent R, G, B IEEE F16 planes containing normalized sRGB.
    pub const PLANAR_RGB_F16_SRGB: Self = Self(2);
    /// Three independent R, G, B IEEE F32 planes containing normalized sRGB.
    pub const PLANAR_RGB_F32_SRGB: Self = Self(3);

    /// Creates an extensible format identifier.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the numeric identifier.
    pub const fn get(self) -> u32 {
        self.0
    }

    const fn known_layout(self) -> Option<(u8, u8)> {
        if self.0 == Self::XRGB8888_SRGB.0 {
            Some((1, 4))
        } else if self.0 == Self::PLANAR_RGB_F16_SRGB.0 {
            Some((3, 2))
        } else if self.0 == Self::PLANAR_RGB_F32_SRGB.0 {
            Some((3, 4))
        } else {
            None
        }
    }
}

/// Opaque platform layout/modifier identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SurfaceModifier(u64);

impl SurfaceModifier {
    /// Ordinary row-major linear storage.
    pub const LINEAR: Self = Self(0);

    /// Creates an opaque modifier identifier.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Declared uses of a surface allocation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct SurfaceUsage(u32);

impl SurfaceUsage {
    /// The surface may be written by a renderer.
    pub const RENDER_TARGET: Self = Self(1 << 0);
    /// The surface may be committed to a presentation authority.
    pub const PRESENT: Self = Self(1 << 1);
    /// The surface may be sampled as an input.
    pub const SAMPLED: Self = Self(1 << 2);
    /// The surface may be scanned out directly by a display plane.
    pub const SCANOUT: Self = Self(1 << 3);

    /// Returns the union of two usage sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether this set contains every requested usage.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no usage is declared.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// One plane inside a surface's exact [`ByteRange`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlaneLayout {
    offset: u64,
    row_stride: u64,
    row_bytes: u64,
    rows: u32,
}

impl PlaneLayout {
    /// Creates a plane. `offset` is relative to the start of the surface
    /// range; `row_bytes` is active content and `row_stride` includes padding.
    pub const fn new(
        offset: u64,
        row_stride: u64,
        row_bytes: u64,
        rows: u32,
    ) -> Result<Self, ContractError> {
        if row_bytes == 0 || rows == 0 || row_stride < row_bytes {
            return Err(ContractError::InvalidPlaneLayout);
        }
        let Some(tail_rows) = ((rows - 1) as u64).checked_mul(row_stride) else {
            return Err(ContractError::RangeOverflow);
        };
        let Some(last_row) = offset.checked_add(tail_rows) else {
            return Err(ContractError::RangeOverflow);
        };
        if last_row.checked_add(row_bytes).is_none() {
            return Err(ContractError::RangeOverflow);
        }
        Ok(Self {
            offset,
            row_stride,
            row_bytes,
            rows,
        })
    }

    /// Returns the surface-range-relative offset.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the byte distance between adjacent rows.
    pub const fn row_stride(self) -> u64 {
        self.row_stride
    }

    /// Returns active bytes in each row.
    pub const fn row_bytes(self) -> u64 {
        self.row_bytes
    }

    /// Returns the number of rows.
    pub const fn rows(self) -> u32 {
        self.rows
    }

    const fn end(self) -> u64 {
        self.offset + (self.rows - 1) as u64 * self.row_stride + self.row_bytes
    }
}

/// Validated physical layout of one host-owned target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceInfo {
    format: PixelFormat,
    modifier: SurfaceModifier,
    extent: Extent2d,
    range: ByteRange,
    usage: SurfaceUsage,
    planes: [PlaneLayout; MAX_SURFACE_PLANES],
    plane_count: u8,
}

impl SurfaceInfo {
    /// Creates the linear, padded XRGB target used by Kore's current display
    /// service and by ordinary CPU-backed host windows.
    pub fn linear_xrgb8888(
        extent: Extent2d,
        range: ByteRange,
        row_stride: u64,
        usage: SurfaceUsage,
    ) -> Result<Self, ContractError> {
        let row_bytes = u64::from(extent.width)
            .checked_mul(4)
            .ok_or(ContractError::RangeOverflow)?;
        let plane = PlaneLayout::new(0, row_stride, row_bytes, extent.height)?;
        Self::new(
            PixelFormat::XRGB8888_SRGB,
            SurfaceModifier::LINEAR,
            extent,
            range,
            usage,
            &[plane],
        )
    }

    /// Validates an exact target description without interpreting native
    /// memory-domain or allocation handles.
    pub fn new(
        format: PixelFormat,
        modifier: SurfaceModifier,
        extent: Extent2d,
        range: ByteRange,
        usage: SurfaceUsage,
        planes: &[PlaneLayout],
    ) -> Result<Self, ContractError> {
        if usage.is_empty() {
            return Err(ContractError::EmptyUsage);
        }
        if planes.is_empty() || planes.len() > MAX_SURFACE_PLANES {
            return Err(ContractError::InvalidPlaneCount);
        }

        if let Some((expected_count, bytes_per_pixel)) = format.known_layout() {
            if planes.len() != usize::from(expected_count) {
                return Err(ContractError::FormatPlaneMismatch);
            }
            let expected_row = u64::from(extent.width)
                .checked_mul(u64::from(bytes_per_pixel))
                .ok_or(ContractError::RangeOverflow)?;
            if planes
                .iter()
                .any(|plane| plane.row_bytes != expected_row || plane.rows != extent.height)
            {
                return Err(ContractError::FormatPlaneMismatch);
            }
        }

        for (index, plane) in planes.iter().enumerate() {
            if plane.end() > range.bytes() {
                return Err(ContractError::PlaneOutOfBounds);
            }
            for prior in &planes[..index] {
                if plane.offset < prior.end() && prior.offset < plane.end() {
                    return Err(ContractError::PlaneOverlap);
                }
            }
        }

        let mut stored = [PlaneLayout::default(); MAX_SURFACE_PLANES];
        stored[..planes.len()].copy_from_slice(planes);
        Ok(Self {
            format,
            modifier,
            extent,
            range,
            usage,
            planes: stored,
            plane_count: planes.len() as u8,
        })
    }

    /// Returns the pixel format.
    pub const fn format(self) -> PixelFormat {
        self.format
    }

    /// Returns the platform modifier.
    pub const fn modifier(self) -> SurfaceModifier {
        self.modifier
    }

    /// Returns the physical extent.
    pub const fn extent(self) -> Extent2d {
        self.extent
    }

    /// Returns the exact allocation range.
    pub const fn range(self) -> ByteRange {
        self.range
    }

    /// Returns declared uses.
    pub const fn usage(self) -> SurfaceUsage {
        self.usage
    }

    /// Returns the active plane descriptions.
    pub fn planes(&self) -> &[PlaneLayout] {
        &self.planes[..usize::from(self.plane_count)]
    }
}

/// Failure to construct a portable surface or geometry descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractError {
    /// A pixel extent was empty.
    ZeroExtent,
    /// A byte range was empty.
    ZeroRange,
    /// Range or rectangle arithmetic overflowed.
    RangeOverflow,
    /// No planes or too many planes were supplied.
    InvalidPlaneCount,
    /// A plane has no rows/bytes or a stride smaller than its active row.
    InvalidPlaneLayout,
    /// A plane extends beyond the declared exact range.
    PlaneOutOfBounds,
    /// Two writable planes overlap.
    PlaneOverlap,
    /// A portable format's plane count or row shape does not match.
    FormatPlaneMismatch,
    /// A surface declared no permissible use.
    EmptyUsage,
    /// A capability limit was zero.
    ZeroLimit,
}

/// Display-list features a graphics backend may advertise.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Features(u64);

impl Features {
    /// Nested rectangular clipping.
    pub const CLIP: Self = Self(1 << 0);
    /// 4×4 transformations.
    pub const TRANSFORM: Self = Self(1 << 1);
    /// Solid-color quads.
    pub const SOLID_QUAD: Self = Self(1 << 2);
    /// Linear-gradient quad paint.
    pub const LINEAR_GRADIENT: Self = Self(1 << 3);
    /// Per-corner inner borders.
    pub const BORDER: Self = Self(1 << 4);
    /// Blurred, offset quad shadows.
    pub const SHADOW: Self = Self(1 << 5);
    /// Physical-pixel snapping intent.
    pub const PIXEL_SNAP: Self = Self(1 << 6);

    /// The complete feature vocabulary in this contract version.
    pub const ALL: Self = Self((1 << 7) - 1);

    /// Returns the union of two feature sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether this set contains every requested feature.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the raw bits.
    pub const fn bits(self) -> u64 {
        self.0
    }
}

/// Allocation-free resource limits for one backend queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsLimits {
    /// Largest target extent.
    pub max_extent: Extent2d,
    /// Maximum commands in one display list.
    pub max_commands: u32,
    /// Maximum quads in one display list.
    pub max_quads: u32,
    /// Maximum combined clip/transform nesting.
    pub max_group_depth: u16,
    /// Maximum explicit damage rectangles; full damage is always one claim.
    pub max_damage_rects: u16,
}

impl GraphicsLimits {
    /// Creates nonzero limits.
    pub const fn new(
        max_extent: Extent2d,
        max_commands: u32,
        max_quads: u32,
        max_group_depth: u16,
        max_damage_rects: u16,
    ) -> Result<Self, ContractError> {
        if max_commands == 0 || max_quads == 0 || max_group_depth == 0 || max_damage_rects == 0 {
            return Err(ContractError::ZeroLimit);
        }
        Ok(Self {
            max_extent,
            max_commands,
            max_quads,
            max_group_depth,
            max_damage_rects,
        })
    }
}

/// Stable capability declaration returned by a graphics backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsCapabilities {
    /// Supported display-list operations.
    pub features: Features,
    /// Queue and target limits.
    pub limits: GraphicsLimits,
}

/// Requirements derived from a validated display list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRequirements {
    features: Features,
    commands: u32,
    quads: u32,
    max_group_depth: u16,
}

impl FrameRequirements {
    /// Returns required features.
    pub const fn features(self) -> Features {
        self.features
    }

    /// Returns the command count.
    pub const fn commands(self) -> u32 {
        self.commands
    }

    /// Returns the quad count.
    pub const fn quads(self) -> u32 {
        self.quads
    }

    /// Returns maximum combined group depth.
    pub const fn max_group_depth(self) -> u16 {
        self.max_group_depth
    }
}

/// Physical damage for rendering and presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Damage<'a> {
    /// The complete target contents changed.
    Full,
    /// Exact nonempty, target-relative physical rectangles changed.
    Rects(&'a [PixelRect]),
}

/// Whether device completion also establishes host/display visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityMode {
    /// The target is coherent; a backend may publish visibility immediately
    /// after device completion.
    Coherent,
    /// A separate cache, ownership, or import transition is required.
    Explicit,
}

/// How long a backend submission must retain its target binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainUntil {
    /// The event may be released after target visibility.
    Visible,
    /// Imported target authority remains live until presentation completes.
    Presented,
}

/// How a backend writes one admitted target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRoute {
    /// Provider execution writes the exact target allocation directly.
    Direct,
    /// The backend explicitly materializes provider output before packing it
    /// into the exact target allocation.
    ///
    /// This is the truthful first route for CPU-mapped Kore display surfaces
    /// while `.ax` providers still expose planar owned outputs. It is not a
    /// zero-copy claim and should not be advertised for scanout admission.
    Staged,
}

/// A backend's truthful route to one opaque target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetSupport {
    /// Whether provider output is direct or explicitly staged.
    pub route: TargetRoute,
    /// Visibility behavior of this target route.
    pub visibility: VisibilityMode,
    /// Minimum lifetime of the submission/target binding.
    pub retain_until: RetainUntil,
}

impl TargetSupport {
    /// Declares a direct route. A direct route writes the exact target range
    /// and may not hide a frame-sized bounce allocation or copy.
    pub const fn direct(visibility: VisibilityMode, retain_until: RetainUntil) -> Self {
        Self {
            route: TargetRoute::Direct,
            visibility,
            retain_until,
        }
    }

    /// Declares an explicit provider-output staging route.
    ///
    /// The final pack remains part of the graphics submission and must finish
    /// before the backend publishes target visibility.
    pub const fn staged(visibility: VisibilityMode, retain_until: RetainUntil) -> Self {
        Self {
            route: TargetRoute::Staged,
            visibility,
            retain_until,
        }
    }
}

/// Strictly increasing client frame epoch committed to a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SurfaceEpoch(NonZeroU64);

impl SurfaceEpoch {
    /// Creates a nonzero epoch.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the numeric epoch.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Monotonic provider completion identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceEpoch(NonZeroU64);

impl DeviceEpoch {
    /// Creates a nonzero provider epoch.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the numeric epoch.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Monotonic cache/ownership visibility identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VisibilityEpoch(NonZeroU64);

impl VisibilityEpoch {
    /// Creates a nonzero visibility epoch.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the numeric epoch.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Backend/runtime failure independent of a platform transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsError {
    /// Malformed frame, viewport, or descriptor.
    InvalidArgument,
    /// A range or rectangle lies outside its authority.
    OutOfBounds,
    /// The backend or target does not implement the request.
    Unsupported,
    /// A declared backend limit would be exceeded.
    ResourceLimit,
    /// The queue cannot currently admit another submission.
    Busy,
    /// A caller-supplied deadline has expired.
    DeadlineExpired,
    /// The complete provider epoch is lost.
    DeviceLost,
    /// Target or resource authority is insufficient.
    PermissionDenied,
    /// Stable backend-owned error namespace.
    External {
        /// Backend error domain.
        domain: u32,
        /// Backend error code.
        code: i64,
    },
}

// ── Independent cursor presentation ───────────────────────────────

/// Pixel layouts accepted for a hardware cursor image.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct CursorFormats(u8);

impl CursorFormats {
    /// Premultiplied alpha in little-endian `0xAARRGGBB` words.
    pub const ARGB8888_PREMULTIPLIED: Self = Self(1 << 0);
    /// Premultiplied RGBA byte order.
    pub const RGBA8888_PREMULTIPLIED: Self = Self(1 << 1);

    /// Returns the union of two format sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every requested format is supported.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the raw bits.
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// Concrete pixel layout of one cursor image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorFormat {
    /// Premultiplied alpha in little-endian `0xAARRGGBB` words.
    Argb8888Premultiplied,
    /// Premultiplied RGBA bytes.
    Rgba8888Premultiplied,
}

impl CursorFormat {
    const fn feature(self) -> CursorFormats {
        match self {
            Self::Argb8888Premultiplied => CursorFormats::ARGB8888_PREMULTIPLIED,
            Self::Rgba8888Premultiplied => CursorFormats::RGBA8888_PREMULTIPLIED,
        }
    }
}

/// Cursor hotspot within its image.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CursorHotspot {
    /// Horizontal image coordinate.
    pub x: u16,
    /// Vertical image coordinate.
    pub y: u16,
}

/// Stable capabilities of an independent cursor plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorPlaneCapabilities {
    /// Largest accepted cursor image.
    pub max_extent: Extent2d,
    /// Accepted alpha layouts.
    pub formats: CursorFormats,
    /// Whether position-only updates avoid a display composition/flip.
    pub independent_position: bool,
}

/// Borrowed cursor pixels uploaded infrequently by a cursor theme.
pub struct CursorImage<'a> {
    /// Pixel dimensions.
    pub extent: Extent2d,
    /// Hotspot within `extent`.
    pub hotspot: CursorHotspot,
    /// Alpha layout.
    pub format: CursorFormat,
    /// Byte stride between rows.
    pub row_stride: u32,
    /// Exact image storage.
    pub pixels: &'a [u8],
}

impl CursorImage<'_> {
    /// Validates image geometry and storage against a plane.
    pub fn validate(&self, capabilities: CursorPlaneCapabilities) -> Result<(), GraphicsError> {
        if self.extent.width() > capabilities.max_extent.width()
            || self.extent.height() > capabilities.max_extent.height()
            || !capabilities.formats.contains(self.format.feature())
        {
            return Err(GraphicsError::Unsupported);
        }
        if u32::from(self.hotspot.x) >= self.extent.width()
            || u32::from(self.hotspot.y) >= self.extent.height()
        {
            return Err(GraphicsError::InvalidArgument);
        }
        let active_row = self
            .extent
            .width()
            .checked_mul(4)
            .ok_or(GraphicsError::InvalidArgument)?;
        if self.row_stride < active_row {
            return Err(GraphicsError::InvalidArgument);
        }
        let required = u64::from(self.row_stride)
            .checked_mul(u64::from(self.extent.height()))
            .ok_or(GraphicsError::InvalidArgument)?;
        let available = u64::try_from(self.pixels.len()).map_err(|_| GraphicsError::OutOfBounds)?;
        if required > available {
            return Err(GraphicsError::OutOfBounds);
        }
        Ok(())
    }
}

/// Image selection in one cursor-plane update.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CursorImageSelection<'a, I> {
    /// Keep the currently selected image. This is the position-only hot path.
    #[default]
    Keep,
    /// Select an uploaded image.
    Select(&'a I),
    /// Deselect any image so its resource can be released.
    Clear,
}

/// Atomic cursor-plane state update.
///
/// Position-only updates use [`CursorImageSelection::Keep`] and must not
/// require allocating or re-uploading image data.
pub struct CursorPlaneUpdate<'a, I> {
    /// Target-relative hotspot position. Negative values keep a partially
    /// off-screen cursor representable.
    pub position: PointerPosition,
    /// Whether the plane is visible.
    pub visible: bool,
    /// Image-selection change.
    pub image: CursorImageSelection<'a, I>,
}

/// Failure to release one owned cursor image.
#[derive(Debug)]
pub enum CursorImageReleaseFailure<I> {
    /// Release was refused and returns the still-live image for retry.
    Rejected {
        /// Stable refusal reason.
        error: GraphicsError,
        /// Still-live image token.
        image: I,
    },
    /// Image state is unknown; invalidate its identity and quarantine any
    /// storage whose ownership cannot be proven.
    Indeterminate(GraphicsError),
}

/// Cross-platform independent-cursor capability.
///
/// `A` is threaded from the final runtime exactly like [`GraphicsBackend`].
/// The implementation may use it for uploaded image storage, but the cursor
/// API never selects a global allocator. Image replacement and state update
/// are separate so high-rate motion remains a bounded, allocation-free call.
pub trait CursorPlane<A: Allocator> {
    /// Backend-owned uploaded image/capability token.
    type Image;

    /// Returns stable plane capabilities.
    fn cursor_capabilities(&self) -> CursorPlaneCapabilities;

    /// Uploads one validated image. The returned token owns any backend
    /// resource until [`destroy_cursor_image`](Self::destroy_cursor_image).
    fn create_cursor_image(&mut self, image: CursorImage<'_>)
    -> Result<Self::Image, GraphicsError>;

    /// Atomically updates image selection, hotspot position, and visibility.
    fn update_cursor(
        &mut self,
        update: CursorPlaneUpdate<'_, Self::Image>,
    ) -> Result<(), GraphicsError>;

    /// Releases an uploaded image after it is no longer selected.
    fn destroy_cursor_image(
        &mut self,
        image: Self::Image,
    ) -> Result<(), CursorImageReleaseFailure<Self::Image>>;
}

/// A render request borrowing one display list and one opaque target.
///
/// `A` is the allocator capability selected by the final runtime and must
/// match the receiving [`GraphicsBackend`]. The request itself owns no dynamic
/// storage.
///
/// The backend may retain native handles in its returned submission, but it
/// must not retain these Rust references. The caller keeps the frame, target,
/// and exact backing alive and does not reuse them until the submission and
/// [`SurfaceRetirement`] permit it.
pub struct RenderRequest<'a, T, A: Allocator> {
    frame: FrameView<'a, A>,
    target: &'a mut T,
    surface: SurfaceInfo,
    surface_epoch: SurfaceEpoch,
    viewport: PixelRect,
    damage: Damage<'a>,
}

impl<'a, T, A: Allocator> RenderRequest<'a, T, A> {
    /// Creates a request. Backend capabilities and target support are checked
    /// by [`validate_render_request`] before admission.
    pub const fn new(
        frame: FrameView<'a, A>,
        target: &'a mut T,
        surface: SurfaceInfo,
        surface_epoch: SurfaceEpoch,
        viewport: PixelRect,
        damage: Damage<'a>,
    ) -> Self {
        Self {
            frame,
            target,
            surface,
            surface_epoch,
            viewport,
            damage,
        }
    }

    /// Returns the display list.
    pub const fn frame(&self) -> FrameView<'a, A> {
        self.frame
    }

    /// Returns the opaque target.
    pub fn target(&self) -> &T {
        self.target
    }

    /// Returns mutable access to the opaque target for admission.
    pub fn target_mut(&mut self) -> &mut T {
        self.target
    }

    /// Returns the validated physical surface description.
    pub const fn surface(&self) -> SurfaceInfo {
        self.surface
    }

    /// Returns the surface epoch this render may eventually commit.
    pub const fn surface_epoch(&self) -> SurfaceEpoch {
        self.surface_epoch
    }

    /// Returns the physical destination viewport.
    pub const fn viewport(&self) -> PixelRect {
        self.viewport
    }

    /// Returns the physical damage claim.
    pub const fn damage(&self) -> Damage<'a> {
        self.damage
    }
}

/// Derives and validates portable requirements from one display list.
pub fn frame_requirements<A: Allocator>(
    frame: FrameView<'_, A>,
) -> Result<FrameRequirements, GraphicsError> {
    let commands =
        u32::try_from(frame.commands().len()).map_err(|_| GraphicsError::ResourceLimit)?;
    let quads = u32::try_from(frame.quads().len()).map_err(|_| GraphicsError::ResourceLimit)?;
    let bounds = frame.bounds();
    if !valid_rect(bounds) || bounds.width <= 0.0 || bounds.height <= 0.0 {
        return Err(GraphicsError::InvalidArgument);
    }
    let scale = frame.scale_factor();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(GraphicsError::InvalidArgument);
    }

    let mut features = Features::default();
    let mut depth = 0_u32;
    let mut max_depth = 0_u32;
    for command in frame.commands() {
        match *command {
            Command::PushClip(rect) => {
                if !valid_rect(rect) {
                    return Err(GraphicsError::InvalidArgument);
                }
                features = features.union(Features::CLIP);
                depth = depth.checked_add(1).ok_or(GraphicsError::ResourceLimit)?;
                max_depth = max_depth.max(depth);
            }
            Command::PushTransform(transform) => {
                if transform.0.iter().any(|value| !value.is_finite()) {
                    return Err(GraphicsError::InvalidArgument);
                }
                features = features.union(Features::TRANSFORM);
                depth = depth.checked_add(1).ok_or(GraphicsError::ResourceLimit)?;
                max_depth = max_depth.max(depth);
            }
            Command::PopClip | Command::PopTransform => {
                depth = depth.checked_sub(1).ok_or(GraphicsError::InvalidArgument)?;
            }
            Command::DrawQuad(index) => {
                if frame.quad(index).is_none() {
                    return Err(GraphicsError::InvalidArgument);
                }
            }
        }
    }
    if depth != 0 {
        return Err(GraphicsError::InvalidArgument);
    }

    for quad in frame.quads() {
        validate_quad(*quad)?;
        features = features.union(match quad.paint {
            Paint::Solid(_) => Features::SOLID_QUAD,
            Paint::LinearGradient(_) => Features::LINEAR_GRADIENT,
        });
        if quad.border.width > 0.0 && quad.border.color.a > 0.0 {
            features = features.union(Features::BORDER);
        }
        if quad.shadow.color.a > 0.0 {
            features = features.union(Features::SHADOW);
        }
        if quad.snap {
            features = features.union(Features::PIXEL_SNAP);
        }
    }

    Ok(FrameRequirements {
        features,
        commands,
        quads,
        max_group_depth: u16::try_from(max_depth).map_err(|_| GraphicsError::ResourceLimit)?,
    })
}

/// Performs all platform-independent checks before a backend accepts work.
pub fn validate_render_request<T, A: Allocator>(
    capabilities: GraphicsCapabilities,
    target_support: Option<TargetSupport>,
    request: &RenderRequest<'_, T, A>,
) -> Result<FrameRequirements, GraphicsError> {
    if target_support.is_none() {
        return Err(GraphicsError::Unsupported);
    }
    let surface = request.surface();
    if !surface.usage().contains(SurfaceUsage::RENDER_TARGET) {
        return Err(GraphicsError::PermissionDenied);
    }
    let extent = surface.extent();
    if extent.width > capabilities.limits.max_extent.width
        || extent.height > capabilities.limits.max_extent.height
    {
        return Err(GraphicsError::ResourceLimit);
    }

    let requirements = frame_requirements(request.frame())?;
    if !capabilities.features.contains(requirements.features)
        || requirements.commands > capabilities.limits.max_commands
        || requirements.quads > capabilities.limits.max_quads
        || requirements.max_group_depth > capabilities.limits.max_group_depth
    {
        return Err(if !capabilities.features.contains(requirements.features) {
            GraphicsError::Unsupported
        } else {
            GraphicsError::ResourceLimit
        });
    }

    let viewport = request.viewport();
    if !viewport.fits(extent) {
        return Err(GraphicsError::OutOfBounds);
    }
    let logical = request.frame().bounds();
    let scale = request.frame().scale_factor();
    if rounded_physical(logical.width * scale) != Some(viewport.width)
        || rounded_physical(logical.height * scale) != Some(viewport.height)
    {
        return Err(GraphicsError::InvalidArgument);
    }

    match request.damage() {
        Damage::Full => {}
        Damage::Rects(rects) => {
            if rects.is_empty() || rects.len() > usize::from(capabilities.limits.max_damage_rects) {
                return Err(GraphicsError::ResourceLimit);
            }
            if rects.iter().any(|rect| !rect.fits(extent)) {
                return Err(GraphicsError::OutOfBounds);
            }
        }
    }

    Ok(requirements)
}

fn rounded_physical(value: f32) -> Option<u32> {
    if !value.is_finite() || value < 1.0 || value >= u32::MAX as f32 {
        return None;
    }
    let truncated = value as u32;
    if value - truncated as f32 >= 0.5 {
        truncated.checked_add(1)
    } else {
        Some(truncated)
    }
}

fn valid_rect(rect: Rect) -> bool {
    [rect.x, rect.y, rect.width, rect.height]
        .iter()
        .all(|value| value.is_finite())
        && rect.width >= 0.0
        && rect.height >= 0.0
}

fn valid_color(color: Color) -> bool {
    [color.r, color.g, color.b, color.a]
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

fn validate_quad(quad: Quad) -> Result<(), GraphicsError> {
    if !valid_rect(quad.bounds)
        || !valid_color(quad.border.color)
        || !valid_color(quad.shadow.color)
        || !quad.border.width.is_finite()
        || quad.border.width < 0.0
        || !quad.shadow.offset.x.is_finite()
        || !quad.shadow.offset.y.is_finite()
        || !quad.shadow.blur_radius.is_finite()
        || quad.shadow.blur_radius < 0.0
        || [
            quad.border.radii.top_left,
            quad.border.radii.top_right,
            quad.border.radii.bottom_right,
            quad.border.radii.bottom_left,
        ]
        .iter()
        .any(|radius| !radius.is_finite() || *radius < 0.0)
    {
        return Err(GraphicsError::InvalidArgument);
    }

    match quad.paint {
        Paint::Solid(color) => {
            if !valid_color(color) {
                return Err(GraphicsError::InvalidArgument);
            }
        }
        Paint::LinearGradient(gradient) => {
            if !gradient.angle.is_finite() {
                return Err(GraphicsError::InvalidArgument);
            }
            let mut previous = None;
            let mut found_empty = false;
            let mut stop_count = 0_u8;
            for stop in gradient.stops {
                match stop {
                    Some(stop) if !found_empty => {
                        if !stop.offset.is_finite()
                            || !(0.0..=1.0).contains(&stop.offset)
                            || !valid_color(stop.color)
                            || previous.is_some_and(|value| stop.offset < value)
                        {
                            return Err(GraphicsError::InvalidArgument);
                        }
                        previous = Some(stop.offset);
                        stop_count += 1;
                    }
                    Some(_) => return Err(GraphicsError::InvalidArgument),
                    None => found_empty = true,
                }
            }
            if stop_count < 2 {
                return Err(GraphicsError::InvalidArgument);
            }
        }
    }

    Ok(())
}

/// Whether an aborted submission has definitely stopped touching its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetReuse {
    /// Provider and cache access are quiescent.
    Safe,
    /// Access is uncertain; the target stays quarantined until platform reset.
    Quarantine,
}

/// Monotonic state of one accepted graphics submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionState {
    /// Accepted and still executing or queued.
    Pending,
    /// Device execution finished, but visibility may still be pending.
    DeviceComplete(DeviceEpoch),
    /// The exact target range is visible to its next owner.
    Visible {
        /// Provider completion that produced these bytes.
        device: DeviceEpoch,
        /// Cache/ownership transition that made them visible.
        visibility: VisibilityEpoch,
    },
    /// Execution failed terminally.
    Failed {
        /// Stable failure reason.
        error: GraphicsError,
        /// Whether target reuse is proven safe.
        target: TargetReuse,
    },
    /// Cancellation reached a terminal state.
    Cancelled {
        /// Whether target reuse is proven safe.
        target: TargetReuse,
    },
}

impl SubmissionState {
    /// Whether the event is terminal and may be considered for release.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Visible { .. } | Self::Failed { .. } | Self::Cancelled { .. }
        )
    }
}

/// Invalid lifecycle transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionError {
    /// The requested transition is not legal from the current state.
    InvalidState,
    /// Visibility named a different device-completion epoch.
    DeviceEpochMismatch,
    /// A presentation observation predates the rendered surface epoch.
    StalePresentation,
}

/// Reusable monotonic state machine for backend event implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmissionTracker {
    surface_epoch: SurfaceEpoch,
    state: SubmissionState,
}

impl SubmissionTracker {
    /// Starts tracking an accepted request.
    pub const fn new(surface_epoch: SurfaceEpoch) -> Self {
        Self {
            surface_epoch,
            state: SubmissionState::Pending,
        }
    }

    /// Returns the client surface epoch.
    pub const fn surface_epoch(self) -> SurfaceEpoch {
        self.surface_epoch
    }

    /// Returns the current event state.
    pub const fn state(self) -> SubmissionState {
        self.state
    }

    /// Records provider execution completion.
    pub fn mark_device_complete(
        &mut self,
        device: DeviceEpoch,
    ) -> Result<SubmissionState, TransitionError> {
        if self.state != SubmissionState::Pending {
            return Err(TransitionError::InvalidState);
        }
        self.state = SubmissionState::DeviceComplete(device);
        Ok(self.state)
    }

    /// Records the separate cache/ownership visibility transition.
    pub fn mark_visible(
        &mut self,
        device: DeviceEpoch,
        visibility: VisibilityEpoch,
    ) -> Result<SubmissionState, TransitionError> {
        match self.state {
            SubmissionState::DeviceComplete(completed) if completed == device => {
                self.state = SubmissionState::Visible { device, visibility };
                Ok(self.state)
            }
            SubmissionState::DeviceComplete(_) => Err(TransitionError::DeviceEpochMismatch),
            _ => Err(TransitionError::InvalidState),
        }
    }

    /// Records terminal failure and whether the target is quiescent.
    pub fn mark_failed(
        &mut self,
        error: GraphicsError,
        target: TargetReuse,
    ) -> Result<SubmissionState, TransitionError> {
        if self.state.is_terminal() {
            return Err(TransitionError::InvalidState);
        }
        self.state = SubmissionState::Failed { error, target };
        Ok(self.state)
    }

    /// Records terminal cancellation and whether the target is quiescent.
    pub fn mark_cancelled(
        &mut self,
        target: TargetReuse,
    ) -> Result<SubmissionState, TransitionError> {
        if self.state.is_terminal() {
            return Err(TransitionError::InvalidState);
        }
        self.state = SubmissionState::Cancelled { target };
        Ok(self.state)
    }
}

/// Tracks when a rendered surface can be committed, released, and reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceRetirement {
    surface_epoch: SurfaceEpoch,
    retain_until: RetainUntil,
    observed: SubmissionState,
    committed: bool,
    presented: Option<SurfaceEpoch>,
    abandoned: bool,
    submission_released: bool,
}

impl SurfaceRetirement {
    /// Creates retirement books for an accepted submission.
    pub const fn new(surface_epoch: SurfaceEpoch, retain_until: RetainUntil) -> Self {
        Self {
            surface_epoch,
            retain_until,
            observed: SubmissionState::Pending,
            committed: false,
            presented: None,
            abandoned: false,
            submission_released: false,
        }
    }

    /// Observes a monotonic backend state.
    pub fn observe(&mut self, next: SubmissionState) -> Result<(), TransitionError> {
        let valid = match (self.observed, next) {
            (current, next) if current == next => true,
            (SubmissionState::Pending, _) => true,
            (SubmissionState::DeviceComplete(current), SubmissionState::Visible { device, .. }) => {
                current == device
            }
            (SubmissionState::DeviceComplete(_), SubmissionState::Failed { .. })
            | (SubmissionState::DeviceComplete(_), SubmissionState::Cancelled { .. }) => true,
            _ => false,
        };
        if !valid || self.submission_released {
            return Err(TransitionError::InvalidState);
        }
        self.observed = next;
        Ok(())
    }

    /// Whether target bytes are visible and may be committed for presentation.
    pub const fn can_commit(self) -> bool {
        matches!(self.observed, SubmissionState::Visible { .. })
            && !self.committed
            && !self.abandoned
    }

    /// Records a successful platform commit of this exact epoch.
    pub fn mark_committed(&mut self) -> Result<(), TransitionError> {
        if !self.can_commit() {
            return Err(TransitionError::InvalidState);
        }
        self.committed = true;
        Ok(())
    }

    /// Records platform presentation truth. A later coalesced epoch also
    /// retires this frame, matching monotonic compositor books.
    pub fn mark_presented(&mut self, epoch: SurfaceEpoch) -> Result<(), TransitionError> {
        if !self.committed {
            return Err(TransitionError::InvalidState);
        }
        if epoch < self.surface_epoch {
            return Err(TransitionError::StalePresentation);
        }
        self.presented = Some(self.presented.map_or(epoch, |prior| prior.max(epoch)));
        Ok(())
    }

    /// Abandons visible bytes without committing them.
    pub fn abandon(&mut self) -> Result<(), TransitionError> {
        if !matches!(self.observed, SubmissionState::Visible { .. }) || self.committed {
            return Err(TransitionError::InvalidState);
        }
        self.abandoned = true;
        Ok(())
    }

    /// Whether backend event/target authority may now be released.
    pub const fn can_release_submission(self) -> bool {
        if self.submission_released {
            return false;
        }
        match self.observed {
            SubmissionState::Visible { .. } => {
                self.abandoned
                    || matches!(self.retain_until, RetainUntil::Visible)
                    || self.presented.is_some()
            }
            SubmissionState::Failed {
                target: TargetReuse::Safe,
                ..
            }
            | SubmissionState::Cancelled {
                target: TargetReuse::Safe,
            } => true,
            _ => false,
        }
    }

    /// Records successful backend event release.
    pub fn mark_submission_released(&mut self) -> Result<(), TransitionError> {
        if !self.can_release_submission() {
            return Err(TransitionError::InvalidState);
        }
        self.submission_released = true;
        Ok(())
    }

    /// Whether the backing may be acquired for another producer epoch.
    pub const fn is_reusable(self) -> bool {
        if !self.submission_released {
            return false;
        }
        match self.observed {
            SubmissionState::Visible { .. } => {
                self.abandoned || (self.committed && self.presented.is_some())
            }
            SubmissionState::Failed {
                target: TargetReuse::Safe,
                ..
            }
            | SubmissionState::Cancelled {
                target: TargetReuse::Safe,
            } => true,
            _ => false,
        }
    }
}

/// Submission failure with an explicit provider acceptance boundary.
#[derive(Debug)]
pub enum SubmitFailure<S> {
    /// Work was not accepted and no target authority was retained.
    Rejected(GraphicsError),
    /// Acceptance is uncertain; the returned event must reach a terminal
    /// state or the complete provider epoch must be quarantined.
    Indeterminate {
        /// Immediate provider failure.
        error: GraphicsError,
        /// Event retaining the uncertain resource lifetime.
        submission: S,
    },
}

/// Failure to release an owned submission event.
#[derive(Debug)]
pub enum ReleaseFailure<S> {
    /// Release was refused and returns the still-live event for retry.
    Rejected {
        /// Stable refusal reason.
        error: GraphicsError,
        /// Still-live submission.
        submission: S,
    },
    /// Event state is unknown; invalidate its ID and quarantine the target.
    Indeterminate(GraphicsError),
}

/// Statically dispatched cross-platform graphics backend.
///
/// `A` is supplied by the final runtime; implementations use it for any owned
/// dynamic lowering or event storage. Implementations do not own windows,
/// presentation, or target allocation. `submit` binds the exact opaque target
/// through the route reported by [`TargetSupport`] or rejects it. Polling is
/// bounded and nonblocking; hosts decide when to advance, wait, commit, and
/// present. Native references passed in a request are valid only for the call
/// and must not be retained as Rust references.
pub trait GraphicsBackend<A: Allocator> {
    /// Platform-owned target handle or lease token.
    type Target;
    /// Owned provider event retaining native binding lifetime.
    type Submission;

    /// Returns stable operation and queue limits.
    fn capabilities(&self) -> GraphicsCapabilities;

    /// Returns a truthful route for this exact target, or `None`.
    fn target_support(&self, target: &Self::Target, surface: SurfaceInfo) -> Option<TargetSupport>;

    /// Validates, compiles, and admits one request without hidden full-frame
    /// staging. A route returned as [`TargetRoute::Staged`] may perform its
    /// declared final materialization; a [`TargetRoute::Direct`] route may
    /// not. Implementations should call [`validate_render_request`] before
    /// crossing their provider acceptance edge.
    fn submit(
        &mut self,
        request: RenderRequest<'_, Self::Target, A>,
    ) -> Result<Self::Submission, SubmitFailure<Self::Submission>>;

    /// Performs bounded progress and returns monotonic event state.
    fn poll_submission(
        &mut self,
        submission: &mut Self::Submission,
    ) -> Result<SubmissionState, GraphicsError>;

    /// Requests cancellation without blocking for terminal completion.
    fn cancel_submission(
        &mut self,
        submission: &mut Self::Submission,
    ) -> Result<SubmissionState, GraphicsError>;

    /// Releases a terminal event after [`SurfaceRetirement`] permits it.
    fn release_submission(
        &mut self,
        submission: Self::Submission,
    ) -> Result<(), ReleaseFailure<Self::Submission>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axui_core::{
        Border, CornerRadii, Frame, GradientStop, LinearGradient, Point, Recorder, Shadow,
    };

    #[derive(Debug, Default)]
    struct MockTarget;

    #[derive(Debug)]
    struct MockSubmission {
        tracker: SubmissionTracker,
    }

    struct MockBackend<A: Allocator> {
        capabilities: GraphicsCapabilities,
        accepted: u64,
        _allocator: A,
    }

    struct MockCursorPlane<A: Allocator> {
        allocator: A,
        selected: Option<u64>,
        position: PointerPosition,
        visible: bool,
        next_image: u64,
    }

    impl<A: Allocator> MockCursorPlane<A> {
        fn new_in(allocator: A) -> Self {
            Self {
                allocator,
                selected: None,
                position: PointerPosition::default(),
                visible: false,
                next_image: 1,
            }
        }
    }

    impl<A: Allocator> CursorPlane<A> for MockCursorPlane<A> {
        type Image = u64;

        fn cursor_capabilities(&self) -> CursorPlaneCapabilities {
            let _ = &self.allocator;
            CursorPlaneCapabilities {
                max_extent: Extent2d::new(64, 64).unwrap(),
                formats: CursorFormats::ARGB8888_PREMULTIPLIED,
                independent_position: true,
            }
        }

        fn create_cursor_image(
            &mut self,
            image: CursorImage<'_>,
        ) -> Result<Self::Image, GraphicsError> {
            image.validate(self.cursor_capabilities())?;
            let id = self.next_image;
            self.next_image += 1;
            Ok(id)
        }

        fn update_cursor(
            &mut self,
            update: CursorPlaneUpdate<'_, Self::Image>,
        ) -> Result<(), GraphicsError> {
            match update.image {
                CursorImageSelection::Keep => {}
                CursorImageSelection::Select(image) => self.selected = Some(*image),
                CursorImageSelection::Clear => self.selected = None,
            }
            self.position = update.position;
            self.visible = update.visible;
            Ok(())
        }

        fn destroy_cursor_image(
            &mut self,
            image: Self::Image,
        ) -> Result<(), CursorImageReleaseFailure<Self::Image>> {
            if self.selected == Some(image) {
                Err(CursorImageReleaseFailure::Rejected {
                    error: GraphicsError::Busy,
                    image,
                })
            } else {
                Ok(())
            }
        }
    }

    impl<A: Allocator> MockBackend<A> {
        fn new_in(features: Features, allocator: A) -> Self {
            Self {
                capabilities: GraphicsCapabilities {
                    features,
                    limits: GraphicsLimits::new(
                        Extent2d::new(4096, 4096).unwrap(),
                        16_384,
                        8_192,
                        256,
                        4,
                    )
                    .unwrap(),
                },
                accepted: 0,
                _allocator: allocator,
            }
        }
    }

    impl<A: Allocator> GraphicsBackend<A> for MockBackend<A> {
        type Target = MockTarget;
        type Submission = MockSubmission;

        fn capabilities(&self) -> GraphicsCapabilities {
            self.capabilities
        }

        fn target_support(
            &self,
            _target: &Self::Target,
            surface: SurfaceInfo,
        ) -> Option<TargetSupport> {
            (surface.format() == PixelFormat::XRGB8888_SRGB
                && surface.modifier() == SurfaceModifier::LINEAR)
                .then(|| TargetSupport::direct(VisibilityMode::Coherent, RetainUntil::Presented))
        }

        fn submit(
            &mut self,
            request: RenderRequest<'_, Self::Target, A>,
        ) -> Result<Self::Submission, SubmitFailure<Self::Submission>> {
            let support = self.target_support(request.target(), request.surface());
            validate_render_request(self.capabilities(), support, &request)
                .map_err(SubmitFailure::Rejected)?;
            self.accepted += 1;
            Ok(MockSubmission {
                tracker: SubmissionTracker::new(request.surface_epoch()),
            })
        }

        fn poll_submission(
            &mut self,
            submission: &mut Self::Submission,
        ) -> Result<SubmissionState, GraphicsError> {
            match submission.tracker.state() {
                SubmissionState::Pending => submission
                    .tracker
                    .mark_device_complete(DeviceEpoch::new(self.accepted).unwrap())
                    .map_err(|_| GraphicsError::InvalidArgument),
                SubmissionState::DeviceComplete(device) => submission
                    .tracker
                    .mark_visible(
                        device,
                        VisibilityEpoch::new(submission.tracker.surface_epoch().get()).unwrap(),
                    )
                    .map_err(|_| GraphicsError::InvalidArgument),
                terminal => Ok(terminal),
            }
        }

        fn cancel_submission(
            &mut self,
            submission: &mut Self::Submission,
        ) -> Result<SubmissionState, GraphicsError> {
            submission
                .tracker
                .mark_cancelled(TargetReuse::Safe)
                .map_err(|_| GraphicsError::InvalidArgument)
        }

        fn release_submission(
            &mut self,
            submission: Self::Submission,
        ) -> Result<(), ReleaseFailure<Self::Submission>> {
            if submission.tracker.state().is_terminal() {
                Ok(())
            } else {
                Err(ReleaseFailure::Rejected {
                    error: GraphicsError::Busy,
                    submission,
                })
            }
        }
    }

    fn surface(extent: Extent2d) -> SurfaceInfo {
        let row_bytes = u64::from(extent.width()) * 4;
        let bytes = row_bytes * u64::from(extent.height());
        SurfaceInfo::linear_xrgb8888(
            extent,
            ByteRange::new(0, bytes).unwrap(),
            row_bytes,
            SurfaceUsage::RENDER_TARGET.union(SurfaceUsage::PRESENT),
        )
        .unwrap()
    }

    fn frame(paint: Paint) -> Frame<std::alloc::System> {
        let mut recorder =
            Recorder::new_in(Rect::new(0.0, 0.0, 320.0, 240.0), 2.0, std::alloc::System);
        recorder.draw_quad(Quad {
            bounds: Rect::new(8.0, 12.0, 100.0, 40.0),
            paint,
            border: Border {
                color: Color::new(0.2, 0.3, 0.4, 1.0),
                width: 1.0,
                radii: CornerRadii {
                    top_left: 4.0,
                    top_right: 4.0,
                    bottom_right: 4.0,
                    bottom_left: 4.0,
                },
            },
            shadow: Shadow {
                color: Color::new(0.0, 0.0, 0.0, 0.4),
                offset: Point::new(0.0, 2.0),
                blur_radius: 6.0,
            },
            snap: true,
        });
        recorder.take_frame().unwrap()
    }

    fn solid_features() -> Features {
        Features::SOLID_QUAD
            .union(Features::BORDER)
            .union(Features::SHADOW)
            .union(Features::PIXEL_SNAP)
    }

    #[test]
    fn linear_xrgb_layout_accepts_kore_style_pitch_padding() {
        let extent = Extent2d::new(1265, 795).unwrap();
        let stride = 5_120;
        let bytes = stride * u64::from(extent.height());
        let surface = SurfaceInfo::linear_xrgb8888(
            extent,
            ByteRange::new(2 * 1024 * 1024, bytes).unwrap(),
            stride,
            SurfaceUsage::RENDER_TARGET.union(SurfaceUsage::PRESENT),
        )
        .unwrap();

        assert_eq!(surface.range().offset(), 2 * 1024 * 1024);
        assert_eq!(surface.planes()[0].row_stride(), 5_120);
    }

    #[test]
    fn staged_target_support_does_not_claim_direct_binding() {
        let support = TargetSupport::staged(VisibilityMode::Coherent, RetainUntil::Visible);

        assert_eq!(support.route, TargetRoute::Staged);
        assert_eq!(support.visibility, VisibilityMode::Coherent);
        assert_eq!(support.retain_until, RetainUntil::Visible);
    }

    #[test]
    fn plane_ranges_are_exact_and_nonoverlapping() {
        let extent = Extent2d::new(4, 4).unwrap();
        let plane = PlaneLayout::new(0, 16, 16, 4).unwrap();
        assert_eq!(
            SurfaceInfo::new(
                PixelFormat::PLANAR_RGB_F32_SRGB,
                SurfaceModifier::LINEAR,
                extent,
                ByteRange::new(0, 192).unwrap(),
                SurfaceUsage::RENDER_TARGET,
                &[plane, plane, plane],
            ),
            Err(ContractError::PlaneOverlap)
        );
    }

    #[test]
    fn backend_and_presentation_epochs_gate_reuse_independently() {
        let mut backend = MockBackend::new_in(solid_features(), std::alloc::System);
        let frame = frame(Paint::Solid(Color::new(0.8, 0.4, 0.2, 1.0)));
        let extent = Extent2d::new(640, 480).unwrap();
        let surface = surface(extent);
        let epoch = SurfaceEpoch::new(7).unwrap();
        let mut target = MockTarget;
        let support = backend.target_support(&target, surface).unwrap();
        let damage = [PixelRect::new(16, 24, 200, 80).unwrap()];
        let request = RenderRequest::new(
            frame.view(),
            &mut target,
            surface,
            epoch,
            PixelRect::full(extent),
            Damage::Rects(&damage),
        );
        let mut submission = backend.submit(request).unwrap();
        let mut retirement = SurfaceRetirement::new(epoch, support.retain_until);

        let device_complete = backend.poll_submission(&mut submission).unwrap();
        retirement.observe(device_complete).unwrap();
        assert!(!retirement.can_commit());
        assert!(!retirement.can_release_submission());

        let visible = backend.poll_submission(&mut submission).unwrap();
        retirement.observe(visible).unwrap();
        retirement.mark_committed().unwrap();
        assert!(!retirement.can_release_submission());
        assert!(!retirement.is_reusable());

        retirement
            .mark_presented(SurfaceEpoch::new(9).unwrap())
            .unwrap();
        assert!(retirement.can_release_submission());
        backend.release_submission(submission).unwrap();
        retirement.mark_submission_released().unwrap();
        assert!(retirement.is_reusable());
    }

    #[test]
    fn unsupported_gradient_is_rejected_before_acceptance() {
        let mut backend = MockBackend::new_in(solid_features(), std::alloc::System);
        let frame = frame(Paint::LinearGradient(LinearGradient {
            angle: 0.0,
            stops: [
                Some(GradientStop {
                    offset: 0.0,
                    color: Color::new(0.0, 0.0, 0.0, 1.0),
                }),
                Some(GradientStop {
                    offset: 1.0,
                    color: Color::new(1.0, 1.0, 1.0, 1.0),
                }),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        }));
        let extent = Extent2d::new(640, 480).unwrap();
        let surface = surface(extent);
        let mut target = MockTarget;
        let request = RenderRequest::new(
            frame.view(),
            &mut target,
            surface,
            SurfaceEpoch::new(1).unwrap(),
            PixelRect::full(extent),
            Damage::Full,
        );

        assert!(matches!(
            backend.submit(request),
            Err(SubmitFailure::Rejected(GraphicsError::Unsupported))
        ));
        assert_eq!(backend.accepted, 0);
    }

    #[test]
    fn viewport_rounding_is_part_of_the_contract() {
        let capabilities = MockBackend::new_in(solid_features(), std::alloc::System).capabilities();
        let frame = frame(Paint::Solid(Color::new(1.0, 1.0, 1.0, 1.0)));
        let extent = Extent2d::new(641, 480).unwrap();
        let surface = surface(extent);
        let mut target = MockTarget;
        let request = RenderRequest::new(
            frame.view(),
            &mut target,
            surface,
            SurfaceEpoch::new(1).unwrap(),
            PixelRect::full(extent),
            Damage::Full,
        );
        let support = Some(TargetSupport::direct(
            VisibilityMode::Coherent,
            RetainUntil::Visible,
        ));

        assert_eq!(
            validate_render_request(capabilities, support, &request),
            Err(GraphicsError::InvalidArgument)
        );
    }

    #[test]
    fn quarantined_failure_never_releases_or_reuses_the_target() {
        let epoch = SurfaceEpoch::new(3).unwrap();
        let mut tracker = SubmissionTracker::new(epoch);
        let failed = tracker
            .mark_failed(GraphicsError::DeviceLost, TargetReuse::Quarantine)
            .unwrap();
        let mut retirement = SurfaceRetirement::new(epoch, RetainUntil::Visible);
        retirement.observe(failed).unwrap();

        assert!(!retirement.can_commit());
        assert!(!retirement.can_release_submission());
        assert!(!retirement.is_reusable());
    }

    #[test]
    fn cursor_image_validation_checks_plane_limits_and_hotspot() {
        let capabilities = CursorPlaneCapabilities {
            max_extent: Extent2d::new(64, 64).unwrap(),
            formats: CursorFormats::ARGB8888_PREMULTIPLIED,
            independent_position: true,
        };
        let pixels = [0_u8; 16 * 12 * 4];
        let image = CursorImage {
            extent: Extent2d::new(16, 12).unwrap(),
            hotspot: CursorHotspot { x: 2, y: 3 },
            format: CursorFormat::Argb8888Premultiplied,
            row_stride: 64,
            pixels: &pixels,
        };

        assert_eq!(image.validate(capabilities), Ok(()));
        assert_eq!(
            CursorImage {
                hotspot: CursorHotspot { x: 16, y: 3 },
                ..image
            }
            .validate(capabilities),
            Err(GraphicsError::InvalidArgument)
        );
    }

    #[test]
    fn cursor_image_validation_refuses_unsupported_alpha_layout() {
        let capabilities = CursorPlaneCapabilities {
            max_extent: Extent2d::new(64, 64).unwrap(),
            formats: CursorFormats::ARGB8888_PREMULTIPLIED,
            independent_position: true,
        };
        let pixels = [0_u8; 4];
        let image = CursorImage {
            extent: Extent2d::new(1, 1).unwrap(),
            hotspot: CursorHotspot::default(),
            format: CursorFormat::Rgba8888Premultiplied,
            row_stride: 4,
            pixels: &pixels,
        };

        assert_eq!(
            image.validate(capabilities),
            Err(GraphicsError::Unsupported)
        );
    }

    #[test]
    fn cursor_plane_position_hot_path_preserves_owned_image_lifecycle() {
        let pixels = [0xff_u8; 4 * 4 * 4];
        let mut plane = MockCursorPlane::new_in(std::alloc::System);
        let image = plane
            .create_cursor_image(CursorImage {
                extent: Extent2d::new(4, 4).unwrap(),
                hotspot: CursorHotspot { x: 1, y: 1 },
                format: CursorFormat::Argb8888Premultiplied,
                row_stride: 16,
                pixels: &pixels,
            })
            .unwrap();
        plane
            .update_cursor(CursorPlaneUpdate {
                position: PointerPosition::new(10, 12),
                visible: true,
                image: CursorImageSelection::Select(&image),
            })
            .unwrap();
        plane
            .update_cursor(CursorPlaneUpdate {
                position: PointerPosition::new(30, 40),
                visible: true,
                image: CursorImageSelection::Keep,
            })
            .unwrap();
        assert_eq!(plane.selected, Some(image));
        assert_eq!(plane.position, PointerPosition::new(30, 40));

        let image = match plane.destroy_cursor_image(image) {
            Err(CursorImageReleaseFailure::Rejected {
                error: GraphicsError::Busy,
                image,
            }) => image,
            other => panic!("selected cursor image release was not rejected: {other:?}"),
        };
        plane
            .update_cursor(CursorPlaneUpdate {
                position: plane.position,
                visible: false,
                image: CursorImageSelection::Clear,
            })
            .unwrap();
        plane.destroy_cursor_image(image).unwrap();
    }
}
