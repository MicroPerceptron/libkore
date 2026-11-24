//! The indexed glyph atlas and its storage seam.

use alloc::alloc::Global;
use alloc::vec::Vec;
use core::alloc::Allocator;

use super::{
    DEFAULT_SUPERSAMPLING, FontError, FontFace, GlyphKey, GlyphMetrics, GlyphSource, valid_px,
};
use crate::PlacedGlyph;

/// Default number of distinct glyph rasters an owned atlas can index.
pub const DEFAULT_GLYPH_CAPACITY: u16 = 1024;

/// One index slot: the key hash and the entry it names.
///
/// The live prefix of the index is sorted by hash; equal hashes sit next to
/// each other and are told apart by a full [`GlyphKey`] compare.
pub type IndexSlot = (u64, u16);

/// A cached raster: its key, where it lives in the atlas, and the placement
/// metrics the layout needs on a hit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtlasGlyph {
    /// The key this entry answers.
    pub key: GlyphKey,
    /// Atlas column of the stored bitmap's top-left pixel.
    pub ax: u16,
    /// Atlas row of the stored bitmap's top-left pixel.
    pub ay: u16,
    /// Stored bitmap width (native width plus one if `phase_x != 0`).
    pub aw: u16,
    /// Stored bitmap height (native height plus one if `phase_y != 0`).
    pub ah: u16,
    /// Whole-pixel left bearing from the pen.
    pub xmin: i16,
    /// Whole-pixel bottom bearing from the baseline (Y up).
    pub ymin: i16,
    /// Native bitmap height before phase expansion.
    pub height: u16,
    /// Horizontal advance in pixels.
    pub advance: f32,
}

impl AtlasGlyph {
    const EMPTY: Self = Self {
        key: GlyphKey {
            font: 0,
            glyph: 0,
            px_bits: 0,
            phase_x: 0,
            phase_y: 0,
        },
        ax: 0,
        ay: 0,
        aw: 0,
        ah: 0,
        xmin: 0,
        ymin: 0,
        height: 0,
        advance: 0.0,
    };

    /// Whether the entry stores no coverage.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.aw == 0 || self.ah == 0
    }

    /// The entry as a [`PlacedGlyph`] at the node origin.
    #[inline]
    pub const fn placed(&self) -> PlacedGlyph {
        PlacedGlyph {
            ax: self.ax,
            ay: self.ay,
            aw: self.aw,
            ah: self.ah,
            dx: 0,
            dy: 0,
            subpixel_x: 0,
            subpixel_y: 0,
        }
    }
}

/// Backing memory for a [`GlyphAtlas`].
///
/// Every method hands out the *full capacity* of a plane or table; the
/// atlas tracks how much of the index and entry tables is live.  The index
/// and entry slices must be the same length; the coverage plane must hold
/// exactly `width × height` samples; the optional packed plane, when
/// present, the same count of `u32`s.
pub trait AtlasStorage {
    /// Linear coverage plane, row-major, `0..=65535`.
    fn coverage(&self) -> &[u16];
    /// Mutable coverage plane.
    fn coverage_mut(&mut self) -> &mut [u16];
    /// Premultiplied-white packed view (`AAAAAAAA` per channel), if this
    /// storage maintains one.
    fn packed(&self) -> Option<&[u32]>;
    /// Both planes at once, so an insert can keep them in step.
    fn planes_mut(&mut self) -> (&mut [u16], Option<&mut [u32]>);
    /// Hash index table (capacity).
    fn index(&self) -> &[IndexSlot];
    /// Mutable hash index table.
    fn index_mut(&mut self) -> &mut [IndexSlot];
    /// Entry table (capacity).
    fn entries(&self) -> &[AtlasGlyph];
    /// Mutable entry table.
    fn entries_mut(&mut self) -> &mut [AtlasGlyph];
}

/// Storage whose planes and tables live in one explicit allocator.
///
/// Everything is reserved once, in the constructor; nothing grows.
#[derive(Clone, Debug)]
pub struct OwnedStorage<A: Allocator> {
    coverage: Vec<u16, A>,
    packed: Option<Vec<u32, A>>,
    index: Vec<IndexSlot, A>,
    entries: Vec<AtlasGlyph, A>,
}

impl<A: Allocator> AtlasStorage for OwnedStorage<A> {
    #[inline]
    fn coverage(&self) -> &[u16] {
        &self.coverage
    }
    #[inline]
    fn coverage_mut(&mut self) -> &mut [u16] {
        &mut self.coverage
    }
    #[inline]
    fn packed(&self) -> Option<&[u32]> {
        self.packed.as_deref()
    }
    #[inline]
    fn planes_mut(&mut self) -> (&mut [u16], Option<&mut [u32]>) {
        (&mut self.coverage, self.packed.as_deref_mut())
    }
    #[inline]
    fn index(&self) -> &[IndexSlot] {
        &self.index
    }
    #[inline]
    fn index_mut(&mut self) -> &mut [IndexSlot] {
        &mut self.index
    }
    #[inline]
    fn entries(&self) -> &[AtlasGlyph] {
        &self.entries
    }
    #[inline]
    fn entries_mut(&mut self) -> &mut [AtlasGlyph] {
        &mut self.entries
    }
}

/// Storage over a caller-owned coverage plane with an inline index of
/// `CAP` glyphs.  Never allocates.
#[derive(Debug)]
pub struct BorrowedStorage<'a, const CAP: usize> {
    coverage: &'a mut [u16],
    index: [IndexSlot; CAP],
    entries: [AtlasGlyph; CAP],
}

impl<const CAP: usize> AtlasStorage for BorrowedStorage<'_, CAP> {
    #[inline]
    fn coverage(&self) -> &[u16] {
        self.coverage
    }
    #[inline]
    fn coverage_mut(&mut self) -> &mut [u16] {
        self.coverage
    }
    #[inline]
    fn packed(&self) -> Option<&[u32]> {
        None
    }
    #[inline]
    fn planes_mut(&mut self) -> (&mut [u16], Option<&mut [u32]>) {
        (self.coverage, None)
    }
    #[inline]
    fn index(&self) -> &[IndexSlot] {
        &self.index
    }
    #[inline]
    fn index_mut(&mut self) -> &mut [IndexSlot] {
        &mut self.index
    }
    #[inline]
    fn entries(&self) -> &[AtlasGlyph] {
        &self.entries
    }
    #[inline]
    fn entries_mut(&mut self) -> &mut [AtlasGlyph] {
        &mut self.entries
    }
}

/// The owned, global-allocator atlas that also keeps a packed `u32` view.
///
/// [`FontAtlas::new`] / [`FontAtlas::with_padding`] /
/// [`FontAtlas::with_supersampling`] build it; `pixels()` and
/// `pixel_bytes()` expose the packed plane for the immediate path.
pub type FontAtlas = GlyphAtlas<OwnedStorage<Global>>;

/// A fixed-size, shelf-packed, hash-indexed glyph atlas.
///
/// Glyphs are packed left to right into rows of the tallest glyph seen on
/// that row, with `padding` transparent pixels around each.  Every raster
/// is keyed by [`GlyphKey`] and found by binary search over the sorted
/// index; a miss rasterises the glyph from its [`GlyphSource`] directly
/// into the plane.  Entries are never evicted: a full plane or index is an
/// error, and the caller decides whether to size a bigger atlas or reset.
#[derive(Clone, Debug)]
pub struct GlyphAtlas<S: AtlasStorage> {
    storage: S,
    width: u16,
    height: u16,
    len: u16,
    cursor_x: u16,
    cursor_y: u16,
    row_height: u16,
    padding: u16,
    supersampling: u8,
}

fn plane_len(width: u16, height: u16) -> Result<usize, FontError> {
    if width == 0 || height == 0 {
        return Err(FontError::InvalidAtlasSize);
    }
    usize::from(width)
        .checked_mul(usize::from(height))
        .ok_or(FontError::AtlasSizeOverflow)
}

fn reserve_zeroed<T: Copy, A: Allocator>(
    len: usize,
    fill: T,
    allocator: A,
) -> Result<Vec<T, A>, FontError> {
    let mut vec = Vec::new_in(allocator);
    vec.try_reserve_exact(len)
        .map_err(|_| FontError::Allocation)?;
    vec.resize(len, fill);
    Ok(vec)
}

impl<A: Allocator + Clone> GlyphAtlas<OwnedStorage<A>> {
    /// An atlas with a coverage plane only, indexed for
    /// [`DEFAULT_GLYPH_CAPACITY`] glyphs, allocated from `allocator`.
    ///
    /// This is the compositor configuration: the graph path samples the
    /// `u16` plane directly and no packed view is maintained.
    pub fn new_in(width: u16, height: u16, padding: u16, allocator: A) -> Result<Self, FontError> {
        Self::build_in(
            width,
            height,
            padding,
            DEFAULT_GLYPH_CAPACITY,
            false,
            allocator,
        )
    }

    /// [`Self::new_in`] with an explicit index capacity.
    pub fn with_capacity_in(
        width: u16,
        height: u16,
        padding: u16,
        glyph_capacity: u16,
        allocator: A,
    ) -> Result<Self, FontError> {
        Self::build_in(width, height, padding, glyph_capacity, false, allocator)
    }

    /// An atlas that also maintains the packed premultiplied-white `u32`
    /// view (`pixels()`), for callers still on the immediate
    /// raster program.
    pub fn new_packed_in(
        width: u16,
        height: u16,
        padding: u16,
        glyph_capacity: u16,
        allocator: A,
    ) -> Result<Self, FontError> {
        Self::build_in(width, height, padding, glyph_capacity, true, allocator)
    }

    fn build_in(
        width: u16,
        height: u16,
        padding: u16,
        glyph_capacity: u16,
        packed: bool,
        allocator: A,
    ) -> Result<Self, FontError> {
        let len = plane_len(width, height)?;
        let coverage = reserve_zeroed(len, 0u16, allocator.clone())?;
        let packed = if packed {
            Some(reserve_zeroed(len, 0u32, allocator.clone())?)
        } else {
            None
        };
        let index = reserve_zeroed(usize::from(glyph_capacity), (0u64, 0u16), allocator.clone())?;
        let entries = reserve_zeroed(usize::from(glyph_capacity), AtlasGlyph::EMPTY, allocator)?;
        Ok(Self::from_storage(
            OwnedStorage {
                coverage,
                packed,
                index,
                entries,
            },
            width,
            height,
            padding,
        ))
    }

    /// Premultiplied-white compatibility pixels in row-major order.
    ///
    /// Empty unless the atlas was built with a packed plane
    /// ([`FontAtlas::new`] and friends, or [`Self::new_packed_in`]).
    #[inline]
    pub fn pixels(&self) -> &[u32] {
        self.storage.packed().unwrap_or(&[])
    }

    /// Native-endian bytes for uploading [`Self::pixels`] into an I32
    /// `[height, width]` tensor.  Zero-copy.
    #[inline]
    pub fn pixel_bytes(&self) -> &[u8] {
        let pixels = self.pixels();
        // `u8` has alignment 1, and a byte slice may view every initialized
        // byte of a `u32` slice. `size_of_val` avoids a second length
        // calculation and cannot overflow.
        unsafe {
            core::slice::from_raw_parts(
                pixels.as_ptr().cast::<u8>(),
                core::mem::size_of_val(pixels),
            )
        }
    }
}

impl FontAtlas {
    /// Build an empty atlas with one transparent-pixel gutter around glyphs.
    pub fn new(width: u16, height: u16) -> Result<Self, FontError> {
        Self::with_supersampling(width, height, 1, DEFAULT_SUPERSAMPLING)
    }

    /// Build an empty atlas with an explicit transparent gutter per glyph.
    pub fn with_padding(width: u16, height: u16, padding: u16) -> Result<Self, FontError> {
        Self::with_supersampling(width, height, padding, DEFAULT_SUPERSAMPLING)
    }

    /// Build an atlas with an explicit gutter and raster supersampling scale.
    ///
    /// The supersampling applies to the [`FontFace`]-specific helpers
    /// ([`FontFace::layout_line`], [`Self::cache_glyph_id`], …).  The generic
    /// [`super::layout_line_into`] path uses the source's own setting
    /// ([`FontFace::with_supersampling`]).
    pub fn with_supersampling(
        width: u16,
        height: u16,
        padding: u16,
        supersampling: u8,
    ) -> Result<Self, FontError> {
        if !(1..=8).contains(&supersampling) {
            return Err(FontError::InvalidSupersampling);
        }
        let mut atlas =
            Self::new_packed_in(width, height, padding, DEFAULT_GLYPH_CAPACITY, Global)?;
        atlas.supersampling = supersampling;
        Ok(atlas)
    }
}

impl<'a, const CAP: usize> GlyphAtlas<BorrowedStorage<'a, CAP>> {
    /// Wrap a caller-owned coverage plane (exactly `width × height`
    /// samples, which are zeroed) with a one-pixel gutter.  Never
    /// allocates; the index lives inline and holds `CAP` glyphs.
    pub fn from_slice(width: u16, height: u16, coverage: &'a mut [u16]) -> Result<Self, FontError> {
        Self::from_slice_with_padding(width, height, 1, coverage)
    }

    /// [`Self::from_slice`] with an explicit gutter.
    pub fn from_slice_with_padding(
        width: u16,
        height: u16,
        padding: u16,
        coverage: &'a mut [u16],
    ) -> Result<Self, FontError> {
        if CAP > usize::from(u16::MAX) {
            return Err(FontError::AtlasSizeOverflow);
        }
        if coverage.len() != plane_len(width, height)? {
            return Err(FontError::InvalidAtlasSize);
        }
        coverage.fill(0);
        Ok(Self::from_storage(
            BorrowedStorage {
                coverage,
                index: [(0, 0); CAP],
                entries: [AtlasGlyph::EMPTY; CAP],
            },
            width,
            height,
            padding,
        ))
    }
}

impl<S: AtlasStorage> GlyphAtlas<S> {
    fn from_storage(storage: S, width: u16, height: u16, padding: u16) -> Self {
        debug_assert_eq!(
            storage.coverage().len(),
            usize::from(width) * usize::from(height)
        );
        debug_assert_eq!(storage.index().len(), storage.entries().len());
        Self {
            storage,
            width,
            height,
            len: 0,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            padding,
            supersampling: DEFAULT_SUPERSAMPLING,
        }
    }

    /// Atlas width in pixels.
    #[inline]
    pub const fn width(&self) -> u16 {
        self.width
    }

    /// Atlas height in pixels.
    #[inline]
    pub const fn height(&self) -> u16 {
        self.height
    }

    /// Transparent gutter around each glyph.
    #[inline]
    pub const fn padding(&self) -> u16 {
        self.padding
    }

    /// Raster samples per output pixel used by the [`FontFace`]-specific
    /// helpers on this atlas.
    #[inline]
    pub const fn supersampling(&self) -> u8 {
        self.supersampling
    }

    /// High-precision linear glyph coverage in row-major order.
    ///
    /// Values span the complete `0..=65535` interval.  This is the plane
    /// both the `.ax` graph and the raster backends sample.
    #[inline]
    pub fn coverage(&self) -> &[u16] {
        self.storage.coverage()
    }

    /// Native-endian bytes of [`Self::coverage`], for uploading into a
    /// `[height, width]` sixteen-bit tensor.  Zero-copy.
    #[inline]
    pub fn coverage_bytes(&self) -> &[u8] {
        let coverage = self.coverage();
        // See `pixel_bytes`: a byte view of an initialized `u16` slice.
        unsafe {
            core::slice::from_raw_parts(
                coverage.as_ptr().cast::<u8>(),
                core::mem::size_of_val(coverage),
            )
        }
    }

    /// The packed premultiplied-white view, when the storage keeps one.
    #[inline]
    pub fn packed_pixels(&self) -> Option<&[u32]> {
        self.storage.packed()
    }

    /// Borrow the underlying storage.
    #[inline]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Number of distinct glyph rasters currently cached.
    #[inline]
    pub fn glyph_count(&self) -> usize {
        usize::from(self.len)
    }

    /// Number of distinct glyph rasters the index can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.storage.index().len()
    }

    /// Live cache entries in insertion order.
    #[inline]
    pub fn entries(&self) -> &[AtlasGlyph] {
        &self.storage.entries()[..usize::from(self.len)]
    }

    /// Find a cached raster by key.
    #[inline]
    pub fn lookup(&self, key: &GlyphKey) -> Option<AtlasGlyph> {
        let live = usize::from(self.len);
        find_in_index(
            &self.storage.index()[..live],
            &self.storage.entries()[..live],
            key,
            key.hash(),
        )
    }

    /// Fetch or build the raster for `glyph` at `px` and phase from `source`.
    ///
    /// A hit is a hash, a binary search and a copy.  A miss reserves plane
    /// space, rasterises into it, and inserts the entry.
    pub fn entry<G: GlyphSource + ?Sized>(
        &mut self,
        source: &G,
        glyph: u16,
        px: f32,
        phase_x: u8,
        phase_y: u8,
    ) -> Result<AtlasGlyph, FontError> {
        let key = GlyphKey {
            font: source.font_id(),
            glyph,
            px_bits: px.to_bits(),
            phase_x,
            phase_y,
        };
        if let Some(hit) = self.lookup(&key) {
            return Ok(hit);
        }
        if !valid_px(px) {
            return Err(FontError::InvalidPixelSize);
        }
        if usize::from(self.len) >= self.capacity() {
            return Err(FontError::IndexFull);
        }
        let metrics = source.metrics(glyph, px);
        let entry = self.insert_raster(key, metrics, |atlas_row, stride| {
            source.rasterize_into(glyph, px, phase_x, phase_y, atlas_row, stride)
        })?;
        self.push_entry(entry);
        Ok(entry)
    }

    /// Cache a glyph ID explicitly, for use with an external text shaper.
    ///
    /// Uses this atlas's [`Self::supersampling`] setting.
    pub fn cache_glyph_id(
        &mut self,
        face: &FontFace<'_>,
        glyph_index: u16,
        px: f32,
    ) -> Result<PlacedGlyph, FontError> {
        self.cache_glyph_id_at_phase(face, glyph_index, px, 0, 0)
    }

    /// Cache an externally shaped glyph directly at its final pixel phase.
    ///
    /// Uses this atlas's [`Self::supersampling`] setting.
    pub fn cache_glyph_id_at_phase(
        &mut self,
        face: &FontFace<'_>,
        glyph_index: u16,
        px: f32,
        phase_x: u8,
        phase_y: u8,
    ) -> Result<PlacedGlyph, FontError> {
        let source = face.with_atlas_supersampling(self.supersampling);
        if !valid_px(px) {
            return Ok(AtlasGlyph::EMPTY.placed());
        }
        self.entry(&source, glyph_index, px, phase_x, phase_y)
            .map(|entry| entry.placed())
    }

    /// Reserve space for `metrics` at the phase in `key`, run `raster` over
    /// the reserved rows, and derive the packed view.  Does not touch the
    /// index.
    fn insert_raster(
        &mut self,
        key: GlyphKey,
        metrics: GlyphMetrics,
        raster: impl FnOnce(&mut [u16], usize) -> Result<GlyphMetrics, FontError>,
    ) -> Result<AtlasGlyph, FontError> {
        let (aw, ah) = metrics.stored_extent(key.phase_x, key.phase_y);
        // The entry is derived from the metrics the rasteriser *reports*,
        // not the ones the reservation was sized from: the raster is what
        // the atlas will place, so its bearings and advance are the truth.
        // The two must agree on placement — extent (the reservation is
        // committed) and bearings (the index would otherwise place a raster
        // it did not draw); a source that drifts is refused rather than
        // cached wrong.  The advance is the rasteriser's alone.
        let entry_for = |produced: GlyphMetrics,
                         ax: u16,
                         ay: u16|
         -> Result<AtlasGlyph, FontError> {
            if produced.width != metrics.width
                || produced.height != metrics.height
                || produced.xmin != metrics.xmin
                || produced.ymin != metrics.ymin
            {
                return Err(FontError::InvalidRaster);
            }
            Ok(AtlasGlyph {
                key,
                ax,
                ay,
                aw,
                ah,
                xmin: i16::try_from(produced.xmin).map_err(|_| FontError::CoordinateOverflow)?,
                ymin: i16::try_from(produced.ymin).map_err(|_| FontError::CoordinateOverflow)?,
                height: produced.height,
                advance: produced.advance,
            })
        };
        if aw == 0 || ah == 0 {
            return entry_for(metrics, 0, 0);
        }
        let (ax, ay) = self.reserve(aw, ah)?;
        let stride = usize::from(self.width);
        let start = usize::from(ay) * stride + usize::from(ax);
        let (coverage, packed) = self.storage.planes_mut();
        let produced = raster(&mut coverage[start..], stride)?;
        let entry = entry_for(produced, ax, ay)?;
        if let Some(packed) = packed {
            for row in 0..usize::from(ah) {
                let row_start = start + row * stride;
                let span = row_start..row_start + usize::from(aw);
                for (dst, &alpha) in packed[span.clone()].iter_mut().zip(&coverage[span]) {
                    let alpha = (u32::from(alpha) + 128) / 257;
                    *dst = (alpha << 24) | (alpha << 16) | (alpha << 8) | alpha;
                }
            }
        }
        Ok(entry)
    }

    /// Append `entry` to the tables and splice its hash into the sorted
    /// index.  The caller has checked capacity.
    fn push_entry(&mut self, entry: AtlasGlyph) {
        let live = usize::from(self.len);
        let hash = entry.key.hash();
        let index = self.storage.index_mut();
        let at = index[..live].partition_point(|slot| slot.0 <= hash);
        index.copy_within(at..live, at + 1);
        index[at] = (hash, self.len);
        self.storage.entries_mut()[live] = entry;
        self.len += 1;
    }

    fn reserve(&mut self, width: u16, height: u16) -> Result<(u16, u16), FontError> {
        let double_padding = self
            .padding
            .checked_mul(2)
            .ok_or(FontError::CoordinateOverflow)?;
        let padded_width = width
            .checked_add(double_padding)
            .ok_or(FontError::CoordinateOverflow)?;
        let padded_height = height
            .checked_add(double_padding)
            .ok_or(FontError::CoordinateOverflow)?;

        if padded_width > self.width || padded_height > self.height {
            return Err(FontError::AtlasFull);
        }
        if self
            .cursor_x
            .checked_add(padded_width)
            .ok_or(FontError::AtlasFull)?
            > self.width
        {
            self.cursor_x = 0;
            self.cursor_y = self
                .cursor_y
                .checked_add(self.row_height)
                .ok_or(FontError::AtlasFull)?;
            self.row_height = 0;
        }
        if self
            .cursor_y
            .checked_add(padded_height)
            .ok_or(FontError::AtlasFull)?
            > self.height
        {
            return Err(FontError::AtlasFull);
        }

        let ax = self
            .cursor_x
            .checked_add(self.padding)
            .ok_or(FontError::CoordinateOverflow)?;
        let ay = self
            .cursor_y
            .checked_add(self.padding)
            .ok_or(FontError::CoordinateOverflow)?;
        self.cursor_x = self
            .cursor_x
            .checked_add(padded_width)
            .ok_or(FontError::AtlasFull)?;
        self.row_height = self.row_height.max(padded_height);
        Ok((ax, ay))
    }

    #[cfg(feature = "graph")]
    /// The coverage plane as a borrowed [`crate::graph::AlphaAtlas`].
    #[inline]
    pub fn alpha_atlas(&self) -> crate::graph::AlphaAtlas<'_> {
        crate::graph::AlphaAtlas::from_u16(
            self.coverage(),
            usize::from(self.width),
            usize::from(self.height),
        )
    }
}

/// Binary-search `index` (sorted by hash) for `key` under `hash`, resolving
/// collisions by comparing the full key of every entry sharing the hash.
fn find_in_index(
    index: &[IndexSlot],
    entries: &[AtlasGlyph],
    key: &GlyphKey,
    hash: u64,
) -> Option<AtlasGlyph> {
    let Ok(hit) = index.binary_search_by_key(&hash, |slot| slot.0) else {
        return None;
    };
    let mut first = hit;
    while first > 0 && index[first - 1].0 == hash {
        first -= 1;
    }
    index[first..]
        .iter()
        .take_while(|slot| slot.0 == hash)
        .map(|slot| entries[usize::from(slot.1)])
        .find(|entry| entry.key == *key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::VerticalMetrics;

    /// A source whose every glyph is a 1×1 solid pixel, so an atlas fills
    /// with many distinct keys cheaply.
    struct Dots(u32);

    impl GlyphSource for Dots {
        fn font_id(&self) -> u32 {
            self.0
        }
        fn glyph_id(&self, c: char) -> u16 {
            c as u16
        }
        fn vertical_metrics(&self, _px: f32) -> VerticalMetrics {
            VerticalMetrics {
                ascent: 1.0,
                descent: 0.0,
                line_gap: 0.0,
            }
        }
        fn metrics(&self, _glyph: u16, _px: f32) -> GlyphMetrics {
            GlyphMetrics {
                xmin: 0,
                ymin: 0,
                width: 1,
                height: 1,
                advance: 1.0,
            }
        }
        fn kern(&self, _l: u16, _r: u16, _px: f32) -> f32 {
            0.0
        }
        fn rasterize_into(
            &self,
            glyph: u16,
            px: f32,
            _phase_x: u8,
            _phase_y: u8,
            out: &mut [u16],
            _stride: usize,
        ) -> Result<GlyphMetrics, FontError> {
            out[0] = glyph.wrapping_mul(977) ^ (px.to_bits() as u16);
            Ok(self.metrics(glyph, px))
        }
    }

    #[test]
    fn atlas_rejects_zero_dimensions() {
        assert!(matches!(
            FontAtlas::new(0, 64),
            Err(FontError::InvalidAtlasSize)
        ));
        assert!(matches!(
            FontAtlas::new(64, 0),
            Err(FontError::InvalidAtlasSize)
        ));
        let mut plane = [0u16; 12];
        assert!(matches!(
            GlyphAtlas::<BorrowedStorage<'_, 4>>::from_slice(4, 4, &mut plane),
            Err(FontError::InvalidAtlasSize)
        ));
    }

    #[test]
    fn raster_entries_are_packed_with_a_transparent_gutter() {
        let mut atlas = FontAtlas::new(8, 4).unwrap();
        let metrics = GlyphMetrics {
            width: 2,
            height: 2,
            ..GlyphMetrics::default()
        };
        let key = GlyphKey {
            font: 1,
            glyph: 7,
            px_bits: 16.0f32.to_bits(),
            phase_x: 0,
            phase_y: 0,
        };
        let glyph = atlas
            .insert_raster(key, metrics, |out, stride| {
                out[0] = 0;
                out[1] = 128 * 257;
                out[stride] = 255 * 257;
                out[stride + 1] = 0;
                Ok(metrics)
            })
            .unwrap();

        assert_eq!((glyph.ax, glyph.ay, glyph.aw, glyph.ah), (1, 1, 2, 2));
        assert_eq!(atlas.pixels()[1 * 8 + 1], 0);
        assert_eq!(atlas.pixels()[1 * 8 + 2], 0x8080_8080);
        assert_eq!(atlas.pixels()[2 * 8 + 1], 0xFFFF_FFFF);
        assert_eq!(atlas.coverage()[1 * 8 + 2], 128 * 257);
        assert_eq!(atlas.pixel_bytes().len(), atlas.pixels().len() * 4);
        assert_eq!(atlas.coverage_bytes().len(), atlas.coverage().len() * 2);
    }

    #[test]
    fn index_lookup_matches_a_linear_scan_over_many_keys() {
        let mut atlas = FontAtlas::new_packed_in(64, 64, 0, 2048, Global).unwrap();
        let dots = [Dots(1), Dots(0xDEAD_BEEF)];
        let sizes = [8.0f32, 12.5, 24.0];
        let mut keys = alloc::vec::Vec::new();
        for (fi, font) in dots.iter().enumerate() {
            for &px in &sizes {
                for glyph in 0..110u16 {
                    let phase = (glyph % 5) as u8 * 51;
                    let entry = atlas.entry(font, glyph, px, phase, 0).unwrap();
                    assert_eq!(entry.key.font, font.font_id());
                    keys.push((fi, entry.key));
                }
            }
        }
        assert!(keys.len() >= 600, "{} keys", keys.len());
        assert_eq!(atlas.glyph_count(), keys.len());

        // Every key: the index and a linear scan agree, and the sample is
        // the one the source wrote.
        for (fi, key) in &keys {
            let linear = atlas.entries().iter().copied().find(|e| e.key == *key);
            let indexed = atlas.lookup(key);
            assert_eq!(indexed, linear, "{key:?}");
            let hit = indexed.unwrap();
            let expected =
                key.glyph.wrapping_mul(977) ^ (f32::from_bits(key.px_bits).to_bits() as u16);
            assert_eq!(
                atlas.coverage()[usize::from(hit.ay) * 64 + usize::from(hit.ax)],
                expected,
                "font {fi} {key:?}",
            );
        }
        // Absent keys miss in both.
        for key in keys.iter().map(|(_, key)| GlyphKey { phase_y: 9, ..*key }) {
            assert!(atlas.lookup(&key).is_none());
            assert!(atlas.entries().iter().all(|e| e.key != key));
        }
        // The index is sorted.
        let index = &atlas.storage().index()[..atlas.glyph_count()];
        assert!(index.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[test]
    fn colliding_hashes_are_resolved_by_full_key_compare() {
        let a = GlyphKey {
            font: 1,
            glyph: 1,
            px_bits: 0,
            phase_x: 0,
            phase_y: 0,
        };
        let b = GlyphKey { glyph: 2, ..a };
        let c = GlyphKey { glyph: 3, ..a };
        let entries = [
            AtlasGlyph {
                key: a,
                ax: 10,
                ..AtlasGlyph::EMPTY
            },
            AtlasGlyph {
                key: b,
                ax: 20,
                ..AtlasGlyph::EMPTY
            },
            AtlasGlyph {
                key: c,
                ax: 30,
                ..AtlasGlyph::EMPTY
            },
        ];
        // Three entries share one (synthetic) hash; one sits below it.
        let shared = 0x5555;
        let index = [(0x1000, 2), (shared, 0), (shared, 1), (shared, 2)];
        assert_eq!(
            find_in_index(&index, &entries, &a, shared).map(|e| e.ax),
            Some(10)
        );
        assert_eq!(
            find_in_index(&index, &entries, &b, shared).map(|e| e.ax),
            Some(20)
        );
        assert_eq!(
            find_in_index(&index, &entries, &c, shared).map(|e| e.ax),
            Some(30)
        );
        let d = GlyphKey { glyph: 4, ..a };
        assert_eq!(find_in_index(&index, &entries, &d, shared), None);
    }

    #[test]
    fn full_plane_and_full_index_are_distinct_errors() {
        // 4×4 plane, no padding: sixteen 1×1 glyphs fit, the seventeenth
        // does not.
        let mut plane = [0u16; 16];
        let mut atlas =
            GlyphAtlas::<BorrowedStorage<'_, 32>>::from_slice_with_padding(4, 4, 0, &mut plane)
                .unwrap();
        let dots = Dots(7);
        for glyph in 0..16 {
            atlas.entry(&dots, glyph, 8.0, 0, 0).unwrap();
        }
        assert_eq!(atlas.entry(&dots, 16, 8.0, 0, 0), Err(FontError::AtlasFull));
        // Hits still answer after the plane filled.
        assert_eq!(atlas.entry(&dots, 3, 8.0, 0, 0).unwrap().ax, 3);

        let mut plane = [0u16; 64];
        let mut atlas =
            GlyphAtlas::<BorrowedStorage<'_, 2>>::from_slice_with_padding(8, 8, 0, &mut plane)
                .unwrap();
        atlas.entry(&dots, 0, 8.0, 0, 0).unwrap();
        atlas.entry(&dots, 1, 8.0, 0, 0).unwrap();
        assert_eq!(atlas.entry(&dots, 2, 8.0, 0, 0), Err(FontError::IndexFull));
        assert_eq!(atlas.capacity(), 2);
        assert_eq!(
            atlas.entry(&dots, 2, 0.0, 0, 0),
            Err(FontError::InvalidPixelSize)
        );
    }

    #[test]
    fn borrowed_storage_does_not_keep_a_packed_view() {
        let mut plane = [0xFFFFu16; 16];
        let atlas = GlyphAtlas::<BorrowedStorage<'_, 4>>::from_slice(4, 4, &mut plane).unwrap();
        assert!(atlas.packed_pixels().is_none());
        assert!(atlas.coverage().iter().all(|&c| c == 0), "plane is zeroed");
        let coverage_only = GlyphAtlas::new_in(4, 4, 1, Global).unwrap();
        assert!(coverage_only.packed_pixels().is_none());
        assert!(coverage_only.pixels().is_empty());
        assert_eq!(
            coverage_only.capacity(),
            usize::from(DEFAULT_GLYPH_CAPACITY)
        );
    }

    /// A source whose raster reports a bearing its metrics did not promise.
    struct Drifting;

    impl GlyphSource for Drifting {
        fn font_id(&self) -> u32 {
            9
        }
        fn glyph_id(&self, c: char) -> u16 {
            c as u16
        }
        fn vertical_metrics(&self, _px: f32) -> VerticalMetrics {
            VerticalMetrics::default()
        }
        fn metrics(&self, _glyph: u16, _px: f32) -> GlyphMetrics {
            GlyphMetrics {
                xmin: 0,
                ymin: 0,
                width: 1,
                height: 1,
                advance: 1.0,
            }
        }
        fn kern(&self, _l: u16, _r: u16, _px: f32) -> f32 {
            0.0
        }
        fn rasterize_into(
            &self,
            glyph: u16,
            px: f32,
            _phase_x: u8,
            _phase_y: u8,
            out: &mut [u16],
            _stride: usize,
        ) -> Result<GlyphMetrics, FontError> {
            out[0] = u16::MAX;
            Ok(GlyphMetrics {
                xmin: 1,
                ..self.metrics(glyph, px)
            })
        }
    }

    #[test]
    fn raster_metrics_that_drift_from_the_promise_are_refused() {
        let mut atlas = FontAtlas::new(8, 8).unwrap();
        assert!(matches!(
            atlas.entry(&Drifting, 1, 8.0, 0, 0),
            Err(FontError::InvalidRaster)
        ));
        // Nothing was indexed for the refused glyph.
        assert_eq!(atlas.len, 0);
    }
}
