use axgfx::{Damage, Extent2d, PixelFormat, PixelRect, SurfaceInfo, SurfaceModifier, SurfaceUsage};

/// Provider-plane encoding accepted by the portable staged materializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarFormat {
    /// Normalized sRGB channels stored as little-endian IEEE F32.
    F32,
    /// Normalized sRGB channels stored as little-endian IEEE F16.
    ///
    /// This encoding is represented so runtimes can reject it truthfully;
    /// the precision-sensitive staged materializer does not consume it.
    F16,
    /// Display bytes stored with the TOSA INT8 `-128` bias.
    U8Biased,
}

/// Invalid provider output or target layout at the staged display boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackError {
    /// The provider format has no portable staged materializer yet.
    UnsupportedFormat,
    /// A provider output plane has the wrong byte length.
    InvalidPlaneLength,
    /// The target is not a linear XRGB8888 surface.
    InvalidTarget,
    /// Provider and target extents differ.
    ExtentMismatch,
    /// A byte count or row offset overflowed the host address space.
    RangeOverflow,
    /// Explicit damage was empty or outside the target extent.
    InvalidDamage,
}

/// Borrowed provider planes in the order `[R, G, B]`.
#[derive(Clone, Copy)]
enum Planes<'a> {
    /// F32 channels still in provider byte form (little-endian).
    F32Bytes([&'a [u8]; 3]),
    /// F32 channels already typed by the caller — sliced straight into the
    /// SIMD pack kernel with no per-chunk decode.
    F32([&'a [f32]; 3]),
    /// Display bytes with the TOSA INT8 `-128` bias.
    U8Biased([&'a [u8]; 3]),
}

/// Three borrowed provider-output planes with an exact physical extent.
#[derive(Clone, Copy)]
pub struct PlanarOutput<'a> {
    planes: Planes<'a>,
    extent: Extent2d,
}

fn extent_pixels(extent: Extent2d) -> Result<usize, PackError> {
    usize::try_from(extent.width())
        .ok()
        .and_then(|width| {
            usize::try_from(extent.height())
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(PackError::RangeOverflow)
}

impl<'a> PlanarOutput<'a> {
    /// Validates a complete planar provider output.
    pub fn new(
        planes: [&'a [u8]; 3],
        format: PlanarFormat,
        extent: Extent2d,
    ) -> Result<Self, PackError> {
        let bytes_per_channel = match format {
            PlanarFormat::F32 => 4,
            PlanarFormat::F16 => return Err(PackError::UnsupportedFormat),
            PlanarFormat::U8Biased => 1,
        };
        let plane_bytes = extent_pixels(extent)?
            .checked_mul(bytes_per_channel)
            .ok_or(PackError::RangeOverflow)?;
        if planes.iter().any(|plane| plane.len() != plane_bytes) {
            return Err(PackError::InvalidPlaneLength);
        }
        let planes = match format {
            PlanarFormat::F32 => Planes::F32Bytes(planes),
            PlanarFormat::U8Biased => Planes::U8Biased(planes),
            PlanarFormat::F16 => unreachable!("rejected above"),
        };
        Ok(Self { planes, extent })
    }

    /// Validates a complete F32 planar provider output from typed planes.
    ///
    /// This is the preferred entry for hosts that already hold `f32`
    /// channel buffers: chunks are handed to the SIMD pack kernel directly,
    /// whereas [`PlanarOutput::new`] must decode little-endian bytes first.
    /// Materialization is bit-identical between the two constructors.
    pub fn f32(planes: [&'a [f32]; 3], extent: Extent2d) -> Result<Self, PackError> {
        let pixels = extent_pixels(extent)?;
        if planes.iter().any(|plane| plane.len() != pixels) {
            return Err(PackError::InvalidPlaneLength);
        }
        Ok(Self {
            planes: Planes::F32(planes),
            extent,
        })
    }

    /// Returns the provider output format.
    pub const fn format(self) -> PlanarFormat {
        match self.planes {
            Planes::F32Bytes(_) | Planes::F32(_) => PlanarFormat::F32,
            Planes::U8Biased(_) => PlanarFormat::U8Biased,
        }
    }

    /// Returns the provider output extent.
    pub const fn extent(self) -> Extent2d {
        self.extent
    }
}

/// A CPU-mapped linear XRGB8888 target expressed as bytes.
pub struct LinearXrgbBytes<'a> {
    bytes: &'a mut [u8],
    offset: usize,
    row_stride: usize,
    extent: Extent2d,
}

impl<'a> LinearXrgbBytes<'a> {
    /// Binds an exact validated surface-range mapping.
    ///
    /// `bytes` begins at [`SurfaceInfo::range`], rather than at the containing
    /// allocation. Plane offsets are relative to that range.
    pub fn from_surface(bytes: &'a mut [u8], surface: SurfaceInfo) -> Result<Self, PackError> {
        if surface.format() != PixelFormat::XRGB8888_SRGB
            || surface.modifier() != SurfaceModifier::LINEAR
            || surface.planes().len() != 1
            || !surface.usage().contains(SurfaceUsage::RENDER_TARGET)
        {
            return Err(PackError::InvalidTarget);
        }
        let plane = surface.planes()[0];
        let offset = usize::try_from(plane.offset()).map_err(|_| PackError::RangeOverflow)?;
        let row_stride =
            usize::try_from(plane.row_stride()).map_err(|_| PackError::RangeOverflow)?;
        let row_bytes = usize::try_from(plane.row_bytes()).map_err(|_| PackError::RangeOverflow)?;
        let rows = usize::try_from(plane.rows()).map_err(|_| PackError::RangeOverflow)?;
        let required = rows
            .checked_sub(1)
            .and_then(|tail| tail.checked_mul(row_stride))
            .and_then(|tail| offset.checked_add(tail))
            .and_then(|last| last.checked_add(row_bytes))
            .ok_or(PackError::RangeOverflow)?;
        let range_bytes =
            usize::try_from(surface.range().bytes()).map_err(|_| PackError::RangeOverflow)?;
        if required > bytes.len() || range_bytes > bytes.len() {
            return Err(PackError::InvalidTarget);
        }
        Ok(Self {
            bytes,
            offset,
            row_stride,
            extent: surface.extent(),
        })
    }
}

/// A host-native word view of a linear XRGB8888 target.
///
/// This is convenient for window APIs such as softbuffer while retaining the
/// same padded-row and damage semantics as [`LinearXrgbBytes`].
pub struct LinearXrgbWords<'a> {
    words: &'a mut [u32],
    row_stride: usize,
    extent: Extent2d,
}

impl<'a> LinearXrgbWords<'a> {
    /// Binds a word target. `row_stride` is measured in `u32` pixels.
    pub fn new(
        words: &'a mut [u32],
        extent: Extent2d,
        row_stride: usize,
    ) -> Result<Self, PackError> {
        let width = usize::try_from(extent.width()).map_err(|_| PackError::RangeOverflow)?;
        let height = usize::try_from(extent.height()).map_err(|_| PackError::RangeOverflow)?;
        let required = height
            .checked_sub(1)
            .and_then(|tail| tail.checked_mul(row_stride))
            .and_then(|tail| tail.checked_add(width))
            .ok_or(PackError::RangeOverflow)?;
        if row_stride < width || required > words.len() {
            return Err(PackError::InvalidTarget);
        }
        Ok(Self {
            words,
            row_stride,
            extent,
        })
    }
}

/// Promotes explicit damage beyond a presenter's bounded rectangle budget to
/// truthful full-surface damage.
pub fn bounded_damage<const N: usize>(damage: Damage<'_>) -> Damage<'_> {
    match damage {
        Damage::Rects(rects) if rects.len() > N => Damage::Full,
        damage => damage,
    }
}

fn validate_damage<'a>(damage: Damage<'a>, extent: Extent2d) -> Result<Damage<'a>, PackError> {
    match damage {
        Damage::Full => Ok(Damage::Full),
        Damage::Rects(rects) if !rects.is_empty() && rects.iter().all(|rect| rect.fits(extent)) => {
            Ok(Damage::Rects(rects))
        }
        Damage::Rects(_) => Err(PackError::InvalidDamage),
    }
}

/// Pixels packed per SIMD call. Bounds the stack staging to 4 KiB (packed)
/// plus 3 KiB (typed decode of byte planes) and keeps a chunk's planes and
/// output resident in L1 while the kernel interleaves them.
const CHUNK: usize = 256;

/// Packs `packed.len()` consecutive pixels starting at plane index `start`.
///
/// Display quantization is `(clamp(v, 0, 1) * 255 + 0.5) as u8` for F32
/// planes and `stored.wrapping_add(128)` for biased bytes — the SIMD kernels
/// are bit-exact with those scalar expressions. The kernels take slices in
/// packed-bit order (first slice → bits 7:0), so planes are passed as
/// `[B, G, R]` to land R in bits 23:16 of the `0xFFRRGGBB` pixel.
fn pack_chunk(planes: Planes<'_>, start: usize, packed: &mut [u32]) {
    let n = packed.len();
    match planes {
        Planes::F32([r, g, b]) => {
            let span = start..start + n;
            axnn_cpu::raster::xrgb_pack_rounded(
                &b[span.clone()],
                &g[span.clone()],
                &r[span],
                packed,
            );
        }
        Planes::F32Bytes(planes) => {
            let mut lanes = [[0f32; CHUNK]; 3];
            for (lane, plane) in lanes.iter_mut().zip(planes) {
                let bytes = &plane[start * 4..(start + n) * 4];
                for (value, quad) in lane.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value = f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
                }
            }
            let [r, g, b] = &lanes;
            axnn_cpu::raster::xrgb_pack_rounded(&b[..n], &g[..n], &r[..n], packed);
        }
        Planes::U8Biased([r, g, b]) => {
            let span = start..start + n;
            axnn_cpu::raster::xrgb_pack_u8_biased(
                &b[span.clone()],
                &g[span.clone()],
                &r[span],
                packed,
            );
        }
    }
}

/// Where packed pixels land: word targets receive each span in place, byte
/// targets stage a span on the stack and copy it as little-endian bytes.
enum RowSink<'t> {
    Words {
        words: &'t mut [u32],
        stride: usize,
    },
    Bytes {
        bytes: &'t mut [u8],
        offset: usize,
        stride: usize,
    },
}

/// Materializes each damaged row in `CHUNK`-pixel spans.
fn materialize(
    output: PlanarOutput<'_>,
    damage: Damage<'_>,
    mut sink: RowSink<'_>,
) -> Result<(), PackError> {
    let extent = output.extent;
    let width = usize::try_from(extent.width()).map_err(|_| PackError::RangeOverflow)?;
    let damage = validate_damage(damage, extent)?;
    let full = PixelRect::full(extent);
    let rects = match damage {
        Damage::Full => core::slice::from_ref(&full),
        Damage::Rects(rects) => rects,
    };
    let mut staged = [0u32; CHUNK];
    for rect in rects {
        let x_start = rect.x() as usize;
        let y_start = rect.y() as usize;
        let x_end = x_start + rect.width() as usize;
        let y_end = y_start + rect.height() as usize;
        for y in y_start..y_end {
            let row = y * width;
            let mut x = x_start;
            while x < x_end {
                let n = (x_end - x).min(CHUNK);
                match &mut sink {
                    RowSink::Words { words, stride } => {
                        let start = y * *stride + x;
                        pack_chunk(output.planes, row + x, &mut words[start..start + n]);
                    }
                    RowSink::Bytes {
                        bytes,
                        offset,
                        stride,
                    } => {
                        pack_chunk(output.planes, row + x, &mut staged[..n]);
                        let start = *offset + y * *stride + x * 4;
                        let span = &mut bytes[start..start + n * 4];
                        for (dst, pixel) in span.chunks_exact_mut(4).zip(&staged[..n]) {
                            dst.copy_from_slice(&pixel.to_le_bytes());
                        }
                    }
                }
                x += n;
            }
        }
    }
    Ok(())
}

/// Materializes provider output into a padded byte-addressed XRGB target.
pub fn pack_planar_xrgb_bytes(
    output: PlanarOutput<'_>,
    target: &mut LinearXrgbBytes<'_>,
    damage: Damage<'_>,
) -> Result<(), PackError> {
    if output.extent != target.extent {
        return Err(PackError::ExtentMismatch);
    }
    materialize(
        output,
        damage,
        RowSink::Bytes {
            bytes: target.bytes,
            offset: target.offset,
            stride: target.row_stride,
        },
    )
}

/// Materializes provider output into a padded word-addressed XRGB target.
pub fn pack_planar_xrgb_words(
    output: PlanarOutput<'_>,
    target: &mut LinearXrgbWords<'_>,
    damage: Damage<'_>,
) -> Result<(), PackError> {
    if output.extent != target.extent {
        return Err(PackError::ExtentMismatch);
    }
    materialize(
        output,
        damage,
        RowSink::Words {
            words: target.words,
            stride: target.row_stride,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axgfx::{ByteRange, SurfaceUsage};
    use std::vec;

    fn biased(values: &[u8]) -> std::vec::Vec<u8> {
        values.iter().map(|value| value.wrapping_sub(128)).collect()
    }

    #[test]
    fn byte_target_honors_padded_stride_and_guard_bytes() {
        let extent = Extent2d::new(2, 2).unwrap();
        let surface = SurfaceInfo::linear_xrgb8888(
            extent,
            ByteRange::new(0, 24).unwrap(),
            12,
            SurfaceUsage::RENDER_TARGET,
        )
        .unwrap();
        let r = biased(&[1, 2, 3, 4]);
        let g = biased(&[5, 6, 7, 8]);
        let b = biased(&[9, 10, 11, 12]);
        let output = PlanarOutput::new([&r, &g, &b], PlanarFormat::U8Biased, extent).unwrap();
        let mut bytes = vec![0x55; 24];
        let mut target = LinearXrgbBytes::from_surface(&mut bytes, surface).unwrap();

        pack_planar_xrgb_bytes(output, &mut target, Damage::Full).unwrap();

        assert_eq!(&bytes[0..8], &[9, 5, 1, 0xff, 10, 6, 2, 0xff]);
        assert_eq!(&bytes[8..12], &[0x55; 4]);
        assert_eq!(&bytes[12..20], &[11, 7, 3, 0xff, 12, 8, 4, 0xff]);
        assert_eq!(&bytes[20..24], &[0x55; 4]);
    }

    #[test]
    fn word_target_only_materializes_explicit_damage() {
        let extent = Extent2d::new(3, 2).unwrap();
        let r = biased(&[10, 20, 30, 40, 50, 60]);
        let g = biased(&[1, 2, 3, 4, 5, 6]);
        let b = biased(&[90, 91, 92, 93, 94, 95]);
        let output = PlanarOutput::new([&r, &g, &b], PlanarFormat::U8Biased, extent).unwrap();
        let mut words = [0xdead_beef; 8];
        let mut target = LinearXrgbWords::new(&mut words, extent, 4).unwrap();
        let rect = [PixelRect::new(1, 0, 1, 2).unwrap()];

        pack_planar_xrgb_words(output, &mut target, Damage::Rects(&rect)).unwrap();

        assert_eq!(words[1], 0xff14_025b);
        assert_eq!(words[5], 0xff32_055e);
        assert_eq!(words[0], 0xdead_beef);
        assert_eq!(words[2], 0xdead_beef);
        assert_eq!(words[4], 0xdead_beef);
        assert_eq!(words[6], 0xdead_beef);
        assert_eq!(words[3], 0xdead_beef, "row padding was overwritten");
        assert_eq!(words[7], 0xdead_beef, "row padding was overwritten");
    }

    #[test]
    fn f32_materialization_clamps_and_rounds_at_the_display_boundary() {
        let extent = Extent2d::new(2, 1).unwrap();
        let plane = |values: [f32; 2]| {
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<std::vec::Vec<_>>()
        };
        let r = plane([-0.25, 0.5]);
        let g = plane([0.25, 0.75]);
        let b = plane([1.25, 1.0]);
        let output = PlanarOutput::new([&r, &g, &b], PlanarFormat::F32, extent).unwrap();
        let mut words = [0_u32; 2];
        let mut target = LinearXrgbWords::new(&mut words, extent, 2).unwrap();

        pack_planar_xrgb_words(output, &mut target, Damage::Full).unwrap();

        assert_eq!(words, [0xff00_40ff, 0xff80_bfff]);
    }

    #[test]
    fn typed_f32_planes_match_byte_planes_and_the_scalar_reference_across_chunks() {
        // 300 px rows straddle the 256-pixel SIMD chunk; the odd height
        // and one-pixel damage rect exercise every tail path.
        let extent = Extent2d::new(300, 3).unwrap();
        let pixels = 300 * 3;
        let value = |i: usize, c: usize| -> f32 {
            match (i + c) % 9 {
                0 => -0.5,
                1 => 1.5,
                2 => f32::NAN,
                3 => 0.5 / 255.0, // rounds down
                4 => 1.5 / 255.0, // rounds up
                5 => 1.0,
                6 => 0.0,
                7 => 0.4999,
                _ => ((i * 7 + c * 31) % 256) as f32 / 255.0,
            }
        };
        let typed: [std::vec::Vec<f32>; 3] =
            core::array::from_fn(|c| (0..pixels).map(|i| value(i, c)).collect());
        let bytes: [std::vec::Vec<u8>; 3] = core::array::from_fn(|c| {
            typed[c]
                .iter()
                .copied()
                .flat_map(f32::to_le_bytes)
                .collect()
        });
        let from_typed = PlanarOutput::f32([&typed[0], &typed[1], &typed[2]], extent).unwrap();
        let from_bytes =
            PlanarOutput::new([&bytes[0], &bytes[1], &bytes[2]], PlanarFormat::F32, extent)
                .unwrap();
        assert_eq!(from_typed.format(), PlanarFormat::F32);

        for damage in [
            Damage::Full,
            Damage::Rects(&[PixelRect::new(255, 1, 2, 2).unwrap()]),
            Damage::Rects(&[PixelRect::new(299, 2, 1, 1).unwrap()]),
        ] {
            let mut a = vec![0xdead_beef_u32; 304 * 3];
            let mut b = vec![0xdead_beef_u32; 304 * 3];
            let mut ta = LinearXrgbWords::new(&mut a, extent, 304).unwrap();
            let mut tb = LinearXrgbWords::new(&mut b, extent, 304).unwrap();
            pack_planar_xrgb_words(from_typed, &mut ta, damage).unwrap();
            pack_planar_xrgb_words(from_bytes, &mut tb, damage).unwrap();
            assert_eq!(a, b);
            let rect = match damage {
                Damage::Full => PixelRect::full(extent),
                Damage::Rects(rects) => rects[0],
            };
            for y in 0..3usize {
                for x in 0..304usize {
                    let inside = (rect.x() as usize..(rect.x() + rect.width()) as usize)
                        .contains(&x)
                        && (rect.y() as usize..(rect.y() + rect.height()) as usize).contains(&y);
                    let want = if inside {
                        let i = y * 300 + x;
                        let q = |c: usize| (value(i, c).clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                        0xff00_0000 | q(0) << 16 | q(1) << 8 | q(2)
                    } else {
                        0xdead_beef
                    };
                    assert_eq!(a[y * 304 + x], want, "pixel ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn typed_f32_planes_validate_their_length() {
        let extent = Extent2d::new(2, 2).unwrap();
        let ok = [0f32; 4];
        let short = [0f32; 3];
        assert!(PlanarOutput::f32([&ok, &ok, &ok], extent).is_ok());
        assert_eq!(
            PlanarOutput::f32([&ok, &short, &ok], extent).err(),
            Some(PackError::InvalidPlaneLength)
        );
    }

    #[test]
    fn biased_bytes_match_the_wrapping_reference_across_chunks() {
        let extent = Extent2d::new(261, 2).unwrap();
        let pixels = 261 * 2;
        let planes: [std::vec::Vec<u8>; 3] =
            core::array::from_fn(|c| (0..pixels).map(|i| (i * 5 + c * 77) as u8).collect());
        let output = PlanarOutput::new(
            [&planes[0], &planes[1], &planes[2]],
            PlanarFormat::U8Biased,
            extent,
        )
        .unwrap();
        let mut words = vec![0u32; pixels];
        let mut target = LinearXrgbWords::new(&mut words, extent, 261).unwrap();
        pack_planar_xrgb_words(output, &mut target, Damage::Full).unwrap();
        for i in 0..pixels {
            let q = |c: usize| u32::from(planes[c][i].wrapping_add(128));
            assert_eq!(
                words[i],
                0xff00_0000 | q(0) << 16 | q(1) << 8 | q(2),
                "pixel {i}"
            );
        }
    }

    #[test]
    fn damage_beyond_presenter_budget_becomes_full() {
        let rect = PixelRect::new(0, 0, 1, 1).unwrap();
        let rects = [rect; 5];
        assert_eq!(bounded_damage::<4>(Damage::Rects(&rects)), Damage::Full);
        assert_eq!(
            bounded_damage::<5>(Damage::Rects(&rects)),
            Damage::Rects(&rects)
        );
    }

    #[test]
    fn output_and_target_shape_mismatches_are_rejected() {
        let small = Extent2d::new(1, 1).unwrap();
        assert_eq!(
            PlanarOutput::new([&[0; 3], &[0; 4], &[0; 4]], PlanarFormat::F32, small,).err(),
            Some(PackError::InvalidPlaneLength)
        );

        let bytes = [0_u8; 4];
        let output = PlanarOutput::new([&bytes, &bytes, &bytes], PlanarFormat::F32, small).unwrap();
        let large = Extent2d::new(2, 1).unwrap();
        let mut words = [0_u32; 2];
        let mut target = LinearXrgbWords::new(&mut words, large, 2).unwrap();
        assert_eq!(
            pack_planar_xrgb_words(output, &mut target, Damage::Full),
            Err(PackError::ExtentMismatch)
        );
    }

    #[test]
    fn byte_target_requires_the_complete_writable_surface_range() {
        let extent = Extent2d::new(2, 2).unwrap();
        let renderable = SurfaceInfo::linear_xrgb8888(
            extent,
            ByteRange::new(0, 24).unwrap(),
            12,
            SurfaceUsage::RENDER_TARGET,
        )
        .unwrap();
        let present_only = SurfaceInfo::linear_xrgb8888(
            extent,
            ByteRange::new(0, 24).unwrap(),
            12,
            SurfaceUsage::PRESENT,
        )
        .unwrap();
        let mut truncated = [0_u8; 20];
        assert_eq!(
            LinearXrgbBytes::from_surface(&mut truncated, renderable).err(),
            Some(PackError::InvalidTarget)
        );
        let mut complete = [0_u8; 24];
        assert_eq!(
            LinearXrgbBytes::from_surface(&mut complete, present_only).err(),
            Some(PackError::InvalidTarget)
        );
    }
}
