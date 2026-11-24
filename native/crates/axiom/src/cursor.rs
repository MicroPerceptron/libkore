//! Retained cursor composition through the same ax/axnn path as other UI.
use crate::{Error, program::Program};
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
pub use axcomp::{CursorController, CursorShape, CursorVisual, FamilyCursor, FamilyPose};
pub use axgfx::{CursorHotspot, Extent2d, PixelRect};
pub use axinput::CursorRequest;
use core::alloc::Allocator;

/// The standard cursor family's display scale, the Dart's from the original
/// WGSL showcase. Keep this rational: every extent stays exact.
pub const SCALE_NUMERATOR: usize = 3;
pub const SCALE_DENOMINATOR: usize = 2;
const PIXELS_PER_UNIT: f32 = SCALE_NUMERATOR as f32 / SCALE_DENOMINATOR as f32;
const SOURCE_WIDTH: usize = axcomp::FAMILY_WIDTH;
const SOURCE_HEIGHT: usize = axcomp::FAMILY_HEIGHT;
pub const WIDTH: usize = SOURCE_WIDTH * SCALE_NUMERATOR / SCALE_DENOMINATOR;
pub const HEIGHT: usize = SOURCE_HEIGHT * SCALE_NUMERATOR / SCALE_DENOMINATOR;
pub const PIXELS: usize = WIDTH * HEIGHT;
const RGBA_BYTES: usize = PIXELS * 4;
/// The Dart's hotspot: the cursor's place before any pose exists. Every
/// member places itself by its own hotspot ([`hotspot_at`]).
pub const HOTSPOT: CursorHotspot = CursorHotspot {
    x: axcomp::DART_HOTSPOT.x * SCALE_NUMERATOR as u16 / SCALE_DENOMINATOR as u16,
    y: axcomp::DART_HOTSPOT.y * SCALE_NUMERATOR as u16 / SCALE_DENOMINATOR as u16,
};

const _: () = assert!(SOURCE_WIDTH * SCALE_NUMERATOR % SCALE_DENOMINATOR == 0);
const _: () = assert!(SOURCE_HEIGHT * SCALE_NUMERATOR % SCALE_DENOMINATOR == 0);

/// Where `pose` puts the pointer's position in its sprite at `density`, in
/// device pixels: the shown member's designed hotspot.
pub fn hotspot_at(pose: &FamilyPose, density: kore_window::appearance::Density) -> CursorHotspot {
    pose.hotspot(PIXELS_PER_UNIT * density.as_f32())
}

pub fn visual() -> CursorVisual {
    CursorVisual {
        request: CursorRequest::DEFAULT,
        extent: Extent2d::new(WIDTH as u32, HEIGHT as u32).expect("static extent"),
        hotspot: HOTSPOT,
    }
}

/// Six canonical nodes selected and prepared once by axnn. Motion supplies the
/// changed patch's background plus the portable family raster; warm
/// execution reuses every tensor and scratch allocation.
pub struct CursorProgram<A: Allocator> {
    prepared: ax_exec::PreparedNative<A>,
}

impl<A: Allocator + Clone> CursorProgram<A> {
    pub fn new(allocator: A) -> Result<Self, Error> {
        let mut graph = GraphBuilder::new_in(allocator.clone());
        let shape = [1, PIXELS as u32];
        // R, G, B background; inverse alpha; premultiplied R, G, B foreground.
        let mut inputs = [ax::TensorId(0); 7];
        for input in &mut inputs {
            *input = graph
                .tensor(DType::F32, TensorFlags::INPUT, &shape, &[])
                .map_err(|_| Error::Graph)?;
        }
        for channel in 0..3 {
            let attenuated = graph
                .tensor(DType::F32, TensorFlags::default(), &shape, &[])
                .map_err(|_| Error::Graph)?;
            let output = graph
                .tensor(DType::F32, TensorFlags::OUTPUT, &shape, &[])
                .map_err(|_| Error::Graph)?;
            graph
                .node(
                    OpCode::MUL,
                    NodeFlags::default(),
                    &[inputs[channel], inputs[3]],
                    &[attenuated],
                    &[],
                )
                .map_err(|_| Error::Graph)?;
            graph
                .node(
                    OpCode::ADD,
                    NodeFlags::default(),
                    &[attenuated, inputs[4 + channel]],
                    &[output],
                    &[],
                )
                .map_err(|_| Error::Graph)?;
        }
        let program = Program::from_graph_in(
            graph.finish().map_err(|_| Error::Graph)?,
            crate::kernel_policy(),
            allocator.clone(),
        )?;
        Ok(Self {
            prepared: program.prepare_native_in(allocator)?,
        })
    }

    pub fn render(
        &mut self,
        background: &[u32; PIXELS],
        pose: FamilyPose,
    ) -> Result<[u32; PIXELS], Error> {
        let mut channels = [[0.; PIXELS]; 3];
        for (i, &word) in background.iter().enumerate() {
            for (c, shift) in [16, 8, 0].into_iter().enumerate() {
                channels[c][i] = ((word >> shift) & 255) as f32 / 255.;
            }
        }
        let mut rgba = [0; RGBA_BYTES];
        pose.rasterize_at_scale(
            visual().extent,
            pose.hotspot(PIXELS_PER_UNIT),
            PIXELS_PER_UNIT,
            &mut rgba,
        )
        .map_err(|_| Error::InvalidBinding)?;
        let mut inverse = [0.; PIXELS];
        let mut foreground = [[0.; PIXELS]; 3];
        premultiplied_inputs(&rgba, &mut inverse, &mut foreground);
        let bindings = [
            f32_bytes(&channels[0]),
            f32_bytes(&channels[1]),
            f32_bytes(&channels[2]),
            f32_bytes(&inverse),
            f32_bytes(&foreground[0]),
            f32_bytes(&foreground[1]),
            f32_bytes(&foreground[2]),
        ];
        self.prepared
            .run(&bindings)
            .map_err(|_| Error::InvalidBinding)?;
        let mut out = [0; PIXELS];
        for c in 0..3 {
            let values = self
                .prepared
                .output_f32(c)
                .filter(|plane| plane.len() == PIXELS)
                .ok_or(Error::InvalidOutput)?;
            for (word, value) in out.iter_mut().zip(values) {
                *word |= ((value.clamp(0., 1.) * 255. + 0.5) as u32) << (16 - c * 8);
            }
        }
        Ok(out)
    }

    pub fn storage_bytes(&self) -> usize {
        self.prepared.buffer_bytes()
    }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    #[cfg(not(target_endian = "little"))]
    compile_error!("Kore's F32 graph input ABI requires a little-endian target");
    // SAFETY: `f32` has no padding; the initialized input is borrowed only for
    // this execution and the prepared runner copies it before returning.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values)) }
}

/// Rasterise the family at `pose` into a premultiplied ARGB sprite (alpha in
/// bits 31:24) — the retained scene's cursor node, blended by the renderer
/// over whatever it covers, so no background reconstruction is needed.
/// Returns the hotspot the sprite was drawn around ([`hotspot_at`]).
pub fn rasterize_sprite(pose: FamilyPose, out: &mut [u32; PIXELS]) -> Result<CursorHotspot, Error> {
    let mut rgba = [0; RGBA_BYTES];
    rasterize_sprite_at_density(
        pose,
        kore_window::appearance::Density::Native,
        &mut rgba,
        out,
    )
}

/// Rasterize the same outline directly at the requested native pixel density,
/// around the pose's own hotspot ([`hotspot_at`]), which it returns.
/// Scratch belongs to the painter and is reused between poses.
pub fn rasterize_sprite_at_density(
    pose: FamilyPose,
    density: kore_window::appearance::Density,
    rgba: &mut [u8],
    out: &mut [u32],
) -> Result<CursorHotspot, Error> {
    let width = density.pixels(WIDTH as u32);
    let height = density.pixels(HEIGHT as u32);
    let count = width as usize * height as usize;
    if out.len() != count || rgba.len() != count * 4 {
        return Err(Error::InvalidBinding);
    }
    let extent = Extent2d::new(width, height).map_err(|_| Error::InvalidBinding)?;
    let hotspot = hotspot_at(&pose, density);
    pose.rasterize_at_scale(extent, hotspot, PIXELS_PER_UNIT * density.as_f32(), rgba)
        .map_err(|_| Error::InvalidBinding)?;
    for (word, pixel) in out.iter_mut().zip(rgba.chunks_exact(4)) {
        let a = u32::from(pixel[3]);
        let premul = |c: u8| (u32::from(c) * a + 127) / 255;
        *word = (a << 24) | (premul(pixel[0]) << 16) | (premul(pixel[1]) << 8) | premul(pixel[2]);
    }
    Ok(hotspot)
}

/// Convert the physical-resolution SDF raster into the graph's source-over
/// planes. Scaling has already happened in signed-distance space, so this
/// step never filters a finished bitmap or softens the keyline.
fn premultiplied_inputs(
    rgba: &[u8; RGBA_BYTES],
    inverse: &mut [f32; PIXELS],
    foreground: &mut [[f32; PIXELS]; 3],
) {
    for (target, pixel) in rgba.chunks_exact(4).enumerate() {
        let alpha = f32::from(pixel[3]) / 255.0;
        inverse[target] = 1.0 - alpha;
        for channel in 0..3 {
            foreground[channel][target] = f32::from(pixel[channel]) / 255.0 * alpha;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        alloc::{AllocError, Layout},
        cell::Cell,
        ptr::NonNull,
    };
    use kore_rt::{Heap, heap::StaticMemSource};

    #[derive(Clone, Copy)]
    struct Counted<'a> {
        heap: &'a Heap<StaticMemSource>,
        calls: &'a Cell<usize>,
        refuse: &'a Cell<bool>,
    }

    // SAFETY: allocation and deallocation delegate to the same live named
    // arena. Refusal and accounting change admission, never pointer layout.
    unsafe impl Allocator for Counted<'_> {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.calls.set(self.calls.get() + 1);
            if self.refuse.get() {
                return Err(AllocError);
            }
            self.heap.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: this pair came from the delegated arena above.
            unsafe { self.heap.deallocate(ptr, layout) }
        }
    }

    fn patterned_background() -> [u32; PIXELS] {
        core::array::from_fn(|index| {
            let x = index % WIDTH;
            let y = index / WIDTH;
            let red = (x * 17 + y * 3 + 11) as u32 & 255;
            let green = (x * 5 + y * 19 + 37) as u32 & 255;
            let blue = (x * 13 + y * 7 + 73) as u32 & 255;
            red << 16 | green << 8 | blue
        })
    }

    fn assert_matches_scalar_source_over(
        background: &[u32; PIXELS],
        pose: FamilyPose,
        actual: &[u32; PIXELS],
    ) {
        let mut rgba = [0; RGBA_BYTES];
        pose.rasterize_at_scale(
            visual().extent,
            pose.hotspot(PIXELS_PER_UNIT),
            PIXELS_PER_UNIT,
            &mut rgba,
        )
        .unwrap();
        let mut inverse = [0.; PIXELS];
        let mut foreground = [[0.; PIXELS]; 3];
        premultiplied_inputs(&rgba, &mut inverse, &mut foreground);

        for (index, (&background, actual)) in background.iter().zip(actual).enumerate() {
            assert_eq!(actual >> 24, 0, "pixel {index} has a non-XRGB high byte");
            for (channel, shift) in [16, 8, 0].into_iter().enumerate() {
                let background = ((background >> shift) & 255) as f32 / 255.0;
                let expected = ((background * inverse[index] + foreground[channel][index])
                    .clamp(0.0, 1.0)
                    * 255.0
                    + 0.5) as u8;
                let actual = ((actual >> shift) & 255) as u8;
                assert!(
                    actual.abs_diff(expected) <= 1,
                    "pixel {index} channel {channel}: graph={actual}, scalar={expected}"
                );
            }
        }
    }

    #[test]
    fn prepared_colored_dart_blend_allocates_nothing_on_the_warm_path() {
        #[repr(align(4096))]
        struct Storage([u8; 512 * 1024]);

        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let mut storage = Storage([0; 512 * 1024]);
                // SAFETY: test-owned aligned storage outlives the program.
                let heap = Heap::new_in(unsafe {
                    StaticMemSource::over(storage.0.as_mut_ptr(), storage.0.len())
                });
                let calls = Cell::new(0);
                let refuse = Cell::new(false);
                let allocator = Counted {
                    heap: &heap,
                    calls: &calls,
                    refuse: &refuse,
                };
                let mut program = CursorProgram::new(allocator).unwrap();
                assert_eq!(program.prepared.input_count(), 7);
                assert_eq!(program.prepared.output_count(), 3);
                assert_eq!(program.prepared.instruction_count(), 3);
                assert_eq!(program.prepared.mul_add_fusion_count(), 3);
                assert!(program.prepared.buffer_count() <= 8);
                assert!(program.storage_bytes() <= 108 * 1024);
                assert!(program.prepared.logical_buffer_count() <= 13);
                assert!(program.prepared.logical_buffer_bytes() <= 162 * 1024);
                assert_eq!(program.prepared.scratch_bytes(), 0);
                let preparation_calls = calls.get();
                let prepared_heap = heap.stats();
                refuse.set(true);

                let background = patterned_background();
                let dart_rest = FamilyPose::rest(CursorShape::Dart);
                let rest = program.render(&background, dart_rest).unwrap();
                let pointer_pose = FamilyPose::rest(CursorShape::Pointer);
                let pointer = program.render(&background, pointer_pose).unwrap();

                assert_ne!(rest, background);
                assert_ne!(rest, pointer, "the member shown is a dynamic graph input");
                assert!(rest.iter().any(|word| {
                    let red = (word >> 16) & 255;
                    let green = (word >> 8) & 255;
                    green > red
                }));
                assert_eq!(rest[0], background[0], "transparent padding is exact");
                assert_matches_scalar_source_over(&background, dart_rest, &rest);
                assert_matches_scalar_source_over(&background, pointer_pose, &pointer);
                assert_eq!(calls.get(), preparation_calls);
                assert_eq!(heap.stats(), prepared_heap);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn dense_sprite_rasterizes_edges_instead_of_repeating_pixels() {
        let mut normal = [0; PIXELS];
        let rest = FamilyPose::rest(CursorShape::Dart);
        rasterize_sprite(rest, &mut normal).unwrap();
        let mut dense = [0; PIXELS * 4];
        let mut rgba = [0; RGBA_BYTES * 4];
        rasterize_sprite_at_density(
            rest,
            kore_window::appearance::Density::Double,
            &mut rgba,
            &mut dense,
        )
        .unwrap();
        assert_eq!(
            dense[(usize::from(HOTSPOT.y) * 2) * WIDTH * 2 + usize::from(HOTSPOT.x) * 2],
            normal[usize::from(HOTSPOT.y) * WIDTH + usize::from(HOTSPOT.x)]
        );
        assert!(dense.iter().enumerate().any(
            |(i, &pixel)| pixel != normal[(i / (WIDTH * 2) / 2) * WIDTH + i % (WIDTH * 2) / 2]
        ));
        assert!(
            rasterize_sprite_at_density(
                rest,
                kore_window::appearance::Density::Double,
                &mut rgba,
                &mut normal
            )
            .is_err()
        );
    }

    #[test]
    fn standard_dart_uses_the_showcase_one_point_five_scale() {
        assert_eq!((WIDTH, HEIGHT), (54, 54));
        assert_eq!(HOTSPOT, CursorHotspot { x: 27, y: 9 });
        assert_eq!(visual().extent, Extent2d::new(54, 54).unwrap());
        assert_eq!(visual().hotspot, HOTSPOT);

        let mut rgba = [0; RGBA_BYTES];
        let rest = FamilyPose::rest(CursorShape::Dart);
        assert_eq!(hotspot_at(&rest, kore_window::appearance::Density::Native), HOTSPOT);
        rest.rasterize_at_scale(visual().extent, HOTSPOT, PIXELS_PER_UNIT, &mut rgba)
            .unwrap();
        let tip = (usize::from(HOTSPOT.y) * WIDTH + usize::from(HOTSPOT.x)) * 4;
        assert_eq!(&rgba[tip..tip + 4], &[0xf9, 0xf4, 0xed, 0xff]);
        assert!(rgba.chunks_exact(4).any(|pixel| {
            let alpha = pixel[3];
            alpha > 0 && alpha < 255
        }));
    }

    /// Each member is placed by its own designed hotspot at every density:
    /// the I-beam and the arrows by their centre, the hands by the palm,
    /// the pointing finger by the Dart's own tip, so changing between the
    /// two never moves the cursor.
    #[test]
    fn every_member_is_placed_by_its_own_hotspot() {
        use kore_window::appearance::Density;
        for shape in CursorShape::ALL {
            let rest = FamilyPose::rest(shape);
            let designed = shape.hotspot();
            for density in Density::ALL {
                let at = hotspot_at(&rest, density);
                let scale = PIXELS_PER_UNIT * density.as_f32();
                let expected = |units: u16| (f32::from(units) * scale + 0.5) as u16;
                assert_eq!(
                    (at.x, at.y),
                    (expected(designed.x), expected(designed.y)),
                    "{shape:?} at {density:?}"
                );
                assert!(u32::from(at.x) < density.pixels(WIDTH as u32), "{shape:?}");
                assert!(u32::from(at.y) < density.pixels(HEIGHT as u32), "{shape:?}");
            }
        }
        let pointer = hotspot_at(&FamilyPose::rest(CursorShape::Pointer), Density::Native);
        assert_eq!(pointer, HOTSPOT, "the finger shares the Dart's tip");
    }
}
