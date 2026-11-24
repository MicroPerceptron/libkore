//! Production WGSL welcome asset on the same catalog/admission path as apps.
//!
//! The welcome is axboot's two graphs: the hole pass draws the black hole at
//! the lens map's resolution, and the compose pass lays the overlay over it
//! at the display's. The compositor prepares one [`BootWorker`] per admitted
//! worker before dispatch; each holds both prepared graphs and reuses their
//! input and intermediate storage, borrowing the completed planes while they
//! are packed directly into a mapped surface. The asset's baked tables — the
//! disk's turbulence and the sky atlas — bind as the hole graph's two
//! resources, validated once per worker.
//!
//! A display no taller than the lens map renders each tile whole
//! ([`BootWorker::render`]): its hole, then its overlay where the overlay
//! reaches it. A taller display renders in two passes: every tile of its
//! hole frame ([`BootWorker::render_hole`]) into a buffer, then every
//! display tile over that buffer ([`BootWorker::render_over`]).
use crate::{
    Error,
    program::{Input, Output, Program},
};
use alloc::vec::Vec;
pub use axboot::{Clock, FRAME_NS, Frame, Hole, HoleImage, TILE_HEIGHT, TILE_PIXELS, TILE_WIDTH, Viewport};
use core::alloc::Allocator;

#[cfg(test)]
#[path = "boot_probe.rs"]
mod probe;

#[cfg(test)]
#[path = "boot_blend_probe.rs"]
mod blend_probe;

pub struct BootProgram<A: Allocator> {
    hole: Program<A>,
    compose: Program<A>,
}

/// The asset's baked tables as the hole graph's RGB-volume resources: the
/// turbulence table and the sky atlas. Validation (dimensions, finite
/// non-negative energy) runs here, once.
#[derive(Clone, Copy)]
struct Volumes {
    turbulence: axnn_cpu::RgbVolumeRef<'static>,
    sky: axnn_cpu::RgbVolumeRef<'static>,
}

fn volumes() -> Result<Volumes, Error> {
    let volume = |points, dims| axnn_cpu::RgbVolumeRef::new_uf16(points, dims).map_err(|_| Error::InvalidBinding);
    Ok(Volumes {
        turbulence: volume(axboot::turbulence(), axboot::TURBULENCE_DIMS)?,
        sky: volume(axboot::sky_atlas(), axboot::SKY_DIMS)?,
    })
}

fn resource_bindings(volumes: Volumes) -> [ax_exec::NativeResourceBinding<'static>; 2] {
    let bind = |(group, binding): (u32, u32), volume| {
        ax_exec::NativeResourceBinding::rgb_volume(ax::sampling::ResourceKey { group, binding }, volume)
    };
    [bind(axboot::TURBULENCE_BINDING, volumes.turbulence), bind(axboot::SKY_BINDING, volumes.sky)]
}

/// F32 planes as a prepared run's byte inputs.
fn bytes<const N: usize>(planes: [&[f32]; N]) -> [&[u8]; N] {
    planes.map(|plane| {
        // SAFETY: the initialized F32 inputs are borrowed only for this
        // execution. Kore's F32 graph ABI requires a little-endian target.
        unsafe { core::slice::from_raw_parts(plane.as_ptr().cast(), core::mem::size_of_val(plane)) }
    })
}

/// The colour planes of a prepared graph's last run.
fn colour<A: Allocator>(prepared: &ax_exec::PreparedNative<A>) -> Result<[&[f32]; 3], Error> {
    let plane = |index| {
        prepared
            .output_f32(index)
            .filter(|p| p.len() == TILE_PIXELS)
            .ok_or(Error::InvalidOutput)
    };
    Ok([plane(0)?, plane(1)?, plane(2)?])
}

impl<A: Allocator + Clone> BootProgram<A> {
    pub fn new_in(allocator: A) -> Result<Self, Error> {
        Ok(Self {
            hole: Program::compile_in(axboot::HOLE_GRAPH, crate::kernel_policy(), allocator.clone())?,
            compose: Program::compile_in(axboot::COMPOSE_GRAPH, crate::kernel_policy(), allocator)?,
        })
    }
    /// One-shot owned hole tile for compatibility and reference checks.
    /// Repeated production rendering prepares a [`BootWorker`] once instead.
    pub fn render_hole_in<R: Allocator + Clone>(
        &self,
        hole: Viewport,
        origin: [u32; 2],
        frame: Frame,
        allocator: R,
    ) -> Result<Vec<Output<R>, R>, Error> {
        let mut input = axboot::HoleInputs::new_in(allocator.clone()).map_err(|_| Error::Allocation)?;
        input.prepare(hole, origin, frame).map_err(|_| Error::InvalidBinding)?;
        self.hole
            .run_with_resources_in(&input.planes().map(Input::F32), &resource_bindings(volumes()?), allocator)
    }
    /// One-shot owned compose tile, always through the graph, even where
    /// nothing covers the hole.
    pub fn render_compose_in<R: Allocator + Clone>(
        &self,
        viewport: Viewport,
        origin: [u32; 2],
        clock: Clock,
        hole: Hole<'_>,
        allocator: R,
    ) -> Result<Vec<Output<R>, R>, Error> {
        let mut input = axboot::ComposeInputs::new_in(allocator.clone()).map_err(|_| Error::Allocation)?;
        input
            .prepare(viewport, origin, clock, hole)
            .map_err(|_| Error::InvalidBinding)?;
        self.compose.run_in(&input.planes().map(Input::F32), allocator)
    }
    /// The hole graph's placement, then the compose graph's.
    pub fn placements(&self) -> [crate::Placement; 2] {
        [self.hole.placement(), self.compose.placement()]
    }
    pub fn native_plan_stats(&self) -> [ax_exec::AxNativePlanStats; 2] {
        [self.hole.native_plan_stats(), self.compose.native_plan_stats()]
    }

    /// Prepare the default repeated-render path on the owning service lane.
    /// Allocation is confined to preparation; rendering never grows an arena.
    /// The returned worker owns its execution plans and may outlive this
    /// program.
    pub fn worker_in<R: Allocator + Clone>(&self, allocator: R) -> Result<BootWorker<R>, Error> {
        Ok(BootWorker {
            hole_input: axboot::HoleInputs::new_in(allocator.clone()).map_err(|_| Error::Allocation)?,
            compose_input: axboot::ComposeInputs::new_in(allocator.clone()).map_err(|_| Error::Allocation)?,
            hole: self.hole.prepare_native_in(allocator.clone())?,
            compose: self.compose.prepare_native_in(allocator)?,
            volumes: volumes()?,
        })
    }
}

/// One exclusive welcome execution lane. The runtime supplies the allocator
/// and the worker's synchronization; no mapping, task or global state lives here.
pub struct BootWorker<A: Allocator> {
    hole_input: axboot::HoleInputs<A>,
    compose_input: axboot::ComposeInputs<A>,
    hole: ax_exec::PreparedNative<A>,
    compose: ax_exec::PreparedNative<A>,
    volumes: Volumes,
}

/// Retained payload counts, excluding executable/allocator metadata. Physical
/// tensors are reused by liveness; logical bytes describe one buffer per value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootStorage {
    pub input_bytes: usize,
    pub tensor_buffers: usize,
    pub tensor_bytes: usize,
    pub logical_tensor_bytes: usize,
}

impl<A: Allocator + Clone> BootWorker<A> {
    pub fn storage(&self) -> BootStorage {
        let size = |planes: &[&[f32]]| planes.iter().map(|p| core::mem::size_of_val(*p)).sum::<usize>();
        BootStorage {
            input_bytes: size(&self.hole_input.planes()) + size(&self.compose_input.planes()),
            tensor_buffers: self.hole.buffer_count() + self.compose.buffer_count(),
            tensor_bytes: self.hole.buffer_bytes() + self.compose.buffer_bytes(),
            logical_tensor_bytes: self.hole.logical_buffer_bytes() + self.compose.logical_buffer_bytes(),
        }
    }

    /// A display no taller than the lens map: the tile at `origin`, its hole
    /// and then, where the overlay reaches it, the overlay over that hole's
    /// own planes. Allocation-free after preparation, including font
    /// coverage, kernel scratch and output planes. The returned tile prevents
    /// another render until packing has finished.
    pub fn render(
        &mut self,
        viewport: Viewport,
        origin: [u32; 2],
        clock: Clock,
        frame: Frame,
    ) -> Result<BootTile<'_>, Error> {
        if viewport.upsamples_hole() {
            return Err(Error::UnsupportedShape);
        }
        self.hole_input
            .prepare(viewport, origin, frame)
            .map_err(|_| Error::InvalidBinding)?;
        self.hole
            .run_with_resources(&bytes(self.hole_input.planes()), &resource_bindings(self.volumes))
            .map_err(|_| Error::InvalidBinding)?;
        let hole = colour(&self.hole)?;
        if !viewport.overlays_tile(origin) {
            return Ok(BootTile { planes: hole });
        }
        self.compose_input
            .prepare(viewport, origin, clock, Hole::Tile(hole))
            .map_err(|_| Error::InvalidBinding)?;
        self.compose
            .run(&bytes(self.compose_input.planes()))
            .map_err(|_| Error::InvalidBinding)?;
        Ok(BootTile { planes: colour(&self.compose)? })
    }

    /// A display taller than the lens map, first pass: the hole alone at the
    /// tile `origin` of `hole`, the display's [`Viewport::hole_frame`].
    pub fn render_hole(&mut self, hole: Viewport, origin: [u32; 2], frame: Frame) -> Result<BootTile<'_>, Error> {
        self.hole_input
            .prepare(hole, origin, frame)
            .map_err(|_| Error::InvalidBinding)?;
        self.hole
            .run_with_resources(&bytes(self.hole_input.planes()), &resource_bindings(self.volumes))
            .map_err(|_| Error::InvalidBinding)?;
        Ok(BootTile { planes: colour(&self.hole)? })
    }

    /// A display taller than the lens map, second pass: the display tile at
    /// `origin` over the hole frame `image`, upsampled. The compose graph
    /// runs only where the overlay reaches the tile; elsewhere it would
    /// return the upsampled hole bit for bit, so the tile is that hole.
    pub fn render_over(
        &mut self,
        viewport: Viewport,
        origin: [u32; 2],
        clock: Clock,
        image: HoleImage<'_>,
    ) -> Result<BootTile<'_>, Error> {
        let overlaid = self
            .compose_input
            .prepare(viewport, origin, clock, Hole::Image(image))
            .map_err(|_| Error::InvalidBinding)?;
        if !overlaid {
            let hole = &self.compose_input.hole;
            return Ok(BootTile { planes: [&hole[0], &hole[1], &hole[2]] });
        }
        self.compose
            .run(&bytes(self.compose_input.planes()))
            .map_err(|_| Error::InvalidBinding)?;
        Ok(BootTile { planes: colour(&self.compose)? })
    }
}

/// Borrowed colour of a successful welcome tile invocation.
pub struct BootTile<'a> {
    planes: [&'a [f32]; 3],
}

impl<'a> BootTile<'a> {
    /// Red, green and blue, each one tile of F32 in [0, 1].
    pub fn planes(&self) -> [&'a [f32]; 3] {
        self.planes
    }

    /// Pack the visible prefix of one tile row directly into a disjoint mapped
    /// XRGB row. Restricting the borrow to one row lets adjacent SPMD tiles write
    /// the same framebuffer without overlapping mutable slices over row padding.
    /// Invalid geometry is rejected before any destination pixel is changed.
    pub fn write_xrgb_row(&self, row: usize, destination: &mut [u32]) -> Result<(), Error> {
        if row >= TILE_HEIGHT || destination.len() > TILE_WIDTH {
            return Err(Error::InvalidBinding);
        }
        let start = row * TILE_WIDTH;
        let end = start + destination.len();
        // The raster kernel stores its first channel in the low byte. Passing
        // B/G/R therefore produces the display's XRGB word without a scalar
        // post-pass while retaining the shader's exact rounded quantization.
        axnn_cpu::raster::xrgb_pack_rounded(
            &self.planes[2][start..end],
            &self.planes[1][start..end],
            &self.planes[0][start..end],
            destination,
        );
        Ok(())
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

    /// A prepared lane's arena: both prepared graphs and their inputs map
    /// 2.15 MB (`welcome_footprint`).
    const WORKER_BYTES: usize = 2304 * 1024;
    /// Both compiled programs map 1.16 MB (`welcome_footprint`).
    const PROGRAM_BYTES: usize = 1280 * 1024;
    /// A one-shot reference invocation's transient arena.
    const REFERENCE_BYTES: usize = 1280 * 1024;

    #[derive(Clone, Copy)]
    struct Counted<'a> {
        heap: &'a Heap<StaticMemSource>,
        calls: &'a Cell<usize>,
        requested: &'a Cell<usize>,
        refuse: &'a Cell<bool>,
    }
    // SAFETY: every allocation and deallocation delegates to the same named
    // arena. Counters and refusal affect admission only, never storage layout.
    unsafe impl Allocator for Counted<'_> {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.calls.set(self.calls.get() + 1);
            self.requested.set(self.requested.get() + layout.size());
            if self.refuse.get() {
                return Err(AllocError);
            }
            self.heap.allocate(layout)
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: the pointer and layout came from this delegated arena.
            unsafe { self.heap.deallocate(ptr, layout) }
        }
    }

    /// Aligned backing for one named arena, on the heap: a debug build would
    /// stage a large array through temporaries on the test thread's stack.
    #[repr(align(4096))]
    struct Backing<const N: usize>([u8; N]);
    fn backing<const N: usize>() -> std::boxed::Box<Backing<N>> {
        // SAFETY: all-zero bytes are a valid `Backing`.
        unsafe { std::boxed::Box::<Backing<N>>::new_zeroed().assume_init() }
    }
    fn arena<const N: usize>(backing: &mut Backing<N>) -> Heap<StaticMemSource> {
        // SAFETY: each caller keeps its backing alive, and unaliased, until
        // every allocation from this heap has been dropped.
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), N) })
    }

    /// A hole frame's worth of packed pixels, varying enough to show a misread.
    fn gradient(viewport: Viewport) -> std::vec::Vec<u32> {
        let [w, h] = viewport.hole_frame().size().map(|v| v as usize);
        (0..w * h).map(|i| ((i % w * 255 / w) as u32) << 16 | ((i / w * 255 / h) as u32) << 8 | 0x40).collect()
    }

    /// `actual` equals `expected` bit for bit, and its visible `width` by
    /// `height` prefix packs exactly that colour without touching a pixel
    /// past it; out-of-tile packing is refused before any write.
    fn assert_tile<R: Allocator>(actual: &BootTile<'_>, expected: &[Output<R>], width: usize, height: usize) {
        for (actual, expected) in actual.planes().into_iter().zip(expected) {
            for (actual, expected) in actual.iter().zip(expected.f32().unwrap()) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        let width = width.min(TILE_WIDTH);
        let mut row = [0x1234_5678; TILE_WIDTH + 2];
        for y in 0..height.min(TILE_HEIGHT) {
            actual.write_xrgb_row(y, &mut row[1..width + 1]).unwrap();
            for x in 0..width {
                let i = y * TILE_WIDTH + x;
                let channel = |c: usize| (expected[c].f32().unwrap()[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                assert_eq!(row[x + 1], 0xff00_0000 | channel(0) << 16 | channel(1) << 8 | channel(2));
            }
            assert_eq!(row[0], 0x1234_5678);
            assert_eq!(row[width + 1], 0x1234_5678);
        }
        let before = row;
        assert!(actual.write_xrgb_row(TILE_HEIGHT, &mut row).is_err());
        assert!(actual.write_xrgb_row(0, &mut row).is_err());
        assert_eq!(row, before);
    }

    #[test]
    fn four_prepared_workers_fit_a_named_service_heap() {
        // The service heap is the compositor's `heap_supply_bytes`.
        const SERVICE_BYTES: usize = 12 * 1024 * 1024;
        struct Arena<'a> {
            heap: Heap<StaticMemSource>,
            backing: NonNull<[u8]>,
            owner: &'a Heap<StaticMemSource>,
        }
        impl Drop for Arena<'_> {
            fn drop(&mut self) {
                // All workers borrow their arenas and are dropped first.
                // StaticMemSource has no owning destructor.
                unsafe {
                    self.owner.deallocate(
                        self.backing.cast(),
                        Layout::from_size_align(WORKER_BYTES, 4096).unwrap(),
                    );
                }
            }
        }
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let mut storage = backing::<SERVICE_BYTES>();
                // The named service heap owns this thread-local arena for
                // longer than its graphs and all four worker subarenas.
                let owner = arena(&mut storage);
                let program = BootProgram::new_in(&owner).unwrap();
                let arenas: [alloc::boxed::Box<Arena<'_>, _>; 4] = core::array::from_fn(|_| {
                    let backing = owner
                        .allocate_zeroed(Layout::from_size_align(WORKER_BYTES, 4096).unwrap())
                        .unwrap();
                    let heap = Heap::new_in(unsafe {
                        StaticMemSource::over(backing.cast().as_ptr(), WORKER_BYTES)
                    });
                    alloc::boxed::Box::try_new_in(Arena { heap, backing, owner: &owner }, &owner).unwrap()
                });
                // Heap metadata is also charged to the named service heap.
                // Worker borrows are tied to the owned boxes by the type
                // system; this sizing test invents no static or self-borrow.
                let mut workers: [BootWorker<_>; 4] =
                    core::array::from_fn(|i| program.worker_in(&arenas[i].heap).unwrap());
                let peak = owner.stats();
                assert!(peak.mapped_bytes <= SERVICE_BYTES);
                let worker_mapped = arenas.each_ref().map(|arena| arena.heap.stats().mapped_bytes);
                assert!(worker_mapped.iter().all(|&n| n <= WORKER_BYTES));
                drop(program);
                let mut packed = [[0_u32; TILE_PIXELS]; 4];
                // Host scoped threads prove independent mutable worker state;
                // they are not evidence of Kore resident task admission. The
                // top row's tiles cross the header's label, so both graphs run.
                std::thread::scope(|scope| {
                    for (i, (worker, pixels)) in workers.iter_mut().zip(&mut packed).enumerate() {
                        scope.spawn(move || {
                            let tile = worker.render(
                                Viewport::new(960, 600).unwrap(),
                                [(i * TILE_WIDTH) as u32, 0],
                                Clock::from_unix_seconds(None), Frame(i as u64),
                            ).unwrap();
                            for (row, output) in pixels.chunks_exact_mut(TILE_WIDTH).enumerate() {
                                tile.write_xrgb_row(row, output).unwrap();
                            }
                        });
                    }
                });
                assert!(packed.iter().flatten().all(|&pixel| pixel >> 24 == 0xff));
                assert_eq!(owner.stats(), peak, "rendering grew the service heap");
                std::eprintln!("welcome four-worker sizing: service mapped {} bytes; worker heaps {worker_mapped:?}", peak.mapped_bytes);
                drop(workers);
                drop(arenas);
            }).unwrap().join().unwrap();
    }

    #[test]
    fn prepared_worker_reuses_bounded_arena_and_matches_production_tiles() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let mut persistent = backing::<PROGRAM_BYTES>();
                let mut prepared = backing::<WORKER_BYTES>();
                let mut reference = backing::<REFERENCE_BYTES>();
                let persistent_heap = arena(&mut persistent);
                let prepared_heap = arena(&mut prepared);
                let reference_heap = arena(&mut reference);
                let program = BootProgram::new_in(&persistent_heap).unwrap();
                let prepared_calls = Cell::new(0);
                let prepared_bytes = Cell::new(0);
                let prepared_refuse = Cell::new(false);
                let mut worker = program.worker_in(Counted {
                    heap: &prepared_heap,
                    calls: &prepared_calls,
                    requested: &prepared_bytes,
                    refuse: &prepared_refuse,
                }).unwrap();
                let storage = worker.storage();
                let preparation_calls = prepared_calls.get();
                let preparation_bytes = prepared_bytes.get();
                let initial_heap = prepared_heap.stats();
                assert!(storage.tensor_bytes < storage.logical_tensor_bytes);
                assert!(initial_heap.mapped_bytes <= WORKER_BYTES);
                // A warm invocation must also work when allocator admission
                // is denied. Stable mapped bytes alone would miss alloc/free
                // churn against the already-mapped heap.
                prepared_refuse.set(true);
                let reference_calls = Cell::new(0);
                let reference_bytes = Cell::new(0);
                let reference_refuse = Cell::new(false);
                let allocator = Counted {
                    heap: &reference_heap,
                    calls: &reference_calls,
                    requested: &reference_bytes,
                    refuse: &reference_refuse,
                };
                let mut reference_elapsed = std::time::Duration::ZERO;
                let mut prepared_elapsed = std::time::Duration::ZERO;
                let mut tiles = 0;
                // Displays drawn at their own size: room, sphere, clock,
                // card, non-aligned edge tiles and a changed/unavailable
                // clock exercise the real shader/font ABI. The reference
                // runs the hole graph and, where the overlay reaches, the
                // compose graph over the hole's own planes.
                let cases = [
                    ([960, 600], [0, 0], Some(76740)),
                    ([960, 600], [640, 256], Some(76740)),
                    ([1920, 1200], [128, 1088], Some(35940)),
                    ([1920, 1200], [1664, 1056], None),
                    ([1307, 803], [1280, 800], Some(36000)),
                ];
                for (index, &(size, origin, seconds)) in cases.iter().enumerate() {
                    let viewport = Viewport::new(size[0], size[1]).unwrap();
                    let clock = Clock::from_unix_seconds(seconds);
                    let frame = Frame(index as u64 * 37);
                    let started = std::time::Instant::now();
                    let hole = program.render_hole_in(viewport, origin, frame, allocator).unwrap();
                    let expected = if viewport.overlays_tile(origin) {
                        let planes = [0, 1, 2].map(|c| hole[c].f32().unwrap());
                        program.render_compose_in(viewport, origin, clock, Hole::Tile(planes), allocator).unwrap()
                    } else {
                        hole
                    };
                    reference_elapsed += started.elapsed();
                    let started = std::time::Instant::now();
                    let actual = worker.render(viewport, origin, clock, frame).unwrap();
                    prepared_elapsed += started.elapsed();
                    assert_tile(&actual, &expected, (size[0] - origin[0]) as usize, (size[1] - origin[1]) as usize);
                    tiles += 1;
                }
                // Biggertron's panel draws its hole at 1920x1200 and composes
                // over it: hole tiles (one clipped at the frame's foot), then
                // display tiles over a synthetic hole frame — the clock's and
                // the corner the pill's shadow reaches, which the compose
                // graph draws, and two the overlay never reaches, which must
                // equal the graph's output bit for bit.
                let panel = Viewport::new(2880, 1800).unwrap();
                assert!(worker.render(panel, [0, 0], Clock::default(), Frame(0)).is_err());
                let hole = panel.hole_frame();
                for origin in [[1664, 1056], [1792, 1184]] {
                    let expected = program.render_hole_in(hole, origin, Frame(90), allocator).unwrap();
                    let actual = worker.render_hole(hole, origin, Frame(90)).unwrap();
                    assert_tile(&actual, &expected, (1920 - origin[0]) as usize, (1200 - origin[1]) as usize);
                    tiles += 1;
                }
                let pixels = gradient(panel);
                let image = HoleImage { pixels: &pixels, width: 1920, height: 1200 };
                let clock = Clock::from_unix_seconds(Some(76740));
                for (origin, overlaid) in [([256, 1184], true), ([2816, 1792], true), ([1408, 320], false), ([0, 1792], false)] {
                    assert_eq!(panel.overlays_tile(origin), overlaid, "tile {origin:?}");
                    let expected = program.render_compose_in(panel, origin, clock, Hole::Image(image), allocator).unwrap();
                    let actual = worker.render_over(panel, origin, clock, image).unwrap();
                    assert_tile(&actual, &expected, (2880 - origin[0]) as usize, (1800 - origin[1]) as usize);
                    tiles += 1;
                }
                assert_eq!(prepared_calls.get(), preparation_calls);
                assert_eq!(prepared_bytes.get(), preparation_bytes);
                assert_eq!(prepared_heap.stats(), initial_heap);
                assert!(reference_calls.get() > tiles);
                std::eprintln!(
                    "welcome prepared: {storage:?}; preparation {preparation_calls} allocations/{preparation_bytes} requested bytes; mapped {}; warm {} allocations; one-shot {} allocations/{} requested bytes across {tiles} tiles; whole-tile render {:?} one-shot vs {:?} prepared",
                    initial_heap.mapped_bytes,
                    prepared_calls.get() - preparation_calls,
                    reference_calls.get(), reference_bytes.get(),
                    reference_elapsed, prepared_elapsed,
                );
            }).unwrap().join().unwrap();
    }

    #[test]
    fn both_graphs_match_their_unfused_reference_from_named_bounded_heaps() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let mut storage = backing::<PROGRAM_BYTES>();
                let mut transient_storage = backing::<REFERENCE_BYTES>();
                let mut reference_storage = backing::<{ 2 * PROGRAM_BYTES }>();
                let mut reference_scratch = backing::<REFERENCE_BYTES>();
                let heap = arena(&mut storage);
                let transient = arena(&mut transient_storage);
                // A separate named arena owns the unfused reference programs
                // and their invocations; like production, each invocation's
                // transient heap is separate from persistent allocation.
                let reference_heap = arena(&mut reference_storage);
                let reference_transient = arena(&mut reference_scratch);
                let program = BootProgram::new_in(&heap).unwrap();
                let stats = program.native_plan_stats();
                std::eprintln!("welcome native plans: {stats:?}");
                for (placement, stats) in program.placements().iter().zip(stats) {
                    assert_eq!(placement.partitions, 1);
                    assert_eq!(stats.logical_nodes, placement.nodes);
                    assert!(stats.instructions <= stats.logical_nodes);
                }
                assert!(stats[0].mul_add_fusions > 0);
                assert!(stats[0].instructions < stats[0].logical_nodes);
                // The production output `out` equals the unfused graph's
                // over the same inputs, bit for bit.
                let assert_unfused = |out: &[Output<_>], graph: &[u8], inputs: &[&[u8]], resources: &[ax_exec::NativeResourceBinding<'_>]| {
                    let graph = ax::Graph::parse(graph).unwrap();
                    let reference = ax_exec::NativeProgram::compile_in(
                        &graph,
                        crate::kernel_policy().into(),
                        false,
                        &reference_heap,
                    )
                    .unwrap();
                    let mut run = reference.begin_with_resources_in(inputs, resources, &reference_transient).unwrap();
                    run.advance(usize::MAX).unwrap();
                    let expected = run.take_retained_in().unwrap();
                    assert_eq!(out.len(), 4);
                    assert_eq!(out.len(), expected.len());
                    for (actual, expected) in out.iter().zip(expected) {
                        let actual = actual.f32().unwrap();
                        assert!(actual.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
                        for (value, bytes) in actual.iter().zip(expected.as_bytes().chunks_exact(4)) {
                            assert_eq!(value.to_le_bytes(), bytes);
                        }
                    }
                };
                let viewport = Viewport::new(960, 600).unwrap();
                for frame in [Frame(150), Frame(1800)] {
                    let out = program.render_hole_in(viewport, [0, 0], frame, &transient).unwrap();
                    let mut inputs = axboot::HoleInputs::new_in(&reference_transient).unwrap();
                    inputs.prepare(viewport, [0, 0], frame).unwrap();
                    assert_unfused(&out, axboot::HOLE_GRAPH, &bytes(inputs.planes()), &resource_bindings(volumes().unwrap()));
                }
                // The clock's tile, over a synthetic hole frame.
                let pixels = gradient(viewport);
                let image = Hole::Image(HoleImage { pixels: &pixels, width: 960, height: 600 });
                let clock = Clock::from_unix_seconds(Some(76740));
                let out = program.render_compose_in(viewport, [0, 384], clock, image, &transient).unwrap();
                let mut inputs = axboot::ComposeInputs::new_in(&reference_transient).unwrap();
                assert!(inputs.prepare(viewport, [0, 384], clock, image).unwrap());
                assert_unfused(&out, axboot::COMPOSE_GRAPH, &bytes(inputs.planes()), &[]);
                assert!(heap.stats().mapped_bytes <= PROGRAM_BYTES);
                assert!(transient.stats().mapped_bytes <= REFERENCE_BYTES);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

#[cfg(test)]
mod sizing_probe {
    use super::*;
    use kore_rt::{Heap, heap::StaticMemSource};

    #[test]
    #[ignore = "sizing probe: prints the welcome's program and worker footprint"]
    fn welcome_footprint() {
        #[repr(align(4096))]
        struct Storage([u8; 16 * 1024 * 1024]);
        std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
            // SAFETY: all-zero bytes are a valid `Storage`.
            let mut program_storage = unsafe { std::boxed::Box::<Storage>::new_zeroed().assume_init() };
            let mut worker_storage = unsafe { std::boxed::Box::<Storage>::new_zeroed().assume_init() };
            let program_heap = Heap::new_in(unsafe { StaticMemSource::over(program_storage.0.as_mut_ptr(), program_storage.0.len()) });
            let worker_heap = Heap::new_in(unsafe { StaticMemSource::over(worker_storage.0.as_mut_ptr(), worker_storage.0.len()) });
            let program = BootProgram::new_in(&program_heap).unwrap();
            std::eprintln!("programs: mapped {} bytes; plans {:?}", program_heap.stats().mapped_bytes, program.native_plan_stats());
            let worker = program.worker_in(&worker_heap).unwrap();
            std::eprintln!("worker: mapped {} bytes; storage {:?}", worker_heap.stats().mapped_bytes, worker.storage());
        }).unwrap().join().unwrap();
    }
}
