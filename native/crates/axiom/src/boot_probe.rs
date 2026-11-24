//! Opt-in investigation of the real welcome graph. Timings are observations,
//! never CI thresholds. This does not select reduced precision in production.
use super::*;
use ax_exec::{NativeProgram, NativeRef};
use kore_rt::{Heap, heap::StaticMemSource};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "release-mode preparation benchmark; prints measurements"]
fn welcome_preparation_probe() {
    #[repr(align(4096))]
    struct Storage([u8; 8 * 1024 * 1024]);
    std::thread::Builder::new()
        .stack_size(48 * 1024 * 1024)
        .spawn(|| {
            let mut storage = Storage([0; 8 * 1024 * 1024]);
            let mut worker_storage = Storage([0; 8 * 1024 * 1024]);
            // SAFETY: the named arena remains live until every allocation is dropped.
            let heap = Heap::new_in(unsafe {
                StaticMemSource::over(storage.0.as_mut_ptr(), storage.0.len())
            });
            let welcome = ax::Graph::parse(axboot::HOLE_GRAPH).unwrap();
            for count in [0, 64, 512] {
                let mut builder = ax::GraphBuilder::new_in(&heap);
                for _ in 0..count {
                    let x = builder
                        .tensor(ax::DType::F32, ax::TensorFlags::INPUT, &[4], &[])
                        .unwrap();
                    let y = builder
                        .tensor(ax::DType::F32, ax::TensorFlags::OUTPUT, &[4], &[])
                        .unwrap();
                    builder
                        .node(ax::OpCode::ABS, ax::NodeFlags::default(), &[x], &[y], &[])
                        .unwrap();
                }
                let bytes = builder.finish().unwrap();
                let graph = if count == 0 {
                    welcome
                } else {
                    bytes.parse().unwrap()
                };
                let native =
                    NativeProgram::compile_in(&graph, crate::kernel_policy().into(), true, &heap)
                        .unwrap();
                let mut samples = [0_u128; 11];
                for sample in 0..12 {
                    let mut elapsed = 0;
                    for _ in 0..20 {
                        // SAFETY: the previous prepared arena and heap have been
                        // dropped; no allocation survives reuse of this backing.
                        let worker_heap = Heap::new_in(unsafe {
                            StaticMemSource::over(
                                worker_storage.0.as_mut_ptr(),
                                worker_storage.0.len(),
                            )
                        });
                        let started = Instant::now();
                        let prepared = native.prepare_native_in(&worker_heap).unwrap();
                        black_box(&prepared);
                        elapsed += started.elapsed().as_nanos();
                    }
                    if sample > 0 {
                        samples[sample - 1] = elapsed / 20;
                    }
                }
                samples.sort_unstable();
                std::eprintln!(
                    "prepare probe inputs={} tensors={} median_ns={} min_ns={} max_ns={}",
                    native.input_ids().len(),
                    graph.tensor_count(),
                    samples[5],
                    samples[0],
                    samples[10]
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
#[ignore = "release-mode welcome census and frame cost; prints measurements"]
fn welcome_frame_probe() {
    #[repr(align(4096))]
    struct Storage([u8; 8 * 1024 * 1024]);
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            // SAFETY: all-zero bytes are a valid `Storage`.
            let mut storage = unsafe { std::boxed::Box::<Storage>::new_zeroed().assume_init() };
            // SAFETY: this named, aligned arena outlives every graph and buffer.
            let heap = Heap::new_in(unsafe {
                StaticMemSource::over(storage.0.as_mut_ptr(), storage.0.len())
            });
            for (name, bytes, changing) in [
                // From frame to frame only the hole's animation time and the
                // camera's swing change (inputs 15 and 16).
                ("hole", axboot::HOLE_GRAPH, (1_u32 << 15) | (1 << 16)),
                // The overlay changes once a minute, with the clock's ink.
                ("compose", axboot::COMPOSE_GRAPH, 1 << 5),
            ] {
                let graph = ax::Graph::parse(bytes).unwrap();
                let native = NativeProgram::compile_in(&graph, crate::kernel_policy().into(), true, &heap).unwrap();
                let mut catalog = Vec::new_in(&heap);
                for plan in native.kernel_plans() {
                    let key = (plan.semantic, plan.implementation);
                    if let Some((_, nodes, elements)) = catalog.iter_mut().find(|(k, _, _)| *k == key) {
                        *nodes += 1;
                        *elements += plan.shape.num_elements();
                    } else {
                        catalog.push((key, 1_usize, plan.shape.num_elements()));
                    }
                }
                // A dependency census, not a proof that a value is cacheable.
                let mut dependencies = Vec::new_in(&heap);
                dependencies.resize(graph.tensor_count(), 0_u32);
                for (i, id) in native.input_ids().iter().enumerate() {
                    dependencies[id.0 as usize] = 1 << i;
                }
                let mut invariant_nodes = 0;
                for &index in native.schedule().order() {
                    let node = graph.node(index as usize).unwrap();
                    // Resource reads sample the asset's baked tables, which no
                    // frame changes: for this census they are pure. Anything
                    // ordered, mutating or resource-writing is not.
                    let impure = ax::NodeFlags::ORDERED.0 | ax::NodeFlags::MUTATING.0 | ax::NodeFlags::WRITES_RESOURCE.0;
                    assert!(node.flags.0 & impure == 0, "dependency census requires pure nodes");
                    let mask = node.inputs().fold(0, |m, id| m | dependencies[id.0 as usize]);
                    invariant_nodes += usize::from(mask & changing == 0);
                    for id in node.outputs() {
                        dependencies[id.0 as usize] = mask;
                    }
                }
                std::eprintln!("welcome {name}: arch={}, native={:?}, invariant_nodes={invariant_nodes}/{}; catalog (semantic/tier, nodes, output elements): {catalog:?}",
                    std::env::consts::ARCH, native.native_plan_stats(), graph.node_count());
            }
            // One worker drawing whole frames exactly as the compositor's
            // lanes do, tile by tile, serially: a display at the lens height
            // or below renders each tile whole; a taller one draws its hole
            // frame, then composes every display tile over it.
            let program = BootProgram::new_in(&heap).unwrap();
            let mut worker = program.worker_in(&heap).unwrap();
            drop(program);
            std::eprintln!("welcome probe storage: {:?}", worker.storage());
            let clock = Clock::from_unix_seconds(Some(76740));
            for size in [[960, 600], [1920, 1200], [1307, 803], [2880, 1800], [3840, 2160]] {
                let viewport = Viewport::new(size[0], size[1]).unwrap();
                let hole = viewport.hole_frame();
                let [hw, hh] = hole.size().map(|v| v as usize);
                let [w, h] = size.map(|v| v as usize);
                let mut hole_pixels = std::vec![0_u32; hw * hh];
                let mut pixels = std::vec![0_u32; w * h];
                let pack = |tile: &BootTile<'_>, origin: [usize; 2], target: &mut [u32], width: usize, height: usize| {
                    let columns = TILE_WIDTH.min(width - origin[0]);
                    for row in 0..TILE_HEIGHT.min(height - origin[1]) {
                        let at = (origin[1] + row) * width + origin[0];
                        tile.write_xrgb_row(row, &mut target[at..at + columns]).unwrap();
                    }
                };
                // Per frame: hole pass, overlaid tiles, bare tiles (ns).
                let mut samples = [[0_u64; 9]; 3];
                let mut overlaid = 0;
                for frame in 0..10 {
                    let frame_number = Frame(frame as u64 * 15);
                    let mut costs = [0_u64; 3];
                    overlaid = 0;
                    if viewport.upsamples_hole() {
                        let started = Instant::now();
                        for y in (0..hh).step_by(TILE_HEIGHT) {
                            for x in (0..hw).step_by(TILE_WIDTH) {
                                let tile = worker.render_hole(hole, [x as u32, y as u32], frame_number).unwrap();
                                pack(&tile, [x, y], &mut hole_pixels, hw, hh);
                            }
                        }
                        costs[0] = started.elapsed().as_nanos() as u64;
                    }
                    for y in (0..h).step_by(TILE_HEIGHT) {
                        for x in (0..w).step_by(TILE_WIDTH) {
                            let origin = [x as u32, y as u32];
                            let over = viewport.overlays_tile(origin);
                            overlaid += usize::from(over);
                            let started = Instant::now();
                            let tile = if viewport.upsamples_hole() {
                                let image = HoleImage { pixels: &hole_pixels, width: hw, height: hh };
                                worker.render_over(viewport, origin, clock, image).unwrap()
                            } else {
                                worker.render(viewport, origin, clock, frame_number).unwrap()
                            };
                            pack(&tile, [x, y], &mut pixels, w, h);
                            costs[if over { 1 } else { 2 }] += started.elapsed().as_nanos() as u64;
                        }
                    }
                    black_box(&pixels);
                    if frame > 0 {
                        for part in 0..3 { samples[part][frame - 1] = costs[part]; }
                    }
                }
                for sample in &mut samples { sample.sort_unstable(); }
                let ms = |ns: u64| ns as f64 / 1e6;
                let median = samples.map(|v| v[4]);
                let tiles = w.div_ceil(TILE_WIDTH) * h.div_ceil(TILE_HEIGHT);
                std::eprintln!("welcome frame {size:?}: hole frame {:?}; serial frame median {:.1} ms = hole pass {:.1} + overlaid tiles {:.1} ({overlaid}/{tiles}) + bare tiles {:.1}; max {:.1} ms",
                    hole.size(), ms(median.iter().sum()), ms(median[0]), ms(median[1]), ms(median[2]),
                    ms(samples.iter().map(|v| v[8]).sum()));
            }
        }).unwrap().join().unwrap();
}
