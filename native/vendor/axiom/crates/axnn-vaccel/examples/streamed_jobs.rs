//! Generic bounded streaming: a packed inference projection and an image tile.
//! RUSTFLAGS='-C target-cpu=native' cargo run --release -p axnn-vaccel --example streamed_jobs
#![feature(allocator_api)]
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use axnn_vaccel::{AxHetero, AxStream, AxStreamProgress, AxTosaAccelerator, Session};
use std::alloc::Global;

fn main() {
    let session = || Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> =
        AxHetero::new(session(), None);
    let mut b = GraphBuilder::new_in(Global);
    let w = b
        .tensor(DType::Q8, TensorFlags::INPUT, &[2, 256], &[])
        .unwrap();
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[1, 256], &[])
        .unwrap();
    let packed = b
        .tensor(DType::Q8, TensorFlags::default(), &[1, 256], &[])
        .unwrap();
    let y = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 1], &[])
        .unwrap();
    b.node(OpCode::CAST, NodeFlags::default(), &[x], &[packed], &[])
        .unwrap();
    b.node(
        OpCode::MATMUL_TRANSPOSED,
        NodeFlags::default(),
        &[w, packed],
        &[y],
        &[],
    )
    .unwrap();
    let projection_bytes = b.finish().unwrap();
    let projection = runtime
        .compile_auto_in(&projection_bytes.parse().unwrap(), Global)
        .unwrap();
    for plan in projection.kernel_plans() {
        println!(
            "{:?}: {} accuracy={:?} scratch={}",
            plan.semantic,
            plan.implementation.name(),
            plan.accuracy,
            plan.temporary_bytes
        );
    }
    let mut b = GraphBuilder::new_in(Global);
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[16, 16], &[])
        .unwrap();
    let y = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[16, 16], &[])
        .unwrap();
    b.node(OpCode::ABS, NodeFlags::default(), &[x], &[y], &[])
        .unwrap();
    let tile_bytes = b.finish().unwrap();
    let tile = runtime
        .compile_auto_in(&tile_bytes.parse().unwrap(), Global)
        .unwrap();
    let mut weights = vec![0u8; 2 * 8 * 34];
    for block in weights.chunks_exact_mut(34) {
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..].fill(1);
    }
    let activation = vec![0u8; 256 * 4];
    let image: Vec<u8> = [-1f32; 256]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut stream = AxStream::new([&projection, &tile], 4096, Global).unwrap();
    stream
        .try_submit(&runtime, 0, vec![weights, activation])
        .unwrap_or_else(|_| panic!("projection admission"));
    stream
        .try_submit(&runtime, 1, vec![image])
        .unwrap_or_else(|_| panic!("tile admission"));
    println!(
        "active={} input allocation bytes={}",
        stream.active_count(),
        stream.input_bytes()
    );
    while stream.active_count() != 0 {
        match stream.poll(&mut runtime) {
            AxStreamProgress::Complete { slot, outputs } => {
                let bytes: usize = outputs.iter().map(|o| o.as_bytes().len()).sum();
                println!("slot {slot} complete: {bytes} bytes");
                if slot == 0 {
                    assert!(outputs[0].as_bytes().iter().all(|&b| b == 0));
                } else {
                    assert!(
                        outputs[0]
                            .as_bytes()
                            .chunks_exact(4)
                            .all(|b| b == 1f32.to_le_bytes())
                    );
                }
            }
            AxStreamProgress::Progress { .. } => std::thread::yield_now(),
            AxStreamProgress::Failed { error, .. } => panic!("{error:?}"),
            AxStreamProgress::Cancelled { .. } | AxStreamProgress::Idle => {
                panic!("unexpected state")
            }
        }
    }
    drop(stream);
    runtime.release_executable(projection).unwrap();
    runtime.release_executable(tile).unwrap();
}
