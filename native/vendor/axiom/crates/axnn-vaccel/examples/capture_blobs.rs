//! Capture what two OpenVINO backends compile for one `.ax` partition and
//! record it for a runtime that only loads offline-compiled programs (kore).
//!
//! The lighting graph is placed and authored into one TOSA partition exactly
//! as `AxHetero` would hand it to a provider; each requested device (`NPU`
//! and `GPU` by default) compiles it through the OpenVINO provider, and
//! `OpenVinoAccelerator::export_program` writes the plugin's compiled form:
//! the NPU plugin's VPUX ELF (kore `FORMAT_VPUX_ELF`), the GPU plugin's
//! serialized graph with its Intel GT zebin kernels embedded. Every blob is
//! bound to the `.ax` graph and TOSA artifact by BLAKE3 in `manifest.json`.
//!
//!     cargo run --release -p axnn-vaccel --features openvino --example capture_blobs
//!
//! `AXNN_VACCEL_OPENVINO_DEVICES` (comma-separated) and `AXNN_VACCEL_BLOB_DIR`
//! (default `target/blobs`) select devices and the output directory.

#![cfg_attr(
    all(feature = "openvino", axnn_vaccel_openvino_runtime),
    feature(allocator_api, f16)
)]

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
#[path = "common/lighting_graph.rs"]
mod lighting_graph;

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
mod capture {
    use std::alloc::Global;
    use std::path::PathBuf;

    use std::collections::HashMap;

    use ax::{DType, Schedule, TensorFlags, TensorId};
    use axnn_vaccel::blob::{BlobRecord, Direction, PortRecord, manifest_json};
    use axnn_vaccel::{AxDeviceModel, Session, artifact};
    use virtio_accel_openvino::{OPENVINO_TOSA_TARGET, OpenVinoAccelerator};

    use super::lighting_graph::{decode, encode, lighting_graph, lighting_inputs};

    pub fn run() -> Result<(), String> {
        let out = PathBuf::from(
            std::env::var("AXNN_VACCEL_BLOB_DIR").unwrap_or_else(|_| "target/blobs".into()),
        );
        std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        let devices =
            std::env::var("AXNN_VACCEL_OPENVINO_DEVICES").unwrap_or_else(|_| "NPU,GPU".into());

        // The partition a provider receives: placed by the scheduler, authored
        // by the same lowering `AxHetero` uses, provider-neutral TOSA bytes.
        let (probes, lights) = (6912_u32, 128_u32);
        let graph_bytes = lighting_graph(DType::F32, probes, lights);
        let graph = graph_bytes.parse().map_err(|e| format!("graph: {e:?}"))?;
        let mut schedule = Schedule::analyze_in(&graph, Global).map_err(|e| format!("{e:?}"))?;
        let model = AxDeviceModel {
            accelerator_launch_cost: 0,
            accelerator_speedup: 32,
            transfer_fixed_cost: 0,
            transfer_byte_cost: 0,
            ..AxDeviceModel::default()
        };
        schedule
            .place_optimized(&graph, &model)
            .map_err(|e| format!("{e:?}"))?;
        let partition = {
            let plan = schedule.build_plan(&graph).map_err(|e| format!("{e:?}"))?;
            if plan.partitions.len() != 1 {
                return Err(format!(
                    "expected one partition, got {}",
                    plan.partitions.len()
                ));
            }
            plan.partitions[0]
        };
        let lowered = artifact::ax_partition_in(
            &graph,
            schedule.order(),
            schedule.producers(),
            partition,
            Global,
        )
        .map_err(|e| format!("partition authoring: {e:?}"))?;
        std::fs::write(out.join("lighting.ax"), graph_bytes.as_bytes())
            .map_err(|e| format!("write lighting.ax: {e}"))?;
        std::fs::write(out.join("lighting.tosa"), &lowered.bytes)
            .map_err(|e| format!("write lighting.tosa: {e}"))?;
        let compiler = OpenVinoAccelerator::runtime_version().unwrap_or_default();
        let (w, rgb) = lighting_inputs(lights);
        let authored = [encode(&w, DType::F32), encode(&rgb, DType::F32)];

        // Bind by tensor id, not by position: the partition declares its own
        // boundary order, which need not match the order the graph was
        // authored in. Feeding positionally would silently compute a
        // different function on any graph whose partition reorders inputs.
        let mut by_tensor: HashMap<TensorId, &[u8]> = HashMap::new();
        let graph_inputs = (0..graph.tensor_count())
            .filter_map(|index| graph.tensor(TensorId(index as u32)).ok())
            .filter(|tensor| tensor.flags.contains(TensorFlags::INPUT));
        for (tensor, bytes) in graph_inputs.zip(&authored) {
            by_tensor.insert(tensor.id, bytes.as_slice());
        }
        let inputs: Vec<&[u8]> = lowered
            .inputs
            .iter()
            .map(|id| {
                by_tensor
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("partition input {id:?} has no authored bytes"))
            })
            .collect::<Result<_, String>>()?;

        // The boundary kore binds device buffers by, in the partition's own
        // order rather than the authoring order.
        let port =
            |direction: Direction, index: usize, id: TensorId| -> Result<PortRecord, String> {
                let tensor = graph
                    .tensor(id)
                    .map_err(|error| format!("boundary tensor {id:?}: {error:?}"))?;
                let shape: Vec<u32> = tensor.dimensions().collect();
                let bytes = ax_exec::native::tensor_storage_bytes(tensor)
                    .ok_or_else(|| format!("boundary tensor {id:?} has no static size"))?;
                Ok(PortRecord {
                    direction,
                    index,
                    tensor: id.0,
                    dtype: format!("{:?}", tensor.dtype).to_lowercase() + "le",
                    shape,
                    bytes: bytes as u64,
                })
            };
        let mut ports = Vec::new();
        for (index, &id) in lowered.inputs.iter().enumerate() {
            ports.push(port(Direction::Input, index, id)?);
        }
        for (index, &id) in lowered.outputs.iter().enumerate() {
            ports.push(port(Direction::Output, index, id)?);
        }

        // The CPU reference for the exact partition bytes each device compiles.
        let reference = axtosa::execute(&lowered.bytes, &inputs)
            .map_err(|e| format!("CPU reference: {e:?}"))?;
        let reference = decode(&reference[0], DType::F32);
        let peak = reference.iter().copied().fold(0.0_f32, f32::max);

        let mut records = Vec::new();
        for device in devices.split(',').map(str::trim).filter(|d| !d.is_empty()) {
            let provider = match OpenVinoAccelerator::with_device(device) {
                Ok(provider) => provider,
                Err(error) => {
                    eprintln!("{device}: unavailable ({error}); skipped");
                    continue;
                }
            };
            let mut session = Session::new(provider, OPENVINO_TOSA_TARGET)
                .map_err(|e| format!("{device}: session: {e:?}"))?;
            if let Err(error) = session.ensure(0, || lowered.bytes.clone()) {
                eprintln!("{device}: did not admit the partition ({error:?}); skipped");
                continue;
            }
            // One execution proves the compiled program runs before its bytes
            // are recorded as a usable artifact.
            let output = session
                .run(0, || unreachable!(), &inputs, probes as usize * 3 * 4)
                .map_err(|e| format!("{device}: execution: {e:?}"))?;
            let got = decode(&output, DType::F32);
            let worst = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let finite = got.iter().all(|v| v.is_finite());
            let resolved = session.provider().device_name().to_owned();
            let path = out.join(format!("lighting-{}.blob", resolved.replace('/', "_")));
            session
                .provider()
                .export_program(session.generic_program(0).expect("loaded"), &path)
                .map_err(|e| format!("{device}: export: {e:?}"))?;
            let blob = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let record = BlobRecord::new(
                "lighting",
                "openvino",
                &resolved,
                &compiler,
                graph_bytes.as_bytes(),
                &lowered.bytes,
                &blob,
            )
            .with_ports(ports.clone());
            // Recorded either way: a blob that executes but disagrees with the
            // CPU reference is evidence about the compiler, and the manifest
            // names the runtime that produced it. The parity is printed, not hidden.
            println!(
                "{resolved}: {} bytes, {} ({} embedded ELF image(s)); kore format {:?}; executed: {}, max |Δ| vs CPU {worst:.6} (peak {peak:.3})",
                record.artifact_bytes,
                record.kind,
                record.kernels.len(),
                record.kore_format,
                if finite { "finite" } else { "NON-FINITE" },
            );
            records.push(record);
        }
        let manifest = out.join("manifest.json");
        std::fs::write(&manifest, manifest_json(&records))
            .map_err(|e| format!("{}: {e}", manifest.display()))?;
        println!("{} record(s) -> {}", records.len(), manifest.display());
        Ok(())
    }
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn main() {
    if let Err(error) = capture::run() {
        eprintln!("capture_blobs: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "openvino", axnn_vaccel_openvino_runtime)))]
fn main() {
    eprintln!(
        "capture_blobs requires --features openvino and an OpenVINO C runtime (VIRTIO_ACCEL_OPENVINO_LIB_DIR or pkg-config)"
    );
}
