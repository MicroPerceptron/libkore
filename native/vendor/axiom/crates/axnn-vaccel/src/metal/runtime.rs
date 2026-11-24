//! Synchronous, owned Metal field storage with checked borrowed resource uploads.

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::slice;
use std::time::Instant;

use ax::sampling::VolumeRgbParams;
use ax::voxel::{ResourceKey, VoxelBrickTraceParams};
use ax::{DType, Graph};
use axnn::lowering::{KernelAccuracy, U8BufferParams};
use axnn_cpu::{BrickGridRef, RgbVolumeRef, U8BufferRef};

use super::compile::{self, MetalFieldPlan, MetalResourceKind, MetalTensorLayout};

const ERROR_BYTES: usize = 2048;

#[derive(Clone, Copy, Debug)]
enum ResourceData<'a> {
    BrickGrid(BrickGridRef<'a>),
    RgbVolume(RgbVolumeRef<'a>),
    U8Buffer(U8BufferRef<'a>),
}

/// A validated resource borrow for one synchronous field invocation.
///
/// The key, kind, dimensions, storage, and exact byte length must match the
/// compiled graph. `run` copies the payload into its resident owned buffer;
/// neither this borrow nor its pointer is retained after the call.
#[derive(Clone, Copy, Debug)]
pub struct MetalResourceBinding<'a> {
    key: ResourceKey,
    data: ResourceData<'a>,
}

impl<'a> MetalResourceBinding<'a> {
    pub const fn brick_grid(key: ResourceKey, grid: BrickGridRef<'a>) -> Self {
        Self {
            key,
            data: ResourceData::BrickGrid(grid),
        }
    }

    pub const fn rgb_volume(key: ResourceKey, volume: RgbVolumeRef<'a>) -> Self {
        Self {
            key,
            data: ResourceData::RgbVolume(volume),
        }
    }

    pub const fn u8_buffer(key: ResourceKey, buffer: U8BufferRef<'a>) -> Self {
        Self {
            key,
            data: ResourceData::U8Buffer(buffer),
        }
    }

    pub const fn key(self) -> ResourceKey {
        self.key
    }

    fn kind(self) -> MetalResourceKind {
        match self.data {
            ResourceData::BrickGrid(grid) => {
                MetalResourceKind::VoxelBrickGrid { dims: grid.dims() }
            }
            ResourceData::RgbVolume(volume) => MetalResourceKind::RgbVolume {
                dims: volume.dims(),
                storage: volume.storage(),
            },
            ResourceData::U8Buffer(buffer) => MetalResourceKind::U8Buffer {
                elements: buffer.elements(),
            },
        }
    }

    // Only padding-free primitive arrays reach this byte-copy boundary. Views
    // validate lengths and contents, and their immutable borrows cover run().
    fn payload(self) -> (*const u8, usize) {
        match self.data {
            ResourceData::BrickGrid(grid) => {
                (grid.bricks().as_ptr().cast(), size_of_val(grid.bricks()))
            }
            ResourceData::RgbVolume(volume) => {
                if let Some(values) = volume.as_f32() {
                    (values.as_ptr().cast(), size_of_val(values))
                } else if let Some(words) = volume.as_uf16() {
                    (words.as_ptr().cast(), size_of_val(words))
                } else {
                    let bytes = volume.as_uf8e5m3().expect("validated RGB storage variant");
                    (bytes.as_ptr().cast(), size_of_val(bytes))
                }
            }
            ResourceData::U8Buffer(buffer) => (buffer.bytes().as_ptr(), buffer.bytes().len()),
        }
    }
}

/// Per-invocation timing, excluding compilation and initial buffer allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetalTiming {
    /// Native GPU execution timestamps; excludes CPU validation and uploads.
    pub gpu_ms: f64,
    /// Validation, uploads, encoding, submission, and synchronous completion.
    pub wall_ms: f64,
}

#[repr(C)]
struct NativeInfo {
    inputs: *mut u8,
    outputs: *mut u8,
    resources: *mut u8,
    device_count: usize,
    device_name: [u8; 256],
}

unsafe extern "C" {
    fn ax_metal_field_create(
        source: *const u8,
        source_length: usize,
        width: u32,
        height: u32,
        input_bytes: usize,
        output_bytes: usize,
        resource_bytes: usize,
        info: *mut NativeInfo,
        error: *mut u8,
        error_capacity: usize,
    ) -> *mut c_void;
    fn ax_metal_field_run(
        handle: *mut c_void,
        gpu_ms: *mut f64,
        error: *mut u8,
        error_capacity: usize,
    ) -> i32;
    fn ax_metal_field_destroy(handle: *mut c_void);
}

#[derive(Debug)]
struct NativeHandle(NonNull<c_void>);

impl Drop for NativeHandle {
    fn drop(&mut self) {
        // SAFETY: successful native creation transfers sole handle ownership.
        // The bridge drains accepted work and releases ARC-owned objects; it
        // contains both C++ and Objective-C exceptions.
        unsafe { ax_metal_field_destroy(self.0.as_ptr()) };
    }
}

/// One compiled `.ax` field with three resident shared Metal buffers.
///
/// The initial implementation requires macOS 15+ and explicit permission for
/// Approximate arithmetic. It never silently falls back to CPU execution.
/// Inputs use the graph's INPUT tensor order, including unused declarations;
/// outputs use its OUTPUT tensor order. Resources are matched by binding key.
///
/// Each run is synchronous. Output slices borrow this object and are available
/// only after successful completion. Any failed run invalidates all outputs;
/// a native execution failure also poisons the object, preventing further
/// writes to storage that could still be referenced by a failed command.
#[derive(Debug)]
pub struct MetalField {
    plan: MetalFieldPlan,
    native: NativeHandle,
    inputs: NonNull<u8>,
    outputs: NonNull<u8>,
    resources: NonNull<u8>,
    device_name: String,
    device_count: usize,
    output_valid: bool,
    poisoned: bool,
}

impl MetalField {
    pub fn compile(
        graph: &Graph<'_>,
        width: u32,
        height: u32,
        minimum_accuracy: KernelAccuracy,
    ) -> Result<Self, String> {
        // All semantic, accuracy, and descriptor rejection happens before
        // opening a native device, compiling MSL, or allocating Metal buffers.
        let plan = compile::compile(graph, width, height, minimum_accuracy)?;
        validate_plan(&plan)?;
        if plan.accuracy < minimum_accuracy {
            return Err("Metal field does not meet the requested minimum accuracy".into());
        }
        let helpers = include_str!("kernels.metal");
        let source_bytes = helpers
            .len()
            .checked_add(plan.source.len())
            .and_then(|n| n.checked_add(1))
            .ok_or("Metal field source length overflow")?;
        let mut source = String::new();
        source
            .try_reserve_exact(source_bytes)
            .map_err(|_| "Cannot allocate Metal field source")?;
        source.push_str(helpers);
        source.push('\n');
        source.push_str(&plan.source);
        let mut error = [0_u8; ERROR_BYTES];
        let mut info = NativeInfo {
            inputs: ptr::null_mut(),
            outputs: ptr::null_mut(),
            resources: ptr::null_mut(),
            device_count: 0,
            device_name: [0; 256],
        };
        // SAFETY: all pointers are valid for this synchronous call. The bridge
        // copies the UTF-8 source and allocates/owns the three returned buffers.
        let raw = unsafe {
            ax_metal_field_create(
                source.as_ptr(),
                source.len(),
                width,
                height,
                plan.initial_inputs.len(),
                plan.output_bytes,
                plan.resource_bytes,
                &mut info,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let native = NativeHandle(
            NonNull::new(raw).ok_or_else(|| native_error(&error, "Metal field creation failed"))?,
        );
        let inputs = checked_mapping(info.inputs, "input")?;
        let outputs = checked_mapping(info.outputs, "output")?;
        let resources = checked_mapping(info.resources, "resource")?;
        // SAFETY: the bridge allocated at least initial_inputs.len() bytes.
        // This allocation is distinct from the plan's initialized byte Vec.
        unsafe {
            ptr::copy_nonoverlapping(
                plan.initial_inputs.as_ptr(),
                inputs.as_ptr(),
                plan.initial_inputs.len(),
            );
        }
        Ok(Self {
            plan,
            native,
            inputs,
            outputs,
            resources,
            device_name: bounded_text(&info.device_name),
            device_count: info.device_count,
            output_valid: false,
            poisoned: false,
        })
    }

    pub fn run(
        &mut self,
        inputs: &[&[u8]],
        resources: &[MetalResourceBinding<'_>],
    ) -> Result<MetalTiming, String> {
        let start = Instant::now();
        self.output_valid = false;
        if self.poisoned {
            return Err("Metal field cannot be reused after a native execution failure".into());
        }
        validate_run(&self.plan, inputs, resources)?;
        // Validation of every binding and every input precedes every upload.
        // There is no per-run payload allocation and no new output plane.
        for (layout, bytes) in self.plan.inputs.iter().zip(inputs) {
            // SAFETY: validate_plan checked this range; validate_run checked
            // exact input length. Public callers cannot borrow this mapping.
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.inputs.as_ptr().add(layout.offset),
                    bytes.len(),
                );
            }
        }
        for requirement in &self.plan.resources {
            let binding = resources
                .iter()
                .find(|binding| binding.key == requirement.key)
                .expect("all resource keys validated before upload");
            let (source, bytes) = binding.payload();
            // SAFETY: typed borrowed slices contain initialized primitive
            // arrays. All ranges/lengths match the owned destination, and no
            // submitted command remains after the previous successful run.
            unsafe {
                ptr::copy_nonoverlapping(
                    source,
                    self.resources.as_ptr().add(requirement.offset),
                    bytes,
                );
            }
        }
        let mut error = [0_u8; ERROR_BYTES];
        let mut gpu_ms = 0.0;
        // SAFETY: this object uniquely owns the live handle. The native call
        // completes synchronously or reports failure without unwinding.
        let success = unsafe {
            ax_metal_field_run(
                self.native.0.as_ptr(),
                &mut gpu_ms,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if success != 1 {
            self.poisoned = true;
            return Err(native_error(&error, "Metal field execution failed"));
        }
        self.output_valid = true;
        Ok(MetalTiming {
            gpu_ms,
            wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn output_f32(&self, index: usize) -> Option<&[f32]> {
        let layout = self.output_layout(index, DType::F32)?;
        // SAFETY: compile validated the 4-byte-aligned range and element
        // count. Completion synchronized GPU writes, and &self ties the slice
        // to its owner while excluding the mutable run() borrow.
        Some(unsafe {
            slice::from_raw_parts(
                self.outputs.as_ptr().add(layout.offset).cast(),
                layout.byte_len / 4,
            )
        })
    }

    pub fn output_i32(&self, index: usize) -> Option<&[i32]> {
        let layout = self.output_layout(index, DType::I32)?;
        // SAFETY: same validated resident mapping and completed-run lifetime
        // as output_f32; all I32 bit patterns are valid values.
        Some(unsafe {
            slice::from_raw_parts(
                self.outputs.as_ptr().add(layout.offset).cast(),
                layout.byte_len / 4,
            )
        })
    }

    pub fn output_u8(&self, index: usize) -> Option<&[u8]> {
        let layout = self.output_layout(index, DType::U8)?;
        // SAFETY: the initialized output range is valid until the next
        // exclusive run or destruction; both require ending this borrow.
        Some(unsafe {
            slice::from_raw_parts(self.outputs.as_ptr().add(layout.offset), layout.byte_len)
        })
    }

    fn output_layout(&self, index: usize, dtype: DType) -> Option<&MetalTensorLayout> {
        self.output_valid.then_some(())?;
        self.plan
            .outputs
            .get(index)
            .filter(|layout| layout.dtype == dtype)
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
    pub fn device_count(&self) -> usize {
        self.device_count
    }
    pub fn node_count(&self) -> usize {
        self.plan.node_count
    }
    pub fn input_count(&self) -> usize {
        self.plan.inputs.len()
    }
    pub fn output_count(&self) -> usize {
        self.plan.outputs.len()
    }
    pub fn resource_count(&self) -> usize {
        self.plan.resources.len()
    }
    pub fn dimensions(&self) -> [u32; 2] {
        [self.plan.width, self.plan.height]
    }
    pub fn accuracy(&self) -> KernelAccuracy {
        self.plan.accuracy
    }
    pub fn output_shape(&self, index: usize) -> Option<&[u32]> {
        self.plan
            .outputs
            .get(index)
            .map(|layout| layout.shape.as_slice())
    }
}

fn checked_mapping(pointer: *mut u8, label: &str) -> Result<NonNull<u8>, String> {
    NonNull::new(pointer)
        .filter(|p| p.as_ptr().addr().is_multiple_of(8))
        .ok_or_else(|| format!("Metal {label} mapping is null or insufficiently aligned"))
}

fn bounded_text(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn native_error(bytes: &[u8], fallback: &str) -> String {
    let message = bounded_text(bytes);
    if message.is_empty() {
        fallback.to_owned()
    } else {
        message
    }
}

fn validate_plan(plan: &MetalFieldPlan) -> Result<(), String> {
    if plan.width == 0 || plan.height == 0 || plan.outputs.is_empty() {
        return Err("Metal field has invalid dimensions or no outputs".into());
    }
    for bytes in [
        plan.initial_inputs.len(),
        plan.output_bytes,
        plan.resource_bytes,
    ] {
        if bytes > isize::MAX as usize {
            return Err("Metal field mapping exceeds the host addressable range".into());
        }
    }
    validate_layouts(&plan.inputs, plan.initial_inputs.len(), "input")?;
    validate_layouts(&plan.outputs, plan.output_bytes, "output")?;
    let mut end = 0;
    for (index, resource) in plan.resources.iter().enumerate() {
        let expected = match resource.kind {
            MetalResourceKind::VoxelBrickGrid { dims } => {
                VoxelBrickTraceParams::new(resource.key, dims)
                    .and_then(|p| p.brick_count())
                    .and_then(|n| n.checked_mul(8))
            }
            MetalResourceKind::RgbVolume { dims, storage } => {
                VolumeRgbParams::new(resource.key, dims, storage).and_then(|p| p.storage_bytes())
            }
            MetalResourceKind::U8Buffer { elements } => {
                U8BufferParams::new(resource.key, elements).map(|p| p.elements as usize)
            }
        };
        let next = resource
            .offset
            .checked_add(resource.byte_len)
            .filter(|&next| next <= plan.resource_bytes);
        if expected != Some(resource.byte_len)
            || !resource.offset.is_multiple_of(8)
            || resource.offset < end
            || next.is_none()
            || plan.resources[..index]
                .iter()
                .any(|previous| previous.key == resource.key)
        {
            return Err(format!(
                "Metal resource {index} has an invalid or overlapping layout"
            ));
        }
        end = next.unwrap();
    }
    Ok(())
}

fn validate_layouts(
    layouts: &[MetalTensorLayout],
    total: usize,
    label: &str,
) -> Result<(), String> {
    let mut end = 0;
    for layout in layouts {
        let element_bytes = match layout.dtype {
            DType::F32 | DType::I32 => 4,
            DType::U8 => 1,
            _ => return Err(format!("Metal {label} has an unsupported storage dtype")),
        };
        let bytes = layout
            .shape
            .iter()
            .try_fold(element_bytes, |count: usize, &extent| {
                (extent != 0 && extent != ax::DYNAMIC_DIM).then_some(())?;
                count.checked_mul(extent as usize)
            });
        let next = layout
            .offset
            .checked_add(layout.byte_len)
            .filter(|&next| next <= total);
        if bytes != Some(layout.byte_len)
            || !layout.offset.is_multiple_of(element_bytes)
            || layout.offset < end
            || next.is_none()
        {
            return Err(format!(
                "Metal {label} tensor {} has an invalid or overlapping layout",
                layout.tensor.0
            ));
        }
        end = next.unwrap();
    }
    Ok(())
}

fn validate_run(
    plan: &MetalFieldPlan,
    inputs: &[&[u8]],
    resources: &[MetalResourceBinding<'_>],
) -> Result<(), String> {
    if inputs.len() != plan.inputs.len() {
        return Err(format!(
            "Metal field expects {} inputs, got {}",
            plan.inputs.len(),
            inputs.len()
        ));
    }
    for (index, (layout, bytes)) in plan.inputs.iter().zip(inputs).enumerate() {
        if bytes.len() != layout.byte_len {
            return Err(format!(
                "Metal input {index} expects {} bytes, got {}",
                layout.byte_len,
                bytes.len()
            ));
        }
    }
    if resources.len() != plan.resources.len() {
        return Err(format!(
            "Metal field expects {} resource bindings, got {}",
            plan.resources.len(),
            resources.len()
        ));
    }
    for (index, binding) in resources.iter().enumerate() {
        if resources[..index]
            .iter()
            .any(|previous| previous.key == binding.key)
        {
            return Err(format!(
                "Duplicate Metal resource group {} binding {}",
                binding.key.group, binding.key.binding
            ));
        }
        let requirement = plan
            .resources
            .iter()
            .find(|r| r.key == binding.key)
            .ok_or_else(|| {
                format!(
                    "Unexpected Metal resource group {} binding {}",
                    binding.key.group, binding.key.binding
                )
            })?;
        if requirement.kind != binding.kind() || requirement.byte_len != binding.payload().1 {
            return Err(format!(
                "Metal resource group {} binding {} has the wrong kind, geometry, storage, or length",
                binding.key.group, binding.key.binding
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Global;

    use super::super::compile::MetalResourceRequirement;
    use ax::sampling::VolumeRgbStorage;
    use ax::{GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId};

    const GRID_KEY: ResourceKey = ResourceKey {
        group: 1,
        binding: 2,
    };
    const RGB_KEY: ResourceKey = ResourceKey {
        group: 1,
        binding: 3,
    };
    const BYTES_KEY: ResourceKey = ResourceKey {
        group: 2,
        binding: 1,
    };

    fn preflight_fixture() -> MetalFieldPlan {
        MetalFieldPlan {
            source: String::new(),
            width: 3,
            height: 2,
            inputs: vec![
                MetalTensorLayout {
                    tensor: TensorId(0),
                    dtype: DType::F32,
                    shape: vec![2, 3],
                    offset: 0,
                    byte_len: 24,
                },
                // Unused input declarations must still have the right size.
                MetalTensorLayout {
                    tensor: TensorId(1),
                    dtype: DType::U8,
                    shape: vec![],
                    offset: 24,
                    byte_len: 1,
                },
            ],
            outputs: vec![MetalTensorLayout {
                tensor: TensorId(2),
                dtype: DType::F32,
                shape: vec![2, 3],
                offset: 0,
                byte_len: 24,
            }],
            initial_inputs: vec![0; 28],
            output_bytes: 24,
            resource_bytes: 28,
            resources: vec![
                MetalResourceRequirement {
                    key: GRID_KEY,
                    kind: MetalResourceKind::VoxelBrickGrid { dims: [4; 3] },
                    offset: 0,
                    byte_len: 8,
                },
                MetalResourceRequirement {
                    key: RGB_KEY,
                    kind: MetalResourceKind::RgbVolume {
                        dims: [1; 3],
                        storage: VolumeRgbStorage::F32,
                    },
                    offset: 8,
                    byte_len: 12,
                },
                MetalResourceRequirement {
                    key: BYTES_KEY,
                    kind: MetalResourceKind::U8Buffer { elements: 4 },
                    offset: 24,
                    byte_len: 4,
                },
            ],
            node_count: 3,
            accuracy: KernelAccuracy::Approximate,
        }
    }

    #[test]
    fn preflight_checks_every_input_and_keyed_typed_resource() {
        let plan = preflight_fixture();
        validate_plan(&plan).unwrap();
        let grid = [0_u64];
        let rgb = [[1.0, 2.0, 3.0]];
        let bytes = [0_u8, 127, 128, 255];
        let grid =
            MetalResourceBinding::brick_grid(GRID_KEY, BrickGridRef::new(&grid, [4; 3]).unwrap());
        let rgb =
            MetalResourceBinding::rgb_volume(RGB_KEY, RgbVolumeRef::new_f32(&rgb, [1; 3]).unwrap());
        let bytes = MetalResourceBinding::u8_buffer(BYTES_KEY, U8BufferRef::new(&bytes).unwrap());
        let inputs: [&[u8]; 2] = [&[0; 24], &[7]];
        let valid = [bytes, grid, rgb]; // Keys, not caller ordering, select storage.
        validate_run(&plan, &inputs, &valid).unwrap();
        assert!(validate_run(&plan, &inputs[..1], &valid).is_err());
        assert!(validate_run(&plan, &[inputs[0], &[]], &valid).is_err());
        assert!(validate_run(&plan, &inputs, &valid[..2]).is_err());
        assert!(validate_run(&plan, &inputs, &[grid, grid, rgb]).is_err());
        let wrong_key = MetalResourceBinding {
            key: ResourceKey {
                group: 99,
                binding: 99,
            },
            ..bytes
        };
        assert!(validate_run(&plan, &inputs, &[wrong_key, grid, rgb]).is_err());
        let wrong_kind = MetalResourceBinding {
            key: RGB_KEY,
            ..grid
        };
        assert!(validate_run(&plan, &inputs, &[bytes, grid, wrong_kind]).is_err());
        let uf = [[0x7800_u16; 3]];
        let wrong_storage =
            MetalResourceBinding::rgb_volume(RGB_KEY, RgbVolumeRef::new_uf16(&uf, [1; 3]).unwrap());
        assert!(validate_run(&plan, &inputs, &[bytes, grid, wrong_storage]).is_err());
        let bigger_grid = [0_u64; 2];
        let wrong_dims = MetalResourceBinding::brick_grid(
            GRID_KEY,
            BrickGridRef::new(&bigger_grid, [8, 4, 4]).unwrap(),
        );
        assert!(validate_run(&plan, &inputs, &[bytes, wrong_dims, rgb]).is_err());
        let short =
            MetalResourceBinding::u8_buffer(BYTES_KEY, U8BufferRef::new(&[3, 4, 5]).unwrap());
        assert!(validate_run(&plan, &inputs, &[short, grid, rgb]).is_err());
    }

    #[test]
    fn invalid_mapping_ranges_and_metadata_are_rejected_before_native_access() {
        let mut p = preflight_fixture();
        p.inputs[0].offset = 1;
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.inputs[1].offset = 0;
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.outputs[0].byte_len = 28;
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.outputs[0].shape = vec![u32::MAX, u32::MAX, u32::MAX];
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.resources[1].offset = usize::MAX;
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.resources[1].offset = 0;
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.resources[1].byte_len = 11;
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.resources[0].kind = MetalResourceKind::VoxelBrickGrid { dims: [5; 3] };
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.resources[2].key = GRID_KEY;
        assert!(validate_plan(&p).is_err());
        let mut p = preflight_fixture();
        p.resources[2].kind = MetalResourceKind::U8Buffer { elements: 0 };
        p.resources[2].byte_len = 0;
        p.resource_bytes = 24;
        validate_plan(&p).unwrap();
        assert!(checked_mapping(ptr::null_mut(), "test").is_err());
        assert!(checked_mapping(ptr::without_provenance_mut(1), "test").is_err());
        assert_eq!(native_error(&[0; 3], "fallback"), "fallback");
        assert_eq!(bounded_text(b"bounded\0ignored"), "bounded");
        assert_eq!(bounded_text(b"no nul"), "no nul");
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
    fn pin(
        builder: &mut GraphBuilder<Global>,
        input: TensorId,
        dtype: DType,
        shape: &[u32],
    ) -> TensorId {
        let output = builder
            .tensor(dtype, TensorFlags::OUTPUT, shape, &[])
            .unwrap();
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[input],
                &[output],
                &[],
            )
            .unwrap();
        output
    }

    // These tests deliberately require explicit native execution. A missing or
    // inaccessible driver is a test failure, never an automatic skip/pass.
    #[test]
    #[ignore = "requires a native Metal device and explicit driver access"]
    fn native_field_reuses_typed_outputs_and_invalidates_before_failed_preflight() {
        let mut builder = GraphBuilder::new_in(Global);
        let x = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[3, 5], &[])
            .unwrap();
        let row = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 5], &[])
            .unwrap();
        let column = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[3, 1], &[])
            .unwrap();
        let integers = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[3, 5], &[])
            .unwrap();
        let mask = builder
            .tensor(DType::U8, TensorFlags::INPUT, &[3, 5], &[])
            .unwrap();
        let _unused = builder
            .tensor(DType::U8, TensorFlags::INPUT, &[], &[])
            .unwrap();
        let constant = builder
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[],
                &2.0_f32.to_le_bytes(),
            )
            .unwrap();
        let sum = builder
            .tensor(DType::F32, TensorFlags::default(), &[3, 5], &[])
            .unwrap();
        let scaled = builder
            .tensor(DType::F32, TensorFlags::default(), &[3, 5], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[3, 5], &[])
            .unwrap();
        builder
            .node(OpCode::ADD, NodeFlags::default(), &[x, row], &[sum], &[])
            .unwrap();
        builder
            .node(
                OpCode::MUL,
                NodeFlags::default(),
                &[sum, column],
                &[scaled],
                &[],
            )
            .unwrap();
        builder
            .node(
                OpCode::ADD,
                NodeFlags::default(),
                &[scaled, constant],
                &[output],
                &[],
            )
            .unwrap();
        pin(&mut builder, integers, DType::I32, &[3, 5]);
        pin(&mut builder, mask, DType::U8, &[3, 5]);
        pin(&mut builder, row, DType::F32, &[1, 5]);
        pin(&mut builder, column, DType::F32, &[3, 1]);
        pin(&mut builder, constant, DType::F32, &[]);
        let graph = builder.finish().unwrap();
        let mut field =
            MetalField::compile(&graph.parse().unwrap(), 5, 3, KernelAccuracy::Approximate)
                .unwrap();
        assert!(!field.device_name().is_empty());
        assert!(field.device_count() > 0);
        assert_eq!(field.input_count(), 6);
        assert_eq!(field.output_count(), 6);
        assert_eq!(field.output_shape(3), Some([1, 5].as_slice()));
        assert!(field.output_f32(0).is_none());
        let row_values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let column_values = [1.0, 2.0, 4.0];
        let ids: Vec<i32> = (0..15).map(|i| 16_777_217 + i).collect();
        let masks: Vec<u8> = (0..15).map(|i| (i * 17) as u8).collect();
        let mut output_pointer = ptr::null();
        for run in 0..3 {
            let values: Vec<f32> = (0..15).map(|i| (i + run * 31) as f32).collect();
            let owned = [
                f32_bytes(&values),
                f32_bytes(&row_values),
                f32_bytes(&column_values),
                i32_bytes(&ids),
                masks.clone(),
                vec![129],
            ];
            let inputs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            let timing = field.run(&inputs, &[]).unwrap();
            assert!(timing.gpu_ms > 0.0 && timing.wall_ms > 0.0);
            let got = field.output_f32(0).unwrap();
            for i in 0..15 {
                assert_eq!(
                    got[i].to_bits(),
                    ((values[i] + row_values[i % 5]) * column_values[i / 5] + 2.0).to_bits()
                );
            }
            if run == 0 {
                output_pointer = got.as_ptr();
            } else {
                assert_eq!(got.as_ptr(), output_pointer);
            }
            assert_eq!(field.output_i32(1).unwrap(), ids);
            assert_eq!(field.output_u8(2).unwrap(), masks);
            assert_eq!(field.output_f32(3).unwrap(), row_values);
            assert_eq!(field.output_f32(4).unwrap(), column_values);
            assert_eq!(field.output_f32(5).unwrap(), [2.0]);
            assert!(field.output_i32(0).is_none());
            assert!(field.output_f32(6).is_none());
            // Invalid *last* input must not upload earlier changed inputs.
            let before = unsafe {
                slice::from_raw_parts(field.inputs.as_ptr(), field.plan.initial_inputs.len())
            }
            .to_vec();
            let changed = f32_bytes(&[99.0; 15]);
            let mut invalid = inputs.clone();
            invalid[0] = &changed;
            invalid[5] = &[];
            assert!(field.run(&invalid, &[]).is_err());
            assert!(field.output_f32(0).is_none());
            assert!(field.output_i32(1).is_none());
            assert!(field.output_u8(2).is_none());
            assert!(!field.poisoned);
            assert_eq!(
                unsafe { slice::from_raw_parts(field.inputs.as_ptr(), before.len()) },
                before
            );
        }
    }

    #[test]
    #[ignore = "requires a native Metal device and explicit driver access"]
    fn native_integer_guards_shifts_and_explicit_casts_match_the_cpu_contract() {
        let a = [
            i32::MIN,
            i32::MIN,
            i32::MAX,
            i32::MAX,
            1,
            -1,
            i32::MIN,
            0x5555_5555,
            -12_345,
            i32::MIN,
            -1,
            0,
            2,
            16_777_217,
            99,
            -99,
            7,
            -7,
        ];
        let b = [
            -1,
            0,
            1,
            2,
            31,
            32,
            -1,
            0,
            31,
            32,
            1,
            0,
            i32::MAX,
            i32::MIN,
            -3,
            3,
            -2,
            2,
        ];
        let floats = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2_147_483_648.0,
            -2_147_483_648.0,
            f32::from_bits(0x4eff_ffff),
            f32::MAX,
            -f32::MAX,
            0.0,
            -0.0,
            1.75,
            -1.75,
            f32::from_bits(1),
            -f32::from_bits(1),
            16_777_218.0,
            -16_777_218.0,
            f32::from_bits(0x7fc0_1234),
            255.5,
        ];
        let mut builder = GraphBuilder::new_in(Global);
        let left = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[2, 9], &[])
            .unwrap();
        let right = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[2, 9], &[])
            .unwrap();
        let float = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[2, 9], &[])
            .unwrap();
        let operations = [
            OpCode::ADD,
            OpCode::SUB,
            OpCode::MUL,
            OpCode::DIV,
            OpCode::REMAINDER,
            OpCode::BITWISE_AND,
            OpCode::BITWISE_OR,
            OpCode::BITWISE_XOR,
            OpCode::SHIFT_LEFT,
            OpCode::SHIFT_RIGHT,
        ];
        for op in operations {
            let out = builder
                .tensor(DType::I32, TensorFlags::OUTPUT, &[2, 9], &[])
                .unwrap();
            builder
                .node(op, NodeFlags::default(), &[left, right], &[out], &[])
                .unwrap();
        }
        for (input, dtype) in [(float, DType::I32), (left, DType::F32), (float, DType::F32)] {
            let out = builder
                .tensor(dtype, TensorFlags::OUTPUT, &[2, 9], &[])
                .unwrap();
            builder
                .node(OpCode::CAST, NodeFlags::default(), &[input], &[out], &[])
                .unwrap();
        }
        let graph = builder.finish().unwrap();
        let mut field =
            MetalField::compile(&graph.parse().unwrap(), 9, 2, KernelAccuracy::Approximate)
                .unwrap();
        field
            .run(&[&i32_bytes(&a), &i32_bytes(&b), &f32_bytes(&floats)], &[])
            .unwrap();
        for i in 0..a.len() {
            let expected = [
                a[i].wrapping_add(b[i]),
                a[i].wrapping_sub(b[i]),
                a[i].wrapping_mul(b[i]),
                if b[i] == 0 {
                    a[i]
                } else {
                    a[i].wrapping_div(b[i])
                },
                if b[i] == 0 {
                    0
                } else {
                    a[i].wrapping_rem(b[i])
                },
                a[i] & b[i],
                a[i] | b[i],
                a[i] ^ b[i],
                a[i].wrapping_shl(b[i] as u32),
                a[i].wrapping_shr(b[i] as u32),
            ];
            for (output, reference) in expected.into_iter().enumerate() {
                assert_eq!(
                    field.output_i32(output).unwrap()[i],
                    reference,
                    "operator {output}, lane {i}"
                );
            }
            assert_eq!(
                field.output_i32(10).unwrap()[i],
                floats[i] as i32,
                "F32->I32 lane {i}"
            );
            assert_eq!(
                field.output_f32(11).unwrap()[i].to_bits(),
                (a[i] as f32).to_bits(),
                "I32->F32 lane {i}"
            );
            assert_eq!(
                field.output_f32(12).unwrap()[i].to_bits(),
                floats[i].to_bits(),
                "same-type CAST lane {i}"
            );
        }
    }

    fn set_cell(bricks: &mut [u64], dims: [u32; 3], cell: [u32; 3], occupied: bool) {
        let b =
            (cell[0] / 4 + (dims[0] / 4) * (cell[1] / 4 + (dims[1] / 4) * (cell[2] / 4))) as usize;
        let bit = 1_u64 << (cell[0] % 4 + 4 * (cell[1] % 4) + 16 * (cell[2] % 4));
        if occupied {
            bricks[b] |= bit;
        } else {
            bricks[b] &= !bit;
        }
    }

    #[test]
    #[ignore = "requires a native Metal device and explicit driver access"]
    fn native_voxel_queries_preserve_ordinary_hits_large_ids_and_live_edits() {
        let width = 17;
        let height = 3;
        let dims = [260; 3];
        let params = VoxelBrickTraceParams::new(GRID_KEY, dims).unwrap();
        let mut builder = GraphBuilder::new_in(Global);
        let planes: Vec<_> = (0..6)
            .map(|_| {
                builder
                    .tensor(DType::F32, TensorFlags::INPUT, &[height, width], &[])
                    .unwrap()
            })
            .collect();
        let limit = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[], &[])
            .unwrap();
        let origins = builder
            .interleave3_f32([planes[0], planes[1], planes[2]], [height, width])
            .unwrap();
        let dirs = builder
            .interleave3_f32([planes[3], planes[4], planes[5]], [height, width])
            .unwrap();
        let hit = builder
            .voxel_brick_trace(origins, dirs, limit, params, [height, width])
            .unwrap();
        pin(&mut builder, hit.t, DType::F32, &[height, width]);
        pin(&mut builder, hit.cell_id, DType::I32, &[height, width]);
        let graph = builder.finish().unwrap();
        let mut field = MetalField::compile(
            &graph.parse().unwrap(),
            width,
            height,
            KernelAccuracy::Approximate,
        )
        .unwrap();
        let cases = [
            ([259.5, 259.5, 259.5], [1.0, 0.0, -0.0]),
            ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            ([260.0; 3], [-1.0; 3]),
            ([4.0, 4.0, 4.0], [-1.0, -1.0, -1.0]),
            ([259.5, 259.5, 261.0], [0.0, -0.0, -2.0]),
            ([120.25, 80.75, -3.0], [0.25, 0.125, 1.0]),
            ([0.0, 0.0, 0.0], [-0.0, 0.0, -0.0]),
            ([f32::NAN, 1.0, 1.0], [1.0, 0.0, 0.0]),
            ([1.0, 1.0, 1.0], [1.0, f32::INFINITY, 0.0]),
        ];
        let rays: Vec<_> = (0..width * height)
            .map(|i| cases[i as usize % cases.len()])
            .collect();
        let ray_bytes: Vec<_> = (0..6)
            .map(|axis| {
                f32_bytes(
                    &rays
                        .iter()
                        .map(|(origin, direction)| {
                            if axis < 3 {
                                origin[axis]
                            } else {
                                direction[axis - 3]
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut bricks = vec![0_u64; params.brick_count().unwrap()];
        for edited in [false, true] {
            set_cell(&mut bricks, dims, [0; 3], !edited);
            set_cell(&mut bricks, dims, [259; 3], !edited);
            set_cell(&mut bricks, dims, [4; 3], edited);
            for tmax in [f32::INFINITY, 0.0, 1.0, 700.0, -f32::from_bits(1), f32::NAN] {
                let limit_bytes = tmax.to_le_bytes();
                let mut inputs: Vec<&[u8]> = ray_bytes.iter().map(Vec::as_slice).collect();
                inputs.push(&limit_bytes);
                field
                    .run(
                        &inputs,
                        &[MetalResourceBinding::brick_grid(
                            GRID_KEY,
                            BrickGridRef::new(&bricks, dims).unwrap(),
                        )],
                    )
                    .unwrap();
                let ts = field.output_f32(0).unwrap();
                let ids = field.output_i32(1).unwrap();
                for (i, &(origin, direction)) in rays.iter().enumerate() {
                    let expected = if tmax.is_nan()
                        || tmax < 0.0
                        || origin
                            .iter()
                            .chain(direction.iter())
                            .any(|v| !v.is_finite())
                    {
                        (f32::INFINITY, u32::MAX)
                    } else {
                        axnn_cpu::voxel::brick_trace(&bricks, dims, origin, direction, tmax)
                    };
                    assert_eq!(
                        ids[i], expected.1 as i32,
                        "ray {i}, edited {edited}, limit {tmax}"
                    );
                    assert_eq!(
                        ts[i].to_bits(),
                        expected.0.to_bits(),
                        "ray {i}, edited {edited}, limit {tmax}"
                    );
                }
                if !edited && tmax.is_infinite() {
                    assert!(ids[0] > 16_777_216);
                }
            }
        }
        assert_eq!(field.resource_count(), 1);
        // Resources are checked before earlier valid input slots are copied.
        let before_inputs = unsafe {
            slice::from_raw_parts(field.inputs.as_ptr(), field.plan.initial_inputs.len())
        }
        .to_vec();
        let before_resource =
            unsafe { slice::from_raw_parts(field.resources.as_ptr(), field.plan.resource_bytes) }
                .to_vec();
        let wrong_grid = [u64::MAX];
        let limit = 999.0_f32.to_le_bytes();
        let mut inputs: Vec<&[u8]> = ray_bytes.iter().map(Vec::as_slice).collect();
        inputs.push(&limit);
        assert!(
            field
                .run(
                    &inputs,
                    &[MetalResourceBinding::brick_grid(
                        GRID_KEY,
                        BrickGridRef::new(&wrong_grid, [4; 3]).unwrap()
                    )]
                )
                .is_err()
        );
        assert!(field.output_i32(1).is_none());
        assert_eq!(
            unsafe { slice::from_raw_parts(field.inputs.as_ptr(), before_inputs.len()) },
            before_inputs
        );
        assert_eq!(
            unsafe { slice::from_raw_parts(field.resources.as_ptr(), before_resource.len()) },
            before_resource
        );
    }

    fn scalar_sample(values: &[[f32; 3]], dims: [u32; 3], point: [f32; 3]) -> [f32; 3] {
        if point.iter().any(|v| v.is_nan()) {
            return [f32::NAN; 3];
        }
        let mut lo = [0; 3];
        let mut hi = [0; 3];
        let mut f = [0.0; 3];
        for axis in 0..3 {
            if dims[axis] != 1 {
                let p = point[axis].clamp(0.0, (dims[axis] - 1) as f32);
                lo[axis] = p as usize;
                hi[axis] = (lo[axis] + 1).min(dims[axis] as usize - 1);
                f[axis] = p - lo[axis] as f32;
            }
        }
        let mut out = [0.0_f32; 3];
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let i = if x == 0 { lo[0] } else { hi[0] };
                    let j = if y == 0 { lo[1] } else { hi[1] };
                    let k = if z == 0 { lo[2] } else { hi[2] };
                    let wx = if x == 0 { 1.0 - f[0] } else { f[0] };
                    let wy = if y == 0 { 1.0 - f[1] } else { f[1] };
                    let wz = if z == 0 { 1.0 - f[2] } else { f[2] };
                    let weight = (wx * wy) * wz;
                    let value = values[i + dims[0] as usize * (j + dims[1] as usize * k)];
                    for channel in 0..3 {
                        out[channel] += value[channel] * weight;
                    }
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "requires a native Metal device and explicit driver access"]
    fn native_rgb_f32_uf16_and_byte_resources_refresh_with_ordered_sampling() {
        let xs = [
            f32::NEG_INFINITY,
            -0.0,
            0.25,
            0.5,
            1.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let ys = [f32::NEG_INFINITY, 0.75, f32::INFINITY];
        let indices: Vec<i32> = (0..21)
            .map(|i| [-1, 0, 1, 2, 3, 4, i32::MAX][i % 7])
            .collect();
        for dims in [[2, 2, 2], [1, 2, 1]] {
            for storage in [
                VolumeRgbStorage::F32,
                VolumeRgbStorage::Uf16E5M11,
                VolumeRgbStorage::Uf8E5M3,
            ] {
                let mut builder = GraphBuilder::new_in(Global);
                let x = builder
                    .tensor(DType::F32, TensorFlags::INPUT, &[1, 7], &[])
                    .unwrap();
                let y = builder
                    .tensor(DType::F32, TensorFlags::INPUT, &[3, 1], &[])
                    .unwrap();
                let z = builder
                    .tensor(DType::F32, TensorFlags::INPUT, &[], &[])
                    .unwrap();
                let index = builder
                    .tensor(DType::I32, TensorFlags::INPUT, &[3, 7], &[])
                    .unwrap();
                let rgb = builder
                    .volume_trilinear_rgb(
                        [x, y, z],
                        VolumeRgbParams::new(RGB_KEY, dims, storage).unwrap(),
                        [3, 7],
                    )
                    .unwrap();
                let loaded = builder
                    .buffer_load_u8(index, U8BufferParams::new(BYTES_KEY, 4).unwrap(), [3, 7])
                    .unwrap();
                pin(&mut builder, rgb.r, DType::F32, &[3, 7]);
                pin(&mut builder, rgb.g, DType::F32, &[3, 7]);
                pin(&mut builder, rgb.b, DType::F32, &[3, 7]);
                pin(&mut builder, loaded, DType::I32, &[3, 7]);
                let graph = builder.finish().unwrap();
                let mut field =
                    MetalField::compile(&graph.parse().unwrap(), 7, 3, KernelAccuracy::Approximate)
                        .unwrap();
                for edited in [false, true] {
                    let mut words = [
                        [0x0001, 0x7801, 0xf7ff],
                        [0x07ff, 0x7802, 0x7001],
                        [0x8001, 0x0800, 0x6801],
                        [0x0000, 0x8801, 0x7803],
                        [0x7000, 0x6000, 0x5800],
                        [0x7807, 0x780b, 0x780d],
                        [0xf7ff, 0x0001, 0x0000],
                        [0x7800, 0x8000, 0x8800],
                    ];
                    if edited {
                        words[0] = [0x7801, 0x7803, 0x7805];
                    }
                    let count = dims.into_iter().product::<u32>() as usize;
                    let words = &words[..count];
                    let mut decoded = vec![0.0_f32; count * 3];
                    let flat: Vec<u16> = words.iter().flatten().copied().collect();
                    axnn_cpu::uf::promote_uf16e5m11_to_f32(&flat, &mut decoded, flat.len());
                    let compact: Vec<[u8; 3]> = (0..count)
                        .map(|i| {
                            core::array::from_fn(|c| {
                                ((i * 11 + c * 13 + usize::from(edited)) % 248) as u8
                            })
                        })
                        .collect();
                    let values: Vec<[f32; 3]> = if storage == VolumeRgbStorage::Uf8E5M3 {
                        compact
                            .iter()
                            .map(|v| v.map(axnn_cpu::uf::uf8e5m3_to_f32))
                            .collect()
                    } else {
                        decoded.as_chunks::<3>().0.to_vec()
                    };
                    let bytes = if edited {
                        [255, 128, 127, 0]
                    } else {
                        [0, 127, 128, 255]
                    };
                    let volume = match storage {
                        VolumeRgbStorage::F32 => RgbVolumeRef::new_f32(&values, dims).unwrap(),
                        VolumeRgbStorage::Uf16E5M11 => RgbVolumeRef::new_uf16(words, dims).unwrap(),
                        VolumeRgbStorage::Uf8E5M3 => {
                            RgbVolumeRef::new_uf8e5m3(&compact, dims).unwrap()
                        }
                    };
                    let z = if edited { f32::INFINITY } else { 0.25 };
                    let owned = [
                        f32_bytes(&xs),
                        f32_bytes(&ys),
                        f32_bytes(&[z]),
                        i32_bytes(&indices),
                    ];
                    let inputs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
                    let resources = [
                        MetalResourceBinding::u8_buffer(
                            BYTES_KEY,
                            U8BufferRef::new(&bytes).unwrap(),
                        ),
                        MetalResourceBinding::rgb_volume(RGB_KEY, volume),
                    ];
                    field.run(&inputs, &resources).unwrap();
                    for i in 0..21 {
                        let reference = scalar_sample(&values, dims, [xs[i % 7], ys[i / 7], z]);
                        for (channel, expected) in reference.into_iter().enumerate() {
                            let actual = field.output_f32(channel).unwrap()[i];
                            if expected.is_nan() {
                                assert!(actual.is_nan());
                            } else {
                                assert_eq!(
                                    actual.to_bits(),
                                    expected.to_bits(),
                                    "pixel {i}, channel {channel}, {storage:?}, dims {dims:?}"
                                );
                            }
                        }
                        let expected = usize::try_from(indices[i])
                            .ok()
                            .and_then(|i| bytes.get(i))
                            .copied()
                            .unwrap_or(0) as i32;
                        assert_eq!(field.output_i32(3).unwrap()[i], expected);
                    }
                    assert!(field.run(&inputs, &resources[..1]).is_err());
                    assert!(field.output_f32(0).is_none());
                    field.run(&inputs, &resources).unwrap();
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a native Metal device and explicit driver access"]
    fn native_full_scene_wgsl_runs_with_synthetic_live_resources() {
        const WIDTH: u32 = 19;
        const HEIGHT: u32 = 7;
        const DIMS: [u32; 3] = [160, 96, 160];
        const LIGHT_DIMS: [u32; 3] = [24, 12, 24];
        let occupancy_key = ResourceKey {
            group: 1,
            binding: 0,
        };
        let light_key = ResourceKey {
            group: 1,
            binding: 1,
        };
        let material_key = ResourceKey {
            group: 1,
            binding: 2,
        };
        let grid_params = VoxelBrickTraceParams::new(occupancy_key, DIMS).unwrap();
        let cells = DIMS.into_iter().product::<u32>() as usize;
        let byte_params = U8BufferParams::new(material_key, cells as u32).unwrap();
        let uniforms = [
            80.0,
            25.0,
            -10.0,
            1.0,
            0.0,
            0.0,
            1.0,
            WIDTH as f32,
            1.0,
            0.0,
            0.0,
            HEIGHT as f32,
            0.0,
            1.0,
            0.0,
            1.0,
        ];
        let mut owned = vec![
            f32_bytes(&(0..WIDTH).map(|v| v as f32 + 0.5).collect::<Vec<_>>()),
            f32_bytes(&(0..HEIGHT).map(|v| v as f32 + 0.5).collect::<Vec<_>>()),
        ];
        owned.extend(uniforms.iter().map(|&v| f32_bytes(&[v])));
        let inputs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let mut reference_pixels: Vec<Vec<i32>> = Vec::new();
        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            let resources = [
                axshader::ShaderResource::VoxelBrickGrid(grid_params),
                axshader::ShaderResource::RgbVolume(
                    VolumeRgbParams::new(light_key, LIGHT_DIMS, storage).unwrap(),
                ),
                axshader::ShaderResource::U8Buffer(byte_params),
            ];
            let compiled = axshader::compile_wgsl_with_resource_specs_in(
                include_str!("../../../../games/aftershock/assets/scene.wgsl"),
                axshader::CompileOptions::for_viewport("main", WIDTH, HEIGHT),
                &resources,
                Global,
            )
            .unwrap();
            let graph = compiled.graph().parse().unwrap();
            let mut field =
                MetalField::compile(&graph, WIDTH, HEIGHT, KernelAccuracy::Approximate).unwrap();
            assert_eq!(field.input_count(), 18);
            assert_eq!(field.resource_count(), 3);
            let mut bricks = vec![0_u64; grid_params.brick_count().unwrap()];
            let mut materials = vec![0_u8; cells];
            let lights = vec![[1.0_f32; 3]; 24 * 12 * 24];
            let words = vec![[0x7800_u16; 3]; 24 * 12 * 24];
            let compact = vec![[0x78_u8; 3]; 24 * 12 * 24];
            let volume = match storage {
                VolumeRgbStorage::F32 => RgbVolumeRef::new_f32(&lights, LIGHT_DIMS).unwrap(),
                VolumeRgbStorage::Uf16E5M11 => RgbVolumeRef::new_uf16(&words, LIGHT_DIMS).unwrap(),
                VolumeRgbStorage::Uf8E5M3 => {
                    RgbVolumeRef::new_uf8e5m3(&compact, LIGHT_DIMS).unwrap()
                }
            };
            for (phase, panel) in [false, true, false].into_iter().enumerate() {
                for y in 5..60 {
                    for x in 50..110 {
                        set_cell(&mut bricks, DIMS, [x, y, 60], panel);
                        materials[(x + DIMS[0] * (y + DIMS[1] * 60)) as usize] = if panel {
                            [3, 4, 8, 10][((x / 4 + y / 3) % 4) as usize]
                        } else {
                            0
                        };
                    }
                }
                field
                    .run(
                        &inputs,
                        &[
                            MetalResourceBinding::rgb_volume(light_key, volume),
                            MetalResourceBinding::u8_buffer(
                                material_key,
                                U8BufferRef::new(&materials).unwrap(),
                            ),
                            MetalResourceBinding::brick_grid(
                                occupancy_key,
                                BrickGridRef::new(&bricks, DIMS).unwrap(),
                            ),
                        ],
                    )
                    .unwrap();
                let pixels = field.output_i32(0).unwrap();
                let depth = field.output_f32(1).unwrap();
                assert_eq!(pixels.len(), (WIDTH * HEIGHT) as usize);
                assert_eq!(depth.len(), pixels.len());
                assert!(pixels.iter().all(|&p| (0..=0x00ff_ffff).contains(&p)));
                assert!(depth.iter().all(|value| !value.is_nan() && *value >= 0.0));
                let center = (WIDTH * (HEIGHT / 2) + WIDTH / 2) as usize;
                if panel {
                    assert_eq!(depth[center].to_bits(), 70.0_f32.to_bits());
                } else {
                    assert_eq!(depth[center], f32::INFINITY);
                }
                if storage == VolumeRgbStorage::F32 {
                    reference_pixels.push(pixels.to_vec());
                } else {
                    assert_eq!(pixels, reference_pixels[phase]);
                }
            }
        }
        assert_eq!(reference_pixels[0], reference_pixels[2]);
        assert_ne!(reference_pixels[0], reference_pixels[1]);
    }
}
