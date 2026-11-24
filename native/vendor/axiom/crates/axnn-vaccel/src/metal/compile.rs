//! Pure, conservative `.ax` field-to-MSL compilation.
//!
//! The runtime prepends `kernels.metal` to `source`, disables fast math, and
//! dispatches a two-dimensional grid. This compiler neither opens a device nor
//! captures resource contents. Every resource read remains an emitted call.

use std::fmt::Write as _;

use ax::sampling::VolumeRgbStorage;
use ax::voxel::ResourceKey;
use ax::{DType, Graph, NodeFlags, TensorFlags, TensorId};
use axnn::lowering::{
    IntegerBinaryOp, KernelAccuracy, KernelEffects, KernelParams, KernelRequest, LogicalOp,
    SemanticOp,
};
use axnn::{BinaryOp, CompareOp, TernaryOp, UnaryOp};

// Bound compiler work and staging before any large allocation. These are
// explicit prototype limits, not limits of the .ax container or Metal itself.
const MAX_NODES: usize = 16_384;
const MAX_TENSORS: usize = 65_536;
const MAX_BUFFER_BYTES: usize = 512 * 1024 * 1024;
// Scalar-only graphs need an independent work limit: their tiny buffers do
// not bound the number of GPU invocations. This admits at most 134,217,728
// lanes per dispatch, independently of the shape of any pinned output.
const MAX_DISPATCH_LANES: usize = MAX_BUFFER_BYTES / size_of::<f32>();

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetalTensorLayout {
    pub tensor: TensorId,
    pub dtype: DType,
    pub shape: Vec<u32>,
    pub offset: usize,
    pub byte_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetalResourceKind {
    VoxelBrickGrid {
        dims: [u32; 3],
    },
    RgbVolume {
        dims: [u32; 3],
        storage: VolumeRgbStorage,
    },
    U8Buffer {
        elements: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MetalResourceRequirement {
    pub key: ResourceKey,
    pub kind: MetalResourceKind,
    pub offset: usize,
    pub byte_len: usize,
}

#[derive(Debug)]
pub(crate) struct MetalFieldPlan {
    /// Kernel source only; the runtime prepends the resource helper source.
    pub source: String,
    pub width: u32,
    pub height: u32,
    /// Caller INPUT tensors, including unused declarations, in tensor-ID order.
    pub inputs: Vec<MetalTensorLayout>,
    /// Pinned OUTPUT tensors in tensor-ID order; shapes are not expanded.
    pub outputs: Vec<MetalTensorLayout>,
    /// Zeroed caller inputs plus immutable constant bytes at their packed offsets.
    pub initial_inputs: Vec<u8>,
    pub output_bytes: usize,
    pub resource_bytes: usize,
    pub resources: Vec<MetalResourceRequirement>,
    pub node_count: usize,
    pub accuracy: KernelAccuracy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldShape {
    Scalar,
    Row,
    Column,
    Field,
    /// A register-only local, never an ingress/egress float3 ABI.
    Packed3,
}

impl FieldShape {
    fn classify(shape: &[u32], width: u32, height: u32) -> Option<Self> {
        match shape {
            [] | [1] | [1, 1] => Some(Self::Scalar),
            [x] if *x == width => Some(Self::Row),
            [y, x] if *y == height && *x == width => Some(Self::Field),
            [1, x] if *x == width => Some(Self::Row),
            [y, 1] if *y == height => Some(Self::Column),
            [y, x, 3] if *y == height && *x == width => Some(Self::Packed3),
            _ => None,
        }
    }

    fn index(self) -> &'static str {
        match self {
            Self::Scalar => "0ul",
            Self::Row => "ulong(gid.x)",
            Self::Column => "ulong(gid.y)",
            Self::Field => "lane",
            Self::Packed3 => unreachable!("packed values have no buffer layout"),
        }
    }

    fn writer_guard(self) -> &'static str {
        match self {
            Self::Scalar => "gid.x == 0u && gid.y == 0u",
            Self::Row => "gid.y == 0u",
            Self::Column => "gid.x == 0u",
            Self::Field => "true",
            Self::Packed3 => unreachable!("packed values cannot be graph outputs"),
        }
    }
}

struct TensorInfo {
    dtype: DType,
    flags: TensorFlags,
    shape: Vec<u32>,
    field: FieldShape,
    byte_len: usize,
    input_offset: Option<usize>,
}

impl TensorInfo {
    fn layout(&self, tensor: TensorId, offset: usize) -> MetalTensorLayout {
        MetalTensorLayout {
            tensor,
            dtype: self.dtype,
            shape: self.shape.clone(),
            offset,
            byte_len: self.byte_len,
        }
    }

    fn msl_type(&self) -> &'static str {
        match (self.dtype, self.field) {
            (DType::F32, FieldShape::Packed3) => "float3",
            (DType::F32, _) => "float",
            (DType::I32, _) => "int",
            (DType::U8, _) => "uchar",
            _ => unreachable!("dtypes are admitted before code generation"),
        }
    }
}

/// Compile a single field with scalar, row, and column broadcasts.
///
/// Only F32, I32, and U8 mask storage is admitted. Packed xyz values must be
/// local results of admitted nodes. Metal's extreme-value resource behavior
/// has not established a faithful/strict contract, so this initial compiler
/// advertises Approximate and rejects stronger minimum-accuracy requests.
pub(crate) fn compile(
    graph: &Graph<'_>,
    width: u32,
    height: u32,
    minimum_accuracy: KernelAccuracy,
) -> Result<MetalFieldPlan, String> {
    if minimum_accuracy > KernelAccuracy::Approximate {
        return Err(format!(
            "Metal field accuracy is Approximate; requested {minimum_accuracy:?}"
        ));
    }
    if width == 0 || height == 0 || width == ax::DYNAMIC_DIM || height == ax::DYNAMIC_DIM {
        return Err("Metal field dimensions must be concrete and nonzero".into());
    }
    if (width as usize)
        .checked_mul(height as usize)
        .is_none_or(|lanes| lanes > MAX_DISPATCH_LANES)
    {
        return Err("Metal field exceeds the prototype dispatch-lane work budget".into());
    }
    if graph.node_count() > MAX_NODES || graph.tensor_count() > MAX_TENSORS {
        return Err("Metal field graph exceeds the compiler node/tensor budget".into());
    }

    let mut infos = Vec::with_capacity(graph.tensor_count());
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut input_bytes = 0;
    let mut output_bytes = 0;
    for index in 0..graph.tensor_count() {
        let id = TensorId(index as u32);
        let tensor = graph
            .tensor(id)
            .map_err(|e| format!("Metal tensor {index}: {e:?}"))?;
        if !matches!(tensor.dtype, DType::F32 | DType::I32 | DType::U8) {
            return Err(format!(
                "Metal tensor {index}: unsupported dtype {:?}",
                tensor.dtype
            ));
        }
        let input = tensor.flags.contains(TensorFlags::INPUT);
        let constant = tensor.flags.contains(TensorFlags::CONSTANT);
        if input && constant {
            return Err(format!(
                "Metal tensor {index}: INPUT and CONSTANT cannot be combined"
            ));
        }
        let shape: Vec<u32> = tensor.dimensions().collect();
        if shape
            .iter()
            .any(|&extent| extent == 0 || extent == ax::DYNAMIC_DIM)
        {
            return Err(format!(
                "Metal tensor {index}: zero/dynamic dimensions are unsupported"
            ));
        }
        let field = FieldShape::classify(&shape, width, height).ok_or_else(|| {
            format!(
                "Metal tensor {index}: unsupported field shape {shape:?} for [{height}, {width}]"
            )
        })?;
        if field == FieldShape::Packed3
            && (tensor.dtype != DType::F32 || tensor.flags != TensorFlags::default())
        {
            return Err(format!(
                "Metal tensor {index}: packed xyz is a local F32 value only"
            ));
        }
        let elements = shape
            .iter()
            .try_fold(1_usize, |count, &extent| count.checked_mul(extent as usize))
            .ok_or_else(|| format!("Metal tensor {index}: element-count overflow"))?;
        let byte_len = elements
            .checked_mul(tensor.dtype.block_byte_size())
            .filter(|&bytes| bytes <= MAX_BUFFER_BYTES)
            .ok_or_else(|| {
                format!("Metal tensor {index}: storage exceeds the compiler buffer budget")
            })?;
        if constant && tensor.constant().len() != byte_len {
            return Err(format!(
                "Metal tensor {index}: constant byte length does not match its descriptor"
            ));
        }
        let input_offset = if input || constant {
            Some(pack(&mut input_bytes, byte_len, 4, "input")?)
        } else {
            None
        };
        let info = TensorInfo {
            dtype: tensor.dtype,
            flags: tensor.flags,
            shape,
            field,
            byte_len,
            input_offset,
        };
        if input {
            inputs.push(info.layout(id, input_offset.expect("INPUT has ingress storage")));
        }
        if tensor.flags.contains(TensorFlags::OUTPUT) {
            let offset = pack(&mut output_bytes, byte_len, 4, "output")?;
            outputs.push(info.layout(id, offset));
        }
        infos.push(info);
    }
    if outputs.is_empty() {
        return Err("Metal field requires at least one graph OUTPUT".into());
    }

    let mut available: Vec<bool> = infos
        .iter()
        .map(|info| info.input_offset.is_some())
        .collect();
    let mut resources = Vec::new();
    let mut resource_bytes = 0;
    // Validate the entire declared graph, including dead nodes. No scheduling,
    // DCE, or physical-slot reuse is allowed to conceal invalid SSA/effects.
    for index in 0..graph.node_count() {
        let node = graph
            .node(index)
            .map_err(|e| format!("Metal node {index}: {e:?}"))?;
        let request = checked_request(graph, node)?;
        for id in node.inputs() {
            if !available[id.0 as usize] {
                return Err(format!(
                    "Metal node {index}: tensor {} used before definition",
                    id.0
                ));
            }
        }
        for id in node.outputs() {
            if available[id.0 as usize] {
                return Err(format!(
                    "Metal node {index}: tensor {} is multiply defined or replaces an input/constant",
                    id.0
                ));
            }
            available[id.0 as usize] = true;
        }
        if let Some((key, kind, bytes)) = resource_for(&request)? {
            if node
                .outputs()
                .any(|id| infos[id.0 as usize].shape != [height, width])
            {
                return Err(format!(
                    "Metal node {index}: resource outputs must be the full field"
                ));
            }
            if let Some(existing) = resources
                .iter()
                .find(|r: &&MetalResourceRequirement| r.key == key)
            {
                if existing.kind != kind || existing.byte_len != bytes {
                    return Err(format!(
                        "Metal node {index}: conflicting resource metadata at group {} binding {}",
                        key.group, key.binding
                    ));
                }
            } else {
                let offset = pack(&mut resource_bytes, bytes, 8, "resource")?;
                resources.push(MetalResourceRequirement {
                    key,
                    kind,
                    offset,
                    byte_len: bytes,
                });
            }
        }
    }
    if let Some(index) = available.iter().position(|&defined| !defined) {
        return Err(format!(
            "Metal tensor {index}: no input, constant, or producer definition"
        ));
    }

    let mut source = String::from(MATH_HELPERS);
    writeln!(source, "\nkernel void ax_field_main(device const uchar* inputs [[buffer(0)]], device uchar* outputs [[buffer(1)]], device const uchar* resources [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {{").unwrap();
    writeln!(
        source,
        "    if (gid.x >= {width}u || gid.y >= {height}u) return;"
    )
    .unwrap();
    writeln!(
        source,
        "    const ulong lane = ulong(gid.y) * {width}ul + ulong(gid.x);"
    )
    .unwrap();
    for (index, info) in infos.iter().enumerate() {
        if let Some(offset) = info.input_offset {
            let ty = info.msl_type();
            let scalar_bytes = info.dtype.block_byte_size();
            writeln!(source, "    const {ty} t{index} = *reinterpret_cast<device const {ty}*>(inputs + {offset}ul + {scalar_bytes}ul * {});", info.field.index()).unwrap();
        }
    }
    for index in 0..graph.node_count() {
        let node = graph
            .node(index)
            .map_err(|e| format!("Metal node {index}: {e:?}"))?;
        let request = checked_request(graph, node)?;
        emit_node(&mut source, node, &request, &infos, &resources)?;
    }
    for output in &outputs {
        let info = &infos[output.tensor.0 as usize];
        let ty = info.msl_type();
        let scalar_bytes = info.dtype.block_byte_size();
        writeln!(source, "    if ({}) *reinterpret_cast<device {ty}*>(outputs + {}ul + {scalar_bytes}ul * {}) = t{};", info.field.writer_guard(), output.offset, info.field.index(), output.tensor.0).unwrap();
    }
    source.push_str("}\n");

    let mut initial_inputs = Vec::new();
    initial_inputs
        .try_reserve_exact(input_bytes)
        .map_err(|_| "Metal input staging allocation failed")?;
    initial_inputs.resize(input_bytes, 0);
    for (index, info) in infos.iter().enumerate() {
        if info.flags.contains(TensorFlags::CONSTANT) {
            let offset = info.input_offset.expect("CONSTANT has ingress storage");
            let tensor = graph
                .tensor(TensorId(index as u32))
                .map_err(|e| format!("Metal constant {index}: {e:?}"))?;
            initial_inputs[offset..offset + info.byte_len].copy_from_slice(tensor.constant());
        }
    }
    Ok(MetalFieldPlan {
        source,
        width,
        height,
        inputs,
        outputs,
        initial_inputs,
        output_bytes,
        resource_bytes,
        resources,
        node_count: graph.node_count(),
        accuracy: KernelAccuracy::Approximate,
    })
}

fn pack(cursor: &mut usize, bytes: usize, alignment: usize, label: &str) -> Result<usize, String> {
    let offset = cursor
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| format!("Metal {label} alignment overflow"))?;
    *cursor = offset
        .checked_add(bytes)
        .filter(|&end| end <= MAX_BUFFER_BYTES)
        .ok_or_else(|| format!("Metal {label} storage exceeds the compiler buffer budget"))?;
    Ok(offset)
}

fn checked_request(graph: &Graph<'_>, node: ax::NodeView<'_>) -> Result<KernelRequest, String> {
    let request = ax::node_kernel_request(graph, node).ok_or_else(|| {
        format!(
            "Metal node {}: unsupported or noncanonical kernel request {:?}",
            node.index, node.op
        )
    })?;
    let resource = matches!(
        request.semantic,
        SemanticOp::VoxelBrickTrace | SemanticOp::VolumeTrilinearRgb | SemanticOp::BufferLoadU8
    );
    let expected_flags = if resource {
        NodeFlags::READS_RESOURCE
    } else {
        NodeFlags::default()
    };
    let expected_effects = if resource {
        KernelEffects::READS_RESOURCE
    } else {
        KernelEffects::PURE
    };
    if node.flags != expected_flags || request.effects() != expected_effects {
        return Err(format!(
            "Metal node {}: ordered, mutating, or unexpected effects are unsupported",
            node.index
        ));
    }
    Ok(request)
}

fn resource_for(
    request: &KernelRequest,
) -> Result<Option<(ResourceKey, MetalResourceKind, usize)>, String> {
    match request.params() {
        KernelParams::VoxelBrickTrace(params) => {
            let bytes = params
                .brick_count()
                .and_then(|count| count.checked_mul(8))
                .ok_or("Metal voxel resource length overflow")?;
            Ok(Some((
                params.resource,
                MetalResourceKind::VoxelBrickGrid { dims: params.dims },
                bytes,
            )))
        }
        KernelParams::VolumeTrilinearRgb(params) => {
            let bytes = params
                .storage_bytes()
                .ok_or("Metal RGB resource length overflow")?;
            Ok(Some((
                params.resource,
                MetalResourceKind::RgbVolume {
                    dims: params.dims,
                    storage: params.storage,
                },
                bytes,
            )))
        }
        KernelParams::BufferLoadU8(params) => Ok(Some((
            params.resource,
            MetalResourceKind::U8Buffer {
                elements: params.elements,
            },
            params.elements as usize,
        ))),
        KernelParams::None => Ok(None),
        _ => Err("Metal field does not admit these kernel parameters".into()),
    }
}

fn emit_node(
    source: &mut String,
    node: ax::NodeView<'_>,
    request: &KernelRequest,
    infos: &[TensorInfo],
    resources: &[MetalResourceRequirement],
) -> Result<(), String> {
    let inputs: Vec<_> = node.inputs().collect();
    let outputs: Vec<_> = node.outputs().collect();
    let input_names: Vec<_> = inputs.iter().map(|id| format!("t{}", id.0)).collect();
    let bad = || {
        format!(
            "Metal node {}: unsupported {:?} dtype/shape combination",
            node.index, request.semantic
        )
    };
    let packed = inputs
        .iter()
        .chain(&outputs)
        .any(|id| infos[id.0 as usize].field == FieldShape::Packed3);
    if packed
        && !matches!(
            request.semantic,
            SemanticOp::Identity | SemanticOp::Interleave3F32 | SemanticOp::VoxelBrickTrace
        )
    {
        return Err(bad());
    }
    if let Some((key, _, _)) = resource_for(request)? {
        let resource = resources
            .iter()
            .find(|resource| resource.key == key)
            .ok_or_else(bad)?;
        match resource.kind {
            MetalResourceKind::VoxelBrickGrid { dims } => {
                writeln!(source, "    const AxMetalVoxelHit q{} = ax_voxel_brick_trace(resources + {}ul, uint3({}u, {}u, {}u), {}, {}, {});", node.index, resource.offset, dims[0], dims[1], dims[2], input_names[0], input_names[1], input_names[2]).unwrap();
                writeln!(
                    source,
                    "    const float t{} = q{}.t;\n    const int t{} = q{}.cell_id;",
                    outputs[0].0, node.index, outputs[1].0, node.index
                )
                .unwrap();
            }
            MetalResourceKind::RgbVolume { dims, storage } => {
                let suffix = match storage {
                    VolumeRgbStorage::F32 => "f32",
                    VolumeRgbStorage::Uf16E5M11 => "uf16",
                    VolumeRgbStorage::Uf8E5M3 => "uf8e5m3",
                };
                writeln!(source, "    const float3 q{} = ax_volume_trilinear_rgb_{suffix}(resources + {}ul, uint3({}u, {}u, {}u), float3({}, {}, {}));", node.index, resource.offset, dims[0], dims[1], dims[2], input_names[0], input_names[1], input_names[2]).unwrap();
                for (id, component) in outputs.iter().zip(["x", "y", "z"]) {
                    writeln!(
                        source,
                        "    const float t{} = q{}.{component};",
                        id.0, node.index
                    )
                    .unwrap();
                }
            }
            MetalResourceKind::U8Buffer { elements } => {
                writeln!(
                    source,
                    "    const int t{} = ax_buffer_load_u8(resources + {}ul, {elements}u, {});",
                    outputs[0].0, resource.offset, input_names[0]
                )
                .unwrap();
            }
        }
        return Ok(());
    }
    let output = outputs[0];
    let info = &infos[output.0 as usize];
    let expression = match request.semantic {
        SemanticOp::Identity => input_names[0].clone(),
        SemanticOp::Cast => match (request.primary_input_dtype(), info.dtype) {
            (DType::F32, DType::I32) => format!("ax_field_cast_i32({})", input_names[0]),
            (DType::I32, DType::F32) => format!("float({})", input_names[0]),
            _ => return Err(bad()),
        },
        SemanticOp::Interleave3F32 => format!(
            "float3({}, {}, {})",
            input_names[0], input_names[1], input_names[2]
        ),
        SemanticOp::Unary(op) if info.dtype == DType::F32 => {
            unary_expression(op, &input_names[0]).ok_or_else(bad)?
        }
        SemanticOp::Binary(op) if info.dtype == DType::F32 => {
            binary_expression(op, &input_names[0], &input_names[1])
        }
        SemanticOp::Binary(op) if info.dtype == DType::I32 => {
            integer_arithmetic(op, &input_names[0], &input_names[1]).ok_or_else(bad)?
        }
        SemanticOp::IntegerBinary(op) => integer_expression(op, &input_names[0], &input_names[1]),
        SemanticOp::Ternary(op) if info.dtype == DType::F32 => match op {
            TernaryOp::Clamp => format!(
                "ax_field_min(ax_field_max({}, {}), {})",
                input_names[0], input_names[1], input_names[2]
            ),
            TernaryOp::Fma => format!(
                "({} * {}) + {}",
                input_names[0], input_names[1], input_names[2]
            ),
        },
        SemanticOp::Compare(op)
            if matches!(request.primary_input_dtype(), DType::F32 | DType::I32) =>
        {
            let operator = match op {
                CompareOp::Equal => "==",
                CompareOp::Greater => ">",
                CompareOp::GreaterEqual => ">=",
            };
            format!("uchar({} {operator} {})", input_names[0], input_names[1])
        }
        SemanticOp::Logical(op) => match op {
            LogicalOp::Not => format!("uchar({} == 0u)", input_names[0]),
            LogicalOp::And => format!(
                "uchar(({} != 0u) && ({} != 0u))",
                input_names[0], input_names[1]
            ),
            LogicalOp::Or => format!(
                "uchar(({} != 0u) || ({} != 0u))",
                input_names[0], input_names[1]
            ),
            LogicalOp::Xor => format!(
                "uchar(({} != 0u) != ({} != 0u))",
                input_names[0], input_names[1]
            ),
        },
        SemanticOp::Select => format!(
            "({} != 0u) ? {} : {}",
            input_names[0], input_names[1], input_names[2]
        ),
        _ => return Err(bad()),
    };
    writeln!(
        source,
        "    const {} t{} = {expression};",
        info.msl_type(),
        output.0
    )
    .unwrap();
    Ok(())
}

fn unary_expression(op: UnaryOp, x: &str) -> Option<String> {
    let function = match op {
        UnaryOp::Abs => "fabs",
        UnaryOp::Acos => "acos",
        UnaryOp::Asin => "asin",
        UnaryOp::Atan => "atan",
        UnaryOp::Ceil => "ceil",
        UnaryOp::Cos => "cos",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Exp => "exp",
        UnaryOp::Exp2 => "exp2",
        UnaryOp::Floor => "floor",
        UnaryOp::Log => "log",
        UnaryOp::Log2 => "log2",
        UnaryOp::Round => "ax_field_round",
        UnaryOp::Sign => "ax_field_sign",
        UnaryOp::Sin => "sin",
        UnaryOp::Tan => "tan",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Trunc => "trunc",
        UnaryOp::Relu => return Some(format!("ax_field_max({x}, 0.0f)")),
        UnaryOp::Neg => return Some(format!("-{x}")),
        UnaryOp::Reciprocal => return Some(format!("1.0f / {x}")),
        UnaryOp::InverseSqrt => return Some(format!("1.0f / sqrt({x})")),
        UnaryOp::Sigmoid => return Some(format!("1.0f / (1.0f + exp(-{x}))")),
        UnaryOp::Fract | UnaryOp::Silu | UnaryOp::Gelu | UnaryOp::Softplus => return None,
    };
    Some(format!("{function}({x})"))
}

fn binary_expression(op: BinaryOp, left: &str, right: &str) -> String {
    let operator = match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Min => return format!("ax_field_min({left}, {right})"),
        BinaryOp::Max => return format!("ax_field_max({left}, {right})"),
        BinaryOp::Atan2 => return format!("atan2({left}, {right})"),
        BinaryOp::Pow => return format!("ax_field_pow({left}, {right})"),
        BinaryOp::Rem => return format!("ax_field_remainder({left}, {right})"),
    };
    format!("{left} {operator} {right}")
}

fn integer_arithmetic(op: BinaryOp, left: &str, right: &str) -> Option<String> {
    let operator = match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => return Some(format!("ax_field_i32_div({left}, {right})")),
        BinaryOp::Rem => return Some(format!("ax_field_i32_rem({left}, {right})")),
        _ => return None,
    };
    // Signed C++ overflow is undefined; the canonical I32 arithmetic wraps.
    Some(format!(
        "as_type<int>(as_type<uint>({left}) {operator} as_type<uint>({right}))"
    ))
}

fn integer_expression(op: IntegerBinaryOp, left: &str, right: &str) -> String {
    let operator = match op {
        IntegerBinaryOp::And => "&",
        IntegerBinaryOp::Or => "|",
        IntegerBinaryOp::Xor => "^",
        IntegerBinaryOp::ShiftLeft => {
            return format!(
                "as_type<int>(as_type<uint>({left}) << (as_type<uint>({right}) & 31u))"
            );
        }
        IntegerBinaryOp::ShiftRight => return format!("ax_field_i32_shr({left}, {right})"),
    };
    format!("as_type<int>(as_type<uint>({left}) {operator} as_type<uint>({right}))")
}

// These small semantic helpers preserve explicit graph operations. In
// particular MSL fmin/fmax suppress NaNs, sign suppresses a NaN, and round has
// a different halfway rule; those defaults are not the canonical operations.
const MATH_HELPERS: &str = r#"#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline int ax_field_cast_i32(float x) {
    if (isnan(x)) return 0;
    if (x >= 2147483648.0f) return as_type<int>(0x7fffffffu);
    if (x <= -2147483648.0f) return as_type<int>(0x80000000u);
    return int(x);
}
inline int ax_field_i32_div(int a, int b) {
    if (b == 0 || (a == as_type<int>(0x80000000u) && b == -1)) return a;
    return a / b;
}
inline int ax_field_i32_rem(int a, int b) {
    if (b == 0 || (a == as_type<int>(0x80000000u) && b == -1)) return 0;
    return a % b;
}
inline int ax_field_i32_shr(int a, int b) {
    const uint shift = as_type<uint>(b) & 31u;
    if (shift == 0u) return a;
    uint bits = as_type<uint>(a) >> shift;
    if (a < 0) bits = bits | (0xffffffffu << (32u - shift));
    return as_type<int>(bits);
}
inline float ax_field_min(float a, float b) {
    return isnan(a) ? a : (isnan(b) ? b : fmin(a, b));
}
inline float ax_field_max(float a, float b) {
    return isnan(a) ? a : (isnan(b) ? b : fmax(a, b));
}
inline float ax_field_sign(float x) {
    return isnan(x) ? x : (x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f));
}
inline float ax_field_round(float x) {
    if (!isfinite(x) || x == 0.0f || fabs(x) >= 8388608.0f) return x;
    const float lo = floor(x);
    const float fraction = x - lo;
    const float rounded = fraction < 0.5f ? lo :
        ((fraction > 0.5f || (int(lo) & 1) != 0) ? lo + 1.0f : lo);
    return rounded == 0.0f ? copysign(0.0f, x) : rounded;
}
inline float ax_field_remainder(float a, float b) {
    const float quotient = trunc(a / b);
    const float product = b * quotient;
    const float value = a - product;
    return value == 0.0f ? copysign(0.0f, a) : value;
}
inline float ax_field_pow(float base, float exponent) {
    if (isfinite(exponent) && fabs(exponent) <= 64.0f && trunc(exponent) == exponent) {
        const int integer = int(exponent);
        const uint count = uint(integer < 0 ? -integer : integer);
        float result = 1.0f;
        for (uint i = 0u; i < count; ++i) result = result * base;
        return integer < 0 ? 1.0f / result : result;
    }
    return exp(exponent * log(base));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use ax::buffer::U8BufferParams;
    use ax::sampling::VolumeRgbParams;
    use ax::voxel::VoxelBrickTraceParams;
    use ax::{GraphBuilder, OpCode};
    use std::alloc::Global;

    fn tensor(
        builder: &mut GraphBuilder<Global>,
        dtype: DType,
        flags: TensorFlags,
        shape: &[u32],
        bytes: &[u8],
    ) -> TensorId {
        builder.tensor(dtype, flags, shape, bytes).unwrap()
    }

    fn pin(
        builder: &mut GraphBuilder<Global>,
        dtype: DType,
        shape: &[u32],
        value: TensorId,
    ) -> TensorId {
        let output = tensor(builder, dtype, TensorFlags::OUTPUT, shape, &[]);
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[value],
                &[output],
                &[],
            )
            .unwrap();
        output
    }

    fn build(
        builder: GraphBuilder<Global>,
        width: u32,
        height: u32,
    ) -> Result<MetalFieldPlan, String> {
        let bytes = builder.finish().unwrap();
        compile(
            &bytes.parse().unwrap(),
            width,
            height,
            KernelAccuracy::Approximate,
        )
    }

    fn single_node(op: OpCode, inputs: &[DType], output_dtype: DType) -> MetalFieldPlan {
        let mut builder = GraphBuilder::new_in(Global);
        let values: Vec<_> = inputs
            .iter()
            .map(|&dtype| tensor(&mut builder, dtype, TensorFlags::INPUT, &[3, 7], &[]))
            .collect();
        let output = tensor(
            &mut builder,
            output_dtype,
            TensorFlags::OUTPUT,
            &[3, 7],
            &[],
        );
        builder
            .node(op, NodeFlags::default(), &values, &[output], &[])
            .unwrap();
        build(builder, 7, 3).unwrap()
    }

    #[test]
    fn actual_aftershock_sources_compile_with_typed_resource_layouts() {
        let rays = axshader::compile_wgsl_with_resources_in(
            include_str!("../../../../games/aftershock/assets/rays.wgsl"),
            axshader::CompileOptions::for_viewport("main", 17, 9),
            &[VoxelBrickTraceParams::new(
                ResourceKey {
                    group: 1,
                    binding: 0,
                },
                [320, 192, 320],
            )
            .unwrap()],
            Global,
        )
        .unwrap();
        let graph = rays.graph().parse().unwrap();
        let plan = compile(&graph, 17, 9, KernelAccuracy::Approximate).unwrap();
        assert_eq!(
            (plan.width, plan.height, plan.node_count),
            (17, 9, graph.node_count())
        );
        assert_eq!(plan.outputs.len(), rays.interface().output_count());
        assert_eq!(plan.outputs[1].dtype, DType::I32);
        assert_eq!(plan.resources[0].byte_len, 2_457_600);
        assert!(
            plan.source
                .contains("ax_voxel_brick_trace(resources + 0ul, uint3(320u, 192u, 320u)")
        );
        assert!(plan.source.contains(".cell_id;"));
        assert!(
            !plan
                .source
                .contains("reinterpret_cast<device const float3*>")
        );
        assert!(!plan.source.contains("reinterpret_cast<device float3*>"));
        assert_eq!(plan.accuracy, KernelAccuracy::Approximate);

        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            let params = VolumeRgbParams::new(
                ResourceKey {
                    group: 1,
                    binding: 1,
                },
                [7, 3, 5],
                storage,
            )
            .unwrap();
            let compiled = axshader::compile_wgsl_with_resource_specs_in(
                include_str!("../../../../games/aftershock/assets/shading.wgsl"),
                axshader::CompileOptions::for_viewport("main", 17, 9),
                &[axshader::ShaderResource::RgbVolume(params)],
                Global,
            )
            .unwrap();
            let graph = compiled.graph().parse().unwrap();
            let plan = compile(&graph, 17, 9, KernelAccuracy::Approximate).unwrap();
            assert_eq!(plan.outputs.len(), compiled.interface().output_count());
            assert!(
                plan.outputs
                    .iter()
                    .all(|output| output.dtype == DType::F32 && output.shape == [9, 17])
            );
            assert_eq!(plan.resource_bytes, params.storage_bytes().unwrap());
            assert_eq!(
                plan.resources[0].kind,
                MetalResourceKind::RgbVolume {
                    dims: params.dims,
                    storage
                }
            );
            let suffix = match storage {
                VolumeRgbStorage::F32 => "f32",
                VolumeRgbStorage::Uf16E5M11 => "uf16",
                VolumeRgbStorage::Uf8E5M3 => "uf8e5m3",
            };
            assert!(
                plan.source
                    .contains(&format!("ax_volume_trilinear_rgb_{suffix}(resources + 0ul"))
            );
            assert!(plan.source.contains("#pragma clang fp contract(off)"));
        }
    }

    #[test]
    fn actual_aftershock_full_scene_compiles_to_one_typed_field() {
        let dimensions = [320, 192, 320];
        let occupancy = ResourceKey {
            group: 1,
            binding: 0,
        };
        let light = ResourceKey {
            group: 1,
            binding: 1,
        };
        let materials = ResourceKey {
            group: 1,
            binding: 2,
        };
        for storage in [VolumeRgbStorage::F32, VolumeRgbStorage::Uf16E5M11] {
            let volume = VolumeRgbParams::new(light, [24, 12, 24], storage).unwrap();
            let compiled = axshader::compile_wgsl_with_resource_specs_in(
                include_str!("../../../../games/aftershock/assets/scene.wgsl"),
                axshader::CompileOptions::for_viewport("main", 17, 9),
                &[
                    axshader::ShaderResource::VoxelBrickGrid(
                        VoxelBrickTraceParams::new(occupancy, dimensions).unwrap(),
                    ),
                    axshader::ShaderResource::RgbVolume(volume),
                    axshader::ShaderResource::U8Buffer(
                        U8BufferParams::new(materials, 19_660_800).unwrap(),
                    ),
                ],
                Global,
            )
            .unwrap();
            let graph = compiled.graph().parse().unwrap();
            let plan = compile(&graph, 17, 9, KernelAccuracy::Approximate).unwrap();
            assert_eq!(plan.inputs.len(), 18);
            assert_eq!(
                plan.outputs
                    .iter()
                    .map(|output| output.dtype)
                    .collect::<Vec<_>>(),
                [DType::I32, DType::F32]
            );
            assert!(plan.outputs.iter().all(|output| output.shape == [9, 17]));
            assert_eq!(plan.node_count, graph.node_count());
            assert!(plan.node_count > 100);
            assert_eq!(plan.source.matches("kernel void ax_field_main(").count(), 1);
            assert_eq!(
                plan.source
                    .matches("ax_voxel_brick_trace(resources +")
                    .count(),
                2
            );
            assert_eq!(
                plan.source.matches("ax_buffer_load_u8(resources +").count(),
                7
            );
            assert_eq!(plan.source.matches("= ax_volume_trilinear_rgb_").count(), 1);
            assert_eq!(plan.resources.len(), 3);
            assert_eq!(
                plan.resources
                    .iter()
                    .find(|resource| resource.key == materials)
                    .unwrap()
                    .byte_len,
                19_660_800
            );
            assert_eq!(
                plan.resources
                    .iter()
                    .find(|resource| resource.key == light)
                    .unwrap()
                    .byte_len,
                volume.storage_bytes().unwrap()
            );
        }
    }

    #[test]
    fn scalar_row_column_and_field_outputs_have_unique_writers() {
        let mut builder = GraphBuilder::new_in(Global);
        let io = TensorFlags::INPUT | TensorFlags::OUTPUT;
        tensor(&mut builder, DType::F32, io, &[], &[]);
        tensor(&mut builder, DType::I32, io, &[1, 7], &[]);
        tensor(&mut builder, DType::U8, io, &[3, 1], &[]);
        tensor(&mut builder, DType::F32, io, &[3, 7], &[]);
        // Even an unused input remains a caller-visible ABI slot.
        let unused = tensor(&mut builder, DType::I32, TensorFlags::INPUT, &[1], &[]);
        let bits = 0x7fc0_00a5_u32.to_le_bytes();
        let constant = tensor(
            &mut builder,
            DType::F32,
            TensorFlags::CONSTANT | TensorFlags::OUTPUT,
            &[],
            &bits,
        );
        let plan = build(builder, 7, 3).unwrap();
        assert_eq!(plan.inputs.len(), 5);
        assert_eq!(plan.inputs[4].tensor, unused);
        assert_eq!(plan.outputs.last().unwrap().tensor, constant);
        assert_eq!(plan.initial_inputs.len(), 128);
        assert_eq!(&plan.initial_inputs[124..], &bits);
        assert!(plan.initial_inputs[..124].iter().all(|&byte| byte == 0));
        assert_eq!(
            plan.outputs
                .iter()
                .map(|layout| layout.offset)
                .collect::<Vec<_>>(),
            [0, 4, 32, 36, 120]
        );
        assert!(
            plan.source
                .contains("if (gid.x == 0u && gid.y == 0u) *reinterpret_cast<device float*>")
        );
        assert!(
            plan.source
                .contains("if (gid.y == 0u) *reinterpret_cast<device int*>")
        );
        assert!(
            plan.source
                .contains("if (gid.x == 0u) *reinterpret_cast<device uchar*>")
        );
        assert!(
            plan.source
                .contains("if (true) *reinterpret_cast<device float*>")
        );
        assert!(plan.source.contains("4ul * ulong(gid.x)"));
        assert!(plan.source.contains("1ul * ulong(gid.y)"));
        assert!(
            plan.source
                .contains("if (gid.x >= 7u || gid.y >= 3u) return;")
        );
    }

    #[test]
    fn canonical_float_operator_corpus_emits_each_operation() {
        let unary = [
            (OpCode::IDENTITY, "t0"),
            (OpCode::COPY, "t0"),
            (OpCode::ABS, "fabs(t0)"),
            (OpCode::ACOS, "acos(t0)"),
            (OpCode::ASIN, "asin(t0)"),
            (OpCode::ATAN, "atan(t0)"),
            (OpCode::CEIL, "ceil(t0)"),
            (OpCode::COS, "cos(t0)"),
            (OpCode::SQRT, "sqrt(t0)"),
            (OpCode::EXP, "exp(t0)"),
            (OpCode::EXP2, "exp2(t0)"),
            (OpCode::FLOOR, "floor(t0)"),
            (OpCode::LOG, "log(t0)"),
            (OpCode::LOG2, "log2(t0)"),
            (OpCode::ROUND, "ax_field_round(t0)"),
            (OpCode::SIGN, "ax_field_sign(t0)"),
            (OpCode::SIN, "sin(t0)"),
            (OpCode::TAN, "tan(t0)"),
            (OpCode::TANH, "tanh(t0)"),
            (OpCode::TRUNC, "trunc(t0)"),
            (OpCode::RELU, "ax_field_max(t0, 0.0f)"),
            (OpCode::NEGATE, "-t0"),
            (OpCode::RECIPROCAL, "1.0f / t0"),
            (OpCode::RSQRT, "1.0f / sqrt(t0)"),
            (OpCode::SIGMOID, "1.0f / (1.0f + exp(-t0))"),
        ];
        for (op, expected) in unary {
            let plan = single_node(op, &[DType::F32], DType::F32);
            assert!(
                plan.source
                    .contains(&format!("const float t1 = {expected};")),
                "{op:?}"
            );
        }
        for (op, expected) in [
            (OpCode::ADD, "t0 + t1"),
            (OpCode::SUB, "t0 - t1"),
            (OpCode::MUL, "t0 * t1"),
            (OpCode::DIV, "t0 / t1"),
            (OpCode::MINIMUM, "ax_field_min(t0, t1)"),
            (OpCode::MAXIMUM, "ax_field_max(t0, t1)"),
            (OpCode::ATAN2, "atan2(t0, t1)"),
            (OpCode::POW, "ax_field_pow(t0, t1)"),
            (OpCode::REMAINDER, "ax_field_remainder(t0, t1)"),
        ] {
            let plan = single_node(op, &[DType::F32; 2], DType::F32);
            assert!(
                plan.source
                    .contains(&format!("const float t2 = {expected};")),
                "{op:?}"
            );
        }
        for (op, expected) in [
            (OpCode::CLAMP, "ax_field_min(ax_field_max(t0, t1), t2)"),
            (OpCode::FMA, "(t0 * t1) + t2"),
        ] {
            let plan = single_node(op, &[DType::F32; 3], DType::F32);
            assert!(
                plan.source
                    .contains(&format!("const float t3 = {expected};")),
                "{op:?}"
            );
        }
    }

    #[test]
    fn integer_masks_casts_and_wrapping_arithmetic_remain_typed() {
        for dtype in [DType::F32, DType::I32] {
            for (op, operator) in [
                (OpCode::EQUAL, "=="),
                (OpCode::GREATER, ">"),
                (OpCode::GREATER_EQUAL, ">="),
            ] {
                let plan = single_node(op, &[dtype; 2], DType::U8);
                assert!(
                    plan.source
                        .contains(&format!("const uchar t2 = uchar(t0 {operator} t1);"))
                );
            }
        }
        for (op, inputs, expected) in [
            (OpCode::LOGICAL_NOT, 1, "uchar(t0 == 0u)"),
            (OpCode::LOGICAL_AND, 2, "uchar((t0 != 0u) && (t1 != 0u))"),
            (OpCode::LOGICAL_OR, 2, "uchar((t0 != 0u) || (t1 != 0u))"),
            (OpCode::LOGICAL_XOR, 2, "uchar((t0 != 0u) != (t1 != 0u))"),
        ] {
            let plan = single_node(op, &vec![DType::U8; inputs], DType::U8);
            assert!(
                plan.source
                    .contains(&format!("const uchar t{inputs} = {expected};"))
            );
        }
        for dtype in [DType::F32, DType::I32, DType::U8] {
            let plan = single_node(OpCode::SELECT, &[DType::U8, dtype, dtype], dtype);
            assert!(plan.source.contains("t3 = (t0 != 0u) ? t1 : t2;"));
        }
        for (op, expected) in [
            (
                OpCode::ADD,
                "as_type<int>(as_type<uint>(t0) + as_type<uint>(t1))",
            ),
            (
                OpCode::SUB,
                "as_type<int>(as_type<uint>(t0) - as_type<uint>(t1))",
            ),
            (
                OpCode::MUL,
                "as_type<int>(as_type<uint>(t0) * as_type<uint>(t1))",
            ),
            (OpCode::DIV, "ax_field_i32_div(t0, t1)"),
            (OpCode::REMAINDER, "ax_field_i32_rem(t0, t1)"),
            (
                OpCode::BITWISE_AND,
                "as_type<int>(as_type<uint>(t0) & as_type<uint>(t1))",
            ),
            (
                OpCode::BITWISE_OR,
                "as_type<int>(as_type<uint>(t0) | as_type<uint>(t1))",
            ),
            (
                OpCode::BITWISE_XOR,
                "as_type<int>(as_type<uint>(t0) ^ as_type<uint>(t1))",
            ),
            (
                OpCode::SHIFT_LEFT,
                "as_type<int>(as_type<uint>(t0) << (as_type<uint>(t1) & 31u))",
            ),
            (OpCode::SHIFT_RIGHT, "ax_field_i32_shr(t0, t1)"),
        ] {
            let plan = single_node(op, &[DType::I32; 2], DType::I32);
            assert!(
                plan.source.contains(&format!("const int t2 = {expected};")),
                "{op:?}"
            );
        }
        let cast = single_node(OpCode::CAST, &[DType::F32], DType::I32);
        assert!(
            cast.source
                .contains("const int t1 = ax_field_cast_i32(t0);")
        );
        let cast = single_node(OpCode::CAST, &[DType::I32], DType::F32);
        assert!(cast.source.contains("const float t1 = float(t0);"));
        for dtype in [DType::F32, DType::I32] {
            let identity = single_node(OpCode::CAST, &[dtype], dtype);
            assert!(identity.source.contains("t1 = t0;"));
        }

        let mut builder = GraphBuilder::new_in(Global);
        let a = tensor(
            &mut builder,
            DType::I32,
            TensorFlags::CONSTANT,
            &[],
            &16_777_217_i32.to_le_bytes(),
        );
        let b = tensor(
            &mut builder,
            DType::I32,
            TensorFlags::CONSTANT,
            &[],
            &16_777_218_i32.to_le_bytes(),
        );
        let equal = tensor(&mut builder, DType::U8, TensorFlags::OUTPUT, &[], &[]);
        builder
            .node(OpCode::EQUAL, NodeFlags::default(), &[a, b], &[equal], &[])
            .unwrap();
        let plan = build(builder, 7, 3).unwrap();
        assert_eq!(&plan.initial_inputs[..4], &16_777_217_i32.to_le_bytes());
        assert_eq!(&plan.initial_inputs[4..], &16_777_218_i32.to_le_bytes());
        let body = plan
            .source
            .split("kernel void ax_field_main")
            .nth(1)
            .unwrap();
        assert!(body.contains("const int t0 ="));
        assert!(body.contains("const uchar t2 = uchar(t0 == t1);"));
        assert!(!body.contains("float(t0)"));
    }

    #[test]
    fn mixed_resource_offsets_are_aligned_and_reads_are_not_elided() {
        let mut builder = GraphBuilder::new_in(Global);
        let scalar = tensor(
            &mut builder,
            DType::F32,
            TensorFlags::CONSTANT,
            &[],
            &0.0_f32.to_le_bytes(),
        );
        let index = tensor(&mut builder, DType::I32, TensorFlags::INPUT, &[3, 7], &[]);
        let rgb = builder
            .volume_trilinear_rgb(
                [scalar; 3],
                VolumeRgbParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 1,
                    },
                    [1; 3],
                    VolumeRgbStorage::F32,
                )
                .unwrap(),
                [3, 7],
            )
            .unwrap();
        let packed = builder.interleave3_f32([scalar; 3], [3, 7]).unwrap();
        let grid = VoxelBrickTraceParams::new(
            ResourceKey {
                group: 1,
                binding: 2,
            },
            [4; 3],
        )
        .unwrap();
        let hit = builder
            .voxel_brick_trace(packed, packed, scalar, grid, [3, 7])
            .unwrap();
        let _second_read = builder
            .voxel_brick_trace(packed, packed, scalar, grid, [3, 7])
            .unwrap();
        let byte = builder
            .buffer_load_u8(
                index,
                U8BufferParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 3,
                    },
                    3,
                )
                .unwrap(),
                [3, 7],
            )
            .unwrap();
        let narrow = builder
            .volume_trilinear_rgb(
                [scalar; 3],
                VolumeRgbParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 4,
                    },
                    [1; 3],
                    VolumeRgbStorage::Uf16E5M11,
                )
                .unwrap(),
                [3, 7],
            )
            .unwrap();
        for (dtype, value) in [
            (DType::F32, rgb.r),
            (DType::F32, hit.t),
            (DType::I32, hit.cell_id),
            (DType::I32, byte),
            (DType::F32, narrow.b),
        ] {
            pin(&mut builder, dtype, &[3, 7], value);
        }
        let plan = build(builder, 7, 3).unwrap();
        assert_eq!(
            plan.resources
                .iter()
                .map(|resource| (resource.offset, resource.byte_len))
                .collect::<Vec<_>>(),
            [(0, 12), (16, 8), (24, 3), (32, 6)]
        );
        assert_eq!(plan.resource_bytes, 38);
        assert_eq!(
            plan.source
                .matches("ax_voxel_brick_trace(resources + 16ul")
                .count(),
            2
        );
        assert!(
            plan.source
                .contains("ax_buffer_load_u8(resources + 24ul, 3u, t1)")
        );
        assert_eq!(plan.outputs[2].dtype, DType::I32);
    }

    #[test]
    fn conflicting_resource_kinds_sizes_and_storage_are_rejected() {
        let key = ResourceKey {
            group: 2,
            binding: 7,
        };
        for conflict in 0..4 {
            let mut builder = GraphBuilder::new_in(Global);
            let scalar = tensor(
                &mut builder,
                DType::F32,
                TensorFlags::CONSTANT | TensorFlags::OUTPUT,
                &[],
                &0.0_f32.to_le_bytes(),
            );
            builder
                .volume_trilinear_rgb(
                    [scalar; 3],
                    VolumeRgbParams::new(key, [1; 3], VolumeRgbStorage::F32).unwrap(),
                    [3, 7],
                )
                .unwrap();
            match conflict {
                0 | 1 => {
                    let (dims, storage) = if conflict == 0 {
                        ([2, 1, 1], VolumeRgbStorage::F32)
                    } else {
                        ([1; 3], VolumeRgbStorage::Uf16E5M11)
                    };
                    builder
                        .volume_trilinear_rgb(
                            [scalar; 3],
                            VolumeRgbParams::new(key, dims, storage).unwrap(),
                            [3, 7],
                        )
                        .unwrap();
                }
                2 => {
                    let packed = builder.interleave3_f32([scalar; 3], [3, 7]).unwrap();
                    builder
                        .voxel_brick_trace(
                            packed,
                            packed,
                            scalar,
                            VoxelBrickTraceParams::new(key, [4; 3]).unwrap(),
                            [3, 7],
                        )
                        .unwrap();
                }
                _ => {
                    let index = tensor(&mut builder, DType::I32, TensorFlags::INPUT, &[], &[]);
                    builder
                        .buffer_load_u8(index, U8BufferParams::new(key, 12).unwrap(), [3, 7])
                        .unwrap();
                }
            }
            assert!(
                build(builder, 7, 3)
                    .unwrap_err()
                    .contains("conflicting resource metadata")
            );
        }
    }

    #[test]
    fn rejects_unordered_ssa_redefinitions_and_missing_values() {
        for case in 0..4 {
            let mut builder = GraphBuilder::new_in(Global);
            let input = tensor(&mut builder, DType::F32, TensorFlags::INPUT, &[3, 7], &[]);
            let middle = tensor(
                &mut builder,
                DType::F32,
                TensorFlags::default(),
                &[3, 7],
                &[],
            );
            let output = tensor(&mut builder, DType::F32, TensorFlags::OUTPUT, &[3, 7], &[]);
            let nodes = match case {
                0 => vec![(middle, output), (input, middle)],
                1 => vec![(input, middle), (input, middle), (middle, output)],
                2 => vec![(input, input), (input, middle), (middle, output)],
                _ => vec![(input, output)],
            };
            for (from, to) in nodes {
                builder
                    .node(OpCode::IDENTITY, NodeFlags::default(), &[from], &[to], &[])
                    .unwrap();
            }
            let error = build(builder, 7, 3).unwrap_err();
            assert!(
                error.contains("before definition")
                    || error.contains("multiply defined")
                    || error.contains("no input, constant, or producer"),
                "{error}"
            );
        }
    }

    #[test]
    fn rejects_effects_noncanonical_shapes_and_unsupported_operations() {
        for flags in [NodeFlags::ORDERED, NodeFlags::MUTATING] {
            let mut builder = GraphBuilder::new_in(Global);
            let input = tensor(&mut builder, DType::F32, TensorFlags::INPUT, &[3, 7], &[]);
            let output = tensor(&mut builder, DType::F32, TensorFlags::OUTPUT, &[3, 7], &[]);
            builder
                .node(OpCode::IDENTITY, flags, &[input], &[output], &[])
                .unwrap();
            assert!(build(builder, 7, 3).unwrap_err().contains("effects"));
        }
        let mut builder = GraphBuilder::new_in(Global);
        let input = tensor(&mut builder, DType::F32, TensorFlags::INPUT, &[3, 7], &[]);
        let output = tensor(&mut builder, DType::F32, TensorFlags::OUTPUT, &[3, 1], &[]);
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[input],
                &[output],
                &[],
            )
            .unwrap();
        assert!(build(builder, 7, 3).unwrap_err().contains("noncanonical"));

        for (op, dtype, dimensions) in [
            (OpCode::MATMUL, DType::F32, [3, 3]),
            (OpCode::ADD, DType::U8, [3, 3]),
            (OpCode::MINIMUM, DType::I32, [3, 3]),
        ] {
            let mut builder = GraphBuilder::new_in(Global);
            let a = tensor(&mut builder, dtype, TensorFlags::INPUT, &dimensions, &[]);
            let b = tensor(&mut builder, dtype, TensorFlags::INPUT, &dimensions, &[]);
            let output = tensor(&mut builder, dtype, TensorFlags::OUTPUT, &dimensions, &[]);
            builder
                .node(op, NodeFlags::default(), &[a, b], &[output], &[])
                .unwrap();
            assert!(build(builder, 3, 3).unwrap_err().contains("unsupported"));
        }
    }

    #[test]
    fn rejects_invalid_tensor_contracts_without_large_staging() {
        for shape in [
            vec![0, 7],
            vec![ax::DYNAMIC_DIM],
            vec![3],
            vec![3, 7, 2],
            vec![1, 3, 7, 1],
        ] {
            let mut builder = GraphBuilder::new_in(Global);
            tensor(
                &mut builder,
                DType::F32,
                TensorFlags::INPUT | TensorFlags::OUTPUT,
                &shape,
                &[],
            );
            assert!(build(builder, 7, 3).is_err(), "{shape:?}");
        }
        for dtype in [DType::F16, DType::I16, DType::I64, DType::F64] {
            let mut builder = GraphBuilder::new_in(Global);
            tensor(
                &mut builder,
                dtype,
                TensorFlags::INPUT | TensorFlags::OUTPUT,
                &[],
                &[],
            );
            assert!(
                build(builder, 7, 3)
                    .unwrap_err()
                    .contains("unsupported dtype")
            );
        }
        let mut builder = GraphBuilder::new_in(Global);
        tensor(
            &mut builder,
            DType::F32,
            TensorFlags::CONSTANT | TensorFlags::OUTPUT,
            &[],
            &[0],
        );
        assert!(
            build(builder, 7, 3)
                .unwrap_err()
                .contains("constant byte length")
        );
        let mut builder = GraphBuilder::new_in(Global);
        tensor(
            &mut builder,
            DType::F32,
            TensorFlags::CONSTANT | TensorFlags::INPUT | TensorFlags::OUTPUT,
            &[],
            &[0; 4],
        );
        assert!(
            build(builder, 7, 3)
                .unwrap_err()
                .contains("INPUT and CONSTANT")
        );
        let mut builder = GraphBuilder::new_in(Global);
        tensor(
            &mut builder,
            DType::F32,
            TensorFlags::INPUT | TensorFlags::OUTPUT,
            &[3, 7, 3],
            &[],
        );
        assert!(build(builder, 7, 3).unwrap_err().contains("local F32"));
        let mut builder = GraphBuilder::new_in(Global);
        tensor(
            &mut builder,
            DType::F32,
            TensorFlags::INPUT | TensorFlags::OUTPUT,
            &[2, u32::MAX - 1],
            &[],
        );
        assert!(
            build(builder, u32::MAX - 1, 2)
                .unwrap_err()
                .contains("budget")
        );
        let mut builder = GraphBuilder::new_in(Global);
        tensor(
            &mut builder,
            DType::F32,
            TensorFlags::CONSTANT | TensorFlags::OUTPUT,
            &[],
            &[0; 4],
        );
        assert!(
            build(builder, 65_536, 65_536)
                .unwrap_err()
                .contains("dispatch-lane work budget")
        );
    }

    #[test]
    fn rejects_stronger_accuracy_and_missing_or_mismatched_fields() {
        let mut builder = GraphBuilder::new_in(Global);
        tensor(
            &mut builder,
            DType::F32,
            TensorFlags::INPUT | TensorFlags::OUTPUT,
            &[3, 7],
            &[],
        );
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        for accuracy in [KernelAccuracy::Faithful, KernelAccuracy::Strict] {
            assert!(
                compile(&graph, 7, 3, accuracy)
                    .unwrap_err()
                    .contains("Approximate")
            );
        }
        for (width, height) in [(0, 3), (7, 0), (ax::DYNAMIC_DIM, 3), (3, 7)] {
            assert!(compile(&graph, width, height, KernelAccuracy::Approximate).is_err());
        }
        let mut builder = GraphBuilder::new_in(Global);
        tensor(&mut builder, DType::F32, TensorFlags::INPUT, &[], &[]);
        assert!(
            build(builder, 7, 3)
                .unwrap_err()
                .contains("at least one graph OUTPUT")
        );
        let mut builder = GraphBuilder::new_in(Global);
        let scalar = tensor(
            &mut builder,
            DType::F32,
            TensorFlags::CONSTANT | TensorFlags::OUTPUT,
            &[],
            &[0; 4],
        );
        builder
            .volume_trilinear_rgb(
                [scalar; 3],
                VolumeRgbParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 0,
                    },
                    [1; 3],
                    VolumeRgbStorage::F32,
                )
                .unwrap(),
                [1, 7],
            )
            .unwrap();
        assert!(
            build(builder, 7, 3)
                .unwrap_err()
                .contains("resource outputs must be the full field")
        );
    }
}
