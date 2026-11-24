//! Host/build-time WGSL to `.ax` compilation.
//!
//! This crate intentionally places standalone Naga on the host side of the
//! asset boundary. Naga and compiler diagnostics may use the process global
//! allocator; the returned [`ax::GraphBuf`] is always owned by the allocator
//! explicitly supplied to [`compile_wgsl_in`]. Runtime and `kore` consumers can
//! therefore ship and execute the resulting `.ax` bytes without linking Naga.

#![feature(allocator_api)]

use std::{collections::HashMap, error::Error, fmt, ops::Range, vec::Vec};

use ax::buffer::U8BufferParams;
use ax::sampling::VolumeRgbParams;
use ax::voxel::VoxelBrickTraceParams;
use ax::{
    AllocationError, DType, DYNAMIC_DIM, GraphBuf, GraphBuilder, NodeFlags, OpCode, TensorFlags,
    TensorId,
};
use core::alloc::Allocator;
use naga::{
    BinaryOperator, Binding, BuiltIn, Expression, Function, Handle, Literal, MathFunction, Module,
    ScalarKind, ShaderStage, Span, Statement, SwizzleComponent, TypeInner, UnaryOperator,
    VectorSize,
    valid::{Capabilities, FunctionInfo, ModuleInfo, ValidationFlags, Validator},
};

mod buffer;
mod sampling;
mod voxel;

/// Typed specialization for a borrowed, explicitly enabled shader resource.
/// Bindings and metadata are serialized in `.ax`; contents are supplied by
/// the runtime. A declared resource need not be called, but every supplied
/// specialization must have one matching declaration of the same kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderResource {
    VoxelBrickGrid(VoxelBrickTraceParams),
    RgbVolume(VolumeRgbParams),
    U8Buffer(U8BufferParams),
}

impl ShaderResource {
    fn resource(self) -> ax::voxel::ResourceKey {
        match self {
            Self::VoxelBrickGrid(params) => params.resource,
            Self::RgbVolume(params) => params.resource,
            Self::U8Buffer(params) => params.resource,
        }
    }

    fn is_valid(self) -> bool {
        match self {
            Self::VoxelBrickGrid(params) => params.is_valid(),
            Self::RgbVolume(params) => params.is_valid(),
            Self::U8Buffer(params) => params.is_valid(),
        }
    }
}

/// Name of the constrained WGSL profile implemented by this compiler.
pub const PROFILE_NAME: &str = "ax-shader-v0";

// Bounds apply to expanded code as well as source nesting. A small, acyclic
// helper DAG can otherwise produce an exponentially large graph when inlined.
const MAX_INLINE_DEPTH: usize = 32;
const MAX_INLINE_CALLS: usize = 1024;
const MAX_CONTROL_DEPTH: usize = 64;
const MAX_EXPRESSION_DEPTH: usize = 256;
const MAX_GRAPH_NODES: usize = 16_384;
const MAX_FRAGMENT_CONSTANT_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of flattened planes in an ordinary fragment result struct.
pub const MAX_FRAGMENT_OUTPUT_LANES: usize = 16;

/// One scalar output plane, identified by its WGSL location and vector component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragmentOutputLane {
    pub location: u32,
    /// Zero for scalar members; x/y/z/w are 0/1/2/3 for F32 vectors.
    pub component: u8,
    pub dtype: DType,
}

/// Explicitly supported fragment output layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderOutputKind {
    RgbaF32,
    /// F32 distance followed by an exact I32 cell ID; misses are `(+inf, -1)`.
    VoxelHit,
    /// Ordinary, immutable result struct, flattened by location then component.
    /// Unused entries are `None`; all occupied entries precede them.
    FragmentStruct {
        lanes: [Option<FragmentOutputLane>; MAX_FRAGMENT_OUTPUT_LANES],
    },
}

/// ABI of a compiled 2D fragment graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderInterface {
    /// Number of F32 coordinate input planes (`position.x`, `position.y`).
    pub position_inputs: u8,
    /// Number of output planes; consult [`Self::output_dtype`] for their types.
    pub output_planes: u8,
    pub output_kind: ShaderOutputKind,
    /// `[height, width]` shape of output planes and the broadcast result of
    /// the two linear coordinate axes.
    pub field_dimensions: [u32; 2],
    /// F32 field lanes following position, ordered by entry argument/location.
    pub varying_inputs: u8,
    /// F32 scalar lanes after varyings, ordered by (group, binding, component).
    pub uniform_inputs: u8,
}

impl ShaderInterface {
    /// Symbolic profile interface. Every compiled shader replaces
    /// `field_dimensions` with its requested concrete viewport.
    pub const AX_SHADER_V0: Self = Self {
        position_inputs: 2,
        output_planes: 4,
        output_kind: ShaderOutputKind::RgbaF32,
        field_dimensions: [DYNAMIC_DIM, DYNAMIC_DIM],
        varying_inputs: 0,
        uniform_inputs: 0,
    };

    /// Broadcast-efficient `[1, width]` plane used for `position.x`.
    pub const fn position_x_dimensions(self) -> [u32; 2] {
        [1, self.field_dimensions[1]]
    }

    /// Broadcast-efficient `[height, 1]` plane used for `position.y`.
    pub const fn position_y_dimensions(self) -> [u32; 2] {
        [self.field_dimensions[0], 1]
    }

    pub const fn output_count(self) -> usize {
        self.output_planes as usize
    }

    pub const fn output_dtype(self, index: usize) -> Option<DType> {
        match self.output_lane(index) {
            Some(lane) => Some(lane.dtype),
            None => None,
        }
    }

    /// Output metadata in tensor order, independent of struct declaration order.
    pub const fn output_lane(self, index: usize) -> Option<FragmentOutputLane> {
        if index >= self.output_count() {
            return None;
        }
        match (self.output_kind, index) {
            (ShaderOutputKind::RgbaF32, 0..=3) => Some(FragmentOutputLane {
                location: 0,
                component: index as u8,
                dtype: DType::F32,
            }),
            (ShaderOutputKind::VoxelHit, 0..=1) => Some(FragmentOutputLane {
                location: index as u32,
                component: 0,
                dtype: if index == 0 { DType::F32 } else { DType::I32 },
            }),
            (ShaderOutputKind::FragmentStruct { lanes }, index)
                if index < MAX_FRAGMENT_OUTPUT_LANES =>
            {
                lanes[index]
            }
            _ => None,
        }
    }
}

/// Successfully compiled WGSL plus its stable runtime ABI.
pub struct CompiledShader<A: Allocator> {
    graph: GraphBuf<A>,
    interface: ShaderInterface,
}

impl<A: Allocator> fmt::Debug for CompiledShader<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledShader")
            .field("graph_bytes", &self.graph.as_bytes().len())
            .field("interface", &self.interface)
            .finish()
    }
}

impl<A: Allocator> CompiledShader<A> {
    pub const fn interface(&self) -> ShaderInterface {
        self.interface
    }

    pub const fn graph(&self) -> &GraphBuf<A> {
        &self.graph
    }

    pub fn into_graph(self) -> GraphBuf<A> {
        self.graph
    }
}

/// Host compiler options. The initial profile requires a concrete viewport so
/// every emitted graph is directly executable by current providers.
#[derive(Clone, Copy, Debug)]
pub struct CompileOptions<'a> {
    pub entry_point: &'a str,
    /// Concrete `[height, width]` for every coordinate/output plane.
    pub field_dimensions: [u32; 2],
}

impl<'a> CompileOptions<'a> {
    /// Select a concrete viewport specialization accepted by current
    /// `axnn-vaccel` artifact builders.
    pub const fn for_viewport(entry_point: &'a str, width: u32, height: u32) -> Self {
        Self {
            entry_point,
            field_dimensions: [height, width],
        }
    }
}

/// Profile rule associated with a source-level rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileRule {
    EntryPoint,
    Stage,
    Interface,
    Type,
    ControlFlow,
    Expression,
    Resource,
    Budget,
}

/// A source-located `ax-shader-v0` diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileDiagnostic {
    pub rule: ProfileRule,
    pub message: String,
    /// UTF-8 byte range in the original WGSL source, when Naga recorded one.
    pub span: Option<Range<usize>>,
    pub line: Option<u32>,
    pub column: Option<u32>,
}

impl ProfileDiagnostic {
    fn new(source: &str, rule: ProfileRule, message: impl Into<String>, span: Span) -> Self {
        // Generated extension declarations and source-less Naga callers may
        // carry spans outside the authored source. Never slice those offsets.
        let range = span.to_range().filter(|range| {
            range.end <= source.len()
                && source.is_char_boundary(range.start)
                && source.is_char_boundary(range.end)
        });
        let (line, column) = range
            .as_ref()
            .map(|_| {
                let location = span.location(source);
                (Some(location.line_number), Some(location.line_position))
            })
            .unwrap_or((None, None));
        Self {
            rule,
            message: message.into(),
            span: range,
            line,
            column,
        }
    }

    fn unspanned(rule: ProfileRule, message: impl Into<String>) -> Self {
        Self {
            rule,
            message: message.into(),
            span: None,
            line: None,
            column: None,
        }
    }
}

impl fmt::Display for ProfileDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let (Some(line), Some(column)) = (self.line, self.column) {
            write!(f, "{PROFILE_NAME}:{line}:{column}: {}", self.message)
        } else {
            write!(f, "{PROFILE_NAME}: {}", self.message)
        }
    }
}

impl Error for ProfileDiagnostic {}

/// Parse, WGSL validation, profile, or output-allocation failure.
#[derive(Debug)]
pub enum CompileError {
    Parse(String),
    Validation(String),
    Profile(ProfileDiagnostic),
    Allocation(AllocationError),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(diagnostic) | Self::Validation(diagnostic) => f.write_str(diagnostic),
            Self::Profile(diagnostic) => diagnostic.fmt(f),
            Self::Allocation(error) => {
                write!(f, "failed to allocate or encode .ax graph: {error:?}")
            }
        }
    }
}

impl Error for CompileError {}

impl From<AllocationError> for CompileError {
    fn from(value: AllocationError) -> Self {
        Self::Allocation(value)
    }
}

/// Parse, validate, profile-check, and lower WGSL into allocator-owned `.ax`.
///
/// Temporary Naga/module state uses host allocation. Only canonical graph
/// storage crosses the returned asset boundary, and that storage uses
/// `allocator` end to end.
pub fn compile_wgsl_in<A: Allocator + Clone>(
    source: &str,
    options: CompileOptions<'_>,
    allocator: A,
) -> Result<CompiledShader<A>, CompileError> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| CompileError::Parse(error.emit_to_string(source)))?;
    lower_naga_module_in(&module, source, options, allocator)
}

/// Compile the explicit `enable ax_voxel_query;` profile with fixed resource
/// binding keys and grid dimensions. Occupancy contents remain a borrowed
/// runtime resource, never an input tensor or serialized constant.
///
/// Unlike [`compile_wgsl_in`], this entry point recognizes `ax_brick_trace`
/// and the known `AxVoxelHit` result. The directive and this opt-in API are
/// both required. Ordinary WGSL and pre-parsed Naga modules cannot register
/// an intrinsic by declaring a function with the same name.
pub fn compile_wgsl_with_resources_in<A: Allocator + Clone>(
    source: &str,
    options: CompileOptions<'_>,
    resources: &[VoxelBrickTraceParams],
    allocator: A,
) -> Result<CompiledShader<A>, CompileError> {
    let prepared = voxel::prepare(source, options.entry_point, resources)?;
    lower_module_in(
        &prepared.module,
        source,
        &prepared.source,
        options,
        Some(&prepared.extension),
        allocator,
    )
}

/// Compile explicitly enabled voxel, RGB-volume and/or byte-buffer operations.
///
/// `enable ax_voxel_query;` registers voxel queries. The directive
/// `enable ax_volume_sampling;` registers opaque `AxRgbVolume` handles with
/// F32 RGB sampling. `enable ax_buffer_access;` registers bounded U8 loads
/// through opaque `AxU8Buffer` handles. The directives may coexist. Unknown operations,
/// authored intrinsic bodies and non-root resource uses remain rejected.
/// Every specialization must match a declaration of the same kind; a
/// declared resource need not be called. Storage format is resource metadata,
/// not a tensor dtype or a promise of native arithmetic in that format.
pub fn compile_wgsl_with_resource_specs_in<A: Allocator + Clone>(
    source: &str,
    options: CompileOptions<'_>,
    resources: &[ShaderResource],
    allocator: A,
) -> Result<CompiledShader<A>, CompileError> {
    let prepared = voxel::prepare_specs(source, options.entry_point, resources)?;
    lower_module_in(
        &prepared.module,
        source,
        &prepared.source,
        options,
        Some(&prepared.extension),
        allocator,
    )
}

/// Validate, profile-check, and lower an existing Naga module.
///
/// This is the shared IR boundary intended for a later SPIR-V frontend. WGSL
/// callers should normally use [`compile_wgsl_in`]. `source` is used only to
/// render Naga/profile source locations; it may be empty for source-less input.
pub fn lower_naga_module_in<A: Allocator + Clone>(
    module: &Module,
    source: &str,
    options: CompileOptions<'_>,
    allocator: A,
) -> Result<CompiledShader<A>, CompileError> {
    voxel::reject_reserved_declarations(module, source)?;
    lower_module_in(module, source, source, options, None, allocator)
}

fn lower_module_in<A: Allocator + Clone>(
    module: &Module,
    source: &str,
    validation_source: &str,
    options: CompileOptions<'_>,
    voxel: Option<&voxel::Extension>,
    allocator: A,
) -> Result<CompiledShader<A>, CompileError> {
    if options
        .field_dimensions
        .iter()
        .any(|&dimension| dimension == 0 || dimension == DYNAMIC_DIM)
    {
        return Err(profile_unspanned(
            ProfileRule::Interface,
            "ax-shader-v0 requires concrete non-zero viewport dimensions; use CompileOptions::for_viewport",
        ));
    }
    let module_info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(module)
        .map_err(|error| CompileError::Validation(error.emit_to_string(validation_source)))?;

    let (entry_index, entry) = select_entry_point(module, source, options.entry_point)?;
    let output_kind = validate_interface(module, source, entry, voxel)?;
    let validated_body = validate_body(module, source, &entry.function)?;
    let mut helpers = vec![None; module.functions.len()];
    validate_helpers(
        module,
        source,
        &validated_body,
        &mut helpers,
        &mut Vec::new(),
        voxel,
    )?;

    let mut lowerer = Lowerer::new(
        source,
        FunctionDefinition {
            function: &entry.function,
            function_info: module_info.get_entry_point(entry_index),
            body: &validated_body,
        },
        ModuleContext {
            module,
            module_info: &module_info,
            helpers: &helpers,
            voxel,
        },
        options.field_dimensions,
        allocator,
    )?;
    match output_kind {
        ShaderOutputKind::RgbaF32 => {
            let colour = lowerer.lower_return(validated_body.return_value)?;
            lowerer.emit_outputs(colour)?;
        }
        ShaderOutputKind::VoxelHit => {
            lowerer.emit_hit_outputs(validated_body.return_value)?;
        }
        ShaderOutputKind::FragmentStruct { .. } => {
            lowerer.emit_fragment_outputs(validated_body.return_value)?;
        }
    }
    let varying_inputs = lowerer
        .frame
        .arguments
        .iter()
        .skip(1)
        .map(|v| v.lanes.len())
        .sum::<usize>() as u8;
    let uniform_inputs = lowerer
        .uniforms
        .iter()
        .flatten()
        .map(|v| v.lanes.len())
        .sum::<usize>() as u8;
    let graph = lowerer.builder.finish()?;
    Ok(CompiledShader {
        graph,
        interface: ShaderInterface {
            output_planes: match output_kind {
                ShaderOutputKind::RgbaF32 => 4,
                ShaderOutputKind::VoxelHit => 2,
                ShaderOutputKind::FragmentStruct { lanes } => lanes.iter().flatten().count() as u8,
            },
            output_kind,
            field_dimensions: options.field_dimensions,
            varying_inputs,
            uniform_inputs,
            ..ShaderInterface::AX_SHADER_V0
        },
    })
}

fn select_entry_point<'a>(
    module: &'a Module,
    _source: &str,
    name: &str,
) -> Result<(usize, &'a naga::EntryPoint), CompileError> {
    let mut named = module
        .entry_points
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.name == name);
    let Some((index, entry)) = named.next() else {
        return Err(CompileError::Profile(ProfileDiagnostic::unspanned(
            ProfileRule::EntryPoint,
            format!("fragment entry point `{name}` was not found"),
        )));
    };
    if named.next().is_some() {
        return Err(CompileError::Profile(ProfileDiagnostic::unspanned(
            ProfileRule::EntryPoint,
            format!("entry point `{name}` is ambiguous across shader stages"),
        )));
    }
    if entry.stage != ShaderStage::Fragment {
        return Err(CompileError::Profile(ProfileDiagnostic::unspanned(
            ProfileRule::Stage,
            format!("entry point `{name}` must be a fragment shader"),
        )));
    }
    Ok((index, entry))
}

fn validate_interface(
    module: &Module,
    source: &str,
    entry: &naga::EntryPoint,
    voxel: Option<&voxel::Extension>,
) -> Result<ShaderOutputKind, CompileError> {
    let function = &entry.function;
    if function.arguments.is_empty() {
        return Err(profile_unspanned(
            ProfileRule::Interface,
            "entry point must take exactly one `@builtin(position) vec4<f32>` argument",
        ));
    }

    // This is a tensor-field ABI, not GPU interpolation. Consecutive locations
    // name full-resolution F32 input planes in component order.
    let mut lanes = 0;
    for (index, argument) in function.arguments.iter().skip(1).enumerate() {
        if !matches!(argument.binding, Some(Binding::Location { location, blend_src: None, .. }) if location == index as u32)
            || f32_lanes(module, argument.ty).is_none()
        {
            return Err(profile_unspanned(
                ProfileRule::Interface,
                "field inputs must be f32/vec2-vec4<f32> at consecutive locations starting at 0",
            ));
        }
        lanes += f32_lanes(module, argument.ty).unwrap();
    }
    if lanes > 16 {
        return Err(profile_unspanned(
            ProfileRule::Interface,
            "at most 16 field input lanes are supported",
        ));
    }
    let mut uniform_lanes = 0;
    for (handle, global) in module.global_variables.iter() {
        if voxel.is_some_and(|extension| extension.resources[handle.index()].is_some()) {
            continue;
        }
        if global.space != naga::AddressSpace::Uniform
            || global.binding.is_none()
            || global.init.is_some()
            || f32_lanes(module, global.ty).is_none()
        {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "only bound, read-only f32/vec2-vec4<f32> uniforms are supported; buffers, textures and pointers remain outside the profile",
                module.global_variables.get_span(handle),
            ));
        }
        uniform_lanes += f32_lanes(module, global.ty).unwrap();
    }
    if uniform_lanes > 32 {
        return Err(profile_unspanned(
            ProfileRule::Resource,
            "at most 32 uniform input lanes are supported",
        ));
    }
    let argument = &function.arguments[0];
    if !matches!(
        argument.binding,
        Some(Binding::BuiltIn(BuiltIn::Position { .. }))
    ) || !is_f32_vector(module, argument.ty, VectorSize::Quad)
    {
        return Err(profile_unspanned(
            ProfileRule::Interface,
            "entry point argument must be `@builtin(position) vec4<f32>`",
        ));
    }

    let Some(result) = &function.result else {
        return Err(profile_unspanned(
            ProfileRule::Interface,
            "entry point must return `@location(0) vec4<f32>`",
        ));
    };
    if voxel.is_some_and(|extension| {
        extension
            .voxel
            .is_some_and(|voxel| result.ty == voxel.hit_type)
    }) && result.binding.is_none()
    {
        return Ok(ShaderOutputKind::VoxelHit);
    }
    if result.binding.is_none() && matches!(module.types[result.ty].inner, TypeInner::Struct { .. })
    {
        return validate_fragment_struct(module, source, result.ty);
    }
    let location_zero = matches!(
        result.binding,
        Some(Binding::Location {
            location: 0,
            blend_src: None,
            ..
        })
    );
    if !location_zero || !is_f32_vector(module, result.ty, VectorSize::Quad) {
        return Err(profile_unspanned(
            ProfileRule::Interface,
            "entry point result must be `@location(0) vec4<f32>` or a supported fragment result struct",
        ));
    }
    Ok(ShaderOutputKind::RgbaF32)
}

fn fragment_member_type(module: &Module, ty: Handle<naga::Type>) -> Option<(ValueKind, usize)> {
    if let Some(lanes) = f32_lanes(module, ty) {
        return Some((ValueKind::Float, lanes));
    }
    matches!(module.types[ty].inner,
        TypeInner::Scalar(scalar) if scalar.kind == ScalarKind::Sint && scalar.width == 4)
    .then_some((ValueKind::I32, 1))
}

fn validate_fragment_struct(
    module: &Module,
    source: &str,
    ty: Handle<naga::Type>,
) -> Result<ShaderOutputKind, CompileError> {
    let TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        unreachable!("result type was checked before validating its members");
    };
    let span = module.types.get_span(ty);
    if members.is_empty() || members.len() > MAX_FRAGMENT_OUTPUT_LANES {
        return Err(profile(
            source,
            ProfileRule::Budget,
            "fragment result structs require 1 to 16 scalar/vector members",
            span,
        ));
    }
    let mut locations = [None; MAX_FRAGMENT_OUTPUT_LANES];
    let mut lane_count = 0;
    for member in members {
        let Some((kind, count)) = fragment_member_type(module, member.ty) else {
            return Err(profile(
                source,
                ProfileRule::Type,
                "fragment result members must be f32, vec2-vec4<f32>, or scalar i32; nested structs, arrays, matrices and other types are unsupported",
                span,
            ));
        };
        let Some(Binding::Location {
            location,
            interpolation,
            sampling,
            blend_src: None,
            per_primitive: false,
        }) = &member.binding
        else {
            return Err(profile(
                source,
                ProfileRule::Interface,
                "fragment result members require explicit locations; builtins, dual-source blending and per-primitive fields are unsupported",
                span,
            ));
        };
        let ordinary_qualifiers = match (interpolation, sampling) {
            (None | Some(naga::Interpolation::Flat), None) => true,
            (Some(naga::Interpolation::Perspective), None | Some(naga::Sampling::Center)) => {
                kind == ValueKind::Float
            }
            _ => false,
        };
        if !ordinary_qualifiers {
            return Err(profile(
                source,
                ProfileRule::Interface,
                "fragment result fields are tensor planes; custom interpolation and sampling are unsupported",
                span,
            ));
        }
        let location = *location as usize;
        if location >= members.len() || locations[location].is_some() {
            return Err(profile(
                source,
                ProfileRule::Interface,
                "fragment result locations must be unique and consecutive starting at 0",
                span,
            ));
        }
        locations[location] = Some((kind.dtype(), count));
        lane_count += count;
        if lane_count > MAX_FRAGMENT_OUTPUT_LANES {
            return Err(profile(
                source,
                ProfileRule::Budget,
                "fragment result structs may flatten to at most 16 output planes",
                span,
            ));
        }
    }
    let mut lanes = [None; MAX_FRAGMENT_OUTPUT_LANES];
    let mut index = 0;
    for (location, member) in locations.iter().take(members.len()).enumerate() {
        let (dtype, count) = member.expect("all consecutive member locations were validated");
        for component in 0..count {
            lanes[index] = Some(FragmentOutputLane {
                location: location as u32,
                component: component as u8,
                dtype,
            });
            index += 1;
        }
    }
    Ok(ShaderOutputKind::FragmentStruct { lanes })
}

fn f32_lanes(module: &Module, ty: Handle<naga::Type>) -> Option<usize> {
    match module.types[ty].inner {
        TypeInner::Scalar(s) if s.kind == ScalarKind::Float && s.width == 4 => Some(1),
        TypeInner::Vector { size, scalar }
            if scalar.kind == ScalarKind::Float && scalar.width == 4 =>
        {
            Some(vector_len(size))
        }
        _ => None,
    }
}

#[cfg(test)]
mod input_abi_tests {
    use super::*;
    use std::alloc::Global;

    #[test]
    fn uniform_lanes_follow_binding_order_not_declaration_or_use_order() {
        let source = r#"
            @group(1) @binding(0) var<uniform> last: f32;
            @group(0) @binding(7) var<uniform> first: vec2<f32>;
            @fragment fn main(@builtin(position) p: vec4<f32>, @location(0) coverage: vec2<f32>) -> @location(0) vec4<f32> {
                return vec4<f32>(coverage.x * last, first.y + p.x, first.x, coverage.y);
            }
        "#;
        let compiled =
            compile_wgsl_in(source, CompileOptions::for_viewport("main", 7, 3), Global).unwrap();
        assert_eq!(compiled.interface().varying_inputs, 2);
        assert_eq!(compiled.interface().uniform_inputs, 3);
        let graph = compiled.graph().parse().unwrap();
        let inputs = (0..graph.tensor_count())
            .map(|i| graph.tensor(TensorId(i as u32)).unwrap())
            .filter(|t| t.flags.contains(TensorFlags::INPUT))
            .map(|t| t.dimensions().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(
            inputs,
            vec![
                vec![1, 7],
                vec![3, 1],
                vec![3, 7],
                vec![3, 7],
                vec![1, 1],
                vec![1, 1],
                vec![1, 1]
            ]
        );
    }

    #[test]
    fn unused_storage_resources_still_refuse_at_the_profile_boundary() {
        let source = r#"
            @group(0) @binding(0) var<storage, read> samples: array<f32>;
            @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> { return p; }
        "#;
        assert!(matches!(
            compile_wgsl_in(source, CompileOptions::for_viewport("main", 7, 3), Global),
            Err(CompileError::Profile(ProfileDiagnostic {
                rule: ProfileRule::Resource,
                ..
            }))
        ));
    }

    #[test]
    fn sparse_varying_locations_are_not_a_silent_abi_reordering() {
        let source = r#"
            @fragment fn main(@builtin(position) p: vec4<f32>, @location(3) v: f32) -> @location(0) vec4<f32> { return vec4<f32>(p.xy,v,1.0); }
        "#;
        assert!(matches!(
            compile_wgsl_in(source, CompileOptions::for_viewport("main", 7, 3), Global),
            Err(CompileError::Profile(ProfileDiagnostic {
                rule: ProfileRule::Interface,
                ..
            }))
        ));
    }
}

#[derive(Clone, Copy, Debug)]
struct ShortCircuit {
    op: OpCode,
    left: Handle<Expression>,
    right: Handle<Expression>,
}

#[derive(Clone)]
struct FunctionCall {
    function: Handle<Function>,
    arguments: Vec<Handle<Expression>>,
    span: Span,
}

#[derive(Clone, Copy)]
enum ReturnValue {
    Expression(Handle<Expression>),
    Select {
        condition: Handle<Expression>,
        accept: usize,
        reject: usize,
    },
}

#[derive(Clone)]
struct ValidatedBody {
    return_value: usize,
    // Indexed nodes share continuation returns instead of copying a tail into
    // every branch. Lowering caches these nodes within each helper invocation.
    returns: Vec<ReturnValue>,
    short_circuits: Vec<Option<ShortCircuit>>,
    calls: Vec<Option<FunctionCall>>,
}

fn validate_body(
    module: &Module,
    source: &str,
    function: &Function,
) -> Result<ValidatedBody, CompileError> {
    // Authored mutable locals retain a source name. Naga's WGSL frontend
    // represents `&&`/`||` with anonymous, uninitialized bool locals whose
    // stores are confined to the exact `if` shape recognized below.
    for (handle, local) in function.local_variables.iter() {
        if local.name.is_some()
            || local.init.is_some()
            || !matches!(
                module.types[local.ty].inner,
                TypeInner::Scalar(scalar) if scalar.kind == ScalarKind::Bool
            )
        {
            return Err(profile(
                source,
                ProfileRule::ControlFlow,
                "mutable local variables are outside ax-shader-v0; use `let` expressions",
                function.local_variables.get_span(handle),
            ));
        }
    }

    let mut body = ValidatedBody {
        return_value: 0,
        returns: Vec::new(),
        short_circuits: vec![None; function.local_variables.len()],
        calls: vec![None; function.expressions.len()],
    };
    let returned = normalize_block(source, function, &function.body, None, &mut body, 0)?;

    for (handle, _) in function.local_variables.iter() {
        if body.short_circuits[handle.index()].is_none() {
            return Err(profile(
                source,
                ProfileRule::ControlFlow,
                "only compiler-generated WGSL short-circuit temporaries are permitted",
                function.local_variables.get_span(handle),
            ));
        }
    }

    body.return_value = returned.ok_or_else(|| {
        profile_unspanned(
            ProfileRule::Interface,
            "every path must return a scalar/vector value",
        )
    })?;
    Ok(body)
}

fn normalize_block(
    source: &str,
    function: &Function,
    block: &naga::Block,
    mut continuation: Option<usize>,
    body: &mut ValidatedBody,
    depth: usize,
) -> Result<Option<usize>, CompileError> {
    if depth > MAX_CONTROL_DEPTH {
        return Err(profile_unspanned(
            ProfileRule::Budget,
            "structured control-flow nesting exceeds the shader budget",
        ));
    }
    let statements: Vec<_> = block.span_iter().collect();
    for (statement, span) in statements.into_iter().rev() {
        match statement {
            Statement::Emit(_) => {}
            Statement::Block(block) => {
                continuation =
                    normalize_block(source, function, block, continuation, body, depth + 1)?;
            }
            Statement::Call {
                function: callee,
                arguments,
                result,
            } => {
                record_call(source, *callee, arguments, *result, *span, &mut body.calls)?;
            }
            Statement::If { reject, .. } if is_short_circuit_reject(function, reject) => {
                validate_short_circuit_if(source, function, statement, *span, body, depth + 1)?;
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                let accept =
                    normalize_block(source, function, accept, continuation, body, depth + 1)?;
                let reject =
                    normalize_block(source, function, reject, continuation, body, depth + 1)?;
                continuation = match (accept, reject) {
                    (Some(accept), Some(reject)) if accept != reject => {
                        let value = body.returns.len();
                        body.returns.push(ReturnValue::Select {
                            condition: *condition,
                            accept,
                            reject,
                        });
                        Some(value)
                    }
                    (Some(value), Some(_)) => Some(value),
                    (None, None) => None,
                    _ => {
                        return Err(profile(
                            source,
                            ProfileRule::ControlFlow,
                            "every branch must return or continue to a value return",
                            *span,
                        ));
                    }
                };
            }
            Statement::Return { value: Some(value) } => {
                continuation = Some(body.returns.len());
                body.returns.push(ReturnValue::Expression(*value));
            }
            _ => {
                return Err(profile(
                    source,
                    ProfileRule::ControlFlow,
                    "only pure helper calls, structured if/else, and value returns are supported; loops and side effects are outside ax-shader-v0",
                    *span,
                ));
            }
        }
    }
    Ok(continuation)
}

fn is_short_circuit_reject(function: &Function, reject: &naga::Block) -> bool {
    let mut statements = reject.span_iter();
    matches!(statements.next(), Some((Statement::Store { pointer, .. }, _))
        if local_pointer(function, *pointer).is_some())
        && statements.next().is_none()
}

fn record_call(
    source: &str,
    function: Handle<Function>,
    arguments: &[Handle<Expression>],
    result: Option<Handle<Expression>>,
    span: Span,
    calls: &mut [Option<FunctionCall>],
) -> Result<(), CompileError> {
    let Some(result) = result else {
        return Err(profile(
            source,
            ProfileRule::ControlFlow,
            "helper calls must return a pure scalar/vector value",
            span,
        ));
    };
    let slot = &mut calls[result.index()];
    if slot.is_some() {
        return Err(profile(
            source,
            ProfileRule::ControlFlow,
            "helper call result is defined more than once",
            span,
        ));
    }
    *slot = Some(FunctionCall {
        function,
        arguments: arguments.to_vec(),
        span,
    });
    Ok(())
}

fn validate_helpers(
    module: &Module,
    source: &str,
    body: &ValidatedBody,
    helpers: &mut [Option<ValidatedBody>],
    active: &mut Vec<Handle<Function>>,
    voxel: Option<&voxel::Extension>,
) -> Result<(), CompileError> {
    // Validate even discarded calls: a return expression must not hide a
    // callee's discard, loop, pointer write, or other unsupported effect.
    for call in body.calls.iter().flatten() {
        // Registration validated the exact root resource and call context
        // before Naga validation. This parser-only declaration is never an
        // ordinary helper and its placeholder body must not be inlined.
        if voxel.is_some_and(|extension| extension.is_intrinsic(call.function)) {
            continue;
        }
        if active.contains(&call.function) {
            return Err(profile(
                source,
                ProfileRule::ControlFlow,
                "recursive helpers are unsupported",
                call.span,
            ));
        }
        if helpers[call.function.index()].is_some() {
            continue;
        }
        if active.len() >= MAX_INLINE_DEPTH {
            return Err(profile(
                source,
                ProfileRule::Budget,
                "helper call depth exceeds the shader budget",
                call.span,
            ));
        }
        let function = &module.functions[call.function];
        if function
            .arguments
            .iter()
            .any(|argument| value_type(module, argument.ty).is_none())
            || function
                .result
                .as_ref()
                .is_none_or(|result| value_type(module, result.ty).is_none())
        {
            return Err(profile(
                source,
                ProfileRule::Type,
                "helpers require f32/i32/bool scalar or vector parameters and results; pointers, arrays, and structs are unsupported",
                call.span,
            ));
        }
        let body = validate_body(module, source, function)?;
        active.push(call.function);
        validate_helpers(module, source, &body, helpers, active, voxel)?;
        active.pop();
        helpers[call.function.index()] = Some(body);
    }
    Ok(())
}

fn validate_short_circuit_if(
    source: &str,
    function: &Function,
    statement: &Statement,
    statement_span: Span,
    body: &mut ValidatedBody,
    depth: usize,
) -> Result<(), CompileError> {
    if depth > MAX_CONTROL_DEPTH {
        return Err(profile(
            source,
            ProfileRule::Budget,
            "short-circuit nesting exceeds the shader budget",
            statement_span,
        ));
    }
    let (condition, accept, reject) = match statement {
        Statement::If {
            condition,
            accept,
            reject,
        } => (*condition, accept, reject),
        _ => return Err(short_circuit_shape_error(source, statement_span)),
    };
    let mut reject_statements = reject.span_iter();
    let Some((
        Statement::Store {
            pointer: reject_pointer,
            value: reject_value,
        },
        _,
    )) = reject_statements.next()
    else {
        return Err(short_circuit_shape_error(source, statement_span));
    };
    if reject_statements.next().is_some() {
        return Err(short_circuit_shape_error(source, statement_span));
    }

    let Some(reject_local) = local_pointer(function, *reject_pointer) else {
        return Err(short_circuit_shape_error(source, statement_span));
    };
    let Some(reject_constant) = bool_literal(function, *reject_value) else {
        return Err(short_circuit_shape_error(source, statement_span));
    };

    let (op, left) = if reject_constant {
        let Expression::Unary {
            op: UnaryOperator::LogicalNot,
            expr,
        } = function.expressions[condition]
        else {
            return Err(short_circuit_shape_error(source, statement_span));
        };
        (OpCode::LOGICAL_OR, expr)
    } else {
        (OpCode::LOGICAL_AND, condition)
    };

    let mut accept_statements: Vec<_> = accept.span_iter().collect();
    let Some((
        Statement::Store {
            pointer: accept_pointer,
            value: right,
        },
        _,
    )) = accept_statements.pop()
    else {
        return Err(short_circuit_shape_error(source, statement_span));
    };
    let Some(accept_local) = local_pointer(function, *accept_pointer) else {
        return Err(short_circuit_shape_error(source, statement_span));
    };
    if accept_local != reject_local {
        return Err(short_circuit_shape_error(source, statement_span));
    }

    for (statement, span) in accept_statements {
        match statement {
            Statement::Emit(_) => {}
            Statement::Call {
                function: callee,
                arguments,
                result,
            } => {
                record_call(source, *callee, arguments, *result, *span, &mut body.calls)?;
            }
            Statement::If { .. } => {
                validate_short_circuit_if(source, function, statement, *span, body, depth + 1)?
            }
            _ => return Err(short_circuit_shape_error(source, *span)),
        }
    }

    let slot = &mut body.short_circuits[reject_local.index()];
    if slot.is_some() {
        return Err(short_circuit_shape_error(source, statement_span));
    }
    *slot = Some(ShortCircuit {
        op,
        left,
        right: *right,
    });
    Ok(())
}

fn local_pointer(
    function: &Function,
    pointer: Handle<Expression>,
) -> Option<Handle<naga::LocalVariable>> {
    match function.expressions[pointer] {
        Expression::LocalVariable(local) => Some(local),
        _ => None,
    }
}

fn bool_literal(function: &Function, value: Handle<Expression>) -> Option<bool> {
    match function.expressions[value] {
        Expression::Literal(Literal::Bool(value)) => Some(value),
        _ => None,
    }
}

fn short_circuit_shape_error(source: &str, span: Span) -> CompileError {
    profile(
        source,
        ProfileRule::ControlFlow,
        "only compiler-generated short-circuit stores are supported; authored local mutation is outside ax-shader-v0",
        span,
    )
}

fn is_f32_vector(module: &Module, ty: Handle<naga::Type>, size: VectorSize) -> bool {
    matches!(
        module.types[ty].inner,
        TypeInner::Vector { size: actual, scalar }
            if actual == size && scalar.kind == ScalarKind::Float && scalar.width == 4
    )
}

fn profile(
    source: &str,
    rule: ProfileRule,
    message: impl Into<String>,
    span: Span,
) -> CompileError {
    CompileError::Profile(ProfileDiagnostic::new(source, rule, message, span))
}

fn profile_unspanned(rule: ProfileRule, message: impl Into<String>) -> CompileError {
    CompileError::Profile(ProfileDiagnostic::unspanned(rule, message))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape([u32; 2]);

impl Shape {
    const SCALAR: Self = Self([1, 1]);

    fn merge(self, other: Self) -> Self {
        Self([
            merge_dimension(self.0[0], other.0[0]),
            merge_dimension(self.0[1], other.0[1]),
        ])
    }

    fn dimensions(&self) -> &[u32; 2] {
        &self.0
    }
}

fn merge_dimension(left: u32, right: u32) -> u32 {
    if left == right || right == 1 {
        left
    } else if left == 1 {
        right
    } else {
        debug_assert!(false, "incompatible internal broadcast dimensions");
        DYNAMIC_DIM
    }
}

#[derive(Clone, Copy, Debug)]
struct Lane {
    tensor: TensorId,
    shape: Shape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    Float,
    Bool,
    I32,
}

impl ValueKind {
    fn dtype(self) -> DType {
        match self {
            Self::Float => DType::F32,
            Self::Bool => DType::U8,
            Self::I32 => DType::I32,
        }
    }
}

#[derive(Clone, Debug)]
struct Value {
    kind: ValueKind,
    lanes: Vec<Lane>,
}

// The known heterogeneous record has its own lowering/cache path. It never
// masquerades as a homogeneous vector or converts cell IDs through F32.
#[derive(Clone, Copy)]
struct HitLanes {
    t: Lane,
    cell_id: Lane,
}

impl Value {
    fn scalar(kind: ValueKind, lane: Lane) -> Self {
        Self {
            kind,
            lanes: vec![lane],
        }
    }
}

fn value_type(module: &Module, ty: Handle<naga::Type>) -> Option<(ValueKind, usize)> {
    match module.types[ty].inner {
        TypeInner::Scalar(scalar) if scalar.kind == ScalarKind::Float && scalar.width == 4 => {
            Some((ValueKind::Float, 1))
        }
        TypeInner::Scalar(scalar) if scalar.kind == ScalarKind::Bool => Some((ValueKind::Bool, 1)),
        TypeInner::Scalar(scalar) if scalar.kind == ScalarKind::Sint && scalar.width == 4 => {
            Some((ValueKind::I32, 1))
        }
        TypeInner::Vector { size, scalar }
            if scalar.kind == ScalarKind::Float && scalar.width == 4 =>
        {
            Some((ValueKind::Float, vector_len(size)))
        }
        TypeInner::Vector { size, scalar } if scalar.kind == ScalarKind::Bool => {
            Some((ValueKind::Bool, vector_len(size)))
        }
        TypeInner::Vector { size, scalar }
            if scalar.kind == ScalarKind::Sint && scalar.width == 4 =>
        {
            Some((ValueKind::I32, vector_len(size)))
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct FunctionDefinition<'a> {
    function: &'a Function,
    function_info: &'a FunctionInfo,
    body: &'a ValidatedBody,
}

struct FunctionFrame<'a> {
    function: &'a Function,
    function_info: &'a FunctionInfo,
    body: &'a ValidatedBody,
    arguments: Vec<Value>,
    expression_cache: Vec<Option<Value>>,
    return_cache: Vec<Option<Value>>,
    short_circuit_cache: Vec<Option<Value>>,
    short_circuit_in_progress: Vec<bool>,
}

impl<'a> FunctionFrame<'a> {
    fn new(definition: FunctionDefinition<'a>, arguments: Vec<Value>) -> Self {
        let FunctionDefinition {
            function,
            function_info,
            body,
        } = definition;
        Self {
            function,
            function_info,
            body,
            arguments,
            expression_cache: vec![None; function.expressions.len()],
            return_cache: vec![None; body.returns.len()],
            short_circuit_cache: vec![None; body.short_circuits.len()],
            short_circuit_in_progress: vec![false; body.short_circuits.len()],
        }
    }
}

struct ShaderBuilder<A: Allocator> {
    graph: GraphBuilder<A>,
    nodes: usize,
}

impl<A: Allocator + Clone> ShaderBuilder<A> {
    fn new_in(allocator: A) -> Self {
        Self {
            graph: GraphBuilder::new_in(allocator),
            nodes: 0,
        }
    }

    fn tensor(
        &mut self,
        dtype: DType,
        flags: TensorFlags,
        dimensions: &[u32],
        constant: &[u8],
    ) -> Result<TensorId, CompileError> {
        Ok(self.graph.tensor(dtype, flags, dimensions, constant)?)
    }

    fn node(
        &mut self,
        op: OpCode,
        flags: NodeFlags,
        inputs: &[TensorId],
        outputs: &[TensorId],
        attributes: &[u8],
    ) -> Result<(), CompileError> {
        self.check_node_budget()?;
        self.graph.node(op, flags, inputs, outputs, attributes)?;
        self.nodes += 1;
        Ok(())
    }

    fn check_node_budget(&self) -> Result<(), CompileError> {
        if self.nodes >= MAX_GRAPH_NODES {
            return Err(profile_unspanned(
                ProfileRule::Budget,
                "expanded graph exceeds the shader node budget",
            ));
        }
        Ok(())
    }

    fn interleave3_f32(
        &mut self,
        components: [TensorId; 3],
        dimensions: [u32; 2],
    ) -> Result<TensorId, CompileError> {
        self.check_node_budget()?;
        let output = self.graph.interleave3_f32(components, dimensions)?;
        self.nodes += 1;
        Ok(output)
    }

    fn voxel_brick_trace(
        &mut self,
        origins: TensorId,
        directions: TensorId,
        limit: TensorId,
        params: VoxelBrickTraceParams,
        dimensions: [u32; 2],
    ) -> Result<ax::voxel::VoxelHit, CompileError> {
        self.check_node_budget()?;
        let hit = self
            .graph
            .voxel_brick_trace(origins, directions, limit, params, dimensions)?;
        self.nodes += 1;
        Ok(hit)
    }

    fn volume_trilinear_rgb(
        &mut self,
        coordinates: [TensorId; 3],
        params: VolumeRgbParams,
        dimensions: [u32; 2],
    ) -> Result<ax::sampling::RgbSample, CompileError> {
        self.check_node_budget()?;
        let sample = self
            .graph
            .volume_trilinear_rgb(coordinates, params, dimensions)?;
        self.nodes += 1;
        Ok(sample)
    }

    fn buffer_load_u8(
        &mut self,
        index: TensorId,
        params: U8BufferParams,
        dimensions: [u32; 2],
    ) -> Result<TensorId, CompileError> {
        self.check_node_budget()?;
        let output = self.graph.buffer_load_u8(index, params, dimensions)?;
        self.nodes += 1;
        Ok(output)
    }

    fn finish(self) -> Result<GraphBuf<A>, CompileError> {
        Ok(self.graph.finish()?)
    }
}

struct ModuleContext<'a> {
    module: &'a Module,
    module_info: &'a ModuleInfo,
    helpers: &'a [Option<ValidatedBody>],
    voxel: Option<&'a voxel::Extension>,
}

struct Lowerer<'a, A: Allocator> {
    source: &'a str,
    module: &'a Module,
    module_info: &'a ModuleInfo,
    helpers: &'a [Option<ValidatedBody>],
    voxel: Option<&'a voxel::Extension>,
    hit_cache: Vec<Option<HitLanes>>,
    frame: FunctionFrame<'a>,
    inline_depth: usize,
    inline_calls: usize,
    expression_depth: usize,
    return_depth: usize,
    builder: ShaderBuilder<A>,
    uniforms: Vec<Option<Value>>,
    domain: Lane,
    i32_output_mask: Option<Lane>,
    global_cache: Vec<Option<Value>>,
    f32_constants: HashMap<u32, Lane>,
    i32_constants: HashMap<i32, Lane>,
    bool_constants: [Lane; 2],
}

impl<'a, A: Allocator + Clone> Lowerer<'a, A> {
    fn new(
        source: &'a str,
        definition: FunctionDefinition<'a>,
        context: ModuleContext<'a>,
        field_dimensions: [u32; 2],
        allocator: A,
    ) -> Result<Self, CompileError> {
        let ModuleContext {
            module,
            module_info,
            helpers,
            voxel,
        } = context;
        let mut builder = ShaderBuilder::new_in(allocator);
        let x_dimensions = [1, field_dimensions[1]];
        let y_dimensions = [field_dimensions[0], 1];
        let x = Lane {
            tensor: builder.tensor(DType::F32, TensorFlags::INPUT, &x_dimensions, &[])?,
            shape: Shape(x_dimensions),
        };
        let y = Lane {
            tensor: builder.tensor(DType::F32, TensorFlags::INPUT, &y_dimensions, &[])?,
            shape: Shape(y_dimensions),
        };
        let zero = Lane {
            tensor: builder.tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &0.0f32.to_le_bytes(),
            )?,
            shape: Shape::SCALAR,
        };
        let one = Lane {
            tensor: builder.tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &1.0f32.to_le_bytes(),
            )?,
            shape: Shape::SCALAR,
        };
        let false_lane = Lane {
            tensor: builder.tensor(DType::U8, TensorFlags::CONSTANT, &[1, 1], &[0])?,
            shape: Shape::SCALAR,
        };
        let true_lane = Lane {
            tensor: builder.tensor(DType::U8, TensorFlags::CONSTANT, &[1, 1], &[1])?,
            shape: Shape::SCALAR,
        };

        let x_zero = emit_node_raw(&mut builder, OpCode::MUL, &[x, zero], ValueKind::Float)?;
        let y_zero = emit_node_raw(&mut builder, OpCode::MUL, &[y, zero], ValueKind::Float)?;
        let domain = emit_node_raw(
            &mut builder,
            OpCode::ADD,
            &[x_zero, y_zero],
            ValueKind::Float,
        )?;

        let mut f32_constants = HashMap::new();
        f32_constants.insert(0.0f32.to_bits(), zero);
        f32_constants.insert(1.0f32.to_bits(), one);
        let mut arguments = vec![Value {
            kind: ValueKind::Float,
            lanes: vec![x, y, zero, one],
        }];
        for argument in definition.function.arguments.iter().skip(1) {
            let mut lanes = Vec::new();
            for _ in 0..f32_lanes(module, argument.ty).unwrap() {
                lanes.push(Lane {
                    tensor: builder.tensor(
                        DType::F32,
                        TensorFlags::INPUT,
                        &field_dimensions,
                        &[],
                    )?,
                    shape: Shape(field_dimensions),
                });
            }
            arguments.push(Value {
                kind: ValueKind::Float,
                lanes,
            });
        }
        let mut uniforms = vec![None; module.global_variables.len()];
        let mut globals: Vec<_> = module
            .global_variables
            .iter()
            .filter(|(_, global)| global.space == naga::AddressSpace::Uniform)
            .collect();
        globals.sort_by_key(|(_, g)| {
            let b = g.binding.as_ref().unwrap();
            (b.group, b.binding)
        });
        for (handle, global) in globals {
            let mut lanes = Vec::new();
            for _ in 0..f32_lanes(module, global.ty).unwrap() {
                lanes.push(Lane {
                    tensor: builder.tensor(DType::F32, TensorFlags::INPUT, &[1, 1], &[])?,
                    shape: Shape::SCALAR,
                });
            }
            uniforms[handle.index()] = Some(Value {
                kind: ValueKind::Float,
                lanes,
            });
        }
        Ok(Self {
            source,
            module,
            module_info,
            helpers,
            voxel,
            hit_cache: vec![None; definition.function.expressions.len()],
            frame: FunctionFrame::new(definition, arguments),
            inline_depth: 0,
            inline_calls: 0,
            expression_depth: 0,
            return_depth: 0,
            builder,
            uniforms,
            domain,
            i32_output_mask: None,
            global_cache: vec![None; module.global_expressions.len()],
            f32_constants,
            i32_constants: HashMap::new(),
            bool_constants: [false_lane, true_lane],
        })
    }

    fn lower_return(&mut self, index: usize) -> Result<Value, CompileError> {
        if let Some(value) = self.frame.return_cache[index].clone() {
            return Ok(value);
        }
        if self.return_depth >= MAX_EXPRESSION_DEPTH {
            return Err(profile_unspanned(
                ProfileRule::Budget,
                "return selection depth exceeds the shader budget",
            ));
        }
        self.return_depth += 1;
        let result = (|| {
            let value = match self.frame.body.returns[index] {
                ReturnValue::Expression(expression) => self.lower_expression(expression)?,
                ReturnValue::Select {
                    condition,
                    accept,
                    reject,
                } => {
                    let predicate = self.lower_expression(condition)?;
                    let accept = self.lower_return(accept)?;
                    let reject = self.lower_return(reject)?;
                    self.select(condition, predicate, accept, reject)?
                }
            };
            self.frame.return_cache[index] = Some(value.clone());
            Ok(value)
        })();
        self.return_depth -= 1;
        result
    }

    fn lower_call(&mut self, handle: Handle<Expression>) -> Result<Value, CompileError> {
        let call = self.frame.body.calls[handle.index()]
            .clone()
            .ok_or_else(|| {
                self.expression_error(handle, "helper result has no validated call statement")
            })?;
        if self.voxel.is_some_and(|extension| {
            extension
                .voxel
                .is_some_and(|voxel| call.function == voxel.intrinsic)
        }) {
            return Err(self.type_error(
                handle,
                "AxVoxelHit is a record; access `.t`/`.cell_id` or return it directly",
            ));
        }
        if self.voxel.is_some_and(|extension| {
            extension
                .sampling
                .is_some_and(|sampling| call.function == sampling.intrinsic)
        }) {
            return self.lower_volume_sample(handle, &call);
        }
        if self.voxel.is_some_and(|extension| {
            extension
                .buffer
                .is_some_and(|buffer| call.function == buffer.intrinsic)
        }) {
            return self.lower_buffer_load(handle, &call);
        }
        if self.inline_depth >= MAX_INLINE_DEPTH || self.inline_calls >= MAX_INLINE_CALLS {
            return Err(profile(
                self.source,
                ProfileRule::Budget,
                "expanded helper calls exceed the shader budget",
                call.span,
            ));
        }
        self.inline_calls += 1;
        // Arguments are evaluated in the caller's frame, including nested
        // calls. Expression handles are function-local and must never share a
        // cache between different invocations of the same helper.
        let arguments = call
            .arguments
            .iter()
            .map(|&argument| self.lower_expression(argument))
            .collect::<Result<Vec<_>, _>>()?;
        let body = self.helpers[call.function.index()]
            .as_ref()
            .ok_or_else(|| self.expression_error(handle, "helper body was not validated"))?;
        let frame = FunctionFrame::new(
            FunctionDefinition {
                function: &self.module.functions[call.function],
                function_info: &self.module_info[call.function],
                body,
            },
            arguments,
        );
        let caller = std::mem::replace(&mut self.frame, frame);
        self.inline_depth += 1;
        let result = self.lower_return(body.return_value);
        self.inline_depth -= 1;
        self.frame = caller;
        result
    }

    fn is_hit_expression(&self, handle: Handle<Expression>) -> bool {
        self.voxel
            .and_then(|extension| extension.voxel)
            .is_some_and(|voxel| {
                matches!(self.frame.function_info[handle].ty,
                naga::proc::TypeResolution::Handle(ty) if ty == voxel.hit_type)
            })
    }

    fn lower_hit(&mut self, handle: Handle<Expression>) -> Result<HitLanes, CompileError> {
        if self.inline_depth != 0 {
            return Err(
                self.type_error(handle, "voxel queries are only admitted in the entry body")
            );
        }
        let extension = self
            .voxel
            .ok_or_else(|| self.expression_error(handle, "voxel query extension is not enabled"))?;
        let voxel = extension
            .voxel
            .ok_or_else(|| self.expression_error(handle, "voxel query extension is not enabled"))?;
        let params = extension
            .calls
            .get(handle.index())
            .copied()
            .flatten()
            .and_then(|resource| match resource {
                ShaderResource::VoxelBrickGrid(params) => Some(params),
                ShaderResource::RgbVolume(_) | ShaderResource::U8Buffer(_) => None,
            })
            .ok_or_else(|| {
                self.type_error(
                    handle,
                    "AxVoxelHit must be the direct result of a registered ax_brick_trace call",
                )
            })?;
        if let Some(hit) = self.hit_cache[handle.index()] {
            return Ok(hit);
        }
        let call = self.frame.body.calls[handle.index()]
            .clone()
            .ok_or_else(|| {
                self.expression_error(handle, "voxel query has no validated call statement")
            })?;
        if call.function != voxel.intrinsic || call.arguments.len() != 3 {
            return Err(
                self.expression_error(handle, "voxel query registration does not match its call")
            );
        }
        let origin = self.lower_expression(call.arguments[0])?;
        let direction = self.lower_expression(call.arguments[1])?;
        let limit = self.lower_expression(call.arguments[2])?;
        if origin.kind != ValueKind::Float
            || origin.lanes.len() != 3
            || direction.kind != ValueKind::Float
            || direction.lanes.len() != 3
            || limit.kind != ValueKind::Float
            || limit.lanes.len() != 1
            || limit.lanes[0].shape != Shape::SCALAR
        {
            return Err(self.expression_error(
                handle,
                "ax_brick_trace requires vec3<f32> rays and a uniform scalar f32 limit",
            ));
        }
        let dimensions = self.domain.shape.0;
        let origins = self.builder.interleave3_f32(
            [
                origin.lanes[0].tensor,
                origin.lanes[1].tensor,
                origin.lanes[2].tensor,
            ],
            dimensions,
        )?;
        let directions = self.builder.interleave3_f32(
            [
                direction.lanes[0].tensor,
                direction.lanes[1].tensor,
                direction.lanes[2].tensor,
            ],
            dimensions,
        )?;
        let hit = self.builder.voxel_brick_trace(
            origins,
            directions,
            limit.lanes[0].tensor,
            params,
            dimensions,
        )?;
        let hit = HitLanes {
            t: Lane {
                tensor: hit.t,
                shape: self.domain.shape,
            },
            cell_id: Lane {
                tensor: hit.cell_id,
                shape: self.domain.shape,
            },
        };
        self.hit_cache[handle.index()] = Some(hit);
        Ok(hit)
    }

    fn lower_volume_sample(
        &mut self,
        handle: Handle<Expression>,
        call: &FunctionCall,
    ) -> Result<Value, CompileError> {
        if self.inline_depth != 0 {
            return Err(
                self.type_error(handle, "volume sampling is only admitted in the entry body")
            );
        }
        let params = self
            .voxel
            .and_then(|extension| extension.calls.get(handle.index()))
            .copied()
            .flatten()
            .and_then(|resource| match resource {
                ShaderResource::RgbVolume(params) => Some(params),
                ShaderResource::VoxelBrickGrid(_) | ShaderResource::U8Buffer(_) => None,
            })
            .ok_or_else(|| {
                self.expression_error(handle, "volume sample has no registered RGB resource")
            })?;
        if call.arguments.len() != 1 {
            return Err(
                self.expression_error(handle, "volume sample registration does not match its call")
            );
        }
        let point = self.lower_expression(call.arguments[0])?;
        if point.kind != ValueKind::Float || point.lanes.len() != 3 {
            return Err(self.expression_error(
                handle,
                "ax_sample_volume_rgb requires vec3<f32> index coordinates",
            ));
        }
        let sample = self.builder.volume_trilinear_rgb(
            [
                point.lanes[0].tensor,
                point.lanes[1].tensor,
                point.lanes[2].tensor,
            ],
            params,
            self.domain.shape.0,
        )?;
        Ok(Value {
            kind: ValueKind::Float,
            lanes: [sample.r, sample.g, sample.b]
                .map(|tensor| Lane {
                    tensor,
                    shape: self.domain.shape,
                })
                .to_vec(),
        })
    }

    fn lower_buffer_load(
        &mut self,
        handle: Handle<Expression>,
        call: &FunctionCall,
    ) -> Result<Value, CompileError> {
        if self.inline_depth != 0 {
            return Err(self.type_error(
                handle,
                "byte-buffer loads are only admitted in the entry body",
            ));
        }
        let params = self
            .voxel
            .and_then(|extension| extension.calls.get(handle.index()))
            .copied()
            .flatten()
            .and_then(|resource| match resource {
                ShaderResource::U8Buffer(params) => Some(params),
                ShaderResource::VoxelBrickGrid(_) | ShaderResource::RgbVolume(_) => None,
            })
            .ok_or_else(|| {
                self.expression_error(handle, "byte-buffer load has no registered U8 resource")
            })?;
        if call.arguments.len() != 1 {
            return Err(
                self.expression_error(handle, "byte-buffer registration does not match its call")
            );
        }
        let index = self.lower_expression(call.arguments[0])?;
        if index.kind != ValueKind::I32 || index.lanes.len() != 1 {
            return Err(self.expression_error(handle, "ax_load_u8 requires a scalar i32 index"));
        }
        let tensor =
            self.builder
                .buffer_load_u8(index.lanes[0].tensor, params, self.domain.shape.0)?;
        Ok(Value::scalar(
            ValueKind::I32,
            Lane {
                tensor,
                shape: self.domain.shape,
            },
        ))
    }

    fn lower_expression(&mut self, handle: Handle<Expression>) -> Result<Value, CompileError> {
        if let Some(value) = self.frame.expression_cache[handle.index()].clone() {
            return Ok(value);
        }
        if self.expression_depth >= MAX_EXPRESSION_DEPTH {
            return Err(profile(
                self.source,
                ProfileRule::Budget,
                "expression nesting exceeds the shader budget",
                self.frame.function.expressions.get_span(handle),
            ));
        }
        self.expression_depth += 1;
        let result = self.lower_expression_uncached(handle);
        self.expression_depth -= 1;
        result
    }

    fn lower_expression_uncached(
        &mut self,
        handle: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        let expression = self.frame.function.expressions[handle].clone();
        let value = match expression {
            Expression::Literal(literal) => self.lower_literal(literal, handle)?,
            Expression::Constant(constant) => {
                self.lower_global_expression(self.module.constants[constant].init)?
            }
            Expression::ZeroValue(ty) => self.zero_value(ty, handle)?,
            Expression::Compose { ty, components } => {
                let (expected_kind, expected_lanes) = self.value_type(ty).ok_or_else(|| {
                    self.type_error(
                        handle,
                        "only f32/i32/bool scalar and vector constructors are supported here; matrices and arrays are unsupported, and fragment result structs may only be constructed for an entry return",
                    )
                })?;
                let mut lanes = Vec::new();
                let mut kind = None;
                for component in components {
                    let component = self.lower_expression(component)?;
                    if kind.is_some_and(|actual| actual != component.kind) {
                        return Err(self
                            .expression_error(handle, "mixed-type constructors are unsupported"));
                    }
                    kind = Some(component.kind);
                    lanes.extend(component.lanes);
                }
                let kind = kind.ok_or_else(|| {
                    self.expression_error(handle, "empty constructors are unsupported")
                })?;
                if kind != expected_kind || lanes.len() != expected_lanes {
                    return Err(self.type_error(
                        handle,
                        "constructor components do not match the scalar/vector result type",
                    ));
                }
                Value { kind, lanes }
            }
            Expression::Splat { size, value } => {
                let value = self.lower_expression(value)?;
                if value.lanes.len() != 1 {
                    return Err(self.expression_error(handle, "vector splat source must be scalar"));
                }
                Value {
                    kind: value.kind,
                    lanes: vec![value.lanes[0]; vector_len(size)],
                }
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let vector = self.lower_expression(vector)?;
                let mut lanes = Vec::with_capacity(vector_len(size));
                for component in pattern.into_iter().take(vector_len(size)) {
                    let index = swizzle_index(component);
                    let Some(lane) = vector.lanes.get(index).copied() else {
                        return Err(self
                            .expression_error(handle, "swizzle component is outside its vector"));
                    };
                    lanes.push(lane);
                }
                Value {
                    kind: vector.kind,
                    lanes,
                }
            }
            Expression::AccessIndex { base, index } => {
                if self.is_hit_expression(base) {
                    let hit = self.lower_hit(base)?;
                    match index {
                        0 => Value::scalar(ValueKind::Float, hit.t),
                        1 => Value::scalar(ValueKind::I32, hit.cell_id),
                        _ => return Err(self.expression_error(handle, "unknown AxVoxelHit member")),
                    }
                } else {
                    self.require_vector_index_base(handle, base)?;
                    let base = self.lower_expression(base)?;
                    let Some(lane) = base.lanes.get(index as usize).copied() else {
                        return Err(self.expression_error(
                            handle,
                            "only constant vector indexing is supported",
                        ));
                    };
                    Value::scalar(base.kind, lane)
                }
            }
            Expression::Access { base, index } => {
                self.require_vector_index_base(handle, base)?;
                let base = self.lower_expression(base)?;
                let index = self.constant_index(index).ok_or_else(|| {
                    self.expression_error(handle, "dynamic vector indexing is outside ax-shader-v0")
                })?;
                let Some(lane) = base.lanes.get(index).copied() else {
                    return Err(self.expression_error(handle, "vector index is out of range"));
                };
                Value::scalar(base.kind, lane)
            }
            Expression::FunctionArgument(index) => self.frame.arguments[index as usize].clone(),
            Expression::CallResult(_) => self.lower_call(handle)?,
            Expression::Load { pointer } => {
                if let Some(value) = self.uniform_value(pointer) {
                    value
                } else {
                    self.lower_short_circuit_load(handle, pointer)?
                }
            }
            Expression::Unary { op, expr } => self.lower_unary(handle, op, expr)?,
            Expression::Binary { op, left, right } => self.lower_binary(handle, op, left, right)?,
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let condition = self.lower_expression(condition)?;
                let accept = self.lower_expression(accept)?;
                let reject = self.lower_expression(reject)?;
                self.select(handle, condition, accept, reject)?
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.lower_math(handle, fun, arg, arg1, arg2, arg3)?,
            Expression::As {
                expr,
                kind,
                convert: Some(4),
            } => self.lower_cast(handle, expr, kind)?,
            other => {
                return Err(self.expression_error(
                    handle,
                    format!(
                        "unsupported expression in {PROFILE_NAME}: {}",
                        expression_name(&other)
                    ),
                ));
            }
        };
        if value.lanes.is_empty() || value.lanes.len() > 4 {
            return Err(
                self.expression_error(handle, "only scalar and vec2-vec4 values are supported")
            );
        }
        self.frame.expression_cache[handle.index()] = Some(value.clone());
        Ok(value)
    }

    // Resolve only direct uniforms and constant vector-component loads. This
    // does not introduce general pointers, aliasing, or memory operations.
    fn uniform_value(&self, pointer: Handle<Expression>) -> Option<Value> {
        match self.frame.function.expressions[pointer] {
            Expression::GlobalVariable(global) => self.uniforms[global.index()].clone(),
            Expression::AccessIndex { base, index } => {
                let value = self.uniform_value(base)?;
                Some(Value::scalar(value.kind, *value.lanes.get(index as usize)?))
            }
            _ => None,
        }
    }

    fn lower_global_expression(
        &mut self,
        handle: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        if let Some(value) = self.global_cache[handle.index()].clone() {
            return Ok(value);
        }
        let expression = self.module.global_expressions[handle].clone();
        let value = match expression {
            Expression::Literal(literal) => self.lower_global_literal(literal, handle)?,
            Expression::ZeroValue(ty) => self.zero_value_global(ty, handle)?,
            Expression::Compose { ty, components } => {
                let (expected_kind, expected_lanes) = self.value_type(ty).ok_or_else(|| {
                    self.global_type_error(
                        handle,
                        "only f32/i32/bool scalar and vector constants are supported; matrices, arrays, and structs are outside ax-shader-v0",
                    )
                })?;
                let mut lanes = Vec::new();
                let mut kind = None;
                for component in components {
                    let component = self.lower_global_expression(component)?;
                    if kind.is_some_and(|actual| actual != component.kind) {
                        return Err(self.global_expression_error(
                            handle,
                            "mixed constant constructor types are unsupported",
                        ));
                    }
                    kind = Some(component.kind);
                    lanes.extend(component.lanes);
                }
                let kind = kind.ok_or_else(|| {
                    self.global_expression_error(handle, "empty constant constructor")
                })?;
                if kind != expected_kind || lanes.len() != expected_lanes {
                    return Err(self.global_type_error(
                        handle,
                        "constant constructor components do not match the scalar/vector result type",
                    ));
                }
                Value { kind, lanes }
            }
            Expression::Splat { size, value } => {
                let value = self.lower_global_expression(value)?;
                if value.lanes.len() != 1 {
                    return Err(self
                        .global_expression_error(handle, "constant splat source must be scalar"));
                }
                Value {
                    kind: value.kind,
                    lanes: vec![value.lanes[0]; vector_len(size)],
                }
            }
            other => {
                return Err(self.global_expression_error(
                    handle,
                    format!(
                        "unsupported constant expression: {}",
                        expression_name(&other)
                    ),
                ));
            }
        };
        self.global_cache[handle.index()] = Some(value.clone());
        Ok(value)
    }

    fn lower_short_circuit_load(
        &mut self,
        handle: Handle<Expression>,
        pointer: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        let local = local_pointer(self.frame.function, pointer).ok_or_else(|| {
            self.expression_error(
                handle,
                "pointer loads are outside ax-shader-v0 short-circuit expressions",
            )
        })?;
        let index = local.index();
        if let Some(value) = self.frame.short_circuit_cache[index].clone() {
            return Ok(value);
        }
        let short_circuit = self.frame.body.short_circuits[index].ok_or_else(|| {
            self.expression_error(
                handle,
                "mutable local loads are outside ax-shader-v0 short-circuit expressions",
            )
        })?;
        if self.frame.short_circuit_in_progress[index] {
            return Err(self.expression_error(handle, "cyclic short-circuit temporary"));
        }

        self.frame.short_circuit_in_progress[index] = true;
        let lowered = (|| {
            let left = self.lower_expression(short_circuit.left)?;
            let right = self.lower_expression(short_circuit.right)?;
            self.bool_binary(handle, short_circuit.op, left, right)
        })();
        self.frame.short_circuit_in_progress[index] = false;
        let value = lowered?;
        self.frame.short_circuit_cache[index] = Some(value.clone());
        Ok(value)
    }

    fn lower_literal(
        &mut self,
        literal: Literal,
        handle: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        match literal {
            Literal::F32(value) => Ok(Value::scalar(ValueKind::Float, self.f32_constant(value)?)),
            Literal::Bool(value) => Ok(Value::scalar(ValueKind::Bool, self.bool_constant(value))),
            Literal::I32(value) => {
                Ok(Value::scalar(ValueKind::I32, self.i32_constant(value)?))
            }
            _ => Err(self.expression_error(
                handle,
                "only f32/i32/bool literals are supported; u32 values are limited to constant shift counts",
            )),
        }
    }

    fn lower_global_literal(
        &mut self,
        literal: Literal,
        handle: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        match literal {
            Literal::F32(value) => Ok(Value::scalar(ValueKind::Float, self.f32_constant(value)?)),
            Literal::Bool(value) => Ok(Value::scalar(ValueKind::Bool, self.bool_constant(value))),
            Literal::I32(value) => {
                Ok(Value::scalar(ValueKind::I32, self.i32_constant(value)?))
            }
            _ => Err(self.global_expression_error(
                handle,
                "only f32/i32/bool constants are supported; u32 values are limited to constant shift counts",
            )),
        }
    }

    fn zero_value(
        &mut self,
        ty: Handle<naga::Type>,
        handle: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        let (kind, lanes) = self.value_type(ty).ok_or_else(|| {
            self.expression_error(
                handle,
                "zero value must be f32, i32, bool, or a vector thereof",
            )
        })?;
        self.zero_of(kind, lanes)
    }

    fn zero_value_global(
        &mut self,
        ty: Handle<naga::Type>,
        handle: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        let (kind, lanes) = self.value_type(ty).ok_or_else(|| {
            self.global_expression_error(
                handle,
                "zero constant must be f32, i32, bool, or a vector thereof",
            )
        })?;
        self.zero_of(kind, lanes)
    }

    fn value_type(&self, ty: Handle<naga::Type>) -> Option<(ValueKind, usize)> {
        value_type(self.module, ty)
    }

    fn zero_of(&mut self, kind: ValueKind, lanes: usize) -> Result<Value, CompileError> {
        let lane = match kind {
            ValueKind::Float => self.f32_constants[&0.0f32.to_bits()],
            ValueKind::Bool => self.bool_constants[0],
            ValueKind::I32 => self.i32_constant(0)?,
        };
        Ok(Value {
            kind,
            lanes: vec![lane; lanes],
        })
    }

    fn lower_unary(
        &mut self,
        handle: Handle<Expression>,
        operator: UnaryOperator,
        expression: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        let value = self.lower_expression(expression)?;
        match operator {
            UnaryOperator::Negate if value.kind == ValueKind::Float => {
                self.unary(handle, OpCode::NEGATE, value, ValueKind::Float)
            }
            UnaryOperator::Negate if value.kind == ValueKind::I32 => {
                let zero = Value::scalar(ValueKind::I32, self.i32_constant(0)?);
                self.numeric_binary(handle, OpCode::SUB, zero, value)
            }
            UnaryOperator::BitwiseNot if value.kind == ValueKind::I32 => {
                let all_bits = Value::scalar(ValueKind::I32, self.i32_constant(-1)?);
                self.integer_binary(handle, OpCode::BITWISE_XOR, value, all_bits)
            }
            UnaryOperator::LogicalNot if value.kind == ValueKind::Bool => {
                self.unary(handle, OpCode::LOGICAL_NOT, value, ValueKind::Bool)
            }
            _ => Err(self.expression_error(handle, "unsupported unary operator or operand type")),
        }
    }

    fn lower_binary(
        &mut self,
        handle: Handle<Expression>,
        operator: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Result<Value, CompileError> {
        let left = self.lower_expression(left)?;
        use BinaryOperator as B;
        // WGSL shift counts are U32 even for I32 values. Only compile-time
        // counts in 0..32 cross this boundary, as exact I32 numeric constants;
        // no U32 value is mislabeled or silently routed through F32.
        let right = if matches!(operator, B::ShiftLeft | B::ShiftRight) {
            self.lower_shift_counts(right)?
        } else {
            self.lower_expression(right)?
        };
        match operator {
            B::Add => self.numeric_binary(handle, OpCode::ADD, left, right),
            B::Subtract => self.numeric_binary(handle, OpCode::SUB, left, right),
            B::Multiply => self.numeric_binary(handle, OpCode::MUL, left, right),
            B::Divide if self.is_float_one(&left) => {
                self.unary_float(handle, OpCode::RECIPROCAL, right)
            }
            B::Divide => self.numeric_binary(handle, OpCode::DIV, left, right),
            B::Modulo => self.numeric_binary(handle, OpCode::REMAINDER, left, right),
            B::And => self.integer_binary(handle, OpCode::BITWISE_AND, left, right),
            B::InclusiveOr => self.integer_binary(handle, OpCode::BITWISE_OR, left, right),
            B::ExclusiveOr => self.integer_binary(handle, OpCode::BITWISE_XOR, left, right),
            B::ShiftLeft => self.integer_binary(handle, OpCode::SHIFT_LEFT, left, right),
            B::ShiftRight => self.integer_binary(handle, OpCode::SHIFT_RIGHT, left, right),
            B::Equal => self.compare(handle, OpCode::EQUAL, left, right, false),
            B::NotEqual => {
                let equal = self.compare(handle, OpCode::EQUAL, left, right, false)?;
                self.unary(handle, OpCode::LOGICAL_NOT, equal, ValueKind::Bool)
            }
            B::Greater => self.compare(handle, OpCode::GREATER, left, right, false),
            B::GreaterEqual => self.compare(handle, OpCode::GREATER_EQUAL, left, right, false),
            B::Less => self.compare(handle, OpCode::GREATER, left, right, true),
            B::LessEqual => self.compare(handle, OpCode::GREATER_EQUAL, left, right, true),
            B::LogicalAnd => self.bool_binary(handle, OpCode::LOGICAL_AND, left, right),
            B::LogicalOr => self.bool_binary(handle, OpCode::LOGICAL_OR, left, right),
        }
    }

    fn lower_cast(
        &mut self,
        handle: Handle<Expression>,
        expression: Handle<Expression>,
        kind: ScalarKind,
    ) -> Result<Value, CompileError> {
        let value = self.lower_expression(expression)?;
        match (value.kind, kind) {
            (ValueKind::Float, ScalarKind::Float) | (ValueKind::I32, ScalarKind::Sint) => Ok(value),
            (ValueKind::I32, ScalarKind::Float) => {
                self.unary(handle, OpCode::CAST, value, ValueKind::Float)
            }
            (ValueKind::Float, ScalarKind::Sint) => {
                // WGSL bounds the input by the largest F32 representable
                // inside I32, not by i32::MAX rounded up to 2^31. Keep this
                // explicit so generic canonical CAST can retain its own
                // saturating conversion contract and other providers see it.
                let low = Value::scalar(ValueKind::Float, self.f32_constant(-2_147_483_648.0)?);
                let high = Value::scalar(ValueKind::Float, self.f32_constant(2_147_483_520.0)?);
                let clamped = self.float_ternary(handle, OpCode::CLAMP, value, low, high)?;
                self.unary(handle, OpCode::CAST, clamped, ValueKind::I32)
            }
            _ => Err(self.expression_error(handle, "only numeric f32/i32 conversions are supported; bool, u32 and bitcasts are outside the profile")),
        }
    }

    fn lower_shift_counts(&mut self, handle: Handle<Expression>) -> Result<Value, CompileError> {
        let counts = constant_u32_lanes(self.module, &self.frame.function.expressions, handle, 0)
            .filter(|values| !values.is_empty() && values.len() <= 4 && values.iter().all(|&value| value < 32))
            .ok_or_else(|| self.expression_error(handle, "I32 shifts require constant u32 counts in 0..32; dynamic u32 values are unsupported"))?;
        let lanes = counts
            .into_iter()
            .map(|count| self.i32_constant(count as i32))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value {
            kind: ValueKind::I32,
            lanes,
        })
    }

    fn numeric_binary(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        left: Value,
        right: Value,
    ) -> Result<Value, CompileError> {
        if left.kind != right.kind || !matches!(left.kind, ValueKind::Float | ValueKind::I32) {
            return Err(
                self.expression_error(handle, "arithmetic requires matching f32 or i32 operands")
            );
        }
        let kind = left.kind;
        self.binary_lanes(handle, op, left, right, kind)
    }

    fn integer_binary(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        left: Value,
        right: Value,
    ) -> Result<Value, CompileError> {
        if left.kind != ValueKind::I32 || right.kind != ValueKind::I32 {
            return Err(self.expression_error(handle, "bitwise operations require i32 operands"));
        }
        self.binary_lanes(handle, op, left, right, ValueKind::I32)
    }

    fn lower_math(
        &mut self,
        handle: Handle<Expression>,
        function: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        _arg3: Option<Handle<Expression>>,
    ) -> Result<Value, CompileError> {
        use MathFunction as M;
        let first = self.lower_expression(arg)?;
        let unary = match function {
            M::Abs => Some(OpCode::ABS),
            M::Cos => Some(OpCode::COS),
            M::Sin => Some(OpCode::SIN),
            M::Tan => Some(OpCode::TAN),
            M::Tanh => Some(OpCode::TANH),
            M::Acos => Some(OpCode::ACOS),
            M::Asin => Some(OpCode::ASIN),
            M::Atan => Some(OpCode::ATAN),
            M::Ceil => Some(OpCode::CEIL),
            M::Floor => Some(OpCode::FLOOR),
            M::Round => Some(OpCode::ROUND),
            M::Trunc => Some(OpCode::TRUNC),
            M::Exp => Some(OpCode::EXP),
            M::Exp2 => Some(OpCode::EXP2),
            M::Log => Some(OpCode::LOG),
            M::Log2 => Some(OpCode::LOG2),
            M::Sign => Some(OpCode::SIGN),
            M::Sqrt => Some(OpCode::SQRT),
            M::InverseSqrt => Some(OpCode::RSQRT),
            _ => None,
        };
        if let Some(op) = unary {
            return self.unary_float(handle, op, first);
        }

        match function {
            M::Min | M::Max | M::Pow | M::Atan2 => {
                let second = self.math_argument(handle, arg1, 2)?;
                let op = match function {
                    M::Min => OpCode::MINIMUM,
                    M::Max => OpCode::MAXIMUM,
                    M::Pow => OpCode::POW,
                    M::Atan2 => OpCode::ATAN2,
                    _ => unreachable!(),
                };
                self.float_binary(handle, op, first, second)
            }
            M::Clamp => {
                let low = self.math_argument(handle, arg1, 2)?;
                let high = self.math_argument(handle, arg2, 3)?;
                self.float_ternary(handle, OpCode::CLAMP, first, low, high)
            }
            M::Saturate => {
                let zero = Value::scalar(ValueKind::Float, self.f32_constant(0.0)?);
                let one = Value::scalar(ValueKind::Float, self.f32_constant(1.0)?);
                self.float_ternary(handle, OpCode::CLAMP, first, zero, one)
            }
            M::Fma => {
                let second = self.math_argument(handle, arg1, 2)?;
                let third = self.math_argument(handle, arg2, 3)?;
                self.float_ternary(handle, OpCode::FMA, first, second, third)
            }
            M::Fract => {
                let floor = self.unary_float(handle, OpCode::FLOOR, first.clone())?;
                self.float_binary(handle, OpCode::SUB, first, floor)
            }
            M::Mix => {
                let second = self.math_argument(handle, arg1, 2)?;
                let weight = self.math_argument(handle, arg2, 3)?;
                let delta = self.float_binary(handle, OpCode::SUB, second, first.clone())?;
                self.float_ternary(handle, OpCode::FMA, delta, weight, first)
            }
            M::Step => {
                let value = self.math_argument(handle, arg1, 2)?;
                let condition = self.compare(handle, OpCode::GREATER_EQUAL, value, first, false)?;
                let zero = Value::scalar(ValueKind::Float, self.f32_constant(0.0)?);
                let one = Value::scalar(ValueKind::Float, self.f32_constant(1.0)?);
                self.select(handle, condition, one, zero)
            }
            M::SmoothStep => {
                let edge1 = self.math_argument(handle, arg1, 2)?;
                let x = self.math_argument(handle, arg2, 3)?;
                let numerator = self.float_binary(handle, OpCode::SUB, x, first.clone())?;
                let denominator = self.float_binary(handle, OpCode::SUB, edge1, first)?;
                let ratio = self.float_binary(handle, OpCode::DIV, numerator, denominator)?;
                let zero = Value::scalar(ValueKind::Float, self.f32_constant(0.0)?);
                let one = Value::scalar(ValueKind::Float, self.f32_constant(1.0)?);
                let t = self.float_ternary(handle, OpCode::CLAMP, ratio, zero, one)?;
                let tt = self.float_binary(handle, OpCode::MUL, t.clone(), t.clone())?;
                let two = Value::scalar(ValueKind::Float, self.f32_constant(2.0)?);
                let three = Value::scalar(ValueKind::Float, self.f32_constant(3.0)?);
                let two_t = self.float_binary(handle, OpCode::MUL, two, t)?;
                let factor = self.float_binary(handle, OpCode::SUB, three, two_t)?;
                self.float_binary(handle, OpCode::MUL, tt, factor)
            }
            M::Dot => {
                let second = self.math_argument(handle, arg1, 2)?;
                self.dot(handle, first, second)
            }
            M::Length => {
                if first.lanes.len() == 1 {
                    self.unary_float(handle, OpCode::ABS, first)
                } else {
                    let square = self.dot(handle, first.clone(), first)?;
                    self.unary_float(handle, OpCode::SQRT, square)
                }
            }
            M::Distance => {
                let second = self.math_argument(handle, arg1, 2)?;
                let delta = self.float_binary(handle, OpCode::SUB, first, second)?;
                if delta.lanes.len() == 1 {
                    self.unary_float(handle, OpCode::ABS, delta)
                } else {
                    let square = self.dot(handle, delta.clone(), delta)?;
                    self.unary_float(handle, OpCode::SQRT, square)
                }
            }
            M::Normalize => {
                let square = self.dot(handle, first.clone(), first.clone())?;
                let inverse = self.unary_float(handle, OpCode::RSQRT, square)?;
                self.float_binary(handle, OpCode::MUL, first, inverse)
            }
            M::Radians | M::Degrees => {
                let factor = if function == M::Radians {
                    core::f32::consts::PI / 180.0
                } else {
                    180.0 / core::f32::consts::PI
                };
                let factor = Value::scalar(ValueKind::Float, self.f32_constant(factor)?);
                self.float_binary(handle, OpCode::MUL, first, factor)
            }
            _ => Err(self.expression_error(
                handle,
                format!("WGSL math function `{function:?}` is outside {PROFILE_NAME}"),
            )),
        }
    }

    fn math_argument(
        &mut self,
        handle: Handle<Expression>,
        argument: Option<Handle<Expression>>,
        position: usize,
    ) -> Result<Value, CompileError> {
        let argument = argument.ok_or_else(|| {
            self.expression_error(
                handle,
                format!("math function is missing argument {position}"),
            )
        })?;
        self.lower_expression(argument)
    }

    fn unary_float(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        value: Value,
    ) -> Result<Value, CompileError> {
        if value.kind != ValueKind::Float {
            return Err(self.expression_error(handle, "math functions require f32 operands"));
        }
        self.unary(handle, op, value, ValueKind::Float)
    }

    fn unary(
        &mut self,
        _handle: Handle<Expression>,
        op: OpCode,
        value: Value,
        output_kind: ValueKind,
    ) -> Result<Value, CompileError> {
        let lanes = value
            .lanes
            .into_iter()
            .map(|lane| emit_node_raw(&mut self.builder, op, &[lane], output_kind))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value {
            kind: output_kind,
            lanes,
        })
    }

    fn float_binary(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        left: Value,
        right: Value,
    ) -> Result<Value, CompileError> {
        if left.kind != ValueKind::Float || right.kind != ValueKind::Float {
            return Err(self.expression_error(handle, "arithmetic operators require f32 operands"));
        }
        self.binary_lanes(handle, op, left, right, ValueKind::Float)
    }

    fn bool_binary(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        left: Value,
        right: Value,
    ) -> Result<Value, CompileError> {
        if left.kind != ValueKind::Bool || right.kind != ValueKind::Bool {
            return Err(self.expression_error(handle, "logical operators require bool operands"));
        }
        self.binary_lanes(handle, op, left, right, ValueKind::Bool)
    }

    fn compare(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        mut left: Value,
        mut right: Value,
        reverse: bool,
    ) -> Result<Value, CompileError> {
        if left.kind != right.kind || !matches!(left.kind, ValueKind::Float | ValueKind::I32) {
            return Err(self.expression_error(
                handle,
                "comparisons require matching f32 or admitted i32 operands",
            ));
        }
        if reverse {
            core::mem::swap(&mut left, &mut right);
        }
        self.binary_lanes(handle, op, left, right, ValueKind::Bool)
    }

    fn binary_lanes(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        left: Value,
        right: Value,
        output_kind: ValueKind,
    ) -> Result<Value, CompileError> {
        let lanes = compatible_lane_count(left.lanes.len(), right.lanes.len())
            .ok_or_else(|| self.expression_error(handle, "vector widths do not match"))?;
        let mut output = Vec::with_capacity(lanes);
        for index in 0..lanes {
            let left = broadcast_lane(&left.lanes, index);
            let right = broadcast_lane(&right.lanes, index);
            output.push(emit_node_raw(
                &mut self.builder,
                op,
                &[left, right],
                output_kind,
            )?);
        }
        Ok(Value {
            kind: output_kind,
            lanes: output,
        })
    }

    fn float_ternary(
        &mut self,
        handle: Handle<Expression>,
        op: OpCode,
        first: Value,
        second: Value,
        third: Value,
    ) -> Result<Value, CompileError> {
        if first.kind != ValueKind::Float
            || second.kind != ValueKind::Float
            || third.kind != ValueKind::Float
        {
            return Err(self.expression_error(handle, "math functions require f32 operands"));
        }
        let lanes = compatible_lane_count(first.lanes.len(), second.lanes.len())
            .and_then(|count| compatible_lane_count(count, third.lanes.len()))
            .ok_or_else(|| self.expression_error(handle, "vector widths do not match"))?;
        let mut output = Vec::with_capacity(lanes);
        for index in 0..lanes {
            output.push(emit_node_raw(
                &mut self.builder,
                op,
                &[
                    broadcast_lane(&first.lanes, index),
                    broadcast_lane(&second.lanes, index),
                    broadcast_lane(&third.lanes, index),
                ],
                ValueKind::Float,
            )?);
        }
        Ok(Value {
            kind: ValueKind::Float,
            lanes: output,
        })
    }

    fn select(
        &mut self,
        handle: Handle<Expression>,
        condition: Value,
        accept: Value,
        reject: Value,
    ) -> Result<Value, CompileError> {
        if condition.kind != ValueKind::Bool || accept.kind != reject.kind {
            return Err(self.expression_error(
                handle,
                "select requires a bool condition and matching values",
            ));
        }
        let lanes = compatible_lane_count(condition.lanes.len(), accept.lanes.len())
            .and_then(|count| compatible_lane_count(count, reject.lanes.len()))
            .ok_or_else(|| self.expression_error(handle, "select vector widths do not match"))?;
        if accept.kind == ValueKind::Bool {
            // The native graph/provider boolean ABI already supports logical
            // masks, while bool-valued SELECT is not universally admitted.
            let inverse = self.unary(
                handle,
                OpCode::LOGICAL_NOT,
                condition.clone(),
                ValueKind::Bool,
            )?;
            let accept = self.bool_binary(handle, OpCode::LOGICAL_AND, condition, accept)?;
            let reject = self.bool_binary(handle, OpCode::LOGICAL_AND, inverse, reject)?;
            return self.bool_binary(handle, OpCode::LOGICAL_OR, accept, reject);
        }
        let mut output = Vec::with_capacity(lanes);
        for index in 0..lanes {
            output.push(emit_node_raw(
                &mut self.builder,
                OpCode::SELECT,
                &[
                    broadcast_lane(&condition.lanes, index),
                    broadcast_lane(&accept.lanes, index),
                    broadcast_lane(&reject.lanes, index),
                ],
                accept.kind,
            )?);
        }
        Ok(Value {
            kind: accept.kind,
            lanes: output,
        })
    }

    fn dot(
        &mut self,
        handle: Handle<Expression>,
        left: Value,
        right: Value,
    ) -> Result<Value, CompileError> {
        if left.kind != ValueKind::Float
            || right.kind != ValueKind::Float
            || left.lanes.len() != right.lanes.len()
            || left.lanes.len() < 2
        {
            return Err(self.expression_error(handle, "dot requires equal-width f32 vectors"));
        }
        let mut terms = Vec::with_capacity(left.lanes.len());
        for (left, right) in left.lanes.into_iter().zip(right.lanes) {
            terms.push(emit_node_raw(
                &mut self.builder,
                OpCode::MUL,
                &[left, right],
                ValueKind::Float,
            )?);
        }
        let mut sum = terms[0];
        for term in terms.into_iter().skip(1) {
            sum = emit_node_raw(
                &mut self.builder,
                OpCode::ADD,
                &[sum, term],
                ValueKind::Float,
            )?;
        }
        Ok(Value::scalar(ValueKind::Float, sum))
    }

    fn emit_outputs(&mut self, colour: Value) -> Result<(), CompileError> {
        if colour.kind != ValueKind::Float || colour.lanes.len() != 4 {
            return Err(profile_unspanned(
                ProfileRule::Interface,
                "fragment return expression must lower to vec4<f32>",
            ));
        }
        for lane in colour.lanes {
            self.emit_output(lane, ValueKind::Float)?;
        }
        Ok(())
    }

    fn emit_output(&mut self, lane: Lane, kind: ValueKind) -> Result<(), CompileError> {
        let output = self.builder.tensor(
            kind.dtype(),
            TensorFlags::OUTPUT,
            self.domain.shape.dimensions(),
            &[],
        )?;
        if lane.shape == self.domain.shape {
            self.builder.node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[lane.tensor],
                &[output],
                &[],
            )?;
        } else if kind == ValueKind::Float {
            self.builder.node(
                OpCode::ADD,
                NodeFlags::default(),
                &[self.domain.tensor, lane.tensor],
                &[output],
                &[],
            )?;
        } else if kind == ValueKind::I32 {
            let mask = match self.i32_output_mask {
                Some(mask) => mask,
                None => {
                    let mask = emit_node_raw(
                        &mut self.builder,
                        OpCode::EQUAL,
                        &[self.domain, self.domain],
                        ValueKind::Bool,
                    )?;
                    self.i32_output_mask = Some(mask);
                    mask
                }
            };
            // Both arms are the same exact integer. The full-field mask only
            // supplies broadcast geometry; even NaN position inputs cannot
            // alter the selected bits or route an ID through F32.
            self.builder.node(
                OpCode::SELECT,
                NodeFlags::default(),
                &[mask.tensor, lane.tensor, lane.tensor],
                &[output],
                &[],
            )?;
        } else {
            return Err(profile_unspanned(
                ProfileRule::Type,
                "non-F32 output fields must already have full viewport shape",
            ));
        }
        Ok(())
    }

    fn emit_fragment_outputs(&mut self, returned: usize) -> Result<(), CompileError> {
        let ReturnValue::Expression(expression) = self.frame.body.returns[returned] else {
            return Err(profile_unspanned(
                ProfileRule::Interface,
                "fragment result structs require a direct constructor or immutable let alias; conditional record returns are unsupported",
            ));
        };
        let Expression::Compose { ty, components } =
            self.frame.function.expressions[expression].clone()
        else {
            return Err(self.type_error(
                expression,
                "fragment result structs require a direct constructor or immutable let alias; general struct operations are unsupported",
            ));
        };
        let TypeInner::Struct { members, .. } = &self.module.types[ty].inner else {
            return Err(self.type_error(expression, "fragment result must be a struct constructor"));
        };
        if self.frame.function.result.as_ref().map(|result| result.ty) != Some(ty)
            || members.len() != components.len()
        {
            return Err(self.type_error(expression, "fragment result constructor type mismatch"));
        }
        // Lower in source constructor order, then emit in location order. This
        // keeps the shared expression cache (including queried hits) intact.
        let members: Vec<_> = members
            .iter()
            .zip(components)
            .map(|(member, component)| {
                let Some(Binding::Location { location, .. }) = member.binding else {
                    unreachable!("fragment member locations were validated");
                };
                (location, member.ty, component)
            })
            .collect();
        let mut values = Vec::with_capacity(members.len());
        for (location, ty, component) in members {
            let value = self.lower_expression(component)?;
            if fragment_member_type(self.module, ty) != Some((value.kind, value.lanes.len())) {
                return Err(self.type_error(component, "fragment result member type mismatch"));
            }
            values.push((location, value));
        }
        values.sort_by_key(|(location, _)| *location);

        let expanded_constants = values
            .iter()
            .filter(|(_, value)| {
                value.kind == ValueKind::I32
                    && value.lanes[0].shape != self.domain.shape
                    && self.i32_literal_value(value.lanes[0]).is_some()
            })
            .count();
        let plane_bytes = if expanded_constants == 0 {
            0
        } else {
            let bytes = self.domain.shape.0[0] as usize;
            bytes
                .checked_mul(self.domain.shape.0[1] as usize)
                .and_then(|elements| elements.checked_mul(size_of::<i32>()))
                .filter(|bytes| {
                    bytes
                        .checked_mul(expanded_constants)
                        .is_some_and(|total| total <= MAX_FRAGMENT_CONSTANT_OUTPUT_BYTES)
                })
                .ok_or_else(|| {
                    profile_unspanned(
                        ProfileRule::Budget,
                        "expanded i32 fragment constants exceed the 64 MiB output payload budget",
                    )
                })?
        };
        for (_, value) in values {
            for lane in value.lanes {
                if value.kind == ValueKind::I32
                    && lane.shape != self.domain.shape
                    && self.i32_literal_value(lane).is_some()
                {
                    self.emit_i32_constant_output(lane, plane_bytes)?;
                } else {
                    self.emit_output(lane, value.kind)?;
                }
            }
        }
        Ok(())
    }

    fn i32_literal_value(&self, lane: Lane) -> Option<i32> {
        self.i32_constants
            .iter()
            .find_map(|(&value, constant)| (constant.tensor == lane.tensor).then_some(value))
            .filter(|_| lane.shape == Shape::SCALAR)
    }

    fn emit_i32_constant_output(&mut self, lane: Lane, bytes: usize) -> Result<(), CompileError> {
        let value = self.i32_literal_value(lane).ok_or_else(|| {
            profile_unspanned(
                ProfileRule::Type,
                "only scalar i32 literals/constants can be broadcast to fragment outputs",
            )
        })?;
        // Native I32 identity is shape-preserving, not an implicit conversion
        // or broadcast. Serialize bounded, exact constant output planes instead.
        let mut constant = Vec::new();
        constant
            .try_reserve_exact(bytes)
            .map_err(|_| AllocationError::AllocationFailed)?;
        for _ in 0..bytes / size_of::<i32>() {
            constant.extend_from_slice(&value.to_le_bytes());
        }
        self.builder.tensor(
            DType::I32,
            TensorFlags::CONSTANT | TensorFlags::OUTPUT,
            self.domain.shape.dimensions(),
            &constant,
        )?;
        Ok(())
    }

    fn emit_hit_outputs(&mut self, returned: usize) -> Result<(), CompileError> {
        let ReturnValue::Expression(expression) = self.frame.body.returns[returned] else {
            return Err(profile_unspanned(
                ProfileRule::Interface,
                "AxVoxelHit outputs must return a query directly; record selection is unsupported",
            ));
        };
        let hit = self.lower_hit(expression)?;
        self.emit_output(hit.t, ValueKind::Float)?;
        self.emit_output(hit.cell_id, ValueKind::I32)?;
        Ok(())
    }

    fn f32_constant(&mut self, value: f32) -> Result<Lane, CompileError> {
        if let Some(lane) = self.f32_constants.get(&value.to_bits()).copied() {
            return Ok(lane);
        }
        let lane = Lane {
            tensor: self.builder.tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &value.to_le_bytes(),
            )?,
            shape: Shape::SCALAR,
        };
        self.f32_constants.insert(value.to_bits(), lane);
        Ok(lane)
    }

    fn i32_constant(&mut self, value: i32) -> Result<Lane, CompileError> {
        if let Some(lane) = self.i32_constants.get(&value).copied() {
            return Ok(lane);
        }
        let lane = Lane {
            tensor: self.builder.tensor(
                DType::I32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &value.to_le_bytes(),
            )?,
            shape: Shape::SCALAR,
        };
        self.i32_constants.insert(value, lane);
        Ok(lane)
    }

    fn bool_constant(&self, value: bool) -> Lane {
        self.bool_constants[usize::from(value)]
    }

    fn is_float_one(&self, value: &Value) -> bool {
        let one = self.f32_constants[&1.0f32.to_bits()].tensor;
        value.kind == ValueKind::Float
            && !value.lanes.is_empty()
            && value.lanes.iter().all(|lane| lane.tensor == one)
    }

    fn constant_index(&self, handle: Handle<Expression>) -> Option<usize> {
        match self.frame.function.expressions[handle] {
            Expression::Literal(Literal::U32(value)) => Some(value as usize),
            Expression::Literal(Literal::I32(value)) => usize::try_from(value).ok(),
            _ => None,
        }
    }

    fn expression_error(
        &self,
        handle: Handle<Expression>,
        message: impl Into<String>,
    ) -> CompileError {
        profile(
            self.source,
            ProfileRule::Expression,
            message,
            self.frame.function.expressions.get_span(handle),
        )
    }

    fn type_error(&self, handle: Handle<Expression>, message: impl Into<String>) -> CompileError {
        profile(
            self.source,
            ProfileRule::Type,
            message,
            self.frame.function.expressions.get_span(handle),
        )
    }

    fn global_type_error(
        &self,
        handle: Handle<Expression>,
        message: impl Into<String>,
    ) -> CompileError {
        profile(
            self.source,
            ProfileRule::Type,
            message,
            self.module.global_expressions.get_span(handle),
        )
    }

    fn require_vector_index_base(
        &self,
        handle: Handle<Expression>,
        base: Handle<Expression>,
    ) -> Result<(), CompileError> {
        if matches!(
            self.frame.function_info[base]
                .ty
                .inner_with(&self.module.types),
            TypeInner::Vector { .. }
        ) {
            Ok(())
        } else {
            Err(self.type_error(
                handle,
                "only vector indexing is supported; matrix, array, and struct indexing is outside ax-shader-v0",
            ))
        }
    }

    fn global_expression_error(
        &self,
        handle: Handle<Expression>,
        message: impl Into<String>,
    ) -> CompileError {
        profile(
            self.source,
            ProfileRule::Resource,
            message,
            self.module.global_expressions.get_span(handle),
        )
    }
}

fn emit_node_raw<A: Allocator + Clone>(
    builder: &mut ShaderBuilder<A>,
    op: OpCode,
    inputs: &[Lane],
    output_kind: ValueKind,
) -> Result<Lane, CompileError> {
    let shape = inputs
        .iter()
        .fold(Shape::SCALAR, |shape, input| shape.merge(input.shape));
    let output = builder.tensor(
        output_kind.dtype(),
        TensorFlags::default(),
        shape.dimensions(),
        &[],
    )?;
    let input_ids: Vec<_> = inputs.iter().map(|lane| lane.tensor).collect();
    builder.node(op, NodeFlags::default(), &input_ids, &[output], &[])?;
    Ok(Lane {
        tensor: output,
        shape,
    })
}

fn compatible_lane_count(left: usize, right: usize) -> Option<usize> {
    if left == right {
        Some(left)
    } else if left == 1 {
        Some(right)
    } else if right == 1 {
        Some(left)
    } else {
        None
    }
}

fn broadcast_lane(lanes: &[Lane], index: usize) -> Lane {
    lanes[if lanes.len() == 1 { 0 } else { index }]
}

fn vector_len(size: VectorSize) -> usize {
    u8::from(size) as usize
}

/// The only unsigned expressions admitted by this profile are constant shift
/// counts. Naga has already validated their type and constant arithmetic;
/// inspect only its folded literals/vector construction without evaluating
/// general U32 expressions or widening the tensor ABI.
fn constant_u32_lanes(
    module: &Module,
    expressions: &naga::Arena<Expression>,
    handle: Handle<Expression>,
    depth: usize,
) -> Option<Vec<u32>> {
    if depth >= MAX_EXPRESSION_DEPTH {
        return None;
    }
    let recurse = |handle| constant_u32_lanes(module, expressions, handle, depth + 1);
    match &expressions[handle] {
        Expression::Literal(Literal::U32(value)) => Some(vec![*value]),
        Expression::Constant(constant) => constant_u32_lanes(
            module,
            &module.global_expressions,
            module.constants[*constant].init,
            depth + 1,
        ),
        Expression::ZeroValue(ty) => match module.types[*ty].inner {
            TypeInner::Scalar(scalar) if scalar.kind == ScalarKind::Uint && scalar.width == 4 => {
                Some(vec![0])
            }
            TypeInner::Vector { size, scalar }
                if scalar.kind == ScalarKind::Uint && scalar.width == 4 =>
            {
                Some(vec![0; vector_len(size)])
            }
            _ => None,
        },
        Expression::Splat { size, value } => {
            let values = recurse(*value)?;
            (values.len() == 1).then(|| vec![values[0]; vector_len(*size)])
        }
        Expression::Compose { ty, components } => {
            let TypeInner::Vector { size, scalar } = module.types[*ty].inner else {
                return None;
            };
            if scalar.kind != ScalarKind::Uint || scalar.width != 4 {
                return None;
            }
            let mut values = Vec::with_capacity(vector_len(size));
            for &component in components {
                values.extend(recurse(component)?);
                if values.len() > vector_len(size) {
                    return None;
                }
            }
            (values.len() == vector_len(size)).then_some(values)
        }
        Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let values = recurse(*vector)?;
            pattern[..vector_len(*size)]
                .iter()
                .map(|&component| values.get(swizzle_index(component)).copied())
                .collect()
        }
        Expression::AccessIndex { base, index } => {
            let values = recurse(*base)?;
            Some(vec![*values.get(*index as usize)?])
        }
        _ => None,
    }
}

fn swizzle_index(component: SwizzleComponent) -> usize {
    match component {
        SwizzleComponent::X => 0,
        SwizzleComponent::Y => 1,
        SwizzleComponent::Z => 2,
        SwizzleComponent::W => 3,
    }
}

fn expression_name(expression: &Expression) -> &'static str {
    match expression {
        Expression::Override(_) => "pipeline override",
        Expression::GlobalVariable(_) => "global variable/resource",
        Expression::LocalVariable(_) | Expression::Load { .. } => "mutable/pointer load",
        Expression::ImageSample { .. }
        | Expression::ImageLoad { .. }
        | Expression::ImageQuery { .. } => "texture/image operation",
        Expression::Derivative { .. } => "derivative",
        Expression::CallResult(_) => "helper function call",
        Expression::As { .. } => "cast or bitcast",
        _ => "expression",
    }
}

#[cfg(test)]
mod tests {
    use core::ptr::NonNull;
    use std::{
        alloc::{AllocError, Global, Layout},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use super::*;

    const PROCEDURAL: &str = r#"
        const CENTER: vec2<f32> = vec2<f32>(320.0, 240.0);
        const WAVE_SCALE: f32 = 0.05;

        @fragment
        fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
            let centered = p.xy - CENTER;
            let radius = sqrt(pow(centered.x, 2.0) + pow(centered.y, 2.0));
            let wave = 0.5 + 0.5 * cos(radius * WAVE_SCALE);
            let warm = clamp(wave, 0.0, 1.0);
            let inside = radius < 120.0;
            return vec4<f32>(
                warm,
                sin(p.x * 0.01) * 0.25 + 0.5,
                abs(cos(p.y * 0.01)),
                select(0.2, 1.0, inside),
            );
        }
    "#;

    fn test_options() -> CompileOptions<'static> {
        CompileOptions::for_viewport("main", 8, 6)
    }

    fn fragment_struct_source(fields: &str, body: &str) -> String {
        format!(
            "struct Result {{ {fields} }}\n@fragment fn main(@builtin(position) p: vec4<f32>) -> Result {{ {body} }}"
        )
    }

    #[test]
    fn integer_helpers_vectors_branches_and_hash_keep_typed_semantics() {
        let source = r#"
            const SHIFTS = vec3<u32>(1u, 15u, 16u);
            fn hash(x: i32) -> i32 {
                let a = (x ^ ((x >> 16u) & 65535i)) * 2146121005i;
                let b = (a ^ ((a >> 15u) & 131071i)) * -2073254261i;
                let h = b ^ ((b >> 16u) & 65535i);
                let low = (h & 2147483647i) % 101i;
                return select(low, (low + 34i) % 101i, h < 0i);
            }
            fn lanes(v: vec3<i32>) -> vec3<i32> {
                let a = (v + vec3<i32>(2i)) * 3i - v / vec3<i32>(3i, 5i, 7i);
                let mixed = (~a & vec3<i32>(65535i)) ^ ((a << SHIFTS) | (a >> SHIFTS.zyx));
                if v.x < 0i { return -mixed; }
                return mixed % vec3<i32>(101i);
            }
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let id = i32(p.x) + 16777217i;
                let value = lanes(vec3<i32>(id, i32(p.y), hash(id)));
                return vec4<f32>(vec3<f32>(value), f32(hash(hash(id))));
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        for op in [
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
            OpCode::SELECT,
        ] {
            assert!(
                (0..graph.node_count())
                    .map(|i| graph.node(i).unwrap())
                    .any(|node| {
                        node.op == op
                            && graph.tensor(node.outputs().next().unwrap()).unwrap().dtype
                                == DType::I32
                    }),
                "missing typed operation {op:?}"
            );
        }
        for index in 0..graph.node_count() {
            assert!(ax::node_kernel_request(&graph, graph.node(index).unwrap()).is_some());
        }
        assert!(
            (0..graph.tensor_count())
                .map(|i| graph.tensor(TensorId(i as u32)).unwrap())
                .any(|tensor| tensor.dtype == DType::I32
                    && tensor.constant() == 16777217i32.to_le_bytes())
        );
    }

    #[test]
    fn integer_casts_expose_wgsl_float_bounds_before_canonical_cast() {
        let source = fragment_struct_source(
            "@location(0) @interpolate(flat) id: i32, @location(1) value: f32,",
            "let id = i32(p.x); return Result(id + 16777217i, f32(id));",
        );
        let compiled = compile_wgsl_in(&source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        let mut integer_casts = 0;
        let mut float_casts = 0;
        for node in (0..graph.node_count()).map(|i| graph.node(i).unwrap()) {
            if node.op != OpCode::CAST {
                continue;
            }
            let input = graph.tensor(node.inputs().next().unwrap()).unwrap();
            let output = graph.tensor(node.outputs().next().unwrap()).unwrap();
            match (input.dtype, output.dtype) {
                (DType::F32, DType::I32) => {
                    integer_casts += 1;
                    let clamp = (0..graph.node_count())
                        .map(|i| graph.node(i).unwrap())
                        .find(|producer| producer.outputs().any(|id| id == input.id))
                        .unwrap();
                    assert_eq!(clamp.op, OpCode::CLAMP);
                    let bounds: Vec<_> = clamp
                        .inputs()
                        .skip(1)
                        .map(|id| {
                            f32::from_le_bytes(
                                graph.tensor(id).unwrap().constant().try_into().unwrap(),
                            )
                        })
                        .collect();
                    assert_eq!(bounds, [-2_147_483_648.0_f32, 2_147_483_520.0_f32]);
                }
                (DType::I32, DType::F32) => float_casts += 1,
                _ => panic!("unexpected cast types"),
            }
        }
        assert_eq!((integer_casts, float_casts), (1, 1));
        assert_eq!(compiled.interface().output_dtype(0), Some(DType::I32));
    }

    #[test]
    fn integer_scalar_and_axis_outputs_broadcast_without_constant_plane_expansion() {
        for value in [
            "i32(p.x) + 1i",
            "i32(scale) + 1i",
            "select(16777217i, -1i, scale > 0.0)",
        ] {
            let source = format!(
                "@group(0) @binding(0) var<uniform> scale: f32;\n{}",
                fragment_struct_source(
                    "@location(0) @interpolate(flat) id: i32,",
                    &format!("return Result({value});")
                )
            );
            // One literal I32 plane at this size exceeds the existing 64 MiB
            // constant budget; computed broadcasts must serialize only nodes.
            let compiled = compile_wgsl_in(
                &source,
                CompileOptions::for_viewport("main", 8192, 4096),
                Global,
            )
            .unwrap();
            assert!(compiled.graph().as_bytes().len() < 8192);
            let graph = compiled.graph().parse().unwrap();
            let output = (0..graph.tensor_count())
                .map(|i| graph.tensor(TensorId(i as u32)).unwrap())
                .find(|tensor| tensor.flags.contains(TensorFlags::OUTPUT))
                .unwrap();
            assert_eq!(output.dtype, DType::I32);
            assert!(output.dimensions().eq([4096, 8192]));
            assert!(!output.flags.contains(TensorFlags::CONSTANT));
            let selected = (0..graph.node_count())
                .map(|i| graph.node(i).unwrap())
                .find(|node| node.outputs().any(|id| id == output.id))
                .unwrap();
            assert_eq!(selected.op, OpCode::SELECT);
            let inputs: Vec<_> = selected.inputs().collect();
            assert_eq!(inputs[1], inputs[2]);
        }
    }

    #[test]
    fn integer_support_keeps_unsigned_bitcasts_and_ingress_outside_the_profile() {
        for (globals, body) in [
            (
                "",
                "let count = u32(p.x); return vec4<f32>(f32(i32(p.y) >> count));",
            ),
            ("", "return vec4<f32>(f32(u32(p.x) + 1u));"),
            ("", "return vec4<f32>(f32(bitcast<i32>(p.x)));"),
            (
                "fn unsigned(x: u32) -> u32 { return x; }",
                "return vec4<f32>(f32(unsigned(1u)));",
            ),
            (
                "@group(0) @binding(0) var<uniform> integer: i32;",
                "return vec4<f32>(f32(integer));",
            ),
        ] {
            let source = format!(
                "{globals}\n@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{ {body} }}"
            );
            let CompileError::Profile(error) =
                compile_wgsl_in(&source, test_options(), Global).unwrap_err()
            else {
                panic!("expected a bounded profile diagnostic");
            };
            assert!(error.to_string().len() < 4096);
        }
    }

    #[test]
    fn fragment_struct_locations_define_typed_output_order_and_aliases() {
        let source = fragment_struct_source(
            "@location(2) xy: vec2<f32>, @location(0) sum: f32, @location(1) @interpolate(flat) id: i32,",
            "let first = Result(p.xy, p.x + p.y, 16777217i); let shared_result = first; let other = Result(vec2<f32>(99.0), 99.0, 16777218i); let returned = shared_result; return returned;",
        );
        let compiled = compile_wgsl_in(&source, test_options(), Global).unwrap();
        let interface = compiled.interface();
        assert!(matches!(
            interface.output_kind,
            ShaderOutputKind::FragmentStruct { .. }
        ));
        assert_eq!(interface.output_count(), 4);
        let expected = [
            (0, 0, DType::F32),
            (1, 0, DType::I32),
            (2, 0, DType::F32),
            (2, 1, DType::F32),
        ];
        for (index, (location, component, dtype)) in expected.into_iter().enumerate() {
            assert_eq!(
                interface.output_lane(index),
                Some(FragmentOutputLane {
                    location,
                    component,
                    dtype
                })
            );
            assert_eq!(interface.output_dtype(index), Some(dtype));
        }
        assert_eq!(interface.output_lane(4), None);
        assert_eq!(interface.output_dtype(usize::MAX), None);
        let graph = compiled.graph().parse().unwrap();
        let outputs: Vec<_> = (0..graph.tensor_count())
            .map(|index| graph.tensor(TensorId(index as u32)).unwrap())
            .filter(|tensor| tensor.flags.contains(TensorFlags::OUTPUT))
            .collect();
        assert_eq!(
            outputs
                .iter()
                .map(|tensor| tensor.dtype)
                .collect::<Vec<_>>(),
            [DType::F32, DType::I32, DType::F32, DType::F32]
        );
        assert!(outputs[1].flags.contains(TensorFlags::CONSTANT));
    }

    #[test]
    fn fragment_struct_limits_flattened_lanes_and_constant_payload() {
        let fields = (0..4)
            .map(|location| format!("@location({location}) field_{location}: vec4<f32>,"))
            .collect::<String>();
        let values = ["p"; 4].join(",");
        let source = fragment_struct_source(&fields, &format!("return Result({values});"));
        let compiled = compile_wgsl_in(&source, test_options(), Global).unwrap();
        assert_eq!(
            compiled.interface().output_count(),
            MAX_FRAGMENT_OUTPUT_LANES
        );
        assert_eq!(
            compiled.interface().output_lane(MAX_FRAGMENT_OUTPUT_LANES),
            None
        );
        let source = fragment_struct_source(
            &format!("{fields} @location(4) extra: f32,"),
            &format!("return Result({values}, 0.0);"),
        );
        let CompileError::Profile(diagnostic) =
            compile_wgsl_in(&source, test_options(), Global).unwrap_err()
        else {
            panic!("expected a flattened output budget diagnostic");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Budget);

        for fields in [
            "@location(0) @interpolate(flat) id: i32,",
            "@location(0) @interpolate(flat) id: i32, @location(1) @interpolate(flat) other: i32,",
        ] {
            let two = fields.contains("other");
            let source = fragment_struct_source(
                fields,
                if two {
                    "return Result(1i, 2i);"
                } else {
                    "return Result(1i);"
                },
            );
            let width = if two { 8_388_609 } else { 16_777_217 };
            let error = compile_wgsl_in(
                &source,
                CompileOptions::for_viewport("main", width, 1),
                Global,
            )
            .unwrap_err();
            let CompileError::Profile(diagnostic) = error else {
                panic!("expected a constant payload budget diagnostic");
            };
            assert_eq!(diagnostic.rule, ProfileRule::Budget);
            assert!(diagnostic.message.contains("64 MiB"));
        }
    }

    #[test]
    fn fragment_struct_rejects_unsupported_field_types_and_bindings() {
        for (name, prefix, fields, value) in [
            ("bool", "", "@location(0) value: bool,", "true"),
            (
                "u32",
                "",
                "@location(0) @interpolate(flat) value: u32,",
                "1u",
            ),
            (
                "i32 vector",
                "",
                "@location(0) @interpolate(flat) value: vec2<i32>,",
                "vec2<i32>(1i)",
            ),
            (
                "u32 vector",
                "",
                "@location(0) @interpolate(flat) value: vec2<u32>,",
                "vec2<u32>(1u)",
            ),
            ("f16", "enable f16;", "@location(0) value: f16,", "f16(1.0)"),
            (
                "matrix",
                "",
                "@location(0) value: mat2x2<f32>,",
                "mat2x2<f32>(vec2<f32>(1.0), vec2<f32>(2.0))",
            ),
            (
                "array",
                "",
                "@location(0) value: array<f32, 2>,",
                "array<f32, 2>(1.0, 2.0)",
            ),
            (
                "nested struct",
                "struct Nested { value: f32 }",
                "@location(0) value: Nested,",
                "Nested(1.0)",
            ),
            ("builtin", "", "@builtin(frag_depth) value: f32,", "1.0"),
            ("missing location", "", "value: f32,", "1.0"),
            ("location gap", "", "@location(1) value: f32,", "1.0"),
            (
                "location overflow",
                "",
                "@location(4294967295) value: f32,",
                "1.0",
            ),
            (
                "duplicate location",
                "",
                "@location(0) value: f32, @location(0) other: f32,",
                "1.0, 2.0",
            ),
            (
                "dual-source blending",
                "enable dual_source_blending;",
                "@location(0) @blend_src(0) value: f32,",
                "1.0",
            ),
            (
                "linear interpolation",
                "",
                "@location(0) @interpolate(linear) value: f32,",
                "1.0",
            ),
            (
                "centroid sampling",
                "",
                "@location(0) @interpolate(perspective, centroid) value: f32,",
                "1.0",
            ),
            (
                "sample sampling",
                "",
                "@location(0) @interpolate(perspective, sample) value: f32,",
                "1.0",
            ),
        ] {
            let source = format!(
                "{prefix}\n{}",
                fragment_struct_source(fields, &format!("return Result({value});"))
            );
            let error = compile_wgsl_in(&source, test_options(), Global).expect_err(name);
            assert!(
                error.to_string().len() < 8192,
                "unbounded {name} diagnostic"
            );
        }

        // This qualifier is not authored by the WGSL frontend, but the public
        // Naga entry point must enforce the same boundary without a panic.
        let source = fragment_struct_source("@location(0) value: f32,", "return Result(1.0);");
        let mut module = naga::front::wgsl::parse_str(&source).unwrap();
        let ty = module.entry_points[0].function.result.as_ref().unwrap().ty;
        let mut result_type = module.types[ty].clone();
        let TypeInner::Struct { members, .. } = &mut result_type.inner else {
            unreachable!()
        };
        let Some(Binding::Location { per_primitive, .. }) = &mut members[0].binding else {
            unreachable!()
        };
        *per_primitive = true;
        module.types.replace(ty, result_type);
        let error = lower_naga_module_in(&module, &source, test_options(), Global).unwrap_err();
        assert!(error.to_string().len() < 8192);
    }

    #[test]
    fn fragment_struct_rejects_general_record_operations() {
        let fields = "@location(0) value: f32, @location(1) @interpolate(flat) id: i32,";
        for (name, helpers, body) in [
            (
                "mutable",
                "",
                "var result = Result(p.x, 1i); return result;",
            ),
            (
                "member access",
                "",
                "let result = Result(p.x, 1i); return Result(result.value, result.id);",
            ),
            (
                "helper result",
                "fn build(v: f32) -> Result { return Result(v, 1i); }",
                "return build(p.x);",
            ),
            (
                "helper argument",
                "fn read(v: Result) -> f32 { return v.value; }",
                "return Result(read(Result(p.x, 1i)), 2i);",
            ),
            (
                "conditional record return",
                "",
                "if p.x > 0.0 { return Result(p.x, 1i); } return Result(p.y, 2i);",
            ),
            (
                "record selection",
                "",
                "return select(Result(p.x, 1i), Result(p.y, 2i), p.x > 0.0);",
            ),
            ("zero record", "", "return Result();"),
            (
                "constant record",
                "const RECORD = Result(1.0, 1i);",
                "return RECORD;",
            ),
        ] {
            let source = format!(
                "struct Result {{ {fields} }}\n{helpers}\n@fragment fn main(@builtin(position) p: vec4<f32>) -> Result {{ {body} }}"
            );
            let error = compile_wgsl_in(&source, test_options(), Global).expect_err(name);
            assert!(
                error.to_string().len() < 8192,
                "unbounded {name} diagnostic"
            );
        }
    }

    #[derive(Default)]
    struct AllocationCounts {
        live: AtomicUsize,
        total: AtomicUsize,
    }

    #[derive(Clone)]
    struct TrackingAllocator(Arc<AllocationCounts>);

    // SAFETY: successful allocations are delegated to `Global` unchanged;
    // the wrapper only observes ownership transitions.
    unsafe impl Allocator for TrackingAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let allocation = Global.allocate(layout)?;
            self.0.live.fetch_add(1, Ordering::Relaxed);
            self.0.total.fetch_add(1, Ordering::Relaxed);
            Ok(allocation)
        }

        unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
            self.0.live.fetch_sub(1, Ordering::Relaxed);
            // SAFETY: this is the pointer/layout pair produced by `Global`.
            unsafe { Global.deallocate(pointer, layout) };
        }
    }

    #[test]
    fn emitted_asset_uses_the_supplied_allocator() {
        let counts = Arc::new(AllocationCounts::default());
        {
            let compiled = compile_wgsl_in(
                PROCEDURAL,
                test_options(),
                TrackingAllocator(Arc::clone(&counts)),
            )
            .unwrap();
            assert!(counts.total.load(Ordering::Relaxed) > 0);
            assert!(counts.live.load(Ordering::Relaxed) > 0);
            assert!(!compiled.graph().as_bytes().is_empty());
        }
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn lowers_procedural_fragment_end_to_end() {
        let compiled = compile_wgsl_in(PROCEDURAL, test_options(), Global).unwrap();
        assert_eq!(compiled.interface().field_dimensions, [6, 8]);
        let graph = compiled.graph().parse().unwrap();
        let inputs = (0..graph.tensor_count())
            .filter(|&index| {
                graph
                    .tensor(TensorId(index as u32))
                    .unwrap()
                    .flags
                    .contains(TensorFlags::INPUT)
            })
            .count();
        let outputs = (0..graph.tensor_count())
            .filter(|&index| {
                graph
                    .tensor(TensorId(index as u32))
                    .unwrap()
                    .flags
                    .contains(TensorFlags::OUTPUT)
            })
            .count();
        assert_eq!(inputs, 2);
        assert_eq!(outputs, 4);

        let ops: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap().op)
            .collect();
        assert!(ops.contains(&OpCode::SQRT));
        assert!(ops.contains(&OpCode::POW));
        assert!(ops.contains(&OpCode::COS));
        assert!(ops.contains(&OpCode::SIN));
        assert!(ops.contains(&OpCode::ABS));
        assert!(ops.contains(&OpCode::CLAMP));
        assert!(ops.contains(&OpCode::SELECT));
    }

    #[test]
    fn emits_a_concrete_viewport_for_current_accelerator_artifacts() {
        let options = CompileOptions::for_viewport("main", 960, 600);
        let compiled = compile_wgsl_in(PROCEDURAL, options, Global).unwrap();
        assert_eq!(compiled.interface().field_dimensions, [600, 960]);
        assert_eq!(compiled.interface().position_x_dimensions(), [1, 960]);
        assert_eq!(compiled.interface().position_y_dimensions(), [600, 1]);
        let graph = compiled.graph().parse().unwrap();
        let mut input_dimensions = Vec::new();
        for index in 0..graph.tensor_count() {
            let tensor = graph.tensor(TensorId(index as u32)).unwrap();
            if tensor.flags.contains(TensorFlags::INPUT) {
                input_dimensions.push(tensor.dimensions().collect::<Vec<_>>());
            } else if tensor.flags.contains(TensorFlags::OUTPUT) {
                assert_eq!(tensor.dimensions().collect::<Vec<_>>(), vec![600, 960]);
            }
        }
        assert_eq!(input_dimensions, vec![vec![1, 960], vec![600, 1]]);
    }

    #[test]
    fn lowers_extended_math_to_semantic_opcodes() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let x = p.x * 0.01;
                return vec4<f32>(
                    floor(x) + ceil(x),
                    round(x) + trunc(x),
                    atan2(sin(x), cos(x)) + exp2(x) + log2(abs(x) + 1.0),
                    fma(sign(x), 0.25, 0.75) + 1.0 / (abs(x) + 1.0) + inverseSqrt(abs(x) + 1.0),
                );
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        let ops: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap().op)
            .collect();
        for op in [
            OpCode::FLOOR,
            OpCode::CEIL,
            OpCode::ROUND,
            OpCode::TRUNC,
            OpCode::ATAN2,
            OpCode::EXP2,
            OpCode::LOG2,
            OpCode::SIGN,
            OpCode::FMA,
            OpCode::RECIPROCAL,
            OpCode::RSQRT,
        ] {
            assert!(ops.contains(&op), "missing {op:?}");
        }
    }

    #[test]
    fn expands_vector_geometry_without_special_ir_ops() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let v = normalize(p.xy - vec2<f32>(10.0, 20.0));
                let d = dot(v, vec2<f32>(0.0, 1.0));
                return vec4<f32>(v, length(v), d);
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        let ops: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap().op)
            .collect();
        assert!(ops.contains(&OpCode::RSQRT));
        assert!(ops.contains(&OpCode::SQRT));
        assert!(ops.contains(&OpCode::MUL));
        assert!(ops.contains(&OpCode::ADD));
    }

    #[test]
    fn lowers_scalar_length_and_distance_to_absolute_value() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let magnitude = length(p.x - 3.0);
                let separation = distance(p.y, 2.0);
                return vec4<f32>(magnitude, separation, 0.0, 1.0);
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        let abs_count = (0..graph.node_count())
            .filter(|&index| graph.node(index).unwrap().op == OpCode::ABS)
            .count();
        assert_eq!(abs_count, 2);
    }

    #[test]
    fn rejects_unresolved_viewport_dimensions() {
        let options = CompileOptions {
            entry_point: "main",
            field_dimensions: [6, DYNAMIC_DIM],
        };
        let error = compile_wgsl_in(PROCEDURAL, options, Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Interface);
        assert!(diagnostic.message.contains("concrete non-zero viewport"));
    }

    #[test]
    fn rejects_matrix_constructor_before_lane_flattening() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let matrix = mat2x2<f32>(vec2<f32>(p.x, 0.0), vec2<f32>(0.0, p.y));
                let value = determinant(matrix);
                return vec4<f32>(value);
            }
        "#;
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Type);
        assert!(diagnostic.message.contains("matrices"));
    }

    #[test]
    fn rejects_matrix_indexing_as_non_vector_access() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let matrix = mat2x2<f32>(vec2<f32>(p.x, 0.0), vec2<f32>(0.0, p.y));
                let value = matrix[1][0];
                return vec4<f32>(value);
            }
        "#;
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Type);
        assert!(diagnostic.message.contains("matrix"));
    }

    #[test]
    fn rejects_loops_with_source_location() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                loop {
                    if p.x > 10.0 { break; }
                }
                return vec4<f32>(0.0);
            }
        "#;
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic");
        };
        assert_eq!(diagnostic.rule, ProfileRule::ControlFlow);
        assert!(diagnostic.span.is_some());
        assert!(diagnostic.line.is_some());
    }

    #[test]
    fn rejects_textures_at_the_profile_boundary() {
        let source = r#"
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var image_sampler: sampler;
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                return textureSample(image, image_sampler, p.xy);
            }
        "#;
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic, got {error}");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Resource);
        assert!(diagnostic.message.contains("texture"));
    }

    #[test]
    fn preserves_naga_parse_diagnostics() {
        let source = "@fragment fn main( -> @location(0) vec4<f32> {}";
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Parse(diagnostic) = error else {
            panic!("expected parse diagnostic");
        };
        assert!(diagnostic.contains("wgsl"));
        assert!(diagnostic.contains("error"));
    }

    #[test]
    fn lowers_nested_short_circuit_boolean_expressions() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let a = p.x > 1.0 && p.y < 2.0;
                let b = a || !(p.z != 3.0 && p.w >= 4.0);
                return vec4<f32>(select(0.0, 1.0, b));
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        let ops: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap().op)
            .collect();

        assert_eq!(
            ops.iter().filter(|&&op| op == OpCode::LOGICAL_AND).count(),
            2
        );
        assert_eq!(
            ops.iter().filter(|&&op| op == OpCode::LOGICAL_OR).count(),
            1
        );
        assert!(ops.contains(&OpCode::EQUAL));
        assert!(ops.iter().filter(|&&op| op == OpCode::LOGICAL_NOT).count() >= 2);
    }

    #[test]
    fn rejects_authored_mutable_locals() {
        let source = r#"
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                var value = p.x;
                value = value + 1.0;
                return vec4<f32>(value);
            }
        "#;
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic");
        };
        assert_eq!(diagnostic.rule, ProfileRule::ControlFlow);
        assert!(diagnostic.message.contains("mutable local"));
        assert!(diagnostic.span.is_some());
    }

    #[test]
    fn inlines_helpers_with_distinct_arguments_and_nested_calls() {
        let source = r#"
            fn shift(value: f32, amount: f32) -> f32 { return value + amount; }
            fn combine(left: f32, right: f32) -> f32 {
                return shift(left, right) * 2.0;
            }
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let x = shift(p.x, 1.0);
                let y = shift(p.y, 2.0);
                return vec4<f32>(x, y, combine(shift(x, y), y), 1.0);
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        assert_eq!(
            compiled.interface(),
            ShaderInterface {
                field_dimensions: [6, 8],
                ..ShaderInterface::AX_SHADER_V0
            }
        );
        // Four distinct shift invocations, one multiply, three domain nodes,
        // and four output nodes; helper calls introduce no extra ABI inputs.
        assert_eq!(graph.node_count(), 12);
    }

    #[test]
    fn normalizes_nested_if_else_and_early_returns() {
        let source = r#"
            fn shade(x: f32) -> vec3<f32> {
                if x < 0.0 {
                    return vec3<f32>(0.0);
                } else if x < 1.0 {
                    return vec3<f32>(x);
                }
                return vec3<f32>(1.0);
            }
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                if p.y < 0.0 { return vec4<f32>(0.0); }
                return vec4<f32>(shade(p.x), 1.0);
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        let selections = (0..graph.node_count())
            .filter(|&index| graph.node(index).unwrap().op == OpCode::SELECT)
            .count();
        assert_eq!(selections, 10);
        assert!(graph.node_count() < 24);
    }

    #[test]
    fn helpers_support_bool_returns_and_short_circuit_arguments() {
        let source = r#"
            fn positive(x: f32) -> bool {
                if x >= 0.0 { return true; }
                return false;
            }
            fn intensity(value: bool) -> f32 {
                if value { return 1.0; }
                return 0.0;
            }
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let visible = positive(p.x) && (positive(p.y) || positive(-p.y));
                return vec4<f32>(intensity(visible));
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        let graph = compiled.graph().parse().unwrap();
        assert!(graph.node_count() < 40);
        let mut logical = 0;
        for index in 0..graph.node_count() {
            let node = graph.node(index).unwrap();
            if node.op == OpCode::SELECT {
                for output in node.outputs() {
                    assert_eq!(graph.tensor(output).unwrap().dtype, DType::F32);
                }
            }
            if node.op == OpCode::LOGICAL_AND {
                logical += 1;
            }
        }
        assert!(logical >= 4);
    }

    #[test]
    fn helper_uniforms_share_the_entry_abi() {
        let source = r#"
            @group(0) @binding(0) var<uniform> tint: vec3<f32>;
            fn paint(shade: f32) -> vec3<f32> { return tint * shade; }
            @fragment
            fn main(@builtin(position) p: vec4<f32>, @location(0) shade: f32)
                -> @location(0) vec4<f32> {
                return vec4<f32>(paint(shade) + paint(p.x), 1.0);
            }
        "#;
        let compiled = compile_wgsl_in(source, test_options(), Global).unwrap();
        assert_eq!(compiled.interface().uniform_inputs, 3);
        assert_eq!(compiled.interface().varying_inputs, 1);
    }

    #[test]
    fn rejects_effects_in_discarded_helper_calls() {
        let source = r#"
            fn hidden_effect(value: f32) -> f32 { discard; return value; }
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let unused = hidden_effect(p.x);
                return vec4<f32>(1.0);
            }
        "#;
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic, got {error}");
        };
        assert_eq!(diagnostic.rule, ProfileRule::ControlFlow);
        assert!(diagnostic.message.contains("side effects"));
        assert!(diagnostic.span.is_some());
    }

    #[test]
    fn rejects_aggregate_helper_parameters() {
        let source = r#"
            fn first(values: array<f32, 2>) -> f32 { return values[0]; }
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                return vec4<f32>(first(array<f32, 2>(p.x, p.y)));
            }
        "#;
        let error = compile_wgsl_in(source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected profile diagnostic, got {error}");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Type);
        assert!(diagnostic.message.contains("parameters"));
    }

    #[test]
    fn recursive_helpers_are_rejected_by_wgsl_validation() {
        let source = r#"
            fn recur(value: f32) -> f32 { return recur(value); }
            @fragment
            fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                return vec4<f32>(recur(p.x));
            }
        "#;
        assert!(matches!(
            compile_wgsl_in(source, test_options(), Global),
            Err(CompileError::Parse(_) | CompileError::Validation(_))
        ));
    }

    #[test]
    fn bounds_exponential_helper_expansion() {
        let mut source = String::from("fn f0(x: f32) -> f32 { return x + 1.0; }\n");
        for level in 1..=11 {
            let previous = level - 1;
            source.push_str(&format!(
                "fn f{level}(x: f32) -> f32 {{ return f{previous}(x) + f{previous}(x); }}\n"
            ));
        }
        source.push_str(
            "@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> { return vec4<f32>(f11(p.x)); }"
        );
        let error = compile_wgsl_in(&source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected budget diagnostic, got {error}");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Budget);
        assert!(diagnostic.message.contains("expanded helper"));
        assert!(diagnostic.span.is_some());
    }

    #[test]
    fn bounds_helper_call_depth() {
        let mut source = String::from("fn chain_0(x: f32) -> f32 { return x; }\n");
        for level in 1..=MAX_INLINE_DEPTH {
            let previous = level - 1;
            source.push_str(&format!(
                "fn chain_{level}(x: f32) -> f32 {{ return chain_{previous}(x); }}\n"
            ));
        }
        source.push_str(&format!(
            "@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{ return vec4<f32>(chain_{MAX_INLINE_DEPTH}(p.x)); }}"
        ));
        let error = compile_wgsl_in(&source, test_options(), Global).unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected budget diagnostic, got {error}");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Budget);
        assert!(diagnostic.message.contains("call depth"));
        assert!(diagnostic.span.is_some());
    }

    #[test]
    fn bounds_expanded_graph_nodes_and_releases_owned_storage() {
        let mut source = String::from("fn wide(x: vec4<f32>) -> vec4<f32> {\n");
        for index in 0..64 {
            source.push_str(&format!("let v{index} = sin(x + vec4<f32>({index}.0));\n"));
        }
        source.push_str("return ");
        source.push_str(
            &(0..64)
                .map(|index| format!("v{index}"))
                .collect::<Vec<_>>()
                .join(" + "),
        );
        source.push_str("; }\n@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> { return ");
        source.push_str(
            &(0..24)
                .map(|index| format!("wide(p + vec4<f32>({index}.0))"))
                .collect::<Vec<_>>()
                .join(" + "),
        );
        source.push_str("; }");
        let counts = Arc::new(AllocationCounts::default());
        let error = compile_wgsl_in(
            &source,
            test_options(),
            TrackingAllocator(Arc::clone(&counts)),
        )
        .unwrap_err();
        let CompileError::Profile(diagnostic) = error else {
            panic!("expected budget diagnostic, got {error}");
        };
        assert_eq!(diagnostic.rule, ProfileRule::Budget);
        assert!(diagnostic.message.contains("node budget"));
        assert!(counts.total.load(Ordering::Relaxed) > 0);
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    }
}
