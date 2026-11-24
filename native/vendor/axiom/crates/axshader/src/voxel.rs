//! Shared registration and pointer normalization for explicit WGSL resources.
//!
//! Naga 30 parses storage-pointer parameters but its validator does not admit
//! them. We bind the exact root resource before removing only that parameter
//! from the generated declaration and registered calls. Naga then validates
//! the remaining numeric arguments normally. Each declaration's body is parser
//! scaffolding and is never eligible for ordinary helper lowering.

use std::collections::{HashMap, HashSet};

use ax::voxel::VoxelBrickTraceParams;
use naga::{
    AddressSpace, ArraySize, Expression, Function, Handle, Module, ScalarKind, ShaderStage, Span,
    Statement, StorageAccess, TypeInner, VectorSize,
};

use super::{
    CompileError, MAX_CONTROL_DEPTH, MAX_EXPRESSION_DEPTH, ProfileRule, ShaderResource, buffer,
    profile, profile_unspanned, sampling, select_entry_point,
};

const DIRECTIVE: &str = "ax_voxel_query";
const INTRINSIC: &str = "ax_brick_trace";
const HIT_TYPE: &str = "AxVoxelHit";
const DECLARATIONS: &str = r#"
struct AxVoxelHit {
    @location(0) t: f32,
    @location(1) @interpolate(flat) cell_id: i32,
}
fn ax_brick_trace(
    grid: ptr<storage, array<vec2<u32>>, read>,
    origin: vec3<f32>, direction: vec3<f32>, limit: f32
) -> AxVoxelHit {
    return AxVoxelHit(0.0, -1i);
}
"#;

#[derive(Clone, Copy)]
pub(super) struct VoxelExtension {
    pub intrinsic: Handle<Function>,
    pub hit_type: Handle<naga::Type>,
}

pub(super) struct Extension {
    pub voxel: Option<VoxelExtension>,
    pub sampling: Option<sampling::Extension>,
    pub buffer: Option<buffer::Extension>,
    pub resources: Vec<Option<ShaderResource>>,
    /// Only selected-entry CallResult handles can have registrations.
    pub calls: Vec<Option<ShaderResource>>,
}

impl Extension {
    pub fn is_intrinsic(&self, function: Handle<Function>) -> bool {
        self.voxel.is_some_and(|voxel| voxel.intrinsic == function)
            || self
                .sampling
                .is_some_and(|sampling| sampling.intrinsic == function)
            || self
                .buffer
                .is_some_and(|buffer| buffer.intrinsic == function)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResourceKind {
    Voxel,
    Rgb,
    U8Buffer,
}

impl ResourceKind {
    fn matches(self, resource: ShaderResource) -> bool {
        matches!(
            (self, resource),
            (Self::Voxel, ShaderResource::VoxelBrickGrid(_))
                | (Self::Rgb, ShaderResource::RgbVolume(_))
                | (Self::U8Buffer, ShaderResource::U8Buffer(_))
        )
    }

    fn intrinsic_name(self) -> &'static str {
        match self {
            Self::Voxel => INTRINSIC,
            Self::Rgb => sampling::INTRINSIC,
            Self::U8Buffer => buffer::INTRINSIC,
        }
    }

    fn argument_count(self) -> usize {
        match self {
            Self::Voxel => 4,
            Self::Rgb | Self::U8Buffer => 2,
        }
    }
}

#[derive(Clone, Copy)]
struct RegisteredIntrinsic {
    function: Handle<Function>,
    kind: ResourceKind,
}

#[derive(Default)]
struct EnabledExtensions {
    voxel: bool,
    sampling: bool,
    buffer: bool,
}

pub(super) struct Prepared {
    pub module: Module,
    /// Same byte offsets as the original source, followed by our declarations.
    pub source: String,
    pub extension: Extension,
}

pub(super) fn prepare(
    source: &str,
    entry_point: &str,
    specializations: &[VoxelBrickTraceParams],
) -> Result<Prepared, CompileError> {
    let specializations: Vec<_> = specializations
        .iter()
        .copied()
        .map(ShaderResource::VoxelBrickGrid)
        .collect();
    prepare_impl(source, entry_point, &specializations, true)
}

pub(super) fn prepare_specs(
    source: &str,
    entry_point: &str,
    specializations: &[ShaderResource],
) -> Result<Prepared, CompileError> {
    prepare_impl(source, entry_point, specializations, false)
}

fn prepare_impl(
    source: &str,
    entry_point: &str,
    specializations: &[ShaderResource],
    voxel_only: bool,
) -> Result<Prepared, CompileError> {
    let (prepared_source, enabled) = prepare_source(source, voxel_only)?;
    let mut module = naga::front::wgsl::parse_str(&prepared_source)
        .map_err(|error| CompileError::Parse(error.emit_to_string(&prepared_source)))?;
    let voxel = enabled
        .voxel
        .then(|| {
            Ok::<_, CompileError>(VoxelExtension {
                intrinsic: generated_function(&module, source.len(), INTRINSIC)?,
                hit_type: generated_type(&module, source.len(), HIT_TYPE)?,
            })
        })
        .transpose()?;
    let sampling = enabled
        .sampling
        .then(|| {
            Ok::<_, CompileError>(sampling::Extension {
                intrinsic: generated_function(&module, source.len(), sampling::INTRINSIC)?,
                resource_type: generated_type(&module, source.len(), sampling::HANDLE_TYPE)?,
            })
        })
        .transpose()?;
    let buffer = enabled
        .buffer
        .then(|| {
            Ok::<_, CompileError>(buffer::Extension {
                intrinsic: generated_function(&module, source.len(), buffer::INTRINSIC)?,
                resource_type: generated_type(&module, source.len(), buffer::HANDLE_TYPE)?,
            })
        })
        .transpose()?;
    let resources = bind_resources(
        &module,
        source,
        specializations,
        voxel.is_some(),
        sampling.map(|s| s.resource_type),
        buffer.map(|b| b.resource_type),
    )?;
    let intrinsics: Vec<_> = voxel
        .map(|voxel| RegisteredIntrinsic {
            function: voxel.intrinsic,
            kind: ResourceKind::Voxel,
        })
        .into_iter()
        .chain(sampling.map(|sampling| RegisteredIntrinsic {
            function: sampling.intrinsic,
            kind: ResourceKind::Rgb,
        }))
        .chain(buffer.map(|buffer| RegisteredIntrinsic {
            function: buffer.intrinsic,
            kind: ResourceKind::U8Buffer,
        }))
        .collect();
    let (entry_index, _) = select_entry_point(&module, source, entry_point)?;

    // Resource calls hidden in helpers, unselected entry points, conditional
    // control flow, or a short-circuit operand are not eagerly executable.
    for (handle, function) in module.functions.iter_mut() {
        if !intrinsics
            .iter()
            .any(|registered| registered.function == handle)
        {
            normalize_function(function, source, &intrinsics, &resources, false)?;
        }
    }
    let mut calls = Vec::new();
    for (index, entry) in module.entry_points.iter_mut().enumerate() {
        let normalized = normalize_function(
            &mut entry.function,
            source,
            &intrinsics,
            &resources,
            index == entry_index && entry.stage == ShaderStage::Fragment,
        )?;
        if index == entry_index {
            calls = normalized;
        }
    }

    for registered in intrinsics {
        let stub = &mut module.functions[registered.function];
        stub.arguments.remove(0);
        for (_, expression) in stub.expressions.iter_mut() {
            if let Expression::FunctionArgument(index) = expression {
                if *index == 0 {
                    // This generated parameter is deliberately unused. Keep
                    // its arena handle without retaining a storage pointer.
                    *expression = Expression::Literal(naga::Literal::F32(0.0));
                } else {
                    *index -= 1;
                }
            }
        }
    }
    Ok(Prepared {
        module,
        source: prepared_source,
        extension: Extension {
            voxel,
            sampling,
            buffer,
            resources,
            calls,
        },
    })
}

fn generated_function(
    module: &Module,
    source_len: usize,
    name: &str,
) -> Result<Handle<Function>, CompileError> {
    let generated = |span: Span| {
        span.to_range()
            .is_some_and(|range| range.start >= source_len)
    };
    module
        .functions
        .iter()
        .find(|(handle, function)| {
            function.name.as_deref() == Some(name) && generated(module.functions.get_span(*handle))
        })
        .map(|(handle, _)| handle)
        .ok_or_else(|| {
            profile_unspanned(
                ProfileRule::Resource,
                format!("generated resource intrinsic `{name}` is missing"),
            )
        })
}

fn generated_type(
    module: &Module,
    source_len: usize,
    name: &str,
) -> Result<Handle<naga::Type>, CompileError> {
    module
        .types
        .iter()
        .find(|(handle, ty)| {
            ty.name.as_deref() == Some(name)
                && module
                    .types
                    .get_span(*handle)
                    .to_range()
                    .is_some_and(|range| range.start >= source_len)
        })
        .map(|(handle, _)| handle)
        .ok_or_else(|| {
            profile_unspanned(
                ProfileRule::Resource,
                format!("generated resource type `{name}` is missing"),
            )
        })
}

pub(super) fn reject_reserved_declarations(
    module: &Module,
    source: &str,
) -> Result<(), CompileError> {
    reject_reserved_source_declarations(source, &token_ranges(source))?;
    for (handle, function) in module.functions.iter() {
        if matches!(
            function.name.as_deref(),
            Some(INTRINSIC) | Some(sampling::INTRINSIC) | Some(buffer::INTRINSIC)
        ) {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "resource intrinsic names require their explicit compiler API and directive; an authored body cannot register them",
                module.functions.get_span(handle),
            ));
        }
    }
    for (handle, ty) in module.types.iter() {
        if matches!(
            ty.name.as_deref(),
            Some(HIT_TYPE) | Some(sampling::HANDLE_TYPE) | Some(buffer::HANDLE_TYPE)
        ) {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "resource intrinsic types are reserved for their explicitly enabled profile",
                module.types.get_span(handle),
            ));
        }
    }
    Ok(())
}

fn bind_resources(
    module: &Module,
    source: &str,
    specializations: &[ShaderResource],
    voxel_enabled: bool,
    rgb_type: Option<Handle<naga::Type>>,
    buffer_type: Option<Handle<naga::Type>>,
) -> Result<Vec<Option<ShaderResource>>, CompileError> {
    let mut pending = HashMap::new();
    for &params in specializations {
        if !params.is_valid() || pending.insert(params.resource(), params).is_some() {
            return Err(profile_unspanned(
                ProfileRule::Resource,
                "resource specializations require unique bindings and valid kind-specific dimensions/storage",
            ));
        }
    }
    let mut bindings = HashSet::new();
    let mut resources = vec![None; module.global_variables.len()];
    for (handle, global) in module.global_variables.iter() {
        let span = module.global_variables.get_span(handle);
        if let Some(binding) = &global.binding
            && !bindings.insert((binding.group, binding.binding))
        {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "global resource bindings must be unique",
                span,
            ));
        }
        if !matches!(global.space, AddressSpace::Storage { .. }) {
            continue;
        }
        let voxel_type = match module.types[global.ty].inner {
            TypeInner::Array {
                base,
                size: ArraySize::Dynamic,
                stride: 8,
            } => {
                matches!(module.types[base].inner, TypeInner::Vector { size: VectorSize::Bi, scalar }
                    if scalar.kind == ScalarKind::Uint && scalar.width == 4)
            }
            _ => false,
        };
        let kind = if voxel_type && voxel_enabled {
            Some(ResourceKind::Voxel)
        } else if rgb_type == Some(global.ty) {
            Some(ResourceKind::Rgb)
        } else if buffer_type == Some(global.ty) {
            Some(ResourceKind::U8Buffer)
        } else {
            None
        };
        if global.space
            != (AddressSpace::Storage {
                access: StorageAccess::LOAD,
            })
            || global.binding.is_none()
            || global.init.is_some()
            || kind.is_none()
        {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "resources must be bound read-only storage globals of an enabled kind: array<vec2<u32>> for voxel grids, opaque AxRgbVolume for RGB volumes, or opaque AxU8Buffer for byte buffers",
                span,
            ));
        }
        let binding = global.binding.as_ref().unwrap();
        let key = ax::voxel::ResourceKey {
            group: binding.group,
            binding: binding.binding,
        };
        let params = pending.remove(&key).ok_or_else(|| {
            profile(
                source,
                ProfileRule::Resource,
                "storage resource has no matching binding/dimension/storage specialization",
                span,
            )
        })?;
        if !kind.unwrap().matches(params) {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "resource specialization kind does not match its storage declaration",
                span,
            ));
        }
        resources[handle.index()] = Some(params);
    }
    if !pending.is_empty() {
        return Err(profile_unspanned(
            ProfileRule::Resource,
            "resource specialization has no matching storage declaration",
        ));
    }
    Ok(resources)
}

fn normalize_function(
    function: &mut Function,
    source: &str,
    intrinsics: &[RegisteredIntrinsic],
    resources: &[Option<ShaderResource>],
    entry: bool,
) -> Result<Vec<Option<ShaderResource>>, CompileError> {
    let mut normalizer = CallNormalizer {
        source,
        intrinsics,
        resources,
        entry,
        expressions: &function.expressions,
        admitted_roots: HashSet::new(),
        calls: vec![None; function.expressions.len()],
    };
    normalizer.block(&mut function.body, false, 0)?;
    for (handle, expression) in function.expressions.iter() {
        let rejected = match *expression {
            Expression::GlobalVariable(global) if resources[global.index()].is_some() => {
                !normalizer.admitted_roots.contains(&handle)
                    || function.named_expressions.contains_key(&handle)
            }
            Expression::Load { pointer } | Expression::ArrayLength(pointer) => {
                resource_pointer(&function.expressions, resources, pointer)
            }
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                resource_pointer(&function.expressions, resources, base)
            }
            _ => false,
        };
        if rejected {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "resources may only appear as the direct root argument of their registered intrinsic; loads, indexing, member access and pointer aliases are unsupported",
                function.expressions.get_span(handle),
            ));
        }
    }
    Ok(normalizer.calls)
}

fn resource_pointer(
    expressions: &naga::Arena<Expression>,
    resources: &[Option<ShaderResource>],
    mut handle: Handle<Expression>,
) -> bool {
    for _ in 0..MAX_EXPRESSION_DEPTH {
        match expressions[handle] {
            Expression::GlobalVariable(global) => return resources[global.index()].is_some(),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => handle = base,
            _ => return false,
        }
    }
    true // An uncertain pointer chain must never bypass the resource boundary.
}

struct CallNormalizer<'a> {
    source: &'a str,
    intrinsics: &'a [RegisteredIntrinsic],
    resources: &'a [Option<ShaderResource>],
    expressions: &'a naga::Arena<Expression>,
    entry: bool,
    admitted_roots: HashSet<Handle<Expression>>,
    calls: Vec<Option<ShaderResource>>,
}

impl CallNormalizer<'_> {
    fn intrinsic_kind(&self, function: Handle<Function>) -> Option<ResourceKind> {
        self.intrinsics
            .iter()
            .find_map(|registered| (registered.function == function).then_some(registered.kind))
    }

    /// Returns whether any path can return, guarding subsequent queries too.
    fn block(
        &mut self,
        block: &mut naga::Block,
        mut guarded: bool,
        depth: usize,
    ) -> Result<bool, CompileError> {
        if depth > MAX_CONTROL_DEPTH {
            return Err(profile_unspanned(
                ProfileRule::Budget,
                "resource call control nesting exceeds the shader budget",
            ));
        }
        let mut can_return = false;
        for (statement, span) in block.span_iter_mut() {
            let span = span.copied().unwrap_or(Span::UNDEFINED);
            let returns = match statement {
                Statement::Call {
                    function,
                    arguments,
                    result,
                } if self.intrinsic_kind(*function).is_some() => {
                    let kind = self.intrinsic_kind(*function).unwrap();
                    let name = kind.intrinsic_name();
                    if !self.entry || guarded {
                        return Err(profile(
                            self.source,
                            ProfileRule::ControlFlow,
                            "resource operations require an unconditional call in the selected entry body; helper and conditional resource calls are unsupported",
                            span,
                        ));
                    }
                    let Some(result) = result else {
                        return Err(profile(
                            self.source,
                            ProfileRule::Expression,
                            "resource operations require a result",
                            span,
                        ));
                    };
                    if arguments.len() != kind.argument_count() {
                        return Err(profile(
                            self.source,
                            ProfileRule::Expression,
                            format!("{name} requires {} arguments", kind.argument_count()),
                            span,
                        ));
                    }
                    let root = arguments[0];
                    let Expression::GlobalVariable(global) = self.expressions[root] else {
                        return Err(profile(
                            self.source,
                            ProfileRule::Resource,
                            format!("{name} requires the direct address of a root resource"),
                            span,
                        ));
                    };
                    let params = self.resources[global.index()]
                        .filter(|params| kind.matches(*params))
                        .ok_or_else(|| profile(self.source, ProfileRule::Resource,
                            format!("{name} first argument must be a specialized resource of the matching kind"), span))?;
                    if self.calls[result.index()].replace(params).is_some() {
                        return Err(profile(
                            self.source,
                            ProfileRule::Expression,
                            "duplicate resource result registration",
                            span,
                        ));
                    }
                    self.admitted_roots.insert(root);
                    arguments.remove(0);
                    false
                }
                Statement::Call { arguments, .. } => {
                    if arguments.iter().any(|&argument| {
                        resource_pointer(self.expressions, self.resources, argument)
                    }) {
                        return Err(profile(
                            self.source,
                            ProfileRule::Resource,
                            "resource pointers cannot be passed to helpers",
                            span,
                        ));
                    }
                    false
                }
                Statement::Block(inner) => self.block(inner, guarded, depth + 1)?,
                Statement::If { accept, reject, .. } => {
                    let accept = self.block(accept, true, depth + 1)?;
                    let reject = self.block(reject, true, depth + 1)?;
                    accept || reject
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        self.block(&mut case.body, true, depth + 1)?;
                    }
                    true
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    self.block(body, true, depth + 1)?;
                    self.block(continuing, true, depth + 1)?;
                    true
                }
                Statement::Return { .. } => true,
                _ => false,
            };
            can_return |= returns;
            guarded |= returns;
        }
        Ok(can_return)
    }
}

fn reject_reserved_source_declarations(
    source: &str,
    tokens: &[std::ops::Range<usize>],
) -> Result<(), CompileError> {
    for (index, token) in tokens.iter().enumerate() {
        let previous = index
            .checked_sub(1)
            .map(|index| &source[tokens[index].clone()]);
        let next = tokens.get(index + 1).map(|range| &source[range.clone()]);
        if matches!(
            &source[token.clone()],
            INTRINSIC
                | HIT_TYPE
                | sampling::INTRINSIC
                | sampling::HANDLE_TYPE
                | sampling::HANDLE_MEMBER
                | buffer::INTRINSIC
                | buffer::HANDLE_TYPE
                | buffer::HANDLE_MEMBER
        ) && (matches!(
            previous,
            Some("fn" | "struct" | "alias" | "const" | "override" | "let" | "var")
        ) || matches!(next, Some(":" | "=")))
        {
            return Err(profile(
                source,
                ProfileRule::Resource,
                "resource intrinsic, type and opaque member names are reserved",
                Span::new(token.start as u32, token.end as u32),
            ));
        }
    }
    Ok(())
}

fn prepare_source(
    source: &str,
    voxel_only: bool,
) -> Result<(String, EnabledExtensions), CompileError> {
    if source
        .len()
        .checked_add(DECLARATIONS.len())
        .and_then(|len| len.checked_add(sampling::DECLARATIONS.len()))
        .and_then(|len| len.checked_add(buffer::DECLARATIONS.len()))
        .is_none_or(|len| len > u32::MAX as usize)
    {
        return Err(profile_unspanned(
            ProfileRule::Budget,
            "WGSL source exceeds the span address range",
        ));
    }
    let tokens = token_ranges(source);
    reject_reserved_source_declarations(source, &tokens)?;
    for (index, token) in tokens.iter().enumerate() {
        if &source[token.clone()] == sampling::HANDLE_TYPE {
            sampling::validate_handle_use(source, &tokens, index)?;
        } else if &source[token.clone()] == buffer::HANDLE_TYPE {
            buffer::validate_handle_use(source, &tokens, index)?;
        }
    }
    let mut enabled = EnabledExtensions::default();
    let mut directives = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        let first = &source[tokens[index].clone()];
        if !matches!(first, "enable" | "requires" | "diagnostic") {
            break;
        }
        let Some(end) = tokens[index..]
            .iter()
            .position(|range| &source[range.clone()] == ";")
            .map(|end| end + index)
        else {
            break;
        };
        if first == "enable"
            && let Some(directive) = tokens[index + 1..end]
                .iter()
                .map(|range| &source[range.clone()])
                .find(|name| matches!(*name, DIRECTIVE | sampling::DIRECTIVE | buffer::DIRECTIVE))
        {
            let range = tokens[index].start..tokens[end].end;
            if directive != DIRECTIVE && voxel_only {
                return Err(profile(
                    source,
                    ProfileRule::Resource,
                    format!("{directive} requires compile_wgsl_with_resource_specs_in"),
                    Span::new(range.start as u32, range.end as u32),
                ));
            }
            let flag = if directive == DIRECTIVE {
                &mut enabled.voxel
            } else if directive == sampling::DIRECTIVE {
                &mut enabled.sampling
            } else {
                &mut enabled.buffer
            };
            if end != index + 2 || *flag {
                return Err(profile(
                    source,
                    ProfileRule::Resource,
                    format!("use exactly one standalone `enable {directive};` directive"),
                    Span::new(range.start as u32, range.end as u32),
                ));
            }
            *flag = true;
            directives.push(range);
        }
        index = end + 1;
    }
    if (voxel_only && !enabled.voxel) || (!enabled.voxel && !enabled.sampling && !enabled.buffer) {
        return Err(profile_unspanned(
            ProfileRule::Resource,
            if voxel_only {
                "compile_wgsl_with_resources_in requires a leading `enable ax_voxel_query;` directive"
            } else {
                "resource compilation requires a leading `enable ax_voxel_query;`, `enable ax_volume_sampling;`, or `enable ax_buffer_access;` directive"
            },
        ));
    }
    let mut bytes = source.as_bytes().to_vec();
    for directive in directives {
        for byte in &mut bytes[directive] {
            if !matches!(*byte, b'\n' | b'\r') {
                *byte = b' ';
            }
        }
    }
    // The complete range is replaced, including any UTF-8 comment bytes.
    let mut prepared = String::from_utf8(bytes).expect("blanking complete tokens preserves UTF-8");
    if enabled.voxel {
        prepared.push_str(DECLARATIONS);
    }
    if enabled.sampling {
        prepared.push_str(sampling::DECLARATIONS);
    }
    if enabled.buffer {
        prepared.push_str(buffer::DECLARATIONS);
    }
    Ok((prepared, enabled))
}

/// Just enough lexical structure to distinguish leading directives and
/// reserved declarations from comments; Naga remains the WGSL parser.
fn token_ranges(source: &str) -> Vec<std::ops::Range<usize>> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    // WGSL identifiers may contain non-ASCII combining characters. Keep them
    // together here so an identifier that merely shares an ASCII reserved prefix
    // is not mistaken for that exact reserved name; Naga checks XID validity.
    let identifier_char = |value: char| {
        value.is_alphanumeric() || value == '_' || (!value.is_ascii() && !value.is_whitespace())
    };
    while index < bytes.len() {
        let first = source[index..].chars().next().unwrap();
        if first.is_whitespace() {
            index += first.len_utf8();
        } else if bytes[index..].starts_with(b"//") {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
        } else if bytes[index..].starts_with(b"/*") {
            let mut depth = 1;
            index += 2;
            while index < bytes.len() && depth != 0 {
                if bytes[index..].starts_with(b"/*") {
                    depth += 1;
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
        } else {
            let start = index;
            index += first.len_utf8();
            if identifier_char(first) {
                while let Some(next) = source[index..].chars().next() {
                    if !identifier_char(next) {
                        break;
                    }
                    index += next.len_utf8();
                }
            }
            tokens.push(start..index);
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompileOptions, ShaderOutputKind, compile_wgsl_in, compile_wgsl_with_resources_in,
    };
    use ax::voxel::{ResourceKey, decode_brick_trace_attributes};
    use ax::{DType, NodeFlags, OpCode, TensorFlags, TensorId};
    use std::alloc::Global;

    const RESOURCE: VoxelBrickTraceParams = VoxelBrickTraceParams {
        resource: ResourceKey {
            group: 1,
            binding: 0,
        },
        dims: [8, 12, 16],
    };
    const PREFIX: &str = "enable ax_voxel_query;\n@group(1) @binding(0) var<storage, read> occupancy: array<vec2<u32>>;\n";
    const RAYS: &str =
        "ax_brick_trace(&occupancy, vec3<f32>(1.0), vec3<f32>(1.0, 0.0, 0.0), 700.0)";

    fn shader(body: &str, result: &str) -> String {
        format!(
            "{PREFIX}@fragment fn main(@builtin(position) p: vec4<f32>) -> {result} {{ {body} }}"
        )
    }

    fn compile(source: &str) -> Result<crate::CompiledShader<Global>, CompileError> {
        compile_wgsl_with_resources_in(
            source,
            CompileOptions::for_viewport("main", 9, 3),
            &[RESOURCE],
            Global,
        )
    }

    #[test]
    fn voxel_hit_and_shared_direction_use_an_ordinary_fragment_record() {
        let source = format!(
            r#"{PREFIX}
            struct Result {{
                @location(2) direction: vec3<f32>,
                @location(1) @interpolate(flat) id: i32,
                @location(0) distance: f32,
            }}
            fn direction(x: f32) -> vec3<f32> {{
                return vec3<f32>(1.0, x * 0.125, 0.0);
            }}
            @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {{
                let ray = direction(p.x);
                let hit = ax_brick_trace(&occupancy, vec3<f32>(0.25), ray, 700.0);
                let result = Result(ray, hit.cell_id, hit.t);
                let shared_result = result;
                return shared_result;
            }}"#
        );
        let compiled = compile(&source).unwrap();
        let interface = compiled.interface();
        assert!(matches!(
            interface.output_kind,
            ShaderOutputKind::FragmentStruct { .. }
        ));
        assert_eq!(interface.output_count(), 5);
        assert_eq!(
            (0..5)
                .map(|index| interface.output_dtype(index).unwrap())
                .collect::<Vec<_>>(),
            [DType::F32, DType::I32, DType::F32, DType::F32, DType::F32]
        );
        let graph = compiled.graph().parse().unwrap();
        let nodes: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap())
            .collect();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.op == OpCode::VOXEL_BRICK_TRACE)
                .count(),
            1
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.op == OpCode::INTERLEAVE3_F32)
                .count(),
            2
        );
        assert_eq!(
            nodes.iter().filter(|node| node.op == OpCode::MUL).count(),
            3
        );
    }

    #[test]
    fn voxel_query_emits_exact_typed_outputs_and_resource_attributes() {
        let source = shader(&format!("return {RAYS};"), "AxVoxelHit");
        let compiled = compile(&source).unwrap();
        let mixed_api = crate::compile_wgsl_with_resource_specs_in(
            &source,
            CompileOptions::for_viewport("main", 9, 3),
            &[ShaderResource::VoxelBrickGrid(RESOURCE)],
            Global,
        )
        .unwrap();
        assert_eq!(compiled.graph().as_bytes(), mixed_api.graph().as_bytes());
        let interface = compiled.interface();
        assert_eq!(interface.output_kind, ShaderOutputKind::VoxelHit);
        assert_eq!(interface.output_count(), 2);
        assert_eq!(
            (
                interface.output_dtype(0),
                interface.output_dtype(1),
                interface.output_dtype(2)
            ),
            (Some(DType::F32), Some(DType::I32), None)
        );
        let graph = compiled.graph().parse().unwrap();
        let nodes: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap())
            .collect();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.op == OpCode::INTERLEAVE3_F32)
                .count(),
            2
        );
        let queries: Vec<_> = nodes
            .iter()
            .filter(|node| node.op == OpCode::VOXEL_BRICK_TRACE)
            .collect();
        assert_eq!(queries.len(), 1);
        assert!(queries[0].flags.contains(NodeFlags::READS_RESOURCE));
        assert_eq!(
            decode_brick_trace_attributes(queries[0].attributes()),
            Some(RESOURCE)
        );
        let tensors: Vec<_> = (0..graph.tensor_count())
            .map(|index| graph.tensor(TensorId(index as u32)).unwrap())
            .collect();
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| tensor.flags.contains(TensorFlags::INPUT))
                .count(),
            2
        );
        let outputs: Vec<_> = tensors
            .iter()
            .filter(|tensor| tensor.flags.contains(TensorFlags::OUTPUT))
            .collect();
        assert_eq!(
            outputs
                .iter()
                .map(|tensor| tensor.dtype)
                .collect::<Vec<_>>(),
            [DType::F32, DType::I32]
        );
        assert!(outputs.iter().all(|tensor| tensor.dimensions().eq([3, 9])));
        assert!(
            tensors
                .iter()
                .filter(|tensor| tensor.rank() == 3)
                .all(|tensor| tensor.dimensions().eq([3, 9, 3]))
        );
    }

    #[test]
    fn voxel_hit_members_share_one_query_and_integer_comparisons_select_f32() {
        let source = shader(
            &format!(
                "let hit = {RAYS}; return vec4<f32>(select(0.0, hit.t, hit.cell_id > -1i), select(0.0, 1.0, hit.cell_id == 16777217i), 0.0, 1.0);"
            ),
            "@location(0) vec4<f32>",
        );
        let compiled = compile(&source).unwrap();
        assert_eq!(compiled.interface().output_kind, ShaderOutputKind::RgbaF32);
        let graph = compiled.graph().parse().unwrap();
        let nodes: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap())
            .collect();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.op == OpCode::VOXEL_BRICK_TRACE)
                .count(),
            1
        );
        for node in nodes
            .iter()
            .filter(|node| matches!(node.op, OpCode::EQUAL | OpCode::GREATER))
        {
            assert!(
                node.inputs()
                    .all(|input| graph.tensor(input).unwrap().dtype == DType::I32)
            );
        }
    }

    #[test]
    fn voxel_extension_requires_both_source_and_api_opt_in() {
        let source = shader(&format!("return {RAYS};"), "AxVoxelHit");
        assert!(matches!(
            compile_wgsl_in(&source, CompileOptions::for_viewport("main", 9, 3), Global),
            Err(CompileError::Parse(_))
        ));
        assert!(matches!(
            compile(source.strip_prefix("enable ax_voxel_query;\n").unwrap()),
            Err(CompileError::Profile(_))
        ));
        let impersonator = "fn ax_brick_trace(v: f32) -> f32 { return v; } @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> { return vec4<f32>(ax_brick_trace(p.x)); }";
        assert!(matches!(
            compile_wgsl_in(
                impersonator,
                CompileOptions::for_viewport("main", 9, 3),
                Global
            ),
            Err(CompileError::Profile(_))
        ));
        let module = naga::front::wgsl::parse_str(impersonator).unwrap();
        assert!(matches!(
            crate::lower_naga_module_in(
                &module,
                impersonator,
                CompileOptions::for_viewport("main", 9, 3),
                Global
            ),
            Err(CompileError::Profile(_))
        ));
    }

    #[test]
    fn voxel_camera_rays_keep_field_and_uniform_abi_without_resource_uploads() {
        let source = format!(
            r#"{PREFIX}
            @group(0) @binding(0) var<uniform> viewport: vec2<f32>;
            @group(0) @binding(1) var<uniform> camera_origin: vec3<f32>;
            @group(0) @binding(2) var<uniform> forward: vec3<f32>;
            @group(0) @binding(3) var<uniform> right: vec3<f32>;
            @group(0) @binding(4) var<uniform> up: vec3<f32>;
            @group(0) @binding(5) var<uniform> grid_scale: f32;
            @group(0) @binding(6) var<uniform> ray_limit: f32;
            fn unit(v: vec3<f32>) -> vec3<f32> {{ return v * (1.0 / max(sqrt(dot(v, v)), 1e-8)); }}
            @fragment fn main(@builtin(position) p: vec4<f32>, @location(0) pixel: vec2<f32>) -> AxVoxelHit {{
                let sx = (pixel.x - viewport.x * 0.5) / viewport.y;
                let sy = (viewport.y * 0.5 - pixel.y) / viewport.y;
                let direction = unit(forward + (right * sx + up * sy));
                return ax_brick_trace(&occupancy, camera_origin * grid_scale, direction * grid_scale, ray_limit);
            }}
        "#
        );
        let compiled = compile(&source).unwrap();
        assert_eq!(
            (
                compiled.interface().varying_inputs,
                compiled.interface().uniform_inputs
            ),
            (2, 16)
        );
        let graph = compiled.graph().parse().unwrap();
        let inputs: Vec<_> = (0..graph.tensor_count())
            .map(|index| graph.tensor(TensorId(index as u32)).unwrap())
            .filter(|tensor| tensor.flags.contains(TensorFlags::INPUT))
            .collect();
        assert_eq!(inputs.len(), 20);
        assert_eq!(
            inputs[..4]
                .iter()
                .map(|tensor| tensor.dimensions().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            [vec![1, 9], vec![3, 1], vec![3, 9], vec![3, 9]]
        );
        assert!(
            inputs[4..]
                .iter()
                .all(|tensor| tensor.dtype == DType::F32 && tensor.dimensions().eq([1, 1]))
        );
        let ops: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap().op)
            .collect();
        assert!(ops.contains(&OpCode::SQRT) && ops.contains(&OpCode::RECIPROCAL));
        assert_eq!(
            ops.iter()
                .filter(|&&op| op == OpCode::VOXEL_BRICK_TRACE)
                .count(),
            1
        );
    }

    #[test]
    fn voxel_distinct_call_sites_keep_their_own_resource_keys() {
        let other = VoxelBrickTraceParams {
            resource: ResourceKey {
                group: 2,
                binding: 7,
            },
            dims: [4, 8, 4],
        };
        let source = format!(
            "{PREFIX}@group(2) @binding(7) var<storage, read> other: array<vec2<u32>>; @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{ let a = {RAYS}; let b = {}; return vec4<f32>(a.t, b.t, select(0.0, 1.0, a.cell_id == b.cell_id), 1.0); }}",
            RAYS.replace("&occupancy", "&other")
        );
        let compiled = compile_wgsl_with_resources_in(
            &source,
            CompileOptions::for_viewport("main", 9, 3),
            &[other, RESOURCE],
            Global,
        )
        .unwrap();
        let graph = compiled.graph().parse().unwrap();
        let params: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap())
            .filter(|node| node.op == OpCode::VOXEL_BRICK_TRACE)
            .map(|node| decode_brick_trace_attributes(node.attributes()).unwrap())
            .collect();
        assert_eq!(params, [RESOURCE, other]);
    }

    #[test]
    fn voxel_resource_pointer_effects_and_speculation_are_rejected() {
        let cases = [
            (
                "pointer alias",
                shader(
                    &format!(
                        "let alias = &occupancy; return {};",
                        RAYS.replace("&occupancy", "alias")
                    ),
                    "AxVoxelHit",
                ),
            ),
            (
                "unused load",
                shader(
                    &format!("let ignored = occupancy[0]; return {RAYS};"),
                    "AxVoxelHit",
                ),
            ),
            (
                "array length",
                shader(
                    &format!("let ignored = arrayLength(&occupancy); return {RAYS};"),
                    "AxVoxelHit",
                ),
            ),
            (
                "indexed pointer",
                shader(
                    &format!("return {};", RAYS.replace("&occupancy", "&occupancy[0]")),
                    "AxVoxelHit",
                ),
            ),
            (
                "conditional query",
                shader(
                    &format!("if p.x > 0.0 {{ return {RAYS}; }} return {RAYS};"),
                    "AxVoxelHit",
                ),
            ),
            (
                "query after early return",
                shader(
                    &format!(
                        "if p.x < 0.0 {{ return vec4<f32>(0.0); }} let hit = {RAYS}; return vec4<f32>(hit.t);"
                    ),
                    "@location(0) vec4<f32>",
                ),
            ),
            (
                "short circuit query",
                shader(
                    &format!(
                        "let visible = p.x > 0.0 && {RAYS}.cell_id >= 0i; return vec4<f32>(select(0.0, 1.0, visible));"
                    ),
                    "@location(0) vec4<f32>",
                ),
            ),
            (
                "helper query",
                format!(
                    "{PREFIX}fn query() -> AxVoxelHit {{ return {RAYS}; }} @fragment fn main(@builtin(position) p: vec4<f32>) -> AxVoxelHit {{ return query(); }}"
                ),
            ),
            (
                "hidden helper query",
                format!(
                    "{PREFIX}fn query() -> AxVoxelHit {{ return {RAYS}; }} @fragment fn main(@builtin(position) p: vec4<f32>) -> AxVoxelHit {{ return {RAYS}; }}"
                ),
            ),
            (
                "pointer helper",
                format!(
                    "{PREFIX}fn inspect(grid: ptr<storage, array<vec2<u32>>, read>) -> f32 {{ return 1.0; }} @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{ return vec4<f32>(inspect(&occupancy)); }}"
                ),
            ),
        ];
        for (name, source) in cases {
            assert!(compile(&source).is_err(), "{name} unexpectedly compiled");
        }
    }

    #[test]
    fn voxel_known_record_and_query_surface_remain_narrow() {
        for (name, body, result) in [
            (
                "constructed hit",
                "return AxVoxelHit(1.0, -1i);".to_owned(),
                "AxVoxelHit",
            ),
            (
                "record selection",
                format!("let hit = {RAYS}; if p.x > 0.0 {{ return hit; }} return hit;"),
                "AxVoxelHit",
            ),
            (
                "varying limit",
                format!("return {};", RAYS.replace("700.0", "p.x")),
                "AxVoxelHit",
            ),
        ] {
            assert!(
                matches!(
                    compile(&shader(&body, result)),
                    Err(CompileError::Profile(_))
                ),
                "{name} did not fail at the explicit profile boundary"
            );
        }
        let bad_type = shader(
            &format!(
                "return {};",
                RAYS.replace("vec3<f32>(1.0)", "vec2<f32>(1.0)")
            ),
            "AxVoxelHit",
        );
        assert!(
            matches!(
                compile(&bad_type),
                Err(CompileError::Validation(_) | CompileError::Parse(_))
            ),
            "Naga must still validate ray argument types"
        );
    }

    #[test]
    fn voxel_declarations_and_specializations_are_checked_even_when_unused() {
        let source = shader(&format!("return {RAYS};"), "AxVoxelHit");
        for resources in [
            vec![],
            vec![RESOURCE, RESOURCE],
            vec![VoxelBrickTraceParams {
                dims: [0, 4, 4],
                ..RESOURCE
            }],
            vec![VoxelBrickTraceParams {
                resource: ResourceKey {
                    group: 9,
                    binding: 0,
                },
                ..RESOURCE
            }],
        ] {
            assert!(matches!(
                compile_wgsl_with_resources_in(
                    &source,
                    CompileOptions::for_viewport("main", 9, 3),
                    &resources,
                    Global
                ),
                Err(CompileError::Profile(_))
            ));
        }
        for (from, to) in [
            ("storage, read", "storage, read_write"),
            ("array<vec2<u32>>", "array<u32>"),
            ("array<vec2<u32>>", "array<vec2<u32>, 4>"),
        ] {
            assert!(compile(&source.replace(from, to)).is_err());
        }
        let unused = shader("return p;", "@location(0) vec4<f32>");
        assert!(compile(&unused.replace("array<vec2<u32>>", "array<u32>")).is_err());
    }

    #[test]
    fn voxel_enable_handling_preserves_original_utf8_locations() {
        let source = "/* π /* nested */ comment */\nenable /* λ */ ax_voxel_query;\n@group(1) @binding(0) var<storage, read> occupancy: array<vec2<u32>>;\n@fragment fn main(@builtin(position) p: vec4<f32>) -> AxVoxelHit {\n    var changed = 1.0;\n    return ax_brick_trace(&occupancy, vec3<f32>(changed), vec3<f32>(1.0), 700.0);\n}";
        let prepared = prepare_source(source, true).unwrap().0;
        let declaration_start = source.find("@group").unwrap();
        assert_eq!(
            &prepared[declaration_start..source.len()],
            &source[declaration_start..]
        );
        let Err(CompileError::Profile(diagnostic)) = compile(source) else {
            panic!("expected mutable-local profile rejection");
        };
        assert_eq!(diagnostic.line, Some(5));
        assert!(source[diagnostic.span.unwrap()].contains("changed"));
        for prefix in [
            "// enable ax_voxel_query;\n",
            "enable ax_voxel_query; enable ax_voxel_query;\n",
            "enable ax_voxel_query, f16;\n",
        ] {
            let text = format!(
                "{prefix}{}",
                source
                    .split_once("@group")
                    .map(|(_, tail)| format!("@group{tail}"))
                    .unwrap()
            );
            assert!(matches!(compile(&text), Err(CompileError::Profile(_))));
        }
        let unknown = format!(
            "enable ax_unknown;\n{}",
            shader(&format!("return {RAYS};"), "AxVoxelHit")
        );
        assert!(matches!(compile(&unknown), Err(CompileError::Parse(_))));
    }

    #[test]
    fn voxel_reserved_names_cannot_replace_generated_declarations() {
        for declaration in [
            "fn ax_brick_trace(x: f32) -> f32 { return x; }",
            "struct AxVoxelHit { t: f32, cell_id: i32 }",
            "alias AxVoxelHit = vec2<f32>;",
        ] {
            let source = shader(&format!("return {RAYS};"), "AxVoxelHit").replacen(
                "@fragment",
                &format!("{declaration}\n@fragment"),
                1,
            );
            assert!(matches!(compile(&source), Err(CompileError::Profile(_))));
        }
    }

    #[test]
    fn voxel_ids_admit_explicit_integer_math_selection_and_float_conversion() {
        for body in [
            format!(
                "let hit = {RAYS}; let id = select(-1i, hit.cell_id, p.x > 0.0); return vec4<f32>(select(0.0, 1.0, id >= 0i));"
            ),
            format!("let hit = {RAYS}; return vec4<f32>(select(0.0, 1.0, hit.cell_id + 1i > 0i));"),
            format!("let hit = {RAYS}; return vec4<f32>(f32(hit.cell_id));"),
        ] {
            let compiled = compile(&shader(&body, "@location(0) vec4<f32>")).unwrap();
            let graph = compiled.graph().parse().unwrap();
            let query = (0..graph.node_count())
                .map(|i| graph.node(i).unwrap())
                .find(|node| node.op == OpCode::VOXEL_BRICK_TRACE)
                .unwrap();
            assert_eq!(
                graph.tensor(query.outputs().nth(1).unwrap()).unwrap().dtype,
                DType::I32
            );
            for index in 0..graph.node_count() {
                assert!(ax::node_kernel_request(&graph, graph.node(index).unwrap()).is_some());
            }
        }
    }
}
