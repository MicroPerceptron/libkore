//! Opaque RGB-volume declaration for the explicit sampling extension.
//!
//! The generated U32 member is parser scaffolding, not a claim about host
//! storage layout. Resource specialization alone selects tightly packed RGB
//! F32 (12 bytes/point) or UF16 E5M11 (6 bytes/point). Authored code cannot
//! construct, alias, inspect or load this handle.

use std::ops::Range;

use naga::{Function, Handle, Span};

use super::{CompileError, ProfileRule, profile};

pub(super) const DIRECTIVE: &str = "ax_volume_sampling";
pub(super) const INTRINSIC: &str = "ax_sample_volume_rgb";
pub(super) const HANDLE_TYPE: &str = "AxRgbVolume";
pub(super) const HANDLE_MEMBER: &str = "_ax_rgb_volume_handle";
pub(super) const DECLARATIONS: &str = r#"
struct AxRgbVolume {
    _ax_rgb_volume_handle: u32,
}
fn ax_sample_volume_rgb(
    volume: ptr<storage, AxRgbVolume, read>, point: vec3<f32>
) -> vec3<f32> {
    return vec3<f32>(0.0);
}
"#;

#[derive(Clone, Copy)]
pub(super) struct Extension {
    pub intrinsic: Handle<Function>,
    pub resource_type: Handle<naga::Type>,
}

/// Naga does not retain source type aliases. Admit the exact opaque spelling
/// only in a root storage annotation before parsing, then validate its global
/// address space, binding and specialization in the shared resource pass.
pub(super) fn validate_handle_use(
    source: &str,
    tokens: &[Range<usize>],
    index: usize,
) -> Result<(), CompileError> {
    let token = |offset: usize| tokens.get(offset).map(|range| &source[range.clone()]);
    let valid_annotation = index >= 8
        && token(index - 8) == Some("var")
        && token(index - 7) == Some("<")
        && token(index - 6) == Some("storage")
        && token(index - 5) == Some(",")
        && token(index - 4) == Some("read")
        && token(index - 3) == Some(">")
        && token(index - 1) == Some(":")
        && token(index + 1) == Some(";");
    if !valid_annotation {
        let range = &tokens[index];
        return Err(profile(
            source,
            ProfileRule::Resource,
            "AxRgbVolume may only annotate a read-only root storage global; handle construction, type aliases and other type uses are unsupported",
            Span::new(range.start as u32, range.end as u32),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::alloc::Global;

    use ax::sampling::{
        ResourceKey, VolumeRgbParams, VolumeRgbStorage, decode_volume_rgb_attributes,
        encode_volume_rgb_attributes,
    };
    use ax::{DType, NodeFlags, OpCode, TensorFlags, TensorId};

    use super::*;
    use crate::{
        CompileOptions, CompiledShader, ShaderResource, compile_wgsl_in,
        compile_wgsl_with_resource_specs_in, compile_wgsl_with_resources_in, lower_naga_module_in,
    };

    const PREFIX: &str = "enable ax_volume_sampling;\n@group(2) @binding(1) var<storage, read> volume: AxRgbVolume;\n";

    fn params(storage: VolumeRgbStorage) -> VolumeRgbParams {
        VolumeRgbParams::new(
            ResourceKey {
                group: 2,
                binding: 1,
            },
            [5, 3, 7],
            storage,
        )
        .unwrap()
    }

    fn source(globals: &str, arguments: &str, body: &str) -> String {
        format!(
            "{PREFIX}{globals}\n@fragment fn main(@builtin(position) p: vec4<f32>{arguments}) -> @location(0) vec4<f32> {{ {body} }}"
        )
    }

    fn compile(
        source: &str,
        resources: &[ShaderResource],
    ) -> Result<CompiledShader<Global>, CompileError> {
        compile_wgsl_with_resource_specs_in(
            source,
            CompileOptions::for_viewport("main", 9, 3),
            resources,
            Global,
        )
    }

    #[test]
    fn rgb_sampling_emits_three_f32_planes_for_each_storage_and_coordinate_shape() {
        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            for (globals, arguments, point, input_count, shapes) in [
                ("", "", "vec3<f32>(1.25, 2.5, 0.5)", 2, [[1, 1]; 3]),
                ("", "", "vec3<f32>(p.xy, 0.5)", 2, [[1, 9], [3, 1], [1, 1]]),
                (
                    "",
                    ", @location(0) point: vec3<f32>",
                    "point",
                    5,
                    [[3, 9]; 3],
                ),
                (
                    "@group(0) @binding(7) var<uniform> point: vec3<f32>;",
                    "",
                    "point",
                    5,
                    [[1, 1]; 3],
                ),
            ] {
                let source = source(
                    globals,
                    arguments,
                    &format!(
                        "let sampled = ax_sample_volume_rgb(&volume, {point}); return vec4<f32>(sampled.zyx, sampled.x + sampled.y);"
                    ),
                );
                let parameters = params(storage);
                let compiled = compile(&source, &[ShaderResource::RgbVolume(parameters)]).unwrap();
                assert_eq!(&compiled.graph().as_bytes()[4..8], &[1, 0, 1, 0]);
                let graph = compiled.graph().parse().unwrap();
                let samples: Vec<_> = (0..graph.node_count())
                    .map(|index| graph.node(index).unwrap())
                    .filter(|node| node.op == OpCode::VOLUME_TRILINEAR_RGB)
                    .collect();
                assert_eq!(
                    samples.len(),
                    1,
                    "sampled members must share one resource call"
                );
                let sample = samples[0];
                assert_eq!(sample.flags, NodeFlags::READS_RESOURCE);
                assert_eq!(
                    decode_volume_rgb_attributes(sample.attributes()),
                    Some(parameters)
                );
                assert_eq!(
                    sample.attributes(),
                    encode_volume_rgb_attributes(parameters).unwrap()
                );
                for (coordinate, shape) in sample.inputs().zip(shapes) {
                    let tensor = graph.tensor(coordinate).unwrap();
                    assert_eq!(tensor.dtype, DType::F32);
                    assert!(tensor.dimensions().eq(shape));
                }
                let outputs: Vec<_> = sample.outputs().collect();
                assert_eq!(outputs.len(), 3);
                for output in outputs {
                    let tensor = graph.tensor(output).unwrap();
                    assert_eq!(tensor.dtype, DType::F32);
                    assert!(tensor.dimensions().eq([3, 9]));
                    assert!(tensor.constant().is_empty());
                }
                assert_eq!(
                    (0..graph.tensor_count())
                        .filter(|&index| graph
                            .tensor(TensorId(index as u32))
                            .unwrap()
                            .flags
                            .contains(TensorFlags::INPUT))
                        .count(),
                    input_count
                );
            }
        }
    }

    #[test]
    fn rgb_sampling_requires_explicit_api_and_rejects_authored_scaffolding() {
        let parameters = [ShaderResource::RgbVolume(params(VolumeRgbStorage::F32))];
        let text = source(
            "",
            "",
            "return vec4<f32>(ax_sample_volume_rgb(&volume, p.xyz), 1.0);",
        );
        assert!(
            compile_wgsl_in(&text, CompileOptions::for_viewport("main", 9, 3), Global).is_err()
        );
        assert!(
            compile_wgsl_with_resources_in(
                &text,
                CompileOptions::for_viewport("main", 9, 3),
                &[],
                Global,
            )
            .is_err()
        );
        assert!(
            compile(
                &text.replacen("enable ax_volume_sampling;", "", 1),
                &parameters
            )
            .is_err()
        );
        let prepared = crate::voxel::prepare_specs(&text, "main", &parameters).unwrap();
        assert!(
            lower_naga_module_in(
                &prepared.module,
                "",
                CompileOptions::for_viewport("main", 9, 3),
                Global,
            )
            .is_err(),
            "a pre-parsed module cannot register the generated intrinsic"
        );
        for declaration in [
            "struct AxRgbVolume { value: u32 }",
            "alias AxRgbVolume = u32;",
            "const AxRgbVolume: u32 = 0u;",
            "fn ax_sample_volume_rgb(point: vec3<f32>) -> vec3<f32> { return point; }",
            "const ax_sample_volume_rgb: f32 = 0.0;",
            "struct Counterfeit { _ax_rgb_volume_handle: u32 }",
        ] {
            let text = source(declaration, "", "return vec4<f32>(1.0);");
            assert!(
                compile(&text, &parameters).is_err(),
                "accepted {declaration}"
            );
            let ordinary = format!(
                "{declaration}\n@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{ return vec4<f32>(1.0); }}"
            );
            assert!(
                compile_wgsl_in(
                    &ordinary,
                    CompileOptions::for_viewport("main", 9, 3),
                    Global
                )
                .is_err()
            );
        }
    }

    #[test]
    fn rgb_sampling_rejects_opaque_type_uses_and_resource_effects_even_when_unused() {
        let parameters = [ShaderResource::RgbVolume(params(VolumeRgbStorage::F32))];
        for (name, globals, body) in [
            (
                "type alias",
                "alias CopyVolume = AxRgbVolume;",
                "return vec4<f32>(1.0);",
            ),
            (
                "nested handle",
                "struct Wrapped { value: AxRgbVolume }",
                "return vec4<f32>(1.0);",
            ),
            (
                "array of handles",
                "@group(3) @binding(0) var<storage, read> nested: array<AxRgbVolume>;",
                "return vec4<f32>(1.0);",
            ),
            (
                "helper handle parameter",
                "fn inspect_volume(value: AxRgbVolume) -> f32 { return 0.0; }",
                "return vec4<f32>(1.0);",
            ),
            (
                "helper pointer parameter",
                "fn inspect_volume(value: ptr<storage, AxRgbVolume, read>) -> f32 { return 0.0; }",
                "return vec4<f32>(1.0);",
            ),
            (
                "private handle",
                "var<private> copy_volume: AxRgbVolume;",
                "return vec4<f32>(1.0);",
            ),
            (
                "constructor",
                "",
                "let made = AxRgbVolume(0u); return vec4<f32>(1.0);",
            ),
            (
                "zero constructor",
                "",
                "let made = AxRgbVolume(); return vec4<f32>(1.0);",
            ),
            (
                "local handle",
                "",
                "var made: AxRgbVolume; return vec4<f32>(1.0);",
            ),
            (
                "copy load",
                "",
                "let copied = volume; return vec4<f32>(1.0);",
            ),
            (
                "member load",
                "",
                "let member = volume._ax_rgb_volume_handle; return vec4<f32>(1.0);",
            ),
            (
                "pointer alias",
                "",
                "let pointer = &volume; return vec4<f32>(ax_sample_volume_rgb(pointer, p.xyz), 1.0);",
            ),
            (
                "unused pointer alias",
                "",
                "let pointer = &volume; return vec4<f32>(1.0);",
            ),
            (
                "member pointer",
                "",
                "let pointer = &volume._ax_rgb_volume_handle; return vec4<f32>(1.0);",
            ),
            (
                "mutation",
                "",
                "volume._ax_rgb_volume_handle = 1u; return vec4<f32>(1.0);",
            ),
            (
                "indexing",
                "",
                "let value = volume[0u]; return vec4<f32>(1.0);",
            ),
            (
                "array length",
                "",
                "let size = arrayLength(&volume); return vec4<f32>(1.0);",
            ),
            (
                "member instead of root",
                "",
                "return vec4<f32>(ax_sample_volume_rgb(&volume._ax_rgb_volume_handle, p.xyz), 1.0);",
            ),
        ] {
            let text = source(globals, "", body);
            let error = compile(&text, &parameters).expect_err(name);
            assert!(
                error.to_string().len() < 16_384,
                "unbounded {name} diagnostic"
            );
        }
        for replacement in [
            "var<storage, read_write> volume: AxRgbVolume;",
            "var<uniform> volume: AxRgbVolume;",
            "var<storage, read> volume: array<vec3<f32>>;",
            "var<storage, read> volume: array<u32>;",
        ] {
            let text = source("", "", "return vec4<f32>(1.0);")
                .replace("var<storage, read> volume: AxRgbVolume;", replacement);
            assert!(
                compile(&text, &parameters).is_err(),
                "accepted {replacement}"
            );
        }
    }

    #[test]
    fn rgb_sampling_keeps_unconditional_entry_only_calls_and_naga_argument_validation() {
        let parameters = [ShaderResource::RgbVolume(params(VolumeRgbStorage::F32))];
        for (name, globals, body) in [
            (
                "helper",
                "fn lookup_volume() -> vec3<f32> { return ax_sample_volume_rgb(&volume, vec3<f32>(0.0)); }",
                "return vec4<f32>(lookup_volume(), 1.0);",
            ),
            (
                "unused helper",
                "fn lookup_volume() -> vec3<f32> { return ax_sample_volume_rgb(&volume, vec3<f32>(0.0)); }",
                "return vec4<f32>(1.0);",
            ),
            (
                "branch",
                "",
                "if p.x > 0.0 { return vec4<f32>(ax_sample_volume_rgb(&volume, p.xyz), 1.0); } return vec4<f32>(0.0);",
            ),
            (
                "after possible return",
                "",
                "if p.x < 0.0 { return vec4<f32>(0.0); } return vec4<f32>(ax_sample_volume_rgb(&volume, p.xyz), 1.0);",
            ),
            (
                "short circuit",
                "",
                "let selected = p.x > 0.0 && ax_sample_volume_rgb(&volume, p.xyz).x > 0.0; return vec4<f32>(select(0.0, 1.0, selected));",
            ),
            (
                "loop",
                "",
                "loop { let sampled = ax_sample_volume_rgb(&volume, p.xyz); break; } return vec4<f32>(1.0);",
            ),
            (
                "unselected fragment",
                "@fragment fn other(@builtin(position) q: vec4<f32>) -> @location(0) vec4<f32> { return vec4<f32>(ax_sample_volume_rgb(&volume, q.xyz), 1.0); }",
                "return vec4<f32>(1.0);",
            ),
            (
                "unselected compute",
                "@compute @workgroup_size(1) fn other() { let sampled = ax_sample_volume_rgb(&volume, vec3<f32>(0.0)); }",
                "return vec4<f32>(1.0);",
            ),
        ] {
            let text = source(globals, "", body);
            let CompileError::Profile(error) = compile(&text, &parameters).expect_err(name) else {
                panic!("{name} must reach resource-call profile validation");
            };
            assert_eq!(error.rule, ProfileRule::ControlFlow, "{name}");
        }
        for point in [
            "1.0",
            "vec2<f32>(0.0)",
            "vec4<f32>(0.0)",
            "vec3<i32>(0i)",
            "true",
        ] {
            let text = source(
                "",
                "",
                &format!("return vec4<f32>(ax_sample_volume_rgb(&volume, {point}), 1.0);"),
            );
            assert!(
                matches!(
                    compile(&text, &parameters),
                    Err(CompileError::Parse(_) | CompileError::Validation(_))
                ),
                "Naga must validate point argument {point}"
            );
        }
        // Resource results may be passed to ordinary pure numeric helpers;
        // only moving the actual resource call into a helper is prohibited.
        let text = source(
            "fn tint(rgb: vec3<f32>) -> vec3<f32> { return rgb * 0.5; }",
            "",
            "return vec4<f32>(tint(ax_sample_volume_rgb(&volume, p.xyz)), 1.0);",
        );
        compile(&text, &parameters).unwrap();
    }

    #[test]
    fn rgb_resource_specs_validate_kind_geometry_bindings_and_unused_declarations() {
        let valid = params(VolumeRgbStorage::Uf16E5M11);
        let text = source("", "", "return vec4<f32>(1.0);");
        for resources in [
            vec![],
            vec![
                ShaderResource::RgbVolume(valid),
                ShaderResource::RgbVolume(valid),
            ],
            vec![
                ShaderResource::RgbVolume(valid),
                ShaderResource::RgbVolume(params(VolumeRgbStorage::F32)),
            ],
            vec![ShaderResource::VoxelBrickGrid(
                ax::voxel::VoxelBrickTraceParams::new(valid.resource, [8; 3]).unwrap(),
            )],
            vec![ShaderResource::RgbVolume(VolumeRgbParams {
                resource: ResourceKey {
                    group: 9,
                    binding: 9,
                },
                ..valid
            })],
            vec![
                ShaderResource::RgbVolume(valid),
                ShaderResource::RgbVolume(VolumeRgbParams {
                    resource: ResourceKey {
                        group: 9,
                        binding: 9,
                    },
                    ..valid
                }),
            ],
        ] {
            assert!(
                compile(&text, &resources).is_err(),
                "accepted invalid spec set {resources:?}"
            );
        }
        for dims in [
            [0, 3, 7],
            [VolumeRgbParams::MAX_DIMENSION + 1, 1, 1],
            [VolumeRgbParams::MAX_DIMENSION; 3],
            [u32::MAX; 3],
        ] {
            assert!(
                compile(
                    &text,
                    &[ShaderResource::RgbVolume(VolumeRgbParams { dims, ..valid })]
                )
                .is_err()
            );
        }
        let duplicate = source(
            "@group(2) @binding(1) var<uniform> duplicate: f32;",
            "",
            "return vec4<f32>(1.0);",
        );
        assert!(compile(&duplicate, &[ShaderResource::RgbVolume(valid)]).is_err());
        // Bindings are declaration contracts, not a requirement to call every
        // resource in every shader variant. Extra undeclared specs still fail.
        let compiled = compile(&text, &[ShaderResource::RgbVolume(valid)]).unwrap();
        let graph = compiled.graph().parse().unwrap();
        assert!(
            (0..graph.node_count())
                .all(|index| graph.node(index).unwrap().op != OpCode::VOLUME_TRILINEAR_RGB)
        );
    }

    #[test]
    fn voxel_and_rgb_sampling_coexist_with_typed_record_outputs_and_distinct_resources() {
        let grid = ax::voxel::VoxelBrickTraceParams::new(
            ResourceKey {
                group: 1,
                binding: 0,
            },
            [8; 3],
        )
        .unwrap();
        let volume = params(VolumeRgbStorage::Uf16E5M11);
        let text = format!(
            r#"enable ax_voxel_query;
            {PREFIX}
            @group(1) @binding(0) var<storage, read> grid: array<vec2<u32>>;
            struct Result {{
                @location(0) distance: f32,
                @location(1) @interpolate(flat) id: i32,
                @location(2) colour: vec3<f32>,
            }}
            @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {{
                let hit = ax_brick_trace(&grid, vec3<f32>(p.xy, 0.5), vec3<f32>(1.0, 0.0, 0.0), 100.0);
                let sampled = ax_sample_volume_rgb(&volume, vec3<f32>(p.xy, hit.t * 0.25));
                return Result(hit.t, hit.cell_id, sampled);
            }}"#
        );
        let compiled = compile(
            &text,
            &[
                ShaderResource::RgbVolume(volume),
                ShaderResource::VoxelBrickGrid(grid),
            ],
        )
        .unwrap();
        assert_eq!(compiled.interface().output_count(), 5);
        assert_eq!(compiled.interface().output_dtype(1), Some(DType::I32));
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
                .filter(|node| node.op == OpCode::VOLUME_TRILINEAR_RGB)
                .count(),
            1
        );
        let sample = nodes
            .iter()
            .find(|node| node.op == OpCode::VOLUME_TRILINEAR_RGB)
            .unwrap();
        assert_eq!(
            decode_volume_rgb_attributes(sample.attributes()),
            Some(volume)
        );
        for wrong in [
            text.replace("ax_sample_volume_rgb(&volume", "ax_sample_volume_rgb(&grid"),
            text.replace("ax_brick_trace(&grid", "ax_brick_trace(&volume"),
        ] {
            assert!(
                compile(
                    &wrong,
                    &[
                        ShaderResource::RgbVolume(volume),
                        ShaderResource::VoxelBrickGrid(grid)
                    ]
                )
                .is_err()
            );
        }
        let other = VolumeRgbParams {
            resource: ResourceKey {
                group: 2,
                binding: 3,
            },
            storage: VolumeRgbStorage::F32,
            ..volume
        };
        let text = source(
            "@group(2) @binding(3) var<storage, read> other_volume: AxRgbVolume;",
            "",
            "let a = ax_sample_volume_rgb(&volume, p.xyz); let b = ax_sample_volume_rgb(&other_volume, p.xyz + vec3<f32>(1.0)); return vec4<f32>(a + b, 1.0);",
        );
        let compiled = compile(
            &text,
            &[
                ShaderResource::RgbVolume(other),
                ShaderResource::RgbVolume(volume),
            ],
        )
        .unwrap();
        let graph = compiled.graph().parse().unwrap();
        let samples: Vec<_> = (0..graph.node_count())
            .map(|index| graph.node(index).unwrap())
            .filter(|node| node.op == OpCode::VOLUME_TRILINEAR_RGB)
            .map(|node| decode_volume_rgb_attributes(node.attributes()).unwrap())
            .collect();
        assert_eq!(samples, [volume, other]);
    }

    #[test]
    fn resource_directives_preserve_utf8_offsets_and_reject_ambiguous_enables() {
        let parameters = [ShaderResource::RgbVolume(params(VolumeRgbStorage::F32))];
        let text = "/* π /* nested */ comment */\nenable /* λ */ ax_volume_sampling;\nenable /* μ */ ax_voxel_query;\n@group(2) @binding(1) var<storage, read> volume: AxRgbVolume;\n@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {\n let sampled = ax_sample_volume_rgb(&volume, p.xyz);\n return vec4<f32>(sampled, 1.0);\n}";
        let prepared = crate::voxel::prepare_specs(text, "main", &parameters).unwrap();
        let declarations = text.find("@group").unwrap();
        assert_eq!(
            &prepared.source[declarations..text.len()],
            &text[declarations..]
        );
        assert_eq!(
            prepared.source[..text.len()]
                .bytes()
                .filter(|&byte| byte == b'\n')
                .count(),
            text.bytes().filter(|&byte| byte == b'\n').count()
        );
        compile(text, &parameters).unwrap();
        let bad = text.replace(
            "let sampled = ax_sample_volume_rgb(&volume, p.xyz);",
            "let unused = volume._ax_rgb_volume_handle; let sampled = vec3<f32>(0.0);",
        );
        let CompileError::Profile(error) = compile(&bad, &parameters).unwrap_err() else {
            panic!("expected original-source resource diagnostic")
        };
        assert_eq!(error.line, Some(6));
        let span = error.span.unwrap();
        assert!(span.end <= bad.len() && bad[span].contains("volume"));
        for directive in [
            "// enable ax_volume_sampling;",
            "enable ax_volume_sampling; enable ax_volume_sampling;",
            "enable ax_volume_sampling, ax_voxel_query;",
            "enable ax_volume_sampling, f16;",
            "enable unknown_resource_extension;",
        ] {
            let text = source("", "", "return vec4<f32>(1.0);").replacen(
                "enable ax_volume_sampling;",
                directive,
                1,
            );
            assert!(compile(&text, &parameters).is_err(), "accepted {directive}");
        }
    }
}
