//! Opaque byte-buffer declaration for bounded, read-only U8 loads.
//!
//! The generated U32 member is parser scaffolding, not host storage. A bound
//! resource is an immutable byte slice; loads widen each byte exactly to I32
//! and return zero for a negative or out-of-range index.

use std::ops::Range;

use naga::{Function, Handle, Span};

use super::{CompileError, ProfileRule, profile};

pub(super) const DIRECTIVE: &str = "ax_buffer_access";
pub(super) const INTRINSIC: &str = "ax_load_u8";
pub(super) const HANDLE_TYPE: &str = "AxU8Buffer";
pub(super) const HANDLE_MEMBER: &str = "_ax_u8_buffer_handle";
pub(super) const DECLARATIONS: &str = r#"
struct AxU8Buffer {
    _ax_u8_buffer_handle: u32,
}
fn ax_load_u8(buffer: ptr<storage, AxU8Buffer, read>, index: i32) -> i32 {
    return 0i;
}
"#;

#[derive(Clone, Copy)]
pub(super) struct Extension {
    pub intrinsic: Handle<Function>,
    pub resource_type: Handle<naga::Type>,
}

/// Reject aliases before Naga erases their source identity. Only the exact
/// opaque annotation of a root read-only storage global is admitted.
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
            "AxU8Buffer may only annotate a read-only root storage global; handle construction, type aliases and other type uses are unsupported",
            Span::new(range.start as u32, range.end as u32),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::alloc::Global;

    use ax::buffer::U8BufferParams;
    use ax::sampling::{VolumeRgbParams, VolumeRgbStorage};
    use ax::voxel::{ResourceKey, VoxelBrickTraceParams};
    use ax::{DType, NodeFlags, OpCode, TensorFlags, TensorId};

    use super::*;
    use crate::{
        CompileOptions, CompiledShader, ShaderResource, compile_wgsl_in,
        compile_wgsl_with_resource_specs_in, compile_wgsl_with_resources_in, lower_naga_module_in,
    };

    const PREFIX: &str = "enable ax_buffer_access;\n@group(1) @binding(2) var<storage, read> materials: AxU8Buffer;\n";

    fn params() -> U8BufferParams {
        U8BufferParams::new(
            ResourceKey {
                group: 1,
                binding: 2,
            },
            4096,
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
    fn byte_loads_widen_to_i32_and_keep_resource_bytes_outside_the_graph() {
        for (globals, arguments, index, shape, inputs) in [
            ("", "", "-1i", [1, 1], 2),
            ("", "", "i32(p.x)", [1, 9], 2),
            ("", "", "i32(p.y)", [3, 1], 2),
            (
                "@group(0) @binding(0) var<uniform> index: f32;",
                "",
                "i32(index)",
                [1, 1],
                3,
            ),
            ("", ", @location(0) index: f32", "i32(index)", [3, 9], 3),
        ] {
            let text = source(
                globals,
                arguments,
                &format!(
                    "let material = ax_load_u8(&materials, {index}); return vec4<f32>(f32(material), f32(material + 1i), select(0.0, 1.0, material != 0i), 1.0);"
                ),
            );
            let compiled = compile(&text, &[ShaderResource::U8Buffer(params())]).unwrap();
            let graph = compiled.graph().parse().unwrap();
            let loads: Vec<_> = (0..graph.node_count())
                .map(|i| graph.node(i).unwrap())
                .filter(|node| node.op == OpCode::BUFFER_LOAD_U8)
                .collect();
            assert_eq!(
                loads.len(),
                1,
                "each use must share its registered call result"
            );
            let load = loads[0];
            assert_eq!(load.flags, NodeFlags::READS_RESOURCE);
            let index = graph.tensor(load.inputs().next().unwrap()).unwrap();
            assert_eq!(index.dtype, DType::I32);
            assert!(index.dimensions().eq(shape));
            let output = graph.tensor(load.outputs().next().unwrap()).unwrap();
            assert_eq!(output.dtype, DType::I32);
            assert!(output.dimensions().eq([3, 9]));
            assert!(output.constant().is_empty());
            assert!(ax::node_kernel_request(&graph, load).is_some());
            assert_eq!(
                (0..graph.tensor_count())
                    .map(|i| graph.tensor(TensorId(i as u32)).unwrap())
                    .filter(|tensor| tensor.flags.contains(TensorFlags::INPUT))
                    .count(),
                inputs
            );
            assert!(compiled.graph().as_bytes().len() < params().elements as usize);
        }
    }

    #[test]
    fn byte_voxel_and_rgb_resources_coexist_with_exact_integer_consumers() {
        let text = r#"
            enable ax_voxel_query;
            enable ax_volume_sampling;
            enable ax_buffer_access;
            @group(1) @binding(0) var<storage, read> grid: array<vec2<u32>>;
            @group(1) @binding(1) var<storage, read> light: AxRgbVolume;
            @group(1) @binding(2) var<storage, read> materials: AxU8Buffer;
            struct Result {
                @location(0) @interpolate(flat) id: i32,
                @location(1) color: vec3<f32>,
            }
            fn palette(code: i32) -> vec3<f32> {
                if code == 4i { return vec3<f32>(0.61, 0.235, 0.13); }
                return vec3<f32>(0.5);
            }
            @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {
                let hit = ax_brick_trace(&grid, p.xyz, vec3<f32>(0.0, 0.0, 1.0), 700.0);
                let code = ax_load_u8(&materials, select(-1i, hit.cell_id, hit.cell_id >= 0i));
                let light_value = ax_sample_volume_rgb(&light, vec3<f32>(f32(code), 0.0, 0.0));
                return Result(hit.cell_id, palette(code) * light_value);
            }
        "#;
        let resources = [
            ShaderResource::VoxelBrickGrid(
                VoxelBrickTraceParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 0,
                    },
                    [4; 3],
                )
                .unwrap(),
            ),
            ShaderResource::RgbVolume(
                VolumeRgbParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 1,
                    },
                    [1; 3],
                    VolumeRgbStorage::Uf16E5M11,
                )
                .unwrap(),
            ),
            ShaderResource::U8Buffer(params()),
        ];
        let compiled = compile(text, &resources).unwrap();
        let graph = compiled.graph().parse().unwrap();
        for op in [
            OpCode::VOXEL_BRICK_TRACE,
            OpCode::VOLUME_TRILINEAR_RGB,
            OpCode::BUFFER_LOAD_U8,
        ] {
            assert_eq!(
                (0..graph.node_count())
                    .map(|i| graph.node(i).unwrap())
                    .filter(|node| node.op == op)
                    .count(),
                1
            );
        }
        assert_eq!(compiled.interface().output_dtype(0), Some(DType::I32));
        for index in 0..graph.node_count() {
            assert!(ax::node_kernel_request(&graph, graph.node(index).unwrap()).is_some());
        }
    }

    #[test]
    fn aftershock_solid_scene_compiles_with_all_three_resource_kinds() {
        let resources = [
            ShaderResource::VoxelBrickGrid(
                VoxelBrickTraceParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 0,
                    },
                    [320, 192, 320],
                )
                .unwrap(),
            ),
            ShaderResource::RgbVolume(
                VolumeRgbParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 1,
                    },
                    [24, 12, 24],
                    VolumeRgbStorage::Uf16E5M11,
                )
                .unwrap(),
            ),
            ShaderResource::U8Buffer(
                U8BufferParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 2,
                    },
                    19_660_800,
                )
                .unwrap(),
            ),
        ];
        let compiled = compile_wgsl_with_resource_specs_in(
            include_str!("../../../games/aftershock/assets/scene.wgsl"),
            CompileOptions::for_viewport("main", 2560, 1440),
            &resources,
            Global,
        )
        .unwrap();
        let interface = compiled.interface();
        assert_eq!(
            interface.position_inputs + interface.varying_inputs + interface.uniform_inputs,
            18
        );
        assert_eq!(compiled.interface().output_dtype(0), Some(DType::I32));
        assert_eq!(compiled.interface().output_dtype(1), Some(DType::F32));
        let graph = compiled.graph().parse().unwrap();
        for (op, expected) in [
            (OpCode::VOXEL_BRICK_TRACE, 2),
            (OpCode::BUFFER_LOAD_U8, 7),
            (OpCode::VOLUME_TRILINEAR_RGB, 1),
        ] {
            assert_eq!(
                (0..graph.node_count())
                    .map(|i| graph.node(i).unwrap())
                    .filter(|node| node.op == op)
                    .count(),
                expected
            );
        }
        for index in 0..graph.node_count() {
            assert!(
                ax::node_kernel_request(&graph, graph.node(index).unwrap()).is_some(),
                "node {index} has no canonical typed request"
            );
        }
    }

    #[test]
    fn byte_buffers_require_explicit_registration_and_valid_matching_resources() {
        let text = source("", "", "return vec4<f32>(f32(ax_load_u8(&materials, 0i)));");
        let specs = [ShaderResource::U8Buffer(params())];
        assert!(
            compile_wgsl_in(&text, CompileOptions::for_viewport("main", 9, 3), Global).is_err()
        );
        assert!(
            compile_wgsl_with_resources_in(
                &text,
                CompileOptions::for_viewport("main", 9, 3),
                &[],
                Global
            )
            .is_err()
        );
        let prepared = crate::voxel::prepare_specs(&text, "main", &specs).unwrap();
        assert!(
            lower_naga_module_in(
                &prepared.module,
                "",
                CompileOptions::for_viewport("main", 9, 3),
                Global
            )
            .is_err()
        );
        for resources in [
            vec![],
            vec![specs[0], specs[0]],
            vec![ShaderResource::U8Buffer(U8BufferParams {
                resource: ResourceKey {
                    group: 1,
                    binding: 3,
                },
                ..params()
            })],
            vec![ShaderResource::VoxelBrickGrid(
                VoxelBrickTraceParams::new(params().resource, [4; 3]).unwrap(),
            )],
        ] {
            assert!(compile(&text, &resources).is_err());
        }
        for text in [
            text.replace("storage, read", "storage, read_write"),
            text.replace("enable ax_buffer_access;", ""),
            text.replace(
                "enable ax_buffer_access;",
                "enable ax_buffer_access; enable ax_buffer_access;",
            ),
            text.replace("AxU8Buffer", "array<u32>"),
        ] {
            assert!(compile(&text, &specs).is_err());
        }
    }

    #[test]
    fn byte_buffers_reject_aliases_effects_and_conditional_calls_even_when_unused() {
        let specs = [ShaderResource::U8Buffer(params())];
        for (globals, body) in [
            ("alias Bytes = AxU8Buffer;", "return vec4<f32>(1.0);"),
            (
                "struct Wrapper { bytes: AxU8Buffer }",
                "return vec4<f32>(1.0);",
            ),
            (
                "",
                "let alias = &materials; return vec4<f32>(f32(ax_load_u8(alias, 0i)));",
            ),
            ("", "let ignored = materials; return vec4<f32>(1.0);"),
            (
                "",
                "let ignored = materials._ax_u8_buffer_handle; return vec4<f32>(1.0);",
            ),
            (
                "",
                "if p.x > 0.0 { return vec4<f32>(f32(ax_load_u8(&materials, 0i))); } return vec4<f32>(0.0);",
            ),
            (
                "",
                "if p.x > 0.0 { return vec4<f32>(0.0); } return vec4<f32>(f32(ax_load_u8(&materials, 0i)));",
            ),
            (
                "",
                "let visible = p.x > 0.0 && ax_load_u8(&materials, 0i) != 0i; return vec4<f32>(select(0.0, 1.0, visible));",
            ),
            (
                "fn hidden() -> i32 { return ax_load_u8(&materials, 0i); }",
                "return vec4<f32>(1.0);",
            ),
            (
                "fn inspect(value: ptr<storage, AxU8Buffer, read>) -> i32 { return 0i; }",
                "return vec4<f32>(1.0);",
            ),
        ] {
            let text = source(globals, "", body);
            assert!(compile(&text, &specs).is_err(), "accepted {globals} {body}");
        }
        for index in ["p.x", "vec2<i32>(0i)", "true", "0u"] {
            let text = source(
                "",
                "",
                &format!("return vec4<f32>(f32(ax_load_u8(&materials, {index})));"),
            );
            assert!(matches!(
                compile(&text, &specs),
                Err(CompileError::Parse(_) | CompileError::Validation(_))
            ));
        }
    }

    #[test]
    fn byte_buffer_names_and_node_expansion_keep_the_existing_bounds() {
        let specs = [ShaderResource::U8Buffer(params())];
        for declaration in [
            "struct AxU8Buffer { value: u32 }",
            "alias AxU8Buffer = u32;",
            "fn ax_load_u8(index: i32) -> i32 { return index; }",
            "const ax_load_u8: i32 = 0i;",
            "struct Fake { _ax_u8_buffer_handle: u32 }",
        ] {
            assert!(compile(&source(declaration, "", "return vec4<f32>(1.0);"), &specs).is_err());
        }
        let mut builder = crate::ShaderBuilder::new_in(Global);
        let index = builder
            .tensor(
                DType::I32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &0i32.to_le_bytes(),
            )
            .unwrap();
        builder.nodes = crate::MAX_GRAPH_NODES;
        let CompileError::Profile(error) =
            builder.buffer_load_u8(index, params(), [3, 9]).unwrap_err()
        else {
            panic!("expected the shared node budget");
        };
        assert_eq!(error.rule, ProfileRule::Budget);
    }
}
