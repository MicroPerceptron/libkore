#![feature(allocator_api)]

use ax::{buffer::U8BufferParams, voxel::ResourceKey};
use std::{alloc::Global, env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=assets/ui/terminal_tile.wgsl");
    let resources = [
        axshader::ShaderResource::U8Buffer(
            U8BufferParams::new(
                ResourceKey {
                    group: 1,
                    binding: 0,
                },
                640 * 180 * 10,
            )
            .unwrap(),
        ),
        axshader::ShaderResource::U8Buffer(
            U8BufferParams::new(
                ResourceKey {
                    group: 1,
                    binding: 1,
                },
                kore_window::appearance::Density::ALL
                    .iter()
                    .map(|d| 224 * d.pixels(12) * d.pixels(24) * 2)
                    .sum(),
            )
            .unwrap(),
        ),
    ];
    let terminal = axshader::compile_wgsl_with_resource_specs_in(
        include_str!("assets/ui/terminal_tile.wgsl"),
        axshader::CompileOptions::for_viewport("main", 128, 32),
        &resources,
        Global,
    )
    .expect("retained terminal tile WGSL must pass the ax profile");
    assert_eq!(terminal.interface().position_inputs, 2);
    assert_eq!(terminal.interface().varying_inputs, 0);
    assert_eq!(terminal.interface().uniform_inputs, 28);
    assert_eq!(terminal.interface().output_count(), 1);
    assert_eq!(terminal.interface().output_dtype(0), Some(ax::DType::I32));
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("terminal_tile.ax"),
        terminal.graph().as_bytes(),
    )
    .unwrap();
}
