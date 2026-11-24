#![cfg(feature = "graph")]

use std::alloc::System;

use axnn_vaccel::{ACCEL_BACKEND, AxDeviceModel, AxHetero, AxTosaAccelerator, Session};
use axrd::{
    Anchor, Bind, Corners, LayoutParams, Offset, Parallax, PlacedGlyph, Rect, Scene,
    graph::{
        AlphaAtlas, GraphBackground, GraphOutputFormat, GraphRadialLight, GraphRadialProfile,
        GraphRadialRim, GraphRadialStyle, compile_scene_graph_in,
        compile_scene_graph_with_output_in,
    },
};

fn f16_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let encoded = if exponent == 0 {
        sign
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(encoded)
}

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[test]
fn planar_u8_display_boundary_executes_as_portable_biased_int8() {
    let mut scene = Scene::<1>::new();
    scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 4,
                h: 4,
            },
            Bind::Composite,
        )
        .unwrap();
    let frame = compile_scene_graph_with_output_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 4,
            h: 4,
        },
        GraphBackground::Inputs,
        None,
        &[],
        GraphOutputFormat::PlanarU8,
        System,
    )
    .unwrap();
    assert_eq!(frame.output_format(), GraphOutputFormat::PlanarU8);

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> =
        AxHetero::new(cpu, Some(accel));
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&frame.graph(), &model, System).unwrap();
    assert_eq!(executable.single_backend(), Some(ACCEL_BACKEND));
    let plane = |value: f32| {
        (0..16)
            .flat_map(|_| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let inputs = [plane(0.0), plane(128.0 / 255.0), plane(1.0)];
    let outputs = runtime
        .run_in(&executable, &inputs.each_ref().map(Vec::as_slice), System)
        .unwrap();

    // Signed storage is biased by -128: wrapping +128 recovers 0, 128, 255.
    assert_eq!(outputs[0].as_slice(), &[i8::MIN as u8; 16]);
    assert_eq!(outputs[1].as_slice(), &[0; 16]);
    assert_eq!(outputs[2].as_slice(), &[i8::MAX as u8; 16]);
}

#[test]
fn scheduled_graph_preserves_fractional_glyph_position() {
    let mut scene = Scene::<2>::new();
    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 2,
                h: 1,
            },
            Bind::Composite,
        )
        .unwrap();
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 2,
                h: 1,
            },
            Bind::TextRun {
                glyph_start: 0,
                glyph_len: 1,
                color: 0x00ff_0000,
            },
        )
        .unwrap();
    let atlas = [u16::MAX];
    let glyphs = [PlacedGlyph {
        ax: 0,
        ay: 0,
        aw: 1,
        ah: 1,
        dx: 0,
        dy: 0,
        subpixel_x: 128,
        subpixel_y: 0,
    }];
    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 2,
            h: 1,
        },
        GraphBackground::Solid(0),
        Some(AlphaAtlas::from_u16(&atlas, 1, 1)),
        &glyphs,
        System,
    )
    .unwrap();

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&frame.graph(), &model, System).unwrap();
    let output = runtime.run_in(&executable, &[], System).unwrap();

    for pixel in 0..2 {
        assert!((f16_at(&output[0], pixel) - 0.5).abs() < 0.002);
        assert_eq!(f16_at(&output[1], pixel), 0.0);
        assert_eq!(f16_at(&output[2], pixel), 0.0);
    }
}

#[test]
fn scheduled_graph_executes_source_over_without_a_raster_backend() {
    let mut scene = Scene::<4>::new();
    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 2,
                h: 1,
            },
            Bind::Composite,
        )
        .unwrap();
    scene
        .add_child_with_opacity(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 1,
                h: 1,
            },
            Bind::Fill { color: 0x00ff_0000 },
            128,
        )
        .unwrap();

    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 2,
            h: 1,
        },
        GraphBackground::Solid(0x00ff_ffff),
        None,
        &[],
        System,
    )
    .unwrap();

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&frame.graph(), &model, System).unwrap();
    let output = runtime.run_in(&executable, &[], System).unwrap();

    assert_eq!(output.len(), 3);
    // Half-covered red over white: R=1, G=B≈0.5 at x=0.
    assert_eq!(f16_at(&output[0], 0), 1.0);
    assert!((f16_at(&output[1], 0) - 0.498).abs() < 0.002);
    assert!((f16_at(&output[2], 0) - 0.498).abs() < 0.002);
    // Untouched x=1 remains white.
    for channel in &output {
        assert_eq!(f16_at(channel, 1), 1.0);
    }
}

#[test]
fn retained_graph_moves_a_shallow_scene_layer_without_recompiling() {
    let mut scene = Scene::<2>::new();
    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 7,
                h: 1,
            },
            Bind::Composite,
        )
        .unwrap();
    let foreground = scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 2, y: 0 },
                w: 3,
                h: 1,
            },
            Bind::Fill { color: 0x00ff_ffff },
        )
        .unwrap();
    scene
        .set_parallax(foreground, Parallax::new(128, 0))
        .unwrap();

    let frame = compile_scene_graph_with_output_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 7,
            h: 1,
        },
        GraphBackground::Solid(0),
        None,
        &[],
        GraphOutputFormat::PlanarF32,
        System,
    )
    .unwrap();
    assert_eq!(frame.background_input_count(), 0);
    assert_eq!(frame.parallax_input_count(), 2);
    assert_eq!(frame.input_count(), 2);

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let executable = runtime
        .compile_in(
            &frame.graph(),
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
            System,
        )
        .unwrap();
    let run = |runtime: &mut AxHetero<AxTosaAccelerator, AxTosaAccelerator>, x: f32| {
        let values = [x.to_le_bytes(), 0.0_f32.to_le_bytes()];
        let inputs = values.each_ref().map(<[u8; 4]>::as_slice);
        runtime.run_in(&executable, &inputs, System).unwrap()
    };
    let left = run(&mut runtime, -1.0);
    let neutral = run(&mut runtime, 0.0);
    let right = run(&mut runtime, 1.0);

    // The neutral pose is the exact prepared field. Positive input transfers
    // edge energy toward positive screen x; negative input mirrors it.
    assert_eq!(f32_at(&neutral[0], 1), 0.0);
    assert_eq!(f32_at(&neutral[0], 2), 1.0);
    assert_eq!(f32_at(&neutral[0], 4), 1.0);
    assert_eq!(f32_at(&neutral[0], 5), 0.0);
    assert!(f32_at(&right[0], 5) > f32_at(&neutral[0], 5));
    assert!(f32_at(&right[0], 2) < f32_at(&neutral[0], 2));
    assert!(f32_at(&left[0], 1) > f32_at(&neutral[0], 1));
    assert!(f32_at(&left[0], 4) < f32_at(&neutral[0], 4));
    for output in [&left, &neutral, &right] {
        for channel in output {
            assert!(channel.as_chunks::<4>().0.iter().all(|bytes| {
                let value = f32::from_le_bytes(*bytes);
                (0.0..=1.0).contains(&value)
            }));
        }
    }
}

#[test]
fn grouped_translucent_fills_preserve_overlap_alpha() {
    let mut scene = Scene::<3>::new();
    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 3,
                h: 1,
            },
            Bind::Composite,
        )
        .unwrap();
    for x in [0, 1] {
        scene
            .add_child_with_opacity(
                root,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset { x, y: 0 },
                    w: 2,
                    h: 1,
                },
                Bind::Fill { color: 0x00ff_0000 },
                128,
            )
            .unwrap();
    }
    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 3,
            h: 1,
        },
        GraphBackground::Solid(0),
        None,
        &[],
        System,
    )
    .unwrap();
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let executable = runtime
        .compile_in(
            &frame.graph(),
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
            System,
        )
        .unwrap();
    let output = runtime.run_in(&executable, &[], System).unwrap();

    assert!((f16_at(&output[0], 0) - 128.0 / 255.0).abs() < 0.002);
    assert!((f16_at(&output[0], 1) - 0.752).abs() < 0.002);
    assert!((f16_at(&output[0], 2) - 128.0 / 255.0).abs() < 0.002);
    for channel in &output[1..] {
        assert_eq!(f16_at(channel, 0), 0.0);
        assert_eq!(f16_at(channel, 1), 0.0);
        assert_eq!(f16_at(channel, 2), 0.0);
    }
}

#[test]
fn rounded_fill_keeps_exact_resolution_edge_coverage_through_scheduled_composition() {
    let mut scene = Scene::<2>::new();
    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 7,
                h: 7,
            },
            Bind::Composite,
        )
        .unwrap();
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 1, y: 1 },
                w: 5,
                h: 5,
            },
            Bind::RoundedFill {
                color: 0x00ff_ffff,
                radii: Corners::uniform(2),
            },
        )
        .unwrap();
    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 7,
            h: 7,
        },
        GraphBackground::Solid(0),
        None,
        &[],
        System,
    )
    .unwrap();
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let executable = runtime
        .compile_in(
            &frame.graph(),
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
            System,
        )
        .unwrap();
    let output = runtime.run_in(&executable, &[], System).unwrap();

    for channel in &output {
        assert_eq!(f16_at(channel, 0), 0.0);
        assert!((f16_at(channel, 8) - 0.3787).abs() < 0.003);
        assert!((f16_at(channel, 9) - 0.9189).abs() < 0.003);
        assert_eq!(f16_at(channel, 24), 1.0);
    }
}

#[test]
fn scheduled_graph_evaluates_radial_depth_field() {
    let mut scene = Scene::<1>::new();
    scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 5,
                h: 5,
            },
            Bind::Composite,
        )
        .unwrap();
    let light = GraphRadialLight {
        x: 2.5,
        y: 2.5,
        radius: 1.0,
        color: 0x00ff_0000,
        strength: 1.0,
        profile: GraphRadialProfile::Rational { sharpness: 1.0 },
        rim: None,
    };
    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 5,
            h: 5,
        },
        GraphBackground::Radial {
            base: 0,
            lights: [light; 3],
        },
        None,
        &[],
        System,
    )
    .unwrap();

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&frame.graph(), &model, System).unwrap();
    let output = runtime.run_in(&executable, &[], System).unwrap();

    assert_eq!(f16_at(&output[0], 12), 1.0);
    assert!(f16_at(&output[0], 0) < 0.35);
    assert_eq!(f16_at(&output[1], 12), 0.0);
    assert_eq!(f16_at(&output[2], 12), 0.0);
}

#[test]
fn radial_disc_profile_preserves_analytical_edge_coverage() {
    let mut scene = Scene::<1>::new();
    scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 5,
                h: 5,
            },
            Bind::Composite,
        )
        .unwrap();
    let light = GraphRadialLight {
        x: 2.5,
        y: 2.5,
        radius: 1.0,
        color: 0x00ff_0000,
        strength: 1.0,
        profile: GraphRadialProfile::Disc { softness: 1.0 },
        rim: None,
    };
    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 5,
            h: 5,
        },
        GraphBackground::Radial {
            base: 0,
            lights: [
                light,
                GraphRadialLight {
                    strength: 0.0,
                    ..light
                },
                GraphRadialLight {
                    strength: 0.0,
                    ..light
                },
            ],
        },
        None,
        &[],
        System,
    )
    .unwrap();

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let executable = runtime
        .compile_in(
            &frame.graph(),
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
            System,
        )
        .unwrap();
    let output = runtime.run_in(&executable, &[], System).unwrap();

    assert_eq!(f16_at(&output[0], 12), 1.0);
    assert_eq!(f16_at(&output[0], 13), 0.5);
    assert_eq!(f16_at(&output[0], 14), 0.0);
    assert_eq!(f16_at(&output[1], 12), 0.0);
    assert_eq!(f16_at(&output[2], 12), 0.0);
}

#[test]
fn retained_executable_rebinds_interactive_radial_state() {
    let mut scene = Scene::<1>::new();
    scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 5,
                h: 5,
            },
            Bind::Composite,
        )
        .unwrap();
    let style = GraphRadialStyle {
        radius: 0.5,
        color: 0x00ff_0000,
        profile: GraphRadialProfile::Disc { softness: 0.5 },
        rim: None,
    };
    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 5,
            h: 5,
        },
        GraphBackground::RadialInputs {
            base: 0,
            lights: [style; 3],
        },
        None,
        &[],
        System,
    )
    .unwrap();
    assert_eq!(frame.background_input_count(), 9);

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&frame.graph(), &model, System).unwrap();

    let run_at = |runtime: &mut AxHetero<AxTosaAccelerator, AxTosaAccelerator>, x: f32| {
        let values = [x, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bytes = values.map(f32::to_le_bytes);
        let inputs: Vec<&[u8]> = bytes.iter().map(<[u8; 4]>::as_slice).collect();
        runtime.run_in(&executable, &inputs, System).unwrap()
    };
    let left = run_at(&mut runtime, 0.5);
    let right = run_at(&mut runtime, 4.5);

    assert_eq!(f16_at(&left[0], 10), 1.0);
    assert!(f16_at(&left[0], 14) < 0.02);
    assert_eq!(f16_at(&right[0], 14), 1.0);
    assert!(f16_at(&right[0], 10) < 0.02);
}

#[test]
fn radial_rim_peaks_at_the_runtime_critical_radius() {
    let mut scene = Scene::<1>::new();
    scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: 5,
                h: 5,
            },
            Bind::Composite,
        )
        .unwrap();
    let style = GraphRadialStyle {
        radius: 1.0,
        color: 0x00ff_0000,
        profile: GraphRadialProfile::Rational { sharpness: 1.0 },
        rim: Some(GraphRadialRim {
            strength: 0.5,
            sharpness: 64.0,
        }),
    };
    let frame = compile_scene_graph_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: 5,
            h: 5,
        },
        GraphBackground::RadialInputs {
            base: 0,
            lights: [style; 3],
        },
        None,
        &[],
        System,
    )
    .unwrap();
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let executable = runtime
        .compile_in(
            &frame.graph(),
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
            System,
        )
        .unwrap();
    let values = [2.5_f32, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let bytes = values.map(f32::to_le_bytes);
    let inputs: Vec<&[u8]> = bytes.iter().map(<[u8; 4]>::as_slice).collect();
    let output = runtime.run_in(&executable, &inputs, System).unwrap();

    assert_eq!(f16_at(&output[0], 12), 1.0);
    assert!((f16_at(&output[0], 13) - 0.75).abs() < 0.002);
    assert!((f16_at(&output[0], 14) - 0.2007).abs() < 0.002);
    assert_eq!(f16_at(&output[1], 13), 0.0);
    assert_eq!(f16_at(&output[2], 13), 0.0);
}
