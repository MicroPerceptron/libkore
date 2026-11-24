#![cfg(all(feature = "coreml", target_os = "macos"))]
use std::alloc::System;

use axnn_vaccel::{ACCEL_BACKEND, AxDeviceModel, AxHetero, AxTosaAccelerator, Session};
use axrd::{
    Anchor, Bind, Corners, LayoutParams, Offset, Parallax, Rect, Scene,
    graph::{
        GraphBackground, GraphOutputFormat, GraphRadialProfile, GraphRadialRim, GraphRadialStyle,
        compile_scene_graph_with_output_in,
    },
};
use virtio_accel_coreml::{COREML_TOSA_TARGET, CoreMlAccelerator, InitError};

#[test]
fn full_viewport_conditioned_depth_field_agrees_with_coreml() {
    const WIDTH: u16 = 1265;
    const HEIGHT: u16 = 795;
    let provider = match CoreMlAccelerator::new_tosa() {
        Ok(provider) => provider,
        Err(InitError::NeuralEngineUnavailable) => {
            eprintln!("skipping: no Neural Engine on this host");
            return;
        }
        Err(error) => panic!("Core ML init failed: {error:?}"),
    };
    let mut scene = Scene::<2>::new();
    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: WIDTH,
                h: HEIGHT,
            },
            Bind::Composite,
        )
        .unwrap();
    let foreground = scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 75, y: 600 },
                w: 300,
                h: 80,
            },
            Bind::RoundedFill {
                color: 0x0045_2416,
                radii: Corners::uniform(12),
            },
        )
        .unwrap();
    scene
        .set_parallax(foreground, Parallax::new(256, 192))
        .unwrap();
    let graph = compile_scene_graph_with_output_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: WIDTH,
            h: HEIGHT,
        },
        GraphBackground::RadialInputs {
            base: 0x00F5E9D8,
            lights: [
                GraphRadialStyle {
                    radius: 472.5,
                    color: 0x00CFD7A9,
                    profile: GraphRadialProfile::Rational { sharpness: 1.0 },
                    rim: None,
                },
                GraphRadialStyle {
                    radius: 217.5,
                    color: 0x00F2A074,
                    profile: GraphRadialProfile::Disc { softness: 1.0 },
                    rim: Some(GraphRadialRim {
                        strength: 0.12,
                        sharpness: 96.0,
                    }),
                },
                GraphRadialStyle {
                    radius: 102.0,
                    color: 0x00FFF8ED,
                    profile: GraphRadialProfile::Rational { sharpness: 2.0 },
                    rim: None,
                },
            ],
        },
        None,
        &[],
        GraphOutputFormat::PlanarF32,
        System,
    )
    .unwrap();
    assert_eq!(graph.background_input_count(), 9);
    assert_eq!(graph.parallax_input_count(), 2);

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accelerator = Session::new(provider, COREML_TOSA_TARGET).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accelerator));
    let cpu_model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let accelerator_model = AxDeviceModel {
        cpu: false,
        accelerator_launch_cost: 0,
        accelerator_speedup: 64,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::coreml()
    };
    let cpu = runtime
        .compile_in(&graph.graph(), &cpu_model, System)
        .unwrap();
    let accelerator = runtime
        .compile_in(&graph.graph(), &accelerator_model, System)
        .unwrap();
    assert_eq!(accelerator.single_backend(), Some(ACCEL_BACKEND));

    let values = [
        247.5_f32, 472.5, 0.42, 945.0, 337.5, 0.82, 867.0, 270.0, 0.85, 0.65, -0.35,
    ];
    let bytes = values.map(f32::to_le_bytes);
    let inputs = bytes.each_ref().map(<[u8; 4]>::as_slice);
    let cpu = runtime.run_in(&cpu, &inputs, System).unwrap();
    let accelerator = runtime.run_in(&accelerator, &inputs, System).unwrap();

    let mut maximum_error = 0.0_f32;
    let mut error_sum = 0.0_f64;
    let mut elements = 0_u64;
    for (cpu, accelerator) in cpu.iter().zip(&accelerator) {
        assert_eq!(cpu.len(), accelerator.len());
        let (cpu, cpu_remainder) = cpu.as_chunks::<4>();
        let (accelerator, accelerator_remainder) = accelerator.as_chunks::<4>();
        assert!(cpu_remainder.is_empty() && accelerator_remainder.is_empty());
        for (cpu, accelerator) in cpu.iter().zip(accelerator) {
            let cpu = f32::from_le_bytes(*cpu);
            let accelerator = f32::from_le_bytes(*accelerator);
            assert!(cpu.is_finite() && accelerator.is_finite());
            let error = (cpu - accelerator).abs();
            maximum_error = maximum_error.max(error);
            error_sum += f64::from(error);
            elements += 1;
        }
    }
    let mean_error = error_sum / elements as f64;
    // Core ML may realize the F32 graph on narrower internal ANE lanes. A
    // one-pixel disc transition therefore has a larger pointwise error than
    // the smooth field, while the whole-frame mean remains the guard against
    // the former catastrophic unconditioned result.
    assert!(
        maximum_error <= 0.18 && mean_error <= 0.02,
        "conditioned depth field diverged: max {maximum_error}, mean {mean_error}"
    );
}
