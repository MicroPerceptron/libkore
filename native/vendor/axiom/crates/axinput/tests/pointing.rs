#![cfg(feature = "pointing")]

use axinput::{frame::*, pointing::*, *};

/// The angle policy reads path shape only, so its tests need no clock.
const UNTIMED: MotionContext =
    MotionContext::new(PointerSource::Mouse, None, TimestampOrigin::Unknown);

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8, "{a} != {b}");
}

#[test]
fn straight_corrections_recovery_and_reset() {
    let mut gain = AngleGain::default();
    for _ in 0..20 {
        close(gain.apply([8.0, 0.0], UNTIMED)[0], 8.0);
    }
    close(gain.status().gain, 1.0);
    for i in 0..16 {
        gain.apply([if i % 2 == 0 { -8.0 } else { 8.0 }, 0.0], UNTIMED);
    }
    assert!(gain.status().gain < 0.6);
    for _ in 0..32 {
        gain.apply([8.0, 0.0], UNTIMED);
    }
    close(gain.status().gain, 1.0);
    gain.reset();
    assert_eq!(gain.status().samples, 0);
    close(gain.apply([0.125, -0.125], UNTIMED)[0], 0.125);
}

#[test]
fn circular_wrap_and_opposite_directions_are_not_confused() {
    let mut gain = AngleGain::default();
    for i in 0..40 {
        // Alternating near +pi / -pi; coherent LEFTWARD travel.
        gain.apply([-8.0, if i % 2 == 0 { 0.01 } else { -0.01 }], UNTIMED);
    }
    assert!(gain.status().gain > 0.99);
    for i in 0..40 {
        gain.apply([if i % 2 == 0 { -8.0 } else { 8.0 }, 0.0], UNTIMED);
    }
    assert!(gain.status().gain < 0.6);
}

#[test]
fn subdivision_zero_reports_and_consumer_batches_preserve_path() {
    fn replay(parts: usize, batch: usize) -> ([f64; 2], AngleStatus) {
        let mut gain = AngleGain::default();
        let path = [
            [40.0, 0.0],
            [-24.0, 0.0],
            [0.0, 32.0],
            [16.0, -16.0],
            [-8.0, 8.0],
        ];
        let deltas: Vec<_> = path
            .iter()
            .flat_map(|d| std::iter::repeat_n([d[0] / parts as f64, d[1] / parts as f64], parts))
            .collect();
        let mut sum = [0.0; 2];
        for group in deltas.chunks(batch) {
            for &d in group {
                let value = gain.apply(d, UNTIMED);
                for axis in 0..2 {
                    sum[axis] += value[axis];
                }
                assert_eq!(gain.apply([0.0; 2], UNTIMED), [0.0; 2]);
            }
        }
        (sum, gain.status())
    }
    let (expected, status) = replay(1, 1);
    for parts in [2, 8, 64, 1000] {
        for batch in [1, 7, 128] {
            let (actual, next) = replay(parts, batch);
            close(actual[0], expected[0]);
            close(actual[1], expected[1]);
            close(next.gain, status.gain);
        }
    }
}

#[test]
fn invalid_and_hostile_input_is_bounded_and_diagnosable() {
    for args in [
        (0.0, 0.2, 1.0),
        (8.0, -1.0, 1.0),
        (8.0, 2.0, 1.0),
        (f64::NAN, 0.2, 1.0),
        (8.0, 0.2, f64::INFINITY),
        (8.0, 0.2, 17.0),
    ] {
        assert!(AngleConfig::new(args.0, args.1, args.2).is_err());
    }
    let mut gain = AngleGain::default();
    for delta in [[f64::NAN, 0.0], [f64::INFINITY, 0.0], [i32::MAX as f64; 2]] {
        assert_eq!(gain.apply(delta, UNTIMED), [0.0; 2]);
        assert_eq!(gain.status().samples, 0);
    }
    assert_eq!(gain.status().rejected_motions, 3);
    close(gain.apply([1.0, 0.0], UNTIMED)[0], 1.0);
}

fn mouse() -> DeviceDescriptor {
    let mut descriptor = DeviceDescriptor::new(
        SeatId(0),
        DeviceId(1),
        DeviceGeneration(1),
        PointerSource::Mouse,
    );
    descriptor.relative_resolution = [Some(AxisResolution::new(100, 1).unwrap()); 2];
    descriptor
}

fn run<M: MotionPolicy>(
    policy: &mut PointerFallback<M>,
    frame: PointerFrame<'_>,
) -> Vec<PointerInputKind> {
    let mut out = Vec::new();
    policy.frame(frame, &mut |event| {
        if let PolicyEvent::Pointer(p) = event {
            out.push(p.kind)
        }
    });
    out
}

#[test]
fn calibrated_mouse_retains_fractions_and_default_is_unchanged() {
    let descriptor = mouse();
    let mut plain = PointerFallback::default();
    let mut gain = PointerFallback::with_motion_policy(10.0, 1000.0, AngleGain::default())
        .unwrap()
        .with_calibrated_mouse();
    let mut sum = 0;
    for i in 0..100 {
        let mut frame = PointerFrame::new(&descriptor, i, &[]);
        frame.relative = [1, 0];
        assert_eq!(
            run(&mut plain, frame)[0],
            PointerInputKind::RelativeMotion { dx: 1, dy: 0 }
        );
        if let PointerInputKind::RelativeMotion { dx, .. } = run(&mut gain, frame)[0] {
            sum += dx;
        }
    }
    assert!((9..=10).contains(&sum));
    let mut partial = descriptor;
    partial.relative_resolution[1] = None;
    let mut policy = PointerFallback::default().with_calibrated_mouse();
    let mut frame = PointerFrame::new(&partial, 0, &[]);
    frame.relative = [10, 10];
    assert_eq!(
        run(&mut policy, frame)[0],
        PointerInputKind::RelativeMotion { dx: 10, dy: 10 }
    );
}

#[test]
fn cancellation_releases_buttons_and_resets_adaptation_before_reanchor() {
    let mut descriptor = mouse();
    descriptor.source = PointerSource::Touchpad;
    descriptor.axes = Some(
        [AbsoluteAxis {
            range: AxisRange::new(0, 1000).unwrap(),
            resolution: None,
        }; 2],
    );
    let mut policy =
        PointerFallback::with_motion_policy(12.0, 1000.0, AngleGain::default()).unwrap();
    for i in 0..20 {
        let contacts = [ContactSample::new(
            1,
            if i % 2 == 0 { 100 } else { 108 },
            100,
            true,
        )];
        let mut frame = PointerFrame::new(&descriptor, i, &contacts);
        frame.buttons = 1;
        run(&mut policy, frame);
    }
    assert!(policy.motion_policy().status().gain < 0.6);
    let mut released = false;
    policy.cancel(CancelReason::FocusLost, &mut |e| {
        if let PolicyEvent::Pointer(PointerInput {
            kind:
                PointerInputKind::Button {
                    state: ButtonState::Released,
                    ..
                },
            ..
        }) = e
        {
            released = true;
        }
    });
    assert!(released);
    assert_eq!(policy.motion_policy().status().samples, 0);
    let contacts = [ContactSample::new(1, 500, 500, true)];
    let mut frame = PointerFrame::new(&descriptor, 20, &contacts);
    frame.buttons = 1;
    assert!(run(&mut policy, frame).is_empty());
}

#[test]
fn touchpad_lift_multitouch_and_contact_replacement_reset_history() {
    let mut descriptor = mouse();
    descriptor.source = PointerSource::Touchpad;
    descriptor.axes = Some(
        [AbsoluteAxis {
            range: AxisRange::new(0, 1000).unwrap(),
            resolution: None,
        }; 2],
    );
    for mode in 0..3 {
        let mut policy =
            PointerFallback::with_motion_policy(12.0, 1000.0, AngleGain::default()).unwrap();
        for i in 0..4 {
            let c = [ContactSample::new(1, i * 8, 0, true)];
            run(&mut policy, PointerFrame::new(&descriptor, i as u32, &c));
        }
        assert!(policy.motion_policy().status().samples > 0);
        let c = [
            ContactSample::new(2, 100, 0, true),
            ContactSample::new(3, 120, 0, true),
        ];
        let contacts = match mode {
            0 => &c[..0],
            1 => &c[..1],
            _ => &c[..],
        };
        run(&mut policy, PointerFrame::new(&descriptor, 4, contacts));
        assert_eq!(policy.motion_policy().status().samples, 0);
    }
}

#[test]
fn direct_touch_and_scroll_bypass_gain() {
    let mut descriptor = mouse();
    descriptor.source = PointerSource::Touchscreen;
    descriptor.axes = Some(
        [AbsoluteAxis {
            range: AxisRange::new(0, 1000).unwrap(),
            resolution: None,
        }; 2],
    );
    let mut policy =
        PointerFallback::with_motion_policy(12.0, 1000.0, AngleGain::default()).unwrap();
    let c = [ContactSample::new(1, 500, 250, true)];
    let mut frame = PointerFrame::new(&descriptor, 0, &c);
    frame.scroll = Some(ScrollSample {
        horizontal: 0.0,
        vertical: 2.0,
        unit: ScrollUnit::Line,
    });
    let out = run(&mut policy, frame);
    assert!(matches!(
        out[0],
        PointerInputKind::AbsoluteMotion { x: 500, y: 250, .. }
    ));
    assert!(out.contains(&PointerInputKind::Scroll {
        horizontal: 0,
        vertical: 2,
        unit: ScrollUnit::Line
    }));
    assert_eq!(policy.motion_policy().status().samples, 0);
}

#[test]
fn trial_counts_crossings_misses_and_rejects_bad_time_without_mutation() {
    let mut trial = PointingTrial::new([0.0, 0.0], [100.0, 0.0], 10.0, InputTimestamp(10)).unwrap();
    trial.observe([120.0, 0.0], InputTimestamp(20)).unwrap();
    assert_eq!(
        trial.observe([500.0, 0.0], InputTimestamp(19)),
        Err(PointingError::TimeReversed)
    );
    assert_eq!(
        trial.observe([f64::NAN, 0.0], InputTimestamp(21)),
        Err(PointingError::InvalidSample)
    );
    assert!(
        trial
            .click([120.0, 0.0], InputTimestamp(21))
            .unwrap()
            .is_none()
    );
    let result = trial
        .click([100.0, 0.0], InputTimestamp(30))
        .unwrap()
        .unwrap();
    assert_eq!(result.elapsed_ns, 20);
    close(result.path_length, 140.0);
    assert_eq!(result.target_entries, 2);
    assert_eq!(result.axial_overshoots, 1);
    assert_eq!(result.missed_clicks, 1);
    close(result.endpoint_error, 0.0);
}

#[test]
fn calibrated_equal_physical_paths_match_across_mouse_resolutions() {
    fn replay(resolution: u32) -> ([i64; 2], AngleStatus) {
        let mut descriptor = mouse();
        descriptor.relative_resolution = [Some(AxisResolution::new(resolution, 1).unwrap()); 2];
        let mut policy = PointerFallback::with_motion_policy(16.0, 1000.0, AngleGain::default())
            .unwrap()
            .with_calibrated_mouse();
        let mut sum = [0_i64; 2];
        let path = [[2, 0], [-1, 0], [0, 2], [1, -1], [-1, 1]];
        for (i, d) in path.iter().enumerate() {
            let mut frame = PointerFrame::new(&descriptor, i as u32, &[]);
            frame.relative = d.map(|v| v * resolution as i32);
            for output in run(&mut policy, frame) {
                if let PointerInputKind::RelativeMotion { dx, dy } = output {
                    sum[0] += i64::from(dx);
                    sum[1] += i64::from(dy);
                }
            }
        }
        (sum, policy.motion_policy().status())
    }
    let a = replay(100);
    let b = replay(800);
    assert_eq!(a.0, b.0);
    close(a.1.gain, b.1.gain);
}

#[test]
fn trial_geometry_admission_keeps_difficulty_finite() {
    assert!(PointingTrial::new([0.0; 2], [100.0, 0.0], 1e-300, InputTimestamp(0)).is_err());
    assert!(PointingTrial::new([0.0; 2], [0.0; 2], 1.0, InputTimestamp(0)).is_err());
    let mut t = PointingTrial::new([-1e9; 2], [1e9; 2], 1e-6, InputTimestamp(0)).unwrap();
    assert!(
        t.click([1e9; 2], InputTimestamp(1))
            .unwrap()
            .unwrap()
            .nominal_difficulty
            .is_finite()
    );
}
