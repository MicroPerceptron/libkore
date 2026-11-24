//! Pointer acceleration through the fallback, as a runtime drives it: framed
//! reports with receipt timestamps, integer output, fractions carried.
#![cfg(feature = "acceleration")]

use axinput::{
    DeviceId, InputTimestamp, PointerInputKind, PointerSource, SeatId,
    acceleration::{
        Acceleration, AccelerationConfig, Calibration, MOTION_TIMEOUT_NS, Profile, UNITS_PER_MM,
    },
    frame::{
        AbsoluteAxis, AxisRange, AxisResolution, ContactSample, DeviceDescriptor, DeviceGeneration,
        FramePolicy, MotionContext, MotionPolicy, PointerFallback, PointerFrame, PolicyEvent,
        TimestampOrigin,
    },
};

const MS: u64 = 1_000_000;

fn device(source: PointerSource) -> DeviceDescriptor {
    DeviceDescriptor::new(SeatId(0), DeviceId(1), DeviceGeneration(1), source)
}

fn accelerated(config: AccelerationConfig) -> PointerFallback<Acceleration> {
    PointerFallback::with_motion_policy(12.0, 1000.0, Acceleration::new(config)).unwrap()
}

/// Feed one mouse report received at `time` and return the pixels it moved.
fn report(
    policy: &mut PointerFallback<Acceleration>,
    descriptor: &DeviceDescriptor,
    sequence: u32,
    relative: [i32; 2],
    time: Option<u64>,
) -> [i64; 2] {
    let mut frame = PointerFrame::new(descriptor, sequence, &[]);
    frame.relative = relative;
    frame.timestamp = time.map(InputTimestamp);
    frame.timestamp_origin = if time.is_some() {
        TimestampOrigin::Received
    } else {
        TimestampOrigin::Unknown
    };
    let mut moved = [0; 2];
    policy.frame(frame, &mut |event| {
        if let PolicyEvent::Pointer(input) = event
            && let PointerInputKind::RelativeMotion { dx, dy } = input.kind
        {
            moved[0] += i64::from(dx);
            moved[1] += i64::from(dy);
        }
    });
    moved
}

/// A steady stream: `count` reports of `relative` counts, `interval` apart.
/// Returns the pixels moved by its second half, once the estimate settled.
fn steady(
    policy: &mut PointerFallback<Acceleration>,
    relative: [i32; 2],
    interval: u64,
    count: u32,
) -> [i64; 2] {
    let descriptor = device(PointerSource::Mouse);
    let mut settled = [0; 2];
    for i in 0..count {
        let moved = report(
            policy,
            &descriptor,
            i,
            relative,
            Some(10 * MS + u64::from(i) * interval),
        );
        if i >= count / 2 {
            settled[0] += moved[0];
            settled[1] += moved[1];
        }
    }
    settled
}

#[test]
fn a_fast_hand_is_multiplied_up_to_the_cap() {
    // Two counts a millisecond is 2 counts/ms: above the plateau, where the
    // default curve gives 1 + 1.1 * (2 - 0.4) = 2.76, capped at 2.
    let mut policy = accelerated(AccelerationConfig::default());
    let moved = steady(&mut policy, [2, 0], MS, 200);
    assert_eq!(moved, [400, 0]);
    let status = policy.motion_policy().status();
    assert!((status.velocity - 2.0).abs() < 1e-9, "{status:?}");
    assert!((status.gain - 2.0).abs() < 1e-9, "{status:?}");
}

#[test]
fn a_moderate_hand_climbs_the_incline() {
    // One count a millisecond: 1 + 1.1 * 0.6 = 1.66 pixels per count.
    let mut policy = accelerated(AccelerationConfig::default());
    let moved = steady(&mut policy, [0, -1], MS, 400);
    assert!((moved[1] + 332).abs() <= 1, "{moved:?}");
}

#[test]
fn the_plateau_is_one_to_one() {
    // One count every 5 ms is 0.2 counts/ms, on the plateau.
    let mut policy = accelerated(AccelerationConfig::default());
    assert_eq!(steady(&mut policy, [1, 1], 5 * MS, 200), [100, 100]);
}

#[test]
fn a_slow_hand_is_slowed_and_no_fraction_is_lost() {
    // One count every 50 ms is 0.02 counts/ms: 10 * 0.02 + 0.3 = 0.5 pixels
    // per count, so two reports make one pixel and the half is carried.
    let mut policy = accelerated(AccelerationConfig::default());
    assert_eq!(steady(&mut policy, [-1, 0], 50 * MS, 400), [-100, 0]);
}

#[test]
fn reports_bunched_on_receipt_read_as_the_hands_speed() {
    // The same hand, one count a millisecond, received evenly or four at a
    // time. A bunch must not read as a burst: the gain stays at the
    // incline's, never the cap's, and the distance stays close.
    let descriptor = device(PointerSource::Mouse);
    let mut even = accelerated(AccelerationConfig::default());
    let mut bunched = accelerated(AccelerationConfig::default());
    let (mut even_moved, mut bunched_moved, mut highest) = (0, 0, 0.0f64);
    for i in 0..400u32 {
        let time = 10 * MS + u64::from(i) * MS;
        let received = 10 * MS + u64::from(i / 4 * 4 + 3) * MS;
        even_moved += report(&mut even, &descriptor, i, [1, 0], Some(time))[0];
        bunched_moved += report(&mut bunched, &descriptor, i, [1, 0], Some(received))[0];
        if i >= 32 {
            highest = highest.max(bunched.motion_policy().status().gain);
        }
    }
    let incline = 1.0 + 1.1 * 0.6;
    assert!(
        highest <= incline + 1e-9,
        "a bunch read as a burst: gain {highest}"
    );
    assert!(
        (bunched_moved - even_moved).abs() * 10 <= even_moved,
        "bunched {bunched_moved} vs even {even_moved}"
    );
}

#[test]
fn the_first_motion_after_a_pause_is_never_a_leap() {
    let descriptor = device(PointerSource::Mouse);
    let mut policy = accelerated(AccelerationConfig::default());
    steady(&mut policy, [5, 0], MS, 100);
    // Two seconds still, then one large report: timed over the whole
    // timeout it is slow, so it is slowed rather than multiplied.
    let moved = report(
        &mut policy,
        &descriptor,
        100,
        [20, 0],
        Some(2 * MOTION_TIMEOUT_NS + 200 * MS),
    );
    let status = policy.motion_policy().status();
    assert!(status.velocity < 0.07, "{status:?}");
    assert!(moved[0] < 20, "{moved:?}");
}

#[test]
fn a_turn_restarts_the_average() {
    let descriptor = device(PointerSource::Mouse);
    let mut policy = accelerated(AccelerationConfig::default());
    steady(&mut policy, [4, 0], MS, 50);
    // Reversing direction: the new motion's speed is measured from the
    // last report alone, not averaged with travel the other way.
    report(
        &mut policy,
        &descriptor,
        50,
        [-1, 0],
        Some(10 * MS + 50 * MS),
    );
    let status = policy.motion_policy().status();
    assert!((status.velocity - 1.0).abs() < 1e-9, "{status:?}");
}

#[test]
fn virtual_sources_are_never_accelerated() {
    // A host's emulated mouse is already accelerated by its host, so it
    // moves one pixel per count whatever its speed or the setting.
    let virtual_mouse = device(PointerSource::Virtual);
    let fastest = AccelerationConfig::new(Profile::Adaptive, 1.0).unwrap();
    let mut policy = accelerated(fastest);
    let mut moved = 0;
    for i in 0..100u32 {
        moved += report(
            &mut policy,
            &virtual_mouse,
            i,
            [7, 0],
            Some(10 * MS + u64::from(i) * MS),
        )[0];
    }
    assert_eq!(moved, 700);
    assert_eq!(policy.motion_policy().status().measured, 0);
}

/// A calibrated touchpad: 40 units a millimetre, so at the fallback's
/// 12 px/mm a unit is 0.3 baseline pixels.
fn touchpad() -> DeviceDescriptor {
    let mut pad = device(PointerSource::Touchpad);
    let axis = AbsoluteAxis {
        range: AxisRange::new(0, 40_000).unwrap(),
        resolution: Some(AxisResolution::new(40, 1).unwrap()),
    };
    pad.axes = Some([axis, axis]);
    pad
}

/// One touchpad frame: finger `id` at `x` (none when `x` is `None`), at
/// `time`. Returns the pixels it moved.
fn touch(
    policy: &mut PointerFallback<Acceleration>,
    pad: &DeviceDescriptor,
    sequence: u32,
    finger: Option<(u32, i32)>,
    time: Option<u64>,
) -> i64 {
    let contact = finger.map(|(id, x)| [ContactSample::new(id, x, 20_000, true)]);
    let contacts: &[ContactSample] = contact.as_ref().map_or(&[], |c| &c[..]);
    let mut frame = PointerFrame::new(pad, sequence, contacts);
    frame.timestamp = time.map(InputTimestamp);
    frame.timestamp_origin = if time.is_some() {
        TimestampOrigin::Received
    } else {
        TimestampOrigin::Unknown
    };
    let mut moved = 0;
    policy.frame(frame, &mut |event| {
        if let PolicyEvent::Pointer(input) = event
            && let PointerInputKind::RelativeMotion { dx, .. } = input.kind
        {
            moved += i64::from(dx);
        }
    });
    moved
}

/// A steady stroke of one finger, `step` units every `interval`; the
/// pixels its second half moved, once the estimate settled.
fn stroke(policy: &mut PointerFallback<Acceleration>, step: i32, interval: u64, count: u32) -> i64 {
    let pad = touchpad();
    let mut settled = 0;
    for i in 0..count {
        let moved = touch(
            policy,
            &pad,
            i,
            Some((1, 1000 + step * i as i32)),
            Some(10 * MS + u64::from(i) * interval),
        );
        if i >= count / 2 {
            settled += moved;
        }
    }
    settled
}

#[test]
fn a_finger_on_the_plateau_moves_at_the_pads_gain() {
    // Half a millimetre every 10 ms is 50 mm/s, on libinput's touchpad
    // plateau (7 to 130 mm/s): exactly the fallback's 12 px/mm.
    let mut policy = accelerated(AccelerationConfig::default());
    assert_eq!(stroke(&mut policy, 20, 10 * MS, 100), 50 * 6);
}

#[test]
fn a_fast_finger_climbs_libinputs_touchpad_curve_to_its_cap() {
    // 2.6 mm every 10 ms is 260 mm/s: libinput's factor is
    // 0.0025 * 2 * 130 + 0.9 = 1.55, so 1.55 / 0.9 of the plateau.
    let mut policy = accelerated(AccelerationConfig::default());
    let moved = stroke(&mut policy, 104, 10 * MS, 100);
    let expected = 50.0 * 31.2 * 1.55 / 0.9;
    assert!(
        (moved as f64 - expected).abs() <= 1.0,
        "{moved} vs {expected}"
    );
    // 5 mm every 8 ms is 625 mm/s, past the 520 mm/s cap: 4.8 / 0.9.
    let mut policy = accelerated(AccelerationConfig::default());
    let moved = stroke(&mut policy, 200, 8 * MS, 100);
    let expected = 50.0 * 60.0 * 4.8 / 0.9;
    assert!(
        (moved as f64 - expected).abs() <= 1.0,
        "{moved} vs {expected}"
    );
    let status = policy.motion_policy().status();
    assert!((status.gain - 4.8 / 0.9).abs() < 1e-9, "{status:?}");
    // The fastest setting scales the whole curve by two, the plateau too.
    let fastest = AccelerationConfig::new(Profile::Adaptive, 1.0).unwrap();
    let mut policy = accelerated(fastest);
    let moved = stroke(&mut policy, 20, 10 * MS, 100);
    let expected = 50.0 * 6.0 * 2.0;
    assert!(
        (moved as f64 - expected).abs() <= 1.0,
        "{moved} vs {expected}"
    );
}

#[test]
fn a_landing_finger_is_timed_from_its_landing() {
    let pad = touchpad();
    let mut policy = accelerated(AccelerationConfig::default());
    // A fast flick, then the finger lifts.
    stroke(&mut policy, 200, 8 * MS, 40);
    let flicked = 10 * MS + 39 * 8 * MS;
    touch(&mut policy, &pad, 40, None, Some(flicked + 8 * MS));
    // It lands again 50 ms later, well inside the one-second timeout, and
    // moves a millimetre in 10 ms: 100 mm/s, the plateau. Neither the
    // flick's speed nor a pause's crawl: exactly 12 pixels.
    let landed = flicked + 58 * MS;
    assert_eq!(
        touch(&mut policy, &pad, 41, Some((2, 5000)), Some(landed)),
        0
    );
    let moved = touch(
        &mut policy,
        &pad,
        42,
        Some((2, 5040)),
        Some(landed + 10 * MS),
    );
    let status = policy.motion_policy().status();
    assert_eq!(moved, 12, "{status:?}");
    assert!((status.gain - 1.0).abs() < 1e-9, "{status:?}");
    assert!(status.restarts >= 2, "{status:?}");
}

#[test]
fn an_unclocked_finger_moves_at_the_plateau() {
    let pad = touchpad();
    let fastest = AccelerationConfig::new(Profile::Adaptive, 1.0).unwrap();
    let mut policy = accelerated(fastest);
    let mut moved = 0;
    for i in 0..11u32 {
        moved += touch(&mut policy, &pad, i, Some((1, 1000 + 400 * i as i32)), None);
    }
    // Ten motions of 10 mm: 120 px each at the plateau, scaled by the
    // fastest setting's two.
    let expected = 1200.0 * 2.0;
    assert!(
        (moved as f64 - expected).abs() <= 1.0,
        "{moved} vs {expected}"
    );
    assert_eq!(policy.motion_policy().status().unclocked, 10);
}

#[test]
fn a_pad_at_another_gain_is_measured_in_millimetres() {
    // A fallback at 6 px/mm with the matching calibration: the same finger
    // speed reads as the same speed, so the plateau is 1:1 at its gain.
    let calibration = Calibration::DEFAULT.with_pixels_per_mm(6.0).unwrap();
    let mut policy = PointerFallback::with_motion_policy(
        6.0,
        1000.0,
        Acceleration::default().with_calibration(calibration),
    )
    .unwrap();
    assert_eq!(stroke(&mut policy, 20, 10 * MS, 100), 50 * 3);
    let status = policy.motion_policy().status();
    // 50 mm/s in units per millisecond.
    assert!(
        (status.velocity - 50.0 * UNITS_PER_MM / 1000.0).abs() < 1e-9,
        "{status:?}"
    );
}

#[test]
fn a_frame_without_a_clock_moves_at_the_plateau() {
    let descriptor = device(PointerSource::Mouse);
    let mut policy = accelerated(AccelerationConfig::default());
    let mut moved = 0;
    for i in 0..10u32 {
        moved += report(&mut policy, &descriptor, i, [9, 0], None)[0];
    }
    assert_eq!(moved, 90);
    let status = policy.motion_policy().status();
    assert_eq!((status.unclocked, status.measured), (10, 0));
}

#[test]
fn time_running_backwards_restarts_the_history() {
    let descriptor = device(PointerSource::Mouse);
    let mut policy = accelerated(AccelerationConfig::default());
    steady(&mut policy, [3, 0], MS, 20);
    let moved = report(&mut policy, &descriptor, 20, [3, 0], Some(MS));
    let status = policy.motion_policy().status();
    assert_eq!(status.time_reversals, 1);
    assert!(
        status.velocity.is_finite() && moved[0] <= 3,
        "{status:?} {moved:?}"
    );
}

#[test]
fn a_hostile_report_stays_bounded() {
    let descriptor = device(PointerSource::Mouse);
    let mut policy = accelerated(AccelerationConfig::new(Profile::Adaptive, 1.0).unwrap());
    report(&mut policy, &descriptor, 0, [1, 1], Some(10 * MS));
    let moved = report(
        &mut policy,
        &descriptor,
        1,
        [i32::MAX, i32::MIN],
        Some(10 * MS + 1),
    );
    let status = policy.motion_policy().status();
    assert!(status.gain.is_finite() && status.gain <= 3.5, "{status:?}");
    assert!(moved[0] > 0 && moved[1] < 0, "{moved:?}");
}

#[test]
fn a_new_setting_applies_to_the_next_report() {
    let descriptor = device(PointerSource::Mouse);
    let mut policy = accelerated(AccelerationConfig::default());
    steady(&mut policy, [2, 0], MS, 50);
    // Flat at the fastest setting: two pixels per count, whatever the speed.
    policy
        .motion_policy_mut()
        .set_config(AccelerationConfig::new(Profile::Flat, 1.0).unwrap());
    let mut moved = 0;
    for i in 50..90u32 {
        moved += report(
            &mut policy,
            &descriptor,
            i,
            [2, 0],
            Some(10 * MS + u64::from(i) * MS),
        )[0];
    }
    assert_eq!(moved, 160);
    // And back to adaptive: the history was kept, so the speed is known at
    // once and the cap applies from the first report.
    policy
        .motion_policy_mut()
        .set_config(AccelerationConfig::default());
    report(
        &mut policy,
        &descriptor,
        90,
        [2, 0],
        Some(10 * MS + 90 * MS),
    );
    let status = policy.motion_policy().status();
    assert!((status.gain - 2.0).abs() < 1e-9, "{status:?}");
}

#[test]
fn the_policy_reads_its_context_directly() {
    // A host that is not a fallback can drive the policy itself.
    let mut policy = Acceleration::default();
    let mut out = [0.0; 2];
    for i in 0..20u64 {
        let context = MotionContext::new(
            PointerSource::Mouse,
            Some(InputTimestamp(5 * MS + i * MS)),
            TimestampOrigin::Captured,
        );
        out = policy.apply([3.0, 4.0], context);
    }
    // 5 counts a millisecond: capped at 2.
    assert!(
        (out[0] - 6.0).abs() < 1e-9 && (out[1] - 8.0).abs() < 1e-9,
        "{out:?}"
    );
    // A timestamp of unknown origin is no clock.
    let context = MotionContext::new(
        PointerSource::Mouse,
        Some(InputTimestamp(1)),
        TimestampOrigin::Unknown,
    );
    assert_eq!(context.time(), None);
    assert_eq!(policy.apply([3.0, 4.0], context), [3.0, 4.0]);
}
