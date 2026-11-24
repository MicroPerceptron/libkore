use axinput::{frame::*, *};

fn pad() -> DeviceDescriptor {
    let mut device = DeviceDescriptor::new(
        SeatId(0),
        DeviceId(7),
        DeviceGeneration(1),
        PointerSource::Touchpad,
    );
    device.axes = Some(
        [AbsoluteAxis {
            range: AxisRange::new(-100, 9900).unwrap(),
            resolution: None,
        }; 2],
    );
    device
}

fn motion(policy: &mut PointerFallback, frame: PointerFrame<'_>) -> [i64; 2] {
    let mut delta = [0; 2];
    policy.frame(frame, &mut |event| {
        if let PolicyEvent::Pointer(PointerInput {
            kind: PointerInputKind::RelativeMotion { dx, dy },
            ..
        }) = event
        {
            delta[0] += i64::from(dx);
            delta[1] += i64::from(dy);
        }
    });
    delta
}

#[test]
fn history_rejects_replay_time_reversal_and_stale_generations_without_advancing() {
    let mut descriptor = pad();
    let mut history = FrameHistory::default();
    fn frame(d: &DeviceDescriptor, sequence: u32, time: u64) -> PointerFrame<'_> {
        let mut f = PointerFrame::new(d, sequence, &[]);
        f.timestamp = Some(InputTimestamp(time));
        f.timestamp_origin = TimestampOrigin::Received;
        f
    }
    assert_eq!(history.accept(frame(&descriptor, u32::MAX, 100)), Ok(None));
    assert_eq!(history.accept(frame(&descriptor, 0, 101)), Ok(None));
    assert_eq!(
        history.accept(frame(&descriptor, 0, 101)),
        Err(FrameError::StaleSequence)
    );
    assert_eq!(
        history.accept(frame(&descriptor, 1, 99)),
        Err(FrameError::TimeReversed)
    );
    assert_eq!(history.accept(frame(&descriptor, 1, 102)), Ok(None));
    assert_eq!(
        history.accept(frame(&descriptor, 3, 103)),
        Ok(Some(CancelReason::StreamLost))
    );
    descriptor.generation = DeviceGeneration(2);
    assert_eq!(
        history.accept(frame(&descriptor, 0, 104)),
        Ok(Some(CancelReason::DeviceReset))
    );
    descriptor.generation = DeviceGeneration(1);
    assert_eq!(
        history.accept(frame(&descriptor, 4, 105)),
        Err(FrameError::StaleGeneration)
    );
}

#[test]
fn validation_preserves_unknowns_and_refuses_ambiguous_input() {
    let descriptor = pad();
    let contact = ContactSample::new(u32::MAX, -99, 10, true);
    assert_eq!(contact.confidence, None);
    let contacts = [contact, contact];
    assert_eq!(
        PointerFrame::new(&descriptor, 1, &contacts).validate(),
        Err(FrameError::DuplicateContact)
    );
    let mut frame = PointerFrame::new(&descriptor, 1, &contacts[..1]);
    frame.timestamp = Some(InputTimestamp(0));
    assert_eq!(frame.validate(), Err(FrameError::InvalidTimestamp));
    frame.timestamp_origin = TimestampOrigin::Captured;
    assert_eq!(frame.validate(), Ok(()));
    frame.scroll = Some(ScrollSample {
        horizontal: f64::NAN,
        vertical: 0.0,
        unit: ScrollUnit::Pixel,
    });
    assert_eq!(frame.validate(), Err(FrameError::InvalidMotion));
    assert!(AxisRange::new(0, 0).is_err());
    assert_eq!(AxisRange::new(i32::MIN, i32::MAX).unwrap().span(), u32::MAX);
    assert!(AxisResolution::new(0, 1).is_err());
    assert!(PointerFallback::new(f64::MAX, 1000.0).is_err());
    assert_eq!(
        TouchpadMotion::with_reference_extent(u32::MAX),
        Err(InputError::ExtentTooLarge)
    );
}

#[test]
fn changing_descriptor_requires_a_new_generation() {
    let mut descriptor = pad();
    let mut history = FrameHistory::default();
    history
        .accept(PointerFrame::new(&descriptor, 0, &[]))
        .unwrap();
    descriptor.axes.as_mut().unwrap()[0].resolution = Some(AxisResolution::new(100, 1).unwrap());
    assert_eq!(
        history.accept(PointerFrame::new(&descriptor, 1, &[])),
        Err(FrameError::DescriptorChanged)
    );
    descriptor.generation = DeviceGeneration(2);
    assert_eq!(
        history.accept(PointerFrame::new(&descriptor, 1, &[])),
        Ok(Some(CancelReason::DeviceReset))
    );
}

#[test]
fn retirement_refuses_the_old_lifetime_but_loss_only_requires_resynchronization() {
    let mut descriptor = pad();
    let mut history = FrameHistory::default();
    history
        .accept(PointerFrame::new(&descriptor, 0, &[]))
        .unwrap();
    history.interrupt();
    assert_eq!(
        history.accept(PointerFrame::new(&descriptor, 1, &[])),
        Ok(Some(CancelReason::StreamLost))
    );
    history.retire();
    history.interrupt(); // a subsequent loss must not undo removal
    assert_eq!(
        history.accept(PointerFrame::new(&descriptor, 2, &[])),
        Err(FrameError::StaleGeneration)
    );
    descriptor.generation = DeviceGeneration(2);
    assert_eq!(
        history.accept(PointerFrame::new(&descriptor, 0, &[])),
        Ok(Some(CancelReason::DeviceReset))
    );
    assert_eq!(history.descriptor(), Some(&descriptor));
}

#[test]
fn impossible_complete_counts_are_not_silently_treated_as_single_finger() {
    let descriptor = pad();
    let contacts = [ContactSample::new(3, 0, 0, true)];
    let mut frame = PointerFrame::new(&descriptor, 0, &contacts);
    frame.reported_contacts = Some(2);
    assert_eq!(frame.validate(), Err(FrameError::Capacity));
    frame.completeness = FrameCompleteness::Fragment;
    assert_eq!(frame.validate(), Ok(()));
}

#[test]
fn physical_calibration_preserves_fractional_movement_and_is_axis_specific() {
    let mut descriptor = pad();
    descriptor.axes.as_mut().unwrap()[0].resolution = Some(AxisResolution::new(100, 1).unwrap());
    descriptor.axes.as_mut().unwrap()[1].resolution = Some(AxisResolution::new(200, 1).unwrap());
    let mut policy = PointerFallback::new(10.0, 1000.0).unwrap();
    let mut total = [0; 2];
    for i in 0..=100 {
        let contacts = [ContactSample::new(300, i, i, true)];
        let delta = motion(
            &mut policy,
            PointerFrame::new(&descriptor, i as u32, &contacts),
        );
        total[0] += delta[0];
        total[1] += delta[1];
    }
    // Floating accumulation may retain a final fraction; the error must remain
    // below a pixel, never a viewport-dependent gain or one rounding per sample.
    assert!((9..=10).contains(&total[0]));
    assert!((4..=5).contains(&total[1]));
}

#[test]
fn deterministic_replay_is_independent_of_consumer_batch_sizes() {
    fn replay(batch: usize) -> [i64; 2] {
        let descriptor = pad();
        let mut history = FrameHistory::default();
        let mut policy = PointerFallback::default();
        let mut total = [0; 2];
        for start in (0..1024).step_by(batch) {
            for i in start..(start + batch).min(1024) {
                let contacts = [ContactSample::new(1, i as i32 * 4, i as i32 * 2, true)];
                let mut f = PointerFrame::new(&descriptor, i as u32, &contacts);
                f.timestamp = Some(InputTimestamp(i as u64 * 5_000_000));
                f.timestamp_origin = TimestampOrigin::Received;
                assert_eq!(history.accept(f), Ok(None));
                let delta = motion(&mut policy, f);
                total[0] += delta[0];
                total[1] += delta[1];
            }
        }
        total
    }
    assert_eq!(replay(1), replay(17));
    assert_eq!(replay(17), replay(128));
}

#[test]
fn fragments_and_palm_or_multitouch_never_become_pointer_motion() {
    let descriptor = pad();
    let mut policy = PointerFallback::default();
    let a = [ContactSample::new(1, 0, 0, true)];
    let b = [
        ContactSample::new(1, 500, 500, true),
        ContactSample::new(2, 900, 500, true),
    ];
    assert_eq!(
        motion(&mut policy, PointerFrame::new(&descriptor, 0, &a)),
        [0; 2]
    );
    assert_eq!(
        motion(&mut policy, PointerFrame::new(&descriptor, 1, &b)),
        [0; 2]
    );
    assert_eq!(
        motion(&mut policy, PointerFrame::new(&descriptor, 2, &b[..1])),
        [0; 2]
    );
    let mut fragment = PointerFrame::new(&descriptor, 3, &a);
    fragment.completeness = FrameCompleteness::Fragment;
    assert_eq!(motion(&mut policy, fragment), [0; 2]);
    assert_eq!(
        motion(&mut policy, PointerFrame::new(&descriptor, 4, &b[..1])),
        [0; 2]
    );
    let mut palm = a;
    palm[0].confidence = Some(false);
    assert_eq!(
        motion(&mut policy, PointerFrame::new(&descriptor, 5, &palm)),
        [0; 2]
    );
}

#[test]
fn loss_releases_buttons_and_requires_a_real_release_before_another_press() {
    let descriptor = DeviceDescriptor::new(
        SeatId(0),
        DeviceId(1),
        DeviceGeneration(1),
        PointerSource::Mouse,
    );
    let mut policy = PointerFallback::default();
    let mut presses = 0;
    let mut releases = 0;
    let mut emit = |event: PolicyEvent<'_>| {
        if let PolicyEvent::Pointer(PointerInput {
            kind: PointerInputKind::Button { state, .. },
            ..
        }) = event
        {
            match state {
                ButtonState::Pressed => presses += 1,
                ButtonState::Released => releases += 1,
            }
        }
    };
    let mut f = PointerFrame::new(&descriptor, 0, &[]);
    f.buttons = 1;
    policy.frame(f, &mut emit);
    policy.cancel(CancelReason::StreamLost, &mut emit);
    policy.frame(f, &mut emit);
    policy.frame(f, &mut emit);
    f.buttons = 0;
    policy.frame(f, &mut emit);
    f.buttons = 1;
    policy.frame(f, &mut emit);
    assert_eq!((presses, releases), (2, 1));
}

#[test]
fn legacy_contacts_keep_device_coordinates_and_resize_does_not_change_gain() {
    let small = Viewport::new(100, 80).unwrap();
    let large = Viewport::new(4000, 3000).unwrap();
    let input = |x| PointerInput {
        seat: SeatId(0),
        device: DeviceId(1),
        source: PointerSource::Touchpad,
        timestamp: None,
        kind: PointerInputKind::Contact {
            id: 4,
            phase: ContactPhase::Moved,
            x,
            y: 5,
            range: AbsoluteRange::new(1000, 1000).unwrap(),
            contacts: 1,
            confident: true,
        },
    };
    let mut motion = TouchpadMotion::default();
    assert_eq!(motion.feed(input(10), small).unwrap(), None);
    assert_eq!(
        motion.feed(input(20), large).unwrap().unwrap().kind,
        PointerInputKind::RelativeMotion { dx: 10, dy: 0 }
    );
    let mut seat = Seat::new(
        SeatId(0),
        small,
        PointerPosition::new(1, 2),
        MotionScale::IDENTITY,
    );
    let event = seat.feed(input(50)).unwrap();
    assert!(matches!(
        event.kind,
        SeatEventKind::Contact { x: 50, y: 5, .. }
    ));
    assert_eq!(event.position, PointerPosition::new(1, 2));
}
