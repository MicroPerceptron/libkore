use super::*;

fn packet(source: u8, sequence: u32, generation: u32, x: i32, count: u16) -> [InputReport; 6] {
    [
        Record::Begin {
            sequence,
            received_ns: Some(u64::from(sequence) * 1_000_000),
            class: 1,
        },
        Record::Axis {
            axis: 0,
            min: 0,
            max: 10_000,
        },
        Record::Axis {
            axis: 1,
            min: 0,
            max: 10_000,
        },
        Record::Contact {
            id: 1000,
            x,
            y: 500,
            touching: true,
            confidence: None,
            in_range: Some(true),
        },
        Record::Buttons(0),
        Record::End {
            sequence,
            generation,
            records: 4,
            reported_contacts: Some(count),
            complete: count == 1,
        },
    ]
    .map(|r| r.encode(source))
}

#[test]
fn frames_reach_policy_only_after_valid_end_and_preserve_metadata() {
    let mut seat = PointerSeat::new(Viewport::new(1000, 800).unwrap());
    let a = packet(128, 1, 1, 100, 1);
    let b = packet(128, 2, 1, 200, 1);
    let initial = seat.position();
    let mut frames = 0;
    for report in &a[..5] {
        seat.feed_with(*report, &mut |e| {
            if matches!(e, Dispatch::Frame(_)) {
                frames += 1;
            }
        });
    }
    assert_eq!(frames, 0);
    assert_eq!(seat.position(), initial);
    seat.feed_with(a[5], &mut |e| {
        if let Dispatch::Frame(f) = e {
            assert_eq!(f.contacts[0].id, 1000);
            assert_eq!(f.contacts[0].confidence, None);
            assert_eq!(f.timestamp, Some(InputTimestamp(1_000_000)));
            frames += 1;
        }
    });
    for report in b {
        seat.feed(report, |_| {});
    }
    assert_eq!(seat.position().x, initial.x + 10);
    assert_eq!(frames, 1);
    assert_eq!(seat.rejected_frames, 0);
}

#[test]
fn replay_gaps_fragments_and_generation_changes_cannot_jump_the_cursor() {
    let mut seat = PointerSeat::new(Viewport::new(1000, 800).unwrap());
    for r in packet(128, 1, 1, 100, 1) {
        seat.feed(r, |_| {});
    }
    for r in packet(128, 3, 1, 1000, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(seat.position().x, 500, "gap re-anchors");
    for r in packet(128, 4, 1, 1200, 2) {
        seat.feed(r, |_| {});
    }
    for r in packet(128, 5, 1, 1500, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(
        seat.position().x,
        500,
        "fragment cannot move pointer or bridge anchors"
    );
    for r in packet(128, 5, 1, 1600, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(seat.rejected_frames, 1, "duplicate frame refused");
    for r in packet(128, 6, 2, 9000, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(seat.position().x, 500, "new generation re-anchors");
    for r in packet(128, 7, 1, 0, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(seat.rejected_frames, 2, "retired generation refused");
}

#[test]
fn every_envelope_truncation_and_unknown_record_is_fail_closed() {
    let good = packet(128, 1, 1, 100, 1);
    for missing in 0..good.len() {
        let mut seat = PointerSeat::new(Viewport::new(100, 80).unwrap());
        let mut frames = 0;
        for (i, r) in good.iter().enumerate() {
            if i != missing {
                seat.feed_with(*r, &mut |e| {
                    if matches!(e, Dispatch::Frame(_)) {
                        frames += 1;
                    }
                });
            }
        }
        assert_eq!(frames, 0);
    }
    let mut seat = PointerSeat::new(Viewport::new(100, 80).unwrap());
    seat.feed(good[0], |_| {});
    let mut bad = good[1];
    bad.kind = 0x1f;
    seat.feed(bad, |_| {});
    for r in &good[2..] {
        seat.feed(*r, |_| {});
    }
    assert!(seat.rejected_frames > 0);
    assert_eq!(seat.position(), PointerPosition::new(50, 40));
}

#[test]
fn button_owners_survive_other_devices_and_loss_never_reclicks() {
    let mut seat = PointerSeat::new(Viewport::new(100, 80).unwrap());
    let mut presses = 0;
    let mut releases = 0;
    let mut observe = |event: SeatEvent| {
        if let SeatEventKind::Button { state, .. } = event.kind {
            match state {
                ButtonState::Pressed => presses += 1,
                ButtonState::Released => releases += 1,
            }
        }
    };
    seat.feed(InputReport::mouse(1, 1, 0, 0, 0), &mut observe);
    seat.feed(InputReport::mouse(2, 1, 0, 0, 0), &mut observe);
    seat.feed(InputReport::mouse(1, 0, 0, 0, 0), &mut observe);
    seat.reset(&mut observe);
    seat.feed(InputReport::mouse(2, 1, 0, 0, 0), &mut observe);
    seat.feed(InputReport::mouse(2, 0, 0, 0, 0), &mut observe);
    seat.feed(InputReport::mouse(2, 1, 0, 0, 0), &mut observe);
    assert_eq!((presses, releases), (2, 1));
}

#[derive(Clone, Default)]
struct Probe {
    samples: usize,
    deadline: Option<InputTimestamp>,
}
impl FramePolicy for Probe {
    fn frame(&mut self, f: PointerFrame<'_>, _emit: &mut impl FnMut(PolicyEvent<'_>)) {
        self.samples += f.contacts.len();
        self.deadline = f.timestamp.map(|t| InputTimestamp(t.0 + 100));
    }
    fn cancel(&mut self, _: CancelReason, _: &mut impl FnMut(PolicyEvent<'_>)) {
        self.deadline = None;
    }
    fn next_deadline(&self) -> Option<InputTimestamp> {
        self.deadline
    }
    fn advance(&mut self, _: InputTimestamp, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        self.deadline = None;
        emit(PolicyEvent::Gesture(GestureEvent::new(
            DeviceId(128),
            DeviceGeneration(1),
            4,
            GestureKind::Hold,
            GesturePhase::Begin,
            &[],
        )));
    }
}

#[test]
fn replacement_policy_and_caller_driven_deadlines_are_reachable() {
    let mut seat = PointerSeat::with_policy(Viewport::new(100, 80).unwrap(), Probe::default());
    for r in packet(128, 1, 1, 100, 1) {
        seat.feed_with(r, &mut |_| {});
    }
    assert_eq!(seat.policy(DeviceId(128)).map(|p| p.samples), Some(1));
    assert_eq!(seat.next_deadline(), Some(InputTimestamp(1_000_100)));
    let mut gestures = 0;
    seat.advance(InputTimestamp(1_000_099), &mut |e| {
        if matches!(e, Dispatch::Gesture(_)) {
            gestures += 1;
        }
    });
    assert_eq!(gestures, 0);
    seat.advance(InputTimestamp(1_000_100), &mut |e| {
        if matches!(e, Dispatch::Gesture(_)) {
            gestures += 1;
        }
    });
    assert_eq!(gestures, 1);
    assert_eq!(seat.next_deadline(), None);
    assert_eq!(
        seat.position(),
        PointerPosition::new(50, 40),
        "custom policy bypasses compatibility cursor mapping"
    );
}

#[derive(Clone)]
struct Forgetful;
impl FramePolicy for Forgetful {
    fn frame(&mut self, frame: PointerFrame<'_>, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        emit(PolicyEvent::Pointer(axinput::PointerInput {
            seat: frame.descriptor.seat,
            device: frame.descriptor.device,
            source: frame.descriptor.source,
            timestamp: frame.timestamp,
            kind: PointerInputKind::Button {
                button: axinput::PointerButton(0),
                state: ButtonState::Pressed,
            },
        }));
    }
    fn cancel(&mut self, _: CancelReason, _: &mut impl FnMut(PolicyEvent<'_>)) {}
}

#[test]
fn cancellation_cannot_strand_button_ownership_even_if_policy_forgets_release() {
    let mut seat = PointerSeat::with_policy(Viewport::new(100, 80).unwrap(), Forgetful);
    for r in packet(128, 1, 1, 100, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(seat.button_owners[0], 1);
    let mut releases = 0;
    seat.cancel_device(DeviceId(128), CancelReason::DeviceRemoved, &mut |event| {
        if matches!(
            event,
            Dispatch::Pointer(SeatEvent {
                kind: SeatEventKind::Button {
                    state: ButtonState::Released,
                    ..
                },
                ..
            })
        ) {
            releases += 1;
        }
    });
    assert_eq!(releases, 1);
    assert_eq!(seat.button_owners[0], 0);
    for r in packet(128, 2, 1, 200, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(
        seat.button_owners[0], 0,
        "removed generation cannot press again"
    );
    assert_eq!(seat.rejected_frames, 1);
}

#[test]
fn relative_density_scales_mouse_and_pad_without_moving_absolute_touch() {
    let mut seat = PointerSeat::new(Viewport::new(2000, 1600).unwrap());
    seat.set_motion_scale(MotionScale::new(2, 1).unwrap());
    seat.feed(InputReport::mouse(1, 0, 7, -3, 0), |_| {});
    assert_eq!(seat.position(), PointerPosition::new(1014, 794));
    for r in packet(128, 1, 1, 100, 1) {
        seat.feed(r, |_| {});
    }
    for r in packet(128, 2, 1, 200, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(seat.position(), PointerPosition::new(1034, 794));
    // A live density change preserves both pointer and trackpad anchor.
    seat.set_motion_scale(MotionScale::IDENTITY);
    for r in packet(128, 3, 1, 300, 1) {
        seat.feed(r, |_| {});
    }
    assert_eq!(seat.position(), PointerPosition::new(1044, 794));
    let mut native = PointerSeat::new(Viewport::new(2000, 1600).unwrap());
    seat.set_motion_scale(MotionScale::new(2, 1).unwrap());
    let mut touch = packet(129, 1, 1, 4000, 1);
    touch[0] = Record::Begin {
        sequence: 1,
        received_ns: None,
        class: 2,
    }
    .encode(129);
    for r in touch {
        seat.feed(r, |_| {});
        native.feed(r, |_| {});
    }
    assert_eq!(seat.position(), native.position());
    assert_eq!(seat.rejected_frames, 0);
}

/// Policies are bound per device in use, not per source byte: a device past
/// the pool takes the least recently used binding, and the device it takes
/// it from is cancelled first, so nothing it held stays pressed.
#[test]
fn a_device_past_the_policy_pool_takes_the_stalest_binding_cleanly() {
    let mut seat = PointerSeat::new(Viewport::new(100, 80).unwrap());
    let mut events = std::vec::Vec::new();
    // Source 1 presses and holds; sources 2.. fill the pool after it.
    seat.feed(InputReport::mouse(1, 1, 0, 0, 0), |_| {});
    for source in 2..=POLICY_SLOTS as u8 {
        seat.feed(InputReport::mouse(source, 0, 1, 0, 0), |_| {});
    }
    assert!(seat.policy(DeviceId(1)).is_some());
    assert_eq!(seat.button_owners[0], 1);
    seat.feed_with(InputReport::mouse(200, 0, 1, 0, 0), &mut |event| {
        events.push(match event {
            Dispatch::Cancelled { device, reason } => std::format!("cancel {} {reason:?}", device.0),
            Dispatch::Pointer(SeatEvent { kind: SeatEventKind::Button { state, .. }, .. }) => {
                std::format!("button {state:?}")
            }
            Dispatch::Pointer(_) => "pointer".into(),
            Dispatch::Frame(_) => "frame".into(),
            _ => "other".into(),
        });
    });
    assert_eq!(events, ["cancel 1 StreamLost", "button Released", "frame", "pointer"]);
    assert!(seat.policy(DeviceId(1)).is_none(), "the stalest device lost its binding");
    assert!(seat.policy(DeviceId(200)).is_some());
    assert_eq!(seat.button_owners[0], 0);
    // The evicted mouse comes back on a fresh binding: still down after the
    // gap, so it does not click again until it is seen up.
    let mut presses = 0;
    for state in [1, 0, 1] {
        seat.feed(InputReport::mouse(1, state, 0, 0, 0), |event| {
            if let SeatEventKind::Button { state: ButtonState::Pressed, .. } = event.kind {
                presses += 1;
            }
        });
    }
    assert_eq!(presses, 1, "still down after the gap: no second click");
    assert_eq!(seat.rejected_frames, 0);
}

#[test]
fn a_removed_device_frees_its_binding() {
    let mut seat = PointerSeat::new(Viewport::new(100, 80).unwrap());
    for r in packet(128, 1, 1, 100, 1) {
        seat.feed(r, |_| {});
    }
    assert!(seat.policy(DeviceId(128)).is_some());
    assert!(seat.next_deadline().is_none());
    seat.cancel_device(DeviceId(128), CancelReason::DeviceRemoved, &mut |_| {});
    assert!(seat.policy(DeviceId(128)).is_none());
    // A queue loss keeps it: the device is still there.
    for r in packet(129, 1, 1, 100, 1) {
        seat.feed(r, |_| {});
    }
    seat.cancel_device(DeviceId(129), CancelReason::StreamLost, &mut |_| {});
    assert!(seat.policy(DeviceId(129)).is_some());
}
