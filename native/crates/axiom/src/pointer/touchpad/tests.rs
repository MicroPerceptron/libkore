//! Deterministic replays of a 100 × 70 mm precision touchpad (10 units per
//! millimetre) through the gesture policy, at a 120 Hz report rate.

use std::vec::Vec;

use axinput::acceleration::Profile;
use axinput::frame::{
    AbsoluteAxis, AxisRange, AxisResolution, CancelReason, ContactSample, DeviceDescriptor,
    DeviceGeneration, FrameCompleteness, FramePolicy, GestureKind, GesturePhase, GestureSpace,
    PointerFrame, PolicyEvent, TimestampOrigin,
};
use axinput::{
    ButtonState, DeviceId, InputTimestamp, PointerInputKind, PointerSource, ScrollUnit, SeatId,
};

use super::*;

const PERIOD: u64 = 8_333_333;

#[derive(Clone, Debug, PartialEq)]
enum Out {
    Motion(i32, i32),
    Press(u16),
    Release(u16),
    Scroll(i32, i32),
    Gesture {
        kind: GestureKind,
        phase: GesturePhase,
        contacts: usize,
        interaction: u64,
        origin: [f64; 2],
        translation: [f64; 2],
        scale: f64,
        rotation: f64,
        velocity: Option<[f64; 2]>,
        space: GestureSpace,
    },
}

fn descriptor(source: PointerSource, calibrated: bool) -> DeviceDescriptor {
    let mut d = DeviceDescriptor::new(SeatId(0), DeviceId(3), DeviceGeneration(1), source);
    let resolution = calibrated.then(|| AxisResolution::new(10, 1).unwrap());
    d.axes = Some([
        AbsoluteAxis {
            range: AxisRange::new(0, 1000).unwrap(),
            resolution,
        },
        AbsoluteAxis {
            range: AxisRange::new(0, 700).unwrap(),
            resolution,
        },
    ]);
    d.max_contacts = Some(5);
    d
}

/// A finger: ID and position in millimetres.
#[derive(Clone, Copy)]
struct F(u32, f64, f64);

fn contact(f: F) -> ContactSample {
    let mut c = ContactSample::new(
        f.0,
        (f.1 * 10.0).round() as i32,
        (f.2 * 10.0).round() as i32,
        true,
    );
    c.confidence = Some(true);
    c
}

struct Pad {
    policy: TouchpadPolicy,
    descriptor: DeviceDescriptor,
    sequence: u32,
    now: u64,
    timed: bool,
    out: Vec<Out>,
}

impl Pad {
    fn new() -> Self {
        Self::with(TouchpadConfig::default(), true)
    }

    /// A pad with the user's acceleration off: a finger moves at exactly
    /// the pad's gain at any speed, so a replay pins the recognizer's
    /// geometry alone. Acceleration has its own replays below.
    fn unaccelerated() -> Self {
        let mut pad = Self::new();
        pad.policy
            .set_acceleration(AccelerationConfig::new(Profile::Flat, 0.0).unwrap());
        pad
    }

    fn with(config: TouchpadConfig, calibrated: bool) -> Self {
        Self {
            policy: TouchpadPolicy::new(config).unwrap(),
            descriptor: descriptor(PointerSource::Touchpad, calibrated),
            sequence: 0,
            now: 1_000_000_000,
            timed: true,
            out: Vec::new(),
        }
    }

    fn record(out: &mut Vec<Out>, event: PolicyEvent<'_>) {
        match event {
            PolicyEvent::Pointer(input) => out.push(match input.kind {
                PointerInputKind::RelativeMotion { dx, dy } => Out::Motion(dx, dy),
                PointerInputKind::Button { button, state } => match state {
                    ButtonState::Pressed => Out::Press(button.0),
                    ButtonState::Released => Out::Release(button.0),
                },
                PointerInputKind::Scroll {
                    horizontal,
                    vertical,
                    unit,
                } => {
                    assert_eq!(unit, ScrollUnit::Pixel, "touchpad scroll is in pixels");
                    Out::Scroll(horizontal, vertical)
                }
                other => panic!("unexpected {other:?}"),
            }),
            PolicyEvent::Gesture(g) => out.push(Out::Gesture {
                kind: g.kind,
                phase: g.phase,
                contacts: g.contacts.len(),
                interaction: g.interaction,
                origin: g.origin,
                translation: g.translation,
                scale: g.scale,
                rotation: g.rotation,
                velocity: g.velocity,
                space: g.space,
            }),
            other => panic!("unexpected {other:?}"),
        }
    }

    fn raw(&mut self, contacts: &[ContactSample], buttons: u64) {
        self.sequence += 1;
        let mut frame = PointerFrame::new(&self.descriptor, self.sequence, contacts);
        if self.timed {
            frame.timestamp = Some(InputTimestamp(self.now));
            frame.timestamp_origin = TimestampOrigin::Received;
        }
        frame.buttons = buttons;
        frame.reported_contacts = Some(contacts.len() as u16);
        let out = &mut self.out;
        self.policy.frame(frame, &mut |e| Self::record(out, e));
        self.now += PERIOD;
        self.service();
    }

    /// Service every deadline the policy asked for, up to the current time.
    fn service(&mut self) {
        while let Some(deadline) = self.policy.next_deadline() {
            if deadline.0 > self.now {
                break;
            }
            let out = &mut self.out;
            self.policy.advance(deadline, &mut |e| Self::record(out, e));
        }
    }

    fn frame(&mut self, fingers: &[F]) {
        self.press_frame(fingers, 0);
    }

    fn press_frame(&mut self, fingers: &[F], buttons: u64) {
        let contacts: Vec<_> = fingers.iter().map(|&f| contact(f)).collect();
        self.raw(&contacts, buttons);
    }

    /// Let `ns` pass with nothing on the pad, servicing deadlines.
    fn wait(&mut self, ns: u64) {
        self.now += ns;
        self.service();
    }

    /// Move `fingers` by (dx, dy) mm in `steps` frames.
    fn stroke(&mut self, fingers: &[F], dx: f64, dy: f64, steps: usize) -> Vec<F> {
        let mut at: Vec<F> = fingers.to_vec();
        for step in 1..=steps {
            let t = step as f64 / steps as f64;
            at = fingers
                .iter()
                .map(|f| F(f.0, f.1 + dx * t, f.2 + dy * t))
                .collect();
            self.frame(&at);
        }
        at
    }

    fn take(&mut self) -> Vec<Out> {
        core::mem::take(&mut self.out)
    }
}

fn motion(out: &[Out]) -> (i32, i32) {
    out.iter().fold((0, 0), |(x, y), o| match o {
        Out::Motion(dx, dy) => (x + dx, y + dy),
        _ => (x, y),
    })
}

fn scrolled(out: &[Out]) -> (i32, i32) {
    out.iter().fold((0, 0), |(x, y), o| match o {
        Out::Scroll(h, v) => (x + h, y + v),
        _ => (x, y),
    })
}

fn buttons(out: &[Out]) -> Vec<Out> {
    out.iter()
        .filter(|o| matches!(o, Out::Press(_) | Out::Release(_)))
        .cloned()
        .collect()
}

#[test]
fn one_finger_moves_the_pointer_at_the_fallback_gain() {
    let mut pad = Pad::unaccelerated();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.stroke(&[F(1, 50.0, 30.0)], 10.0, -5.0, 10);
    pad.frame(&[]);
    let out = pad.take();
    assert_eq!(motion(&out), (120, -60), "12 px/mm, landing anchors");
    assert!(buttons(&out).is_empty(), "a stroke is not a tap");
    assert_eq!(scrolled(&out), (0, 0));
}

/// libinput's touchpad factor at `mm_s`, relative to its 0.9 plateau.
fn touchpad_gain(mm_s: f64) -> f64 {
    let v = mm_s.min(520.0);
    (0.0025 * (v / 130.0) * (v - 130.0) + 0.9) / 0.9
}

#[test]
fn with_acceleration_on_a_finger_on_the_plateau_moves_at_the_pads_gain() {
    // 60 mm/s, inside libinput's plateau (7 to 130 mm/s): exactly 12 px/mm,
    // from the first report on, as it moved before acceleration.
    let mut pad = Pad::new();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.stroke(&[F(1, 50.0, 30.0)], 10.0, 0.0, 20);
    pad.frame(&[]);
    assert_eq!(motion(&pad.take()), (120, 0));
}

#[test]
fn a_fast_finger_is_accelerated_on_libinputs_touchpad_curve() {
    // 4 mm a frame at 120 Hz is 480 mm/s, high on the curve; each report is
    // timed from the last, the first from the landing.
    let mut pad = Pad::new();
    pad.frame(&[F(1, 10.0, 30.0)]);
    pad.stroke(&[F(1, 10.0, 30.0)], 40.0, 0.0, 10);
    pad.frame(&[]);
    let (x, y) = motion(&pad.take());
    let expected = 480.0 * touchpad_gain(480.0);
    assert!((f64::from(x) - expected).abs() <= 1.0, "{x} vs {expected}");
    assert_eq!(y, 0);
    // The same stroke with acceleration off is the pad's gain.
    let mut pad = Pad::unaccelerated();
    pad.frame(&[F(1, 10.0, 30.0)]);
    pad.stroke(&[F(1, 10.0, 30.0)], 40.0, 0.0, 10);
    pad.frame(&[]);
    assert_eq!(motion(&pad.take()), (480, 0));
}

#[test]
fn a_finger_landing_after_a_flick_is_timed_from_its_landing() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 10.0, 30.0)]);
    pad.stroke(&[F(1, 10.0, 30.0)], 40.0, 0.0, 10);
    pad.frame(&[]);
    assert!(motion(&pad.take()).0 > 480, "the flick is accelerated");
    // 50 ms later another touch moves at 60 mm/s: neither the flick's gain
    // nor a pause's crawl, exactly the pad's gain.
    pad.wait(50 * MS);
    pad.frame(&[F(2, 50.0, 30.0)]);
    pad.stroke(&[F(2, 50.0, 30.0)], -10.0, 0.0, 20);
    pad.frame(&[]);
    assert_eq!(motion(&pad.take()), (-120, 0));
}

#[test]
fn travel_owed_before_a_finger_points_is_timed_from_when_it_began() {
    // Two fingers land; one rests, the other moves up a millimetre a frame
    // (120 mm/s). Once it has clearly travelled alone it points, from where
    // it began: those five millimetres took five frames, so they are on the
    // plateau too, not a burst and not a crawl.
    let mut pad = Pad::new();
    let rest = F(2, 70.0, 30.0);
    pad.frame(&[F(1, 30.0, 30.0), rest]);
    for step in 1..=8 {
        pad.frame(&[F(1, 30.0, 30.0 - f64::from(step)), rest]);
    }
    let out = pad.take();
    assert_eq!(motion(&out), (0, -96), "{out:?}");
    assert_eq!(scrolled(&out), (0, 0));
}

#[test]
fn an_unclocked_pad_moves_at_the_plateau() {
    // Without timestamps no speed is measured: a fast stroke moves at the
    // pad's gain.
    let mut pad = Pad::new();
    pad.timed = false;
    pad.frame(&[F(1, 10.0, 30.0)]);
    pad.stroke(&[F(1, 10.0, 30.0)], 40.0, 0.0, 10);
    pad.frame(&[]);
    assert_eq!(motion(&pad.take()), (480, 0));
}

#[test]
fn the_speed_setting_scales_the_whole_touchpad_curve() {
    // The fastest setting's even scale, 2^1, multiplies the whole curve,
    // even the plateau.
    let mut pad = Pad::new();
    pad.policy
        .set_acceleration(AccelerationConfig::new(Profile::Adaptive, 1.0).unwrap());
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.stroke(&[F(1, 50.0, 30.0)], 10.0, 0.0, 20);
    pad.frame(&[]);
    let (x, _) = motion(&pad.take());
    let expected = 120.0 * 2.0;
    assert!((f64::from(x) - expected).abs() <= 1.0, "{x} vs {expected}");
}

#[test]
fn an_uncalibrated_pad_points_through_the_reference_stroke() {
    let mut pad = Pad::with(TouchpadConfig::default(), false);
    pad.policy
        .set_acceleration(AccelerationConfig::new(Profile::Flat, 0.0).unwrap());
    pad.frame(&[F(1, 0.0, 0.0)]);
    // 1000 units across = 1000 px; 100 units is 100 px.
    pad.stroke(&[F(1, 0.0, 0.0)], 10.0, 0.0, 5);
    pad.frame(&[]);
    assert_eq!(motion(&pad.take()), (100, 0));
    // 700 units down = 1000 px: 70 units is exactly 100 px, though no single
    // step's share of it is a whole pixel.
    pad.wait(DRAG_WINDOW_NS + MS);
    pad.frame(&[F(1, 0.0, 0.0)]);
    pad.stroke(&[F(1, 0.0, 0.0)], 0.0, 7.0, 7);
    pad.frame(&[]);
    assert_eq!(motion(&pad.take()), (0, 100));
}

#[test]
fn two_fingers_scroll_vertically_and_content_follows_them() {
    let mut pad = Pad::new();
    let pair = [F(1, 40.0, 20.0), F(2, 60.0, 20.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 0.0, 20.0, 20);
    let out = pad.take();
    assert_eq!(motion(&out), (0, 0), "two fingers never move the pointer");
    let (h, v) = scrolled(&out);
    assert_eq!(h, 0, "a vertical stroke stays vertical");
    // The first 2 mm decide the gesture; every millimetre after scrolls 12 px.
    assert!((v - (20.0 - 3.0) as i32 * 12).abs() <= 12, "scrolled {v}");
    assert!(v > 0, "fingers down show what is above (natural)");
    assert!(out.iter().all(|o| !matches!(o, Out::Press(_))));
}

#[test]
fn the_wheel_direction_inverts_without_natural_scrolling() {
    let config = TouchpadConfig {
        natural_scroll: false,
        ..TouchpadConfig::default()
    };
    let mut pad = Pad::with(config, true);
    let pair = [F(1, 40.0, 20.0), F(2, 60.0, 20.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 0.0, 20.0, 20);
    let (_, v) = scrolled(&pad.take());
    assert!(v < 0, "fingers down scroll down, as a wheel does");
}

#[test]
fn a_sideways_stroke_scrolls_horizontally() {
    let mut pad = Pad::new();
    let pair = [F(1, 30.0, 20.0), F(2, 30.0, 40.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 25.0, 0.0, 25);
    let (h, v) = scrolled(&pad.take());
    assert_eq!(v, 0);
    assert!(h < -200, "fingers right reveal what is left: {h}");
}

#[test]
fn the_axis_lock_lets_go_once_the_other_axis_clearly_moves() {
    let mut pad = Pad::new();
    let pair = [F(1, 40.0, 10.0), F(2, 60.0, 10.0)];
    pad.frame(&pair);
    let pair = pad.stroke(&pair, 0.0, 10.0, 10);
    let locked = pad.take();
    assert_eq!(scrolled(&locked).0, 0, "locked to vertical");
    // Drift sideways 5 mm: still locked. Then 6 mm more: past the 8 mm unlock.
    let pair = pad.stroke(&pair, 5.0, 2.0, 5);
    assert_eq!(
        scrolled(&pad.take()).0,
        0,
        "a drift under the unlock stays vertical"
    );
    pad.stroke(&pair, 6.0, 0.0, 6);
    assert!(
        scrolled(&pad.take()).0 < 0,
        "past the unlock both axes scroll"
    );
}

#[test]
fn taps_click_the_button_their_finger_count_names() {
    for (fingers, button) in [(1usize, 0u16), (2, 1), (3, 2)] {
        let mut pad = Pad::new();
        let at: Vec<F> = (0..fingers)
            .map(|i| F(i as u32 + 1, 30.0 + 15.0 * i as f64, 25.0))
            .collect();
        pad.frame(&at);
        pad.frame(&at);
        pad.frame(&[]);
        if fingers == 1 {
            // The primary press is held through the drag window.
            assert_eq!(buttons(&pad.take()), [Out::Press(0)]);
            pad.wait(DRAG_WINDOW_NS);
        }
        assert_eq!(
            buttons(&pad.take()),
            [Out::Press(button), Out::Release(button)][usize::from(fingers == 1)..],
            "{fingers}-finger tap"
        );
        assert!(
            pad.policy.next_deadline().is_none(),
            "nothing left to decide"
        );
    }
}

#[test]
fn without_tap_drag_a_tap_clicks_at_once() {
    let config = TouchpadConfig {
        tap_drag: false,
        ..TouchpadConfig::default()
    };
    let mut pad = Pad::with(config, true);
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[]);
    assert_eq!(buttons(&pad.take()), [Out::Press(0), Out::Release(0)]);
    assert!(pad.policy.next_deadline().is_none());
}

#[test]
fn a_slow_touch_or_a_moving_one_is_not_a_tap() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.now += TAP_TIMEOUT_NS;
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[]);
    assert!(buttons(&pad.take()).is_empty(), "held past the tap timeout");

    let mut pad = Pad::new();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[F(1, 52.0, 30.0)]);
    pad.frame(&[]);
    assert!(buttons(&pad.take()).is_empty(), "moved past the tap slop");
}

#[test]
fn without_timestamps_taps_are_off_and_scrolling_still_works() {
    let mut pad = Pad::new();
    pad.timed = false;
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[]);
    assert!(buttons(&pad.take()).is_empty(), "no invented durations");
    let pair = [F(1, 40.0, 20.0), F(2, 60.0, 20.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 0.0, 10.0, 10);
    pad.frame(&[]);
    assert!(scrolled(&pad.take()).1 > 0);
    assert!(
        pad.policy.next_deadline().is_none(),
        "no timing, no inertia"
    );
}

#[test]
fn a_touch_after_a_tap_that_moves_drags_with_the_tap_s_own_press() {
    let mut pad = Pad::unaccelerated();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[]);
    assert_eq!(buttons(&pad.take()), [Out::Press(0)]);
    pad.wait(50 * MS);
    pad.frame(&[F(2, 50.0, 30.0)]);
    pad.stroke(&[F(2, 50.0, 30.0)], 20.0, 0.0, 10);
    let out = pad.take();
    assert!(
        buttons(&out).is_empty(),
        "one press for the tap and the drag: no double-click"
    );
    assert_eq!(motion(&out), (240, 0), "and the pointer moves with it");
    pad.wait(DRAG_WINDOW_NS);
    assert!(
        buttons(&pad.take()).is_empty(),
        "a drag outlasts the window"
    );
    pad.frame(&[]);
    assert_eq!(
        buttons(&pad.take()),
        [Out::Release(0)],
        "lifting ends the drag"
    );
}

#[test]
fn a_touch_held_after_a_tap_drags_once_the_tap_timeout_passes() {
    let mut pad = Pad::unaccelerated();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[]);
    assert_eq!(buttons(&pad.take()), [Out::Press(0)]);
    pad.frame(&[F(2, 50.0, 30.0)]);
    assert!(
        pad.policy.next_deadline().is_some(),
        "the hold asks for a wake"
    );
    pad.wait(TAP_TIMEOUT_NS + DRAG_WINDOW_NS);
    assert!(
        buttons(&pad.take()).is_empty(),
        "held without moving: still the tap's press"
    );
    pad.stroke(&[F(2, 50.0, 30.0)], 10.0, 0.0, 5);
    assert_eq!(motion(&pad.take()), (120, 0));
    pad.frame(&[]);
    assert_eq!(buttons(&pad.take()), [Out::Release(0)]);
}

#[test]
fn two_quick_taps_are_two_clicks_and_a_late_touch_is_not_a_drag() {
    let mut pad = Pad::unaccelerated();
    for _ in 0..2 {
        pad.frame(&[F(1, 50.0, 30.0)]);
        pad.frame(&[]);
    }
    assert_eq!(
        buttons(&pad.take()),
        [Out::Press(0), Out::Release(0), Out::Press(0)],
        "the second tap is a second click, held for a drag of its own"
    );
    pad.wait(DRAG_WINDOW_NS);
    assert_eq!(buttons(&pad.take()), [Out::Release(0)]);
    pad.wait(DRAG_WINDOW_NS + MS);
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.stroke(&[F(1, 50.0, 30.0)], 10.0, 0.0, 5);
    pad.frame(&[]);
    let out = pad.take();
    assert!(
        buttons(&out).is_empty(),
        "outside the drag window it only points"
    );
    assert_eq!(motion(&out), (120, 0));
}

#[test]
fn a_physical_click_counts_the_fingers_on_the_pad() {
    for (fingers, button) in [(1usize, 0u16), (2, 1), (3, 2)] {
        let mut pad = Pad::new();
        let at: Vec<F> = (0..fingers)
            .map(|i| F(i as u32 + 1, 30.0 + 15.0 * i as f64, 25.0))
            .collect();
        pad.frame(&at);
        pad.press_frame(&at, 1);
        pad.press_frame(&at, 1);
        pad.press_frame(&at, 0);
        pad.frame(&[]);
        assert_eq!(
            buttons(&pad.take()),
            [Out::Press(button), Out::Release(button)],
            "{fingers} fingers on a clicked pad, and no tap after it"
        );
    }
}

#[test]
fn a_resting_thumb_neither_adds_a_finger_nor_moves_the_pointer() {
    let mut pad = Pad::unaccelerated();
    // The thumb rests in the bottom band (y > 60 mm on a 70 mm pad).
    let thumb = F(9, 50.0, 66.0);
    pad.frame(&[thumb]);
    pad.wait(300 * MS);
    pad.frame(&[thumb]);
    pad.take();
    // The index taps: a primary click, not a secondary one.
    pad.frame(&[thumb, F(1, 50.0, 30.0)]);
    pad.frame(&[thumb]);
    pad.wait(DRAG_WINDOW_NS);
    assert_eq!(buttons(&pad.take()), [Out::Press(0), Out::Release(0)]);
    // The index points while the thumb rests.
    pad.wait(DRAG_WINDOW_NS + MS);
    pad.frame(&[thumb, F(2, 50.0, 30.0)]);
    pad.stroke(&[thumb, F(2, 50.0, 30.0)], 0.0, 0.0, 1);
    let index_only: Vec<F> = (1..=5).map(|i| F(2, 50.0 + 2.0 * i as f64, 30.0)).collect();
    for f in index_only {
        pad.frame(&[thumb, f]);
    }
    let out = pad.take();
    assert_eq!(motion(&out), (120, 0), "only the index moved the pointer");
    assert_eq!(scrolled(&out), (0, 0), "the thumb did not make it a scroll");
    // Clicking with the thumb while the index rests is a primary click.
    pad.press_frame(&[thumb, F(2, 60.0, 30.0)], 1);
    pad.press_frame(&[thumb, F(2, 60.0, 30.0)], 0);
    assert_eq!(buttons(&pad.take()), [Out::Press(0), Out::Release(0)]);
}

#[test]
fn a_thumb_landing_alone_still_taps_and_points() {
    let mut pad = Pad::unaccelerated();
    pad.frame(&[F(1, 50.0, 66.0)]);
    pad.frame(&[]);
    pad.wait(DRAG_WINDOW_NS);
    assert_eq!(buttons(&pad.take()), [Out::Press(0), Out::Release(0)]);
    pad.wait(MS);
    pad.frame(&[F(1, 50.0, 66.0)]);
    pad.stroke(&[F(1, 50.0, 66.0)], 10.0, -20.0, 10);
    assert_eq!(motion(&pad.take()), (120, -240));
    // A lone finger resting in the band is still a finger when it moves.
    pad.frame(&[]);
    pad.wait(DRAG_WINDOW_NS);
    pad.frame(&[F(2, 50.0, 66.0)]);
    for _ in 0..50 {
        pad.frame(&[F(2, 50.0, 66.0)]);
    }
    pad.stroke(&[F(2, 50.0, 66.0)], 10.0, 0.0, 10);
    assert_eq!(
        motion(&pad.take()),
        (120, 0),
        "no thumb dead zone for a lone finger"
    );
}

#[test]
fn a_resting_thumb_does_not_catch_a_flick() {
    let mut pad = Pad::new();
    let thumb = F(9, 50.0, 66.0);
    let pair = [thumb, F(1, 40.0, 10.0), F(2, 60.0, 10.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 0.0, 0.0, 1);
    let moved: Vec<F> = (1..=10)
        .map(|i| i as f64 * 4.0)
        .flat_map(|d| [F(1, 40.0, 10.0 + d), F(2, 60.0, 10.0 + d)])
        .collect();
    for step in moved.chunks(2) {
        pad.frame(&[thumb, step[0], step[1]]);
    }
    pad.take();
    for _ in 0..6 {
        pad.frame(&[thumb]);
    }
    pad.take();
    // Still coasting several frames on, with the thumb reported every frame.
    for _ in 0..3 {
        pad.frame(&[thumb]);
    }
    assert!(
        scrolled(&pad.take()).1 > 0,
        "it coasts with the thumb still down"
    );
    assert!(pad.policy.next_deadline().is_some());
    // A finger landing catches it.
    pad.frame(&[thumb, F(3, 50.0, 30.0)]);
    pad.wait(100 * MS);
    assert_eq!(scrolled(&pad.take()), (0, 0));
}

#[test]
fn a_touch_after_a_tap_with_two_fingers_is_not_a_drag() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[]);
    assert_eq!(buttons(&pad.take()), [Out::Press(0)]);
    let pair = [F(2, 40.0, 20.0), F(3, 60.0, 20.0)];
    pad.frame(&pair);
    assert_eq!(
        buttons(&pad.take()),
        [Out::Release(0)],
        "the tap was only a click"
    );
    pad.stroke(&pair, 0.0, 10.0, 10);
    let out = pad.take();
    assert!(scrolled(&out).1 > 0, "the pair scrolls");
    assert!(buttons(&out).is_empty());
}

#[test]
fn a_second_finger_during_a_drag_neither_scrolls_nor_drops_it() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 50.0, 30.0)]);
    pad.frame(&[]);
    pad.frame(&[F(2, 50.0, 30.0)]);
    let at = pad.stroke(&[F(2, 50.0, 30.0)], 5.0, 0.0, 5);
    assert_eq!(buttons(&pad.take()), [Out::Press(0)]);
    // A second finger joins and both travel: still the drag, pointing.
    let pair = [at[0], F(3, 30.0, 30.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 0.0, 10.0, 10);
    let out = pad.take();
    assert_eq!(scrolled(&out), (0, 0));
    assert!(buttons(&out).is_empty());
    assert_eq!(motion(&out).1, 120);
    pad.frame(&[]);
    assert_eq!(buttons(&pad.take()), [Out::Release(0)]);
}

#[test]
fn the_finger_left_by_a_two_finger_tap_does_not_nudge_the_pointer() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 40.0, 30.0), F(2, 60.0, 30.0)]);
    pad.frame(&[F(1, 40.0, 30.0)]);
    pad.frame(&[F(1, 41.0, 30.5)]);
    pad.frame(&[]);
    let out = pad.take();
    assert_eq!(motion(&out), (0, 0));
    assert_eq!(buttons(&out), [Out::Press(1), Out::Release(1)]);
}

#[test]
fn a_palm_never_counts_or_moves() {
    let mut pad = Pad::new();
    let mut palm = contact(F(7, 20.0, 40.0));
    palm.confidence = Some(false);
    let finger = |x: f64| contact(F(1, x, 30.0));
    pad.raw(&[palm, finger(50.0)], 0);
    for i in 1..=5 {
        let mut moved = palm;
        moved.x += 30 * i;
        pad.raw(&[moved, finger(50.0 + i as f64)], 0);
    }
    pad.raw(&[], 0);
    let out = pad.take();
    assert_eq!(motion(&out), (60, 0), "only the finger moved the pointer");
    assert_eq!(scrolled(&out), (0, 0));
    // A palm-only touch taps nothing.
    pad.wait(DRAG_WINDOW_NS + MS);
    pad.raw(&[palm], 0);
    pad.raw(&[], 0);
    assert!(buttons(&pad.take()).is_empty());
}

#[test]
fn a_flick_coasts_on_and_a_touch_catches_it() {
    let mut pad = Pad::new();
    let pair = [F(1, 40.0, 10.0), F(2, 60.0, 10.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 0.0, 40.0, 10); // 40 mm in 83 ms: a flick
    let during = scrolled(&pad.take()).1;
    pad.frame(&[]);
    assert!(
        pad.policy.next_deadline().is_some(),
        "inertia asks for wakes"
    );
    pad.wait(50 * MS);
    let coasted = scrolled(&pad.take()).1;
    assert!(coasted > 0, "it keeps going the same way");
    // A finger lands: the coast stops at once.
    pad.frame(&[F(3, 50.0, 30.0)]);
    pad.wait(200 * MS);
    assert_eq!(scrolled(&pad.take()), (0, 0), "caught");
    assert!(during > 0);
}

#[test]
fn a_coast_decays_to_rest_by_itself() {
    let mut pad = Pad::new();
    let pair = [F(1, 40.0, 10.0), F(2, 60.0, 10.0)];
    pad.frame(&pair);
    pad.stroke(&pair, 0.0, 30.0, 8);
    pad.frame(&[]);
    pad.take();
    let mut total = 0;
    let mut last = i32::MAX;
    for _ in 0..400 {
        pad.wait(INERTIA_STEP_NS);
        let step = scrolled(&pad.take()).1;
        // Whole pixels out of a decaying curve: one step may round up by one.
        assert!(step <= last.saturating_add(1), "it never speeds up");
        if step > 0 {
            last = step;
        }
        total += step;
    }
    assert!(pad.policy.next_deadline().is_none(), "it comes to rest");
    assert!(total > 0);
}

#[test]
fn a_paused_scroll_does_not_coast() {
    let mut pad = Pad::new();
    let pair = [F(1, 40.0, 10.0), F(2, 60.0, 10.0)];
    pad.frame(&pair);
    let pair = pad.stroke(&pair, 0.0, 30.0, 8);
    pad.now += 100 * MS;
    pad.frame(&pair);
    pad.frame(&[]);
    pad.take();
    assert!(pad.policy.next_deadline().is_none());
}

#[test]
fn lifting_two_scrolling_fingers_one_at_a_time_keeps_the_pointer_and_the_flick() {
    let mut pad = Pad::new();
    let pair = [F(1, 40.0, 10.0), F(2, 60.0, 10.0)];
    pad.frame(&pair);
    let pair = pad.stroke(&pair, 0.0, 40.0, 10);
    pad.take();
    pad.frame(&pair[..1]);
    pad.frame(&[]);
    let out = pad.take();
    assert_eq!(motion(&out), (0, 0), "the last finger off did not point");
    assert!(pad.policy.next_deadline().is_some(), "the flick survived");
}

#[test]
fn one_finger_resting_and_one_moving_points() {
    let mut pad = Pad::new();
    let pair = [F(1, 30.0, 30.0), F(2, 60.0, 30.0)];
    pad.frame(&pair);
    for i in 1..=10 {
        pad.frame(&[F(1, 30.0, 30.0), F(2, 60.0 + i as f64, 30.0)]);
    }
    let out = pad.take();
    assert_eq!(scrolled(&out), (0, 0));
    // Deciding took 4 mm of travel; the pointer catches up on all of it.
    assert_eq!(
        motion(&out),
        (120, 0),
        "the moving finger points, every millimetre"
    );
}

#[test]
fn spreading_two_fingers_is_a_transform_not_a_scroll() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 45.0, 30.0), F(2, 55.0, 30.0)]);
    for i in 1..=10 {
        let d = i as f64 * 2.0;
        pad.frame(&[F(1, 45.0 - d, 30.0), F(2, 55.0 + d, 30.0)]);
    }
    pad.frame(&[]);
    let out = pad.take();
    assert_eq!(scrolled(&out), (0, 0));
    let gestures: Vec<_> = out
        .iter()
        .filter_map(|o| match o {
            Out::Gesture {
                kind,
                phase,
                scale,
                contacts,
                space,
                ..
            } => Some((*kind, *phase, *scale, *contacts, *space)),
            _ => None,
        })
        .collect();
    assert_eq!(
        gestures.first().map(|g| (g.0, g.1)),
        Some((GestureKind::Transform, GesturePhase::Begin))
    );
    let end = gestures.last().unwrap();
    assert_eq!(
        (end.0, end.1, end.3, end.4),
        (
            GestureKind::Transform,
            GesturePhase::End,
            2,
            GestureSpace::Millimeters
        )
    );
    // Recognized at 11 mm from the centroid (4 mm of change), spread to 25.
    assert!((end.2 - 25.0 / 11.0).abs() < 0.01, "scale {}", end.2);
    assert!(buttons(&out).is_empty());
}

#[test]
fn three_fingers_moving_together_swipe() {
    let mut pad = Pad::new();
    let trio = [F(1, 30.0, 30.0), F(2, 45.0, 30.0), F(3, 60.0, 30.0)];
    pad.frame(&trio);
    pad.stroke(&trio, 20.0, 0.0, 10);
    pad.frame(&[]);
    let out = pad.take();
    let swipe: Vec<_> = out
        .iter()
        .filter_map(|o| match o {
            Out::Gesture {
                kind: GestureKind::Swipe,
                phase,
                translation,
                contacts,
                ..
            } => Some((*phase, *translation, *contacts)),
            _ => None,
        })
        .collect();
    assert_eq!(swipe.first().unwrap().0, GesturePhase::Begin);
    let end = swipe.last().unwrap();
    assert_eq!((end.0, end.2), (GesturePhase::End, 3));
    // Recognized at 6 mm (past the 4 mm start); 14 mm more since Begin.
    assert!(
        (end.1[0] - 14.0).abs() < 0.01,
        "totals since Begin: {:?}",
        end.1
    );
    let velocity = out.iter().rev().find_map(|o| match o {
        Out::Gesture {
            phase: GesturePhase::End,
            velocity,
            ..
        } => Some(*velocity),
        _ => None,
    });
    // 2 mm a frame at 120 Hz.
    let [vx, vy] = velocity.flatten().expect("timed frames carry a velocity");
    assert!(
        (vx - 240.0).abs() < 1.0 && vy.abs() < 1.0,
        "velocity {vx} {vy}"
    );
    assert_eq!(motion(&out), (0, 0));
    assert_eq!(scrolled(&out), (0, 0));
    assert!(buttons(&out).is_empty(), "a swipe is not a tap");
}

#[test]
fn hybrid_fragments_assemble_by_scan_and_a_broken_scan_is_dropped_whole() {
    let mut pad = Pad::new();
    let fragment = |pad: &mut Pad, fingers: &[F], reported: u16, scan: u32| {
        let contacts: Vec<_> = fingers.iter().map(|&f| contact(f)).collect();
        pad.sequence += 1;
        let mut frame = PointerFrame::new(&pad.descriptor, pad.sequence, &contacts);
        frame.timestamp = Some(InputTimestamp(pad.now));
        frame.timestamp_origin = TimestampOrigin::Received;
        frame.completeness = FrameCompleteness::Fragment;
        frame.reported_contacts = Some(reported);
        frame.scan_time = Some(scan);
        let out = &mut pad.out;
        pad.policy.frame(frame, &mut |e| Pad::record(out, e));
        pad.now += PERIOD / 2;
    };
    // Two fingers reported one per packet, scrolling down.
    for i in 0..12u32 {
        let y = 10.0 + i as f64 * 2.0;
        fragment(&mut pad, &[F(1, 40.0, y)], 2, 100 + i);
        fragment(&mut pad, &[F(2, 60.0, y)], 0, 100 + i);
    }
    assert!(scrolled(&pad.take()).1 > 0, "assembled scans scroll");
    assert_eq!(pad.policy.dropped_scans(), 0);
    // A continuation from another scan breaks this one: dropped, not guessed.
    fragment(&mut pad, &[F(1, 40.0, 40.0)], 2, 200);
    fragment(&mut pad, &[F(2, 60.0, 40.0)], 0, 201);
    assert_eq!(scrolled(&pad.take()), (0, 0));
    assert!(pad.policy.dropped_scans() >= 1);
}

#[test]
fn a_cancel_releases_what_is_held_and_a_held_click_must_lift_first() {
    let mut pad = Pad::new();
    pad.press_frame(&[F(1, 50.0, 30.0)], 1);
    assert_eq!(buttons(&pad.take()), [Out::Press(0)]);
    let out = &mut pad.out;
    pad.policy
        .cancel(CancelReason::StreamLost, &mut |e| Pad::record(out, e));
    assert_eq!(buttons(&pad.take()), [Out::Release(0)]);
    // The button is still physically down after the gap: no second click.
    pad.press_frame(&[F(1, 50.0, 30.0)], 1);
    assert!(buttons(&pad.take()).is_empty());
    pad.press_frame(&[F(1, 50.0, 30.0)], 0);
    pad.press_frame(&[F(1, 50.0, 30.0)], 1);
    assert_eq!(buttons(&pad.take()), [Out::Press(0)]);
}

#[test]
fn a_cancelled_transform_says_so() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 45.0, 30.0), F(2, 55.0, 30.0)]);
    for i in 1..=5 {
        let d = i as f64 * 2.0;
        pad.frame(&[F(1, 45.0 - d, 30.0), F(2, 55.0 + d, 30.0)]);
    }
    pad.take();
    let out = &mut pad.out;
    pad.policy
        .cancel(CancelReason::DeviceReset, &mut |e| Pad::record(out, e));
    let out = pad.take();
    assert!(matches!(
        out.last(),
        Some(Out::Gesture { phase: GesturePhase::Cancel, scale, .. }) if *scale == 1.0
    ));
}

#[test]
fn other_sources_keep_the_fallback() {
    let mut policy = TouchpadPolicy::default();
    let mouse = DeviceDescriptor::new(
        SeatId(0),
        DeviceId(1),
        DeviceGeneration(0),
        PointerSource::Mouse,
    );
    let mut frame = PointerFrame::new(&mouse, 1, &[]);
    frame.relative = [5, -3];
    frame.buttons = 1;
    let mut out = Vec::new();
    policy.frame(frame, &mut |e| Pad::record(&mut out, e));
    assert_eq!(out, [Out::Motion(5, -3), Out::Press(0)]);
}

#[test]
fn turning_two_fingers_is_a_transform_that_rotates() {
    let mut pad = Pad::new();
    // Two fingers 20 mm apart, turning about their midpoint 5° a frame.
    let at = |degrees: f64| {
        let (sin, cos) = degrees.to_radians().sin_cos();
        [
            F(1, 50.0 - 10.0 * cos, 30.0 - 10.0 * sin),
            F(2, 50.0 + 10.0 * cos, 30.0 + 10.0 * sin),
        ]
    };
    for step in 0..=12 {
        pad.frame(&at(5.0 * step as f64));
    }
    pad.frame(&[]);
    let out = pad.take();
    assert_eq!(scrolled(&out), (0, 0), "a turn is not a scroll");
    assert_eq!(motion(&out), (0, 0));
    let shapes: Vec<_> = out
        .iter()
        .filter_map(|o| match o {
            Out::Gesture {
                kind: GestureKind::Transform,
                phase,
                rotation,
                scale,
                translation,
                ..
            } => Some((*phase, *rotation, *scale, *translation)),
            _ => None,
        })
        .collect();
    assert_eq!(shapes.first().unwrap().0, GesturePhase::Begin);
    let &(phase, rotation, scale, translation) = shapes.last().unwrap();
    assert_eq!(phase, GesturePhase::End);
    // Each finger's arc passes 4 mm at 25° (4.36 mm); 35° more since Begin,
    // positive from +X toward +Y.
    assert!(
        (rotation - 35f64.to_radians()).abs() < 0.01,
        "rotation {rotation}"
    );
    assert!((scale - 1.0).abs() < 0.01, "scale {scale}");
    assert!(translation[0].abs() < 0.1 && translation[1].abs() < 0.1);
    assert!(buttons(&out).is_empty());
}

#[test]
fn an_uncalibrated_pad_reports_gestures_in_its_own_coordinates() {
    let mut pad = Pad::with(TouchpadConfig::default(), false);
    // On this pad a millimetre of the harness is 10 device units, and the
    // policy measures 12 units (one reference pixel each) as its millimetre.
    let trio = [F(1, 30.0, 30.0), F(2, 45.0, 30.0), F(3, 60.0, 30.0)];
    pad.frame(&trio);
    pad.stroke(&trio, 20.0, 0.0, 10);
    pad.frame(&[]);
    let out = pad.take();
    let end = out
        .iter()
        .rev()
        .find_map(|o| match o {
            Out::Gesture {
                phase: GesturePhase::End,
                origin,
                translation,
                space,
                ..
            } => Some((*origin, *translation, *space)),
            _ => None,
        })
        .expect("a swipe");
    assert_eq!(
        end.2,
        GestureSpace::DeviceCoordinates,
        "no guessed millimetres"
    );
    // Recognized after 60 units (5 of its millimetres), at centroid x 510.
    assert!(
        (end.0[0] - 510.0).abs() < 0.01 && (end.0[1] - 300.0).abs() < 0.01,
        "origin {:?}",
        end.0
    );
    assert!(
        (end.1[0] - 140.0).abs() < 0.01 && end.1[1].abs() < 0.01,
        "translation {:?}",
        end.1
    );
}

/// One hybrid-mode report as the I2C-HID transport delivers it: a report
/// counting more contacts than it carries is a fragment, and one counting
/// zero is a fragment only when it carries a touching contact. Its lifting
/// contacts are not delivered at all.
fn hybrid(pad: &mut Pad, fingers: &[(F, bool)], reported: u16, scan: u32) {
    let delivered: Vec<ContactSample> = fingers
        .iter()
        .filter(|(_, touching)| reported != 0 || *touching)
        .map(|&(f, touching)| {
            let mut c = contact(f);
            c.touching = touching;
            c
        })
        .collect();
    let touching = delivered.iter().any(|c| c.touching);
    pad.sequence += 1;
    let mut frame = PointerFrame::new(&pad.descriptor, pad.sequence, &delivered);
    if pad.timed {
        frame.timestamp = Some(InputTimestamp(pad.now));
        frame.timestamp_origin = TimestampOrigin::Received;
    }
    frame.completeness = if usize::from(reported) > fingers.len() || (reported == 0 && touching) {
        FrameCompleteness::Fragment
    } else {
        FrameCompleteness::Complete
    };
    frame.reported_contacts = Some(reported);
    frame.scan_time = Some(scan);
    let out = &mut pad.out;
    pad.policy.frame(frame, &mut |e| Pad::record(out, e));
    // Reports of one scan arrive back to back.
    pad.now += MS;
    pad.service();
}

#[test]
fn a_finger_lifting_in_a_hybrid_continuation_is_a_lift_not_an_empty_pad() {
    let mut pad = Pad::new();
    let (a, b) = (F(1, 40.0, 30.0), F(2, 60.0, 30.0));
    // Two fingers tap; the second lifts first, in a continuation the
    // transport delivers as an empty, complete report.
    for scan in 1..=2 {
        hybrid(&mut pad, &[(a, true)], 2, scan);
        hybrid(&mut pad, &[(b, true)], 0, scan);
        pad.now += PERIOD;
    }
    hybrid(&mut pad, &[(a, true)], 2, 3);
    hybrid(&mut pad, &[(b, false)], 0, 3);
    assert!(buttons(&pad.take()).is_empty(), "one finger is still down");
    pad.now += PERIOD;
    hybrid(&mut pad, &[(a, true)], 1, 4);
    pad.now += PERIOD;
    hybrid(&mut pad, &[(a, false)], 1, 5);
    let out = pad.take();
    assert_eq!(
        buttons(&out),
        [Out::Press(1), Out::Release(1)],
        "one secondary click, no primary"
    );
    assert_eq!(motion(&out), (0, 0));
    assert_eq!(pad.policy.dropped_scans(), 0);
}

#[test]
fn a_hybrid_scan_whose_last_report_lifts_settles_without_a_next_one() {
    let mut pad = Pad::new();
    let (a, b) = (F(1, 40.0, 30.0), F(2, 60.0, 30.0));
    hybrid(&mut pad, &[(a, true)], 2, 1);
    hybrid(&mut pad, &[(b, true)], 0, 1);
    pad.now += PERIOD;
    // Both lift in one scan, and the pad falls silent.
    hybrid(&mut pad, &[(a, false)], 2, 2);
    hybrid(&mut pad, &[(b, false)], 0, 2);
    assert!(buttons(&pad.take()).is_empty());
    assert!(
        pad.policy.next_deadline().is_some(),
        "the short scan asks for a wake"
    );
    pad.wait(HYBRID_SETTLE_NS);
    assert_eq!(buttons(&pad.take()), [Out::Press(1), Out::Release(1)]);
    assert_eq!(pad.policy.dropped_scans(), 0);

    // Without a clock there is no settle deadline: the short scan settles
    // when the next report arrives, ahead of it, so the next finger starts
    // on an empty pad.
    let mut pad = Pad::new();
    pad.timed = false;
    let pair = [(a, true), (b, true)];
    for scan in 1..=3 {
        hybrid(&mut pad, &pair[..1], 2, scan);
        hybrid(&mut pad, &pair[1..], 0, scan);
    }
    hybrid(&mut pad, &[(a, false)], 2, 4);
    hybrid(&mut pad, &[(b, false)], 0, 4);
    pad.take();
    pad.frame(&[F(3, 50.0, 30.0)]);
    pad.stroke(&[F(3, 50.0, 30.0)], 10.0, 0.0, 5);
    assert_eq!(
        motion(&pad.take()),
        (120, 0),
        "a lone new finger, not a third"
    );
}

#[test]
fn a_hybrid_scan_missing_its_continuation_is_dropped_not_guessed() {
    let mut pad = Pad::new();
    let (a, b) = (F(1, 40.0, 30.0), F(2, 60.0, 30.0));
    hybrid(&mut pad, &[(a, true)], 2, 1);
    hybrid(&mut pad, &[(b, true)], 0, 1);
    pad.now += PERIOD;
    // A first report, and then the next scan: its continuation never came,
    // so nothing says its other contact lifted.
    hybrid(&mut pad, &[(a, true)], 2, 2);
    pad.now += PERIOD;
    hybrid(&mut pad, &[(a, true)], 2, 3);
    hybrid(&mut pad, &[(b, true)], 0, 3);
    assert_eq!(pad.policy.dropped_scans(), 1);
    pad.now += PERIOD;
    hybrid(&mut pad, &[(a, false)], 2, 4);
    hybrid(&mut pad, &[(b, false)], 0, 4);
    pad.wait(HYBRID_SETTLE_NS);
    assert_eq!(
        buttons(&pad.take()),
        [Out::Press(1), Out::Release(1)],
        "still one two-finger tap"
    );
}

#[test]
fn a_palm_the_firmware_flags_after_it_lands_never_taps() {
    let mut pad = Pad::new();
    // A contact lands confident, and is flagged a palm before it moves.
    let mut late = contact(F(7, 30.0, 40.0));
    pad.raw(&[late], 0);
    late.confidence = Some(false);
    pad.raw(&[late], 0);
    pad.raw(&[], 0);
    pad.wait(DRAG_WINDOW_NS);
    assert!(buttons(&pad.take()).is_empty(), "no click from the palm");
    // Beside a tapping finger it does not make the tap a secondary click.
    pad.wait(DRAG_WINDOW_NS);
    let mut late = contact(F(8, 30.0, 40.0));
    pad.raw(&[late, contact(F(1, 60.0, 30.0))], 0);
    late.confidence = Some(false);
    pad.raw(&[late, contact(F(1, 60.0, 30.0))], 0);
    pad.raw(&[], 0);
    pad.wait(DRAG_WINDOW_NS);
    assert_eq!(buttons(&pad.take()), [Out::Press(0), Out::Release(0)]);
}

#[test]
fn clicking_the_pad_mid_pinch_ends_the_pinch() {
    let mut pad = Pad::new();
    pad.frame(&[F(1, 45.0, 30.0), F(2, 55.0, 30.0)]);
    let mut pair = [F(1, 45.0, 30.0), F(2, 55.0, 30.0)];
    for i in 1..=5 {
        let d = i as f64 * 2.0;
        pair = [F(1, 45.0 - d, 30.0), F(2, 55.0 + d, 30.0)];
        pad.frame(&pair);
    }
    pad.press_frame(&pair, 1);
    pad.press_frame(&pair, 0);
    pad.frame(&[]);
    let out = pad.take();
    let phases: Vec<_> = out
        .iter()
        .filter_map(|o| match o {
            Out::Gesture { phase, .. } => Some(*phase),
            _ => None,
        })
        .collect();
    assert_eq!(phases.first(), Some(&GesturePhase::Begin));
    assert_eq!(
        phases.last(),
        Some(&GesturePhase::End),
        "the press ends it: {phases:?}"
    );
    assert_eq!(
        phases.iter().filter(|p| **p == GesturePhase::End).count(),
        1
    );
    assert_eq!(buttons(&out), [Out::Press(1), Out::Release(1)]);
}

#[test]
fn fingers_down_across_a_cancel_never_tap() {
    let mut pad = Pad::new();
    let pair = [F(1, 40.0, 30.0), F(2, 60.0, 30.0)];
    pad.frame(&pair);
    let out = &mut pad.out;
    pad.policy
        .cancel(CancelReason::StreamLost, &mut |e| Pad::record(out, e));
    // The stream resumes with the same fingers, which lift at once.
    pad.frame(&pair);
    pad.frame(&[]);
    pad.wait(DRAG_WINDOW_NS);
    assert!(
        buttons(&pad.take()).is_empty(),
        "a touch begun before the gap is no tap"
    );
    // The next touch is an ordinary one.
    pad.frame(&[F(3, 50.0, 30.0)]);
    pad.frame(&[]);
    pad.wait(DRAG_WINDOW_NS);
    assert_eq!(buttons(&pad.take()), [Out::Press(0), Out::Release(0)]);
}

#[test]
fn gesture_interactions_keep_counting_across_a_cancel() {
    let mut pad = Pad::new();
    let pinch = |pad: &mut Pad| {
        pad.frame(&[F(1, 45.0, 30.0), F(2, 55.0, 30.0)]);
        for i in 1..=5 {
            let d = i as f64 * 2.0;
            pad.frame(&[F(1, 45.0 - d, 30.0), F(2, 55.0 + d, 30.0)]);
        }
    };
    let begun = |out: &[Out]| {
        out.iter()
            .find_map(|o| match o {
                Out::Gesture {
                    phase: GesturePhase::Begin,
                    interaction,
                    ..
                } => Some(*interaction),
                _ => None,
            })
            .expect("a transform")
    };
    pinch(&mut pad);
    let first = begun(&pad.take());
    let out = &mut pad.out;
    pad.policy
        .cancel(CancelReason::DeviceReset, &mut |e| Pad::record(out, e));
    pad.frame(&[]);
    pinch(&mut pad);
    assert!(
        begun(&pad.take()) > first,
        "a new gesture is never an old one's id"
    );
}
