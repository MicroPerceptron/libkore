//! These regressions intentionally use the old public discovery/drain API,
//! so they can run unchanged against the pre-fix driver for an A/B witness.
use super::*;
use crate::mock::{MockPool, MockXhciDevice};

pub(super) const KEYBOARD_MOUSE: [u8; 59] = [
    9, 2, 59, 0, 2, 1, 0, 0xA0, 50, 9, 4, 7, 0, 1, 3, 1, 1, 0, 9, 0x21, 0x11, 1, 0, 1, 0x22, 63, 0,
    7, 5, 0x81, 3, 8, 0, 10, 9, 4, 42, 0, 1, 3, 1, 2, 0, 9, 0x21, 0x11, 1, 0, 1, 0x22, 52, 0, 7, 5,
    0x82, 3, 4, 0, 10,
];

const MOUSE_KEYBOARD: [u8; 59] = {
    let mut bytes = KEYBOARD_MOUSE;
    bytes[16] = 2;
    bytes[31] = 4;
    bytes[41] = 1;
    bytes[56] = 8;
    bytes
};

fn spin() {}

#[test]
fn usb_separate_keyboard_mouse_baseline() {
    let regs = MockXhciDevice::new();
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let mut input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(input.source_count(), 2);
    assert_eq!(input.discovery_errors(), 0);
    regs.inject_report(1, &[0, 0, 4, 0, 0, 0, 0, 0]);
    regs.inject_report(2, &[1, 5, 0xfd, 0]);
    let mut keyboard = 0;
    let mut mouse = 0;
    let result = input.drain(&mut ctl, &regs, || None, spin, &mut |report| {
        if report == InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]) {
            keyboard += 1;
        }
        if report == InputReport::mouse(2, 1, 5, -3, 0) {
            mouse += 1;
        }
        true
    });
    assert_eq!((keyboard, mouse, result.errors), (1, 1, 0));
}

#[test]
fn usb_mouse_optional_set_idle_stall_does_not_reject_mouse() {
    let regs = MockXhciDevice::new();
    regs.fail_hid_request(2, 0x0a, 0, 6);
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let mut input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(
        input.source_count(),
        2,
        "optional mouse STALL cannot remove its source"
    );
    assert_eq!(input.discovery_errors(), 0);
    assert_eq!(regs.ep0_recoveries(), 1, "EP0 really left its halted TD");
    regs.inject_report(1, &[0, 0, 4, 0, 0, 0, 0, 0]);
    regs.inject_report(2, &[0, 5, 0xfd]);
    let mut keyboard = 0;
    let mut mouse = 0;
    let result = input.drain(&mut ctl, &regs, || None, spin, &mut |report| {
        if report.source == 1 {
            keyboard += 1;
        }
        if report == InputReport::mouse(2, 0, 5, -3, 0) {
            mouse += 1;
        }
        true
    });
    assert_eq!((keyboard, mouse, result.errors), (1, 1, 0));
}

#[test]
fn usb_composite_discovers_keyboard_and_mouse() {
    let regs = MockXhciDevice::with_hid_config(&KEYBOARD_MOUSE);
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(
        input.source_count(),
        2,
        "discovery must not stop at the keyboard"
    );
    assert_eq!(
        regs.last_slot(),
        1,
        "one affine device/slot, not one per interface"
    );
    assert_eq!(input.discovery_errors(), 0);
    let requests = regs.class_requests();
    assert!(requests.contains(&(0x0b, 0, 7)));
    assert!(requests.contains(&(0x0b, 0, 42)));
}

#[test]
fn usb_optional_stall_recovers_ep0_before_later_keyboard_request() {
    let regs = MockXhciDevice::with_hid_config(&MOUSE_KEYBOARD);
    regs.fail_hid_request(1, 0x0a, 7, 6);
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(input.source_count(), 2);
    assert_eq!(input.discovery_errors(), 0);
    assert_eq!(regs.ep0_recoveries(), 1);
    assert!(regs.class_requests().contains(&(0x0a, 0, 42)));
}

#[test]
fn usb_optional_request_does_not_hide_transport_failure() {
    let regs = MockXhciDevice::new();
    regs.fail_hid_request(2, 0x0a, 0, 4);
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(input.source_count(), 1);
    assert_eq!(input.discovery_census().err_completion, 1);
    assert_eq!(input.discovery_census().last_completion_code, 4);
    assert_eq!(regs.ep0_recoveries(), 0);
}

#[test]
fn usb_unsupported_sibling_cannot_remove_valid_keyboard() {
    let regs = MockXhciDevice::with_hid_config(&MOUSE_KEYBOARD);
    regs.fail_hid_request(1, 0x0b, 7, 6); // boot protocol is required
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let mut input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(input.source_count(), 1);
    assert_eq!(regs.ep0_recoveries(), 1);
    regs.inject_report_at(1, 5, &[0, 0, 4, 0, 0, 0, 0, 0]);
    let result = input.drain(&mut ctl, &regs, || None, spin, &mut |report| {
        assert_eq!(report, InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]));
        true
    });
    assert_eq!((result.delivered, result.errors), (1, 0));
}

#[test]
fn usb_keyboard_set_idle_is_not_treated_as_optional() {
    let regs = MockXhciDevice::with_hid_config(&KEYBOARD_MOUSE);
    regs.fail_hid_request(1, 0x0a, 7, 6);
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let mut input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(input.source_count(), 1);
    regs.inject_report_at(1, 5, &[0, 5, 0xfd, 0]);
    let result = input.drain(&mut ctl, &regs, || None, spin, &mut |report| {
        assert_eq!(report, InputReport::mouse(1, 0, 5, -3, 0));
        true
    });
    assert_eq!((result.delivered, result.errors), (1, 0));
}

#[test]
fn usb_full_speed_mps_is_evaluated_before_long_descriptors() {
    for mps in [8, 16, 32, 64] {
        for csz64 in [false, true] {
            let regs = if csz64 {
                MockXhciDevice::new_csz64()
            } else {
                MockXhciDevice::new()
            };
            regs.set_hid_mps0(mps);
            let pool = MockPool;
            let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
            let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
            assert_eq!(input.source_count(), 2, "MPS{mps}, CSZ64={csz64}");
            assert_eq!(input.discovery_errors(), 0);
            assert_eq!(regs.ep0_evaluations(), if mps == 8 { 0 } else { 2 });
        }
    }
}

#[test]
fn usb_short_device_prefix_cannot_program_mps_from_stale_bytes() {
    for length in [0, 1, 7] {
        let regs = MockXhciDevice::new();
        regs.set_device_prefix_len(length);
        let pool = MockPool;
        let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
        let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
        assert_eq!(input.source_count(), 0);
        assert_eq!(input.discovery_census().err_unsupported, 2);
        assert_eq!(regs.ep0_evaluations(), 0);
        assert!(regs.class_requests().is_empty());
    }
}

#[test]
fn usb_reserved_mps_and_failed_evaluation_refuse_before_class_requests() {
    for mps in [0, 9, 12, 255] {
        let regs = MockXhciDevice::new();
        regs.set_hid_mps0(mps);
        let pool = MockPool;
        let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
        let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
        assert_eq!(input.source_count(), 0);
        assert_eq!(input.discovery_census().err_unsupported, 2);
        assert_eq!(regs.ep0_evaluations(), 0);
        assert!(regs.class_requests().is_empty());
    }
    let regs = MockXhciDevice::new();
    regs.set_hid_mps0(64);
    regs.fail_command(13, 17); // Evaluate Context / Parameter Error
    let pool = MockPool;
    let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
    assert_eq!(input.source_count(), 0);
    assert_eq!(input.discovery_census().last_completion_code, 17);
    assert!(regs.class_requests().is_empty());
}

#[test]
fn usb_failed_ep0_recovery_is_not_optional() {
    for command in [14, 16] {
        // Reset Endpoint and Set TR Dequeue Pointer
        let regs = MockXhciDevice::with_hid_config(&MOUSE_KEYBOARD);
        regs.fail_hid_request(1, 0x0a, 7, 6);
        regs.fail_command(command, 17);
        let pool = MockPool;
        let mut ctl = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
        let input = InputDriver::discover(&mut ctl, &regs, &pool, || None, spin);
        assert_eq!(input.source_count(), 0);
        assert_eq!(input.discovery_census().last_completion_code, 17);
        assert!(!regs.class_requests().contains(&(0x0b, 0, 42)));
    }
}

#[test]
fn usb_control_data_short_then_status_success_reports_real_length() {
    let regs = MockXhciDevice::new();
    let mut controller = Controller::bring_up(&regs, &MockPool, || None, spin).unwrap();
    let mut input = InputDriver::discover(&mut controller, &regs, &MockPool, || None, spin);
    regs.inject_report(1, &[0, 0, 4, 0, 0, 0, 0, 0]);
    regs.set_device_prefix_len(7);
    let received = controller
        .get_descriptor(&regs, &mut input.sources[1].device, 1, 0, 8, || None, spin)
        .unwrap();
    assert_eq!(
        received, 7,
        "Data residual cannot be inferred from the separate Status TD"
    );
    let out = input.drain(&mut controller, &regs, || None, spin, &mut |report| {
        assert_eq!(report, InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]));
        true
    });
    assert_eq!(
        (out.events, out.delivered, out.errors),
        (1, 1, 0),
        "short and status consumed, unrelated keyboard retained"
    );
    let received = controller
        .get_descriptor(&regs, &mut input.sources[1].device, 2, 0, 9, || None, spin)
        .unwrap();
    assert_eq!(received, 9, "old status must not complete the next request");
}

#[test]
fn usb_control_no_data_and_data_stalls_are_correlated_and_consumed() {
    for length in [0, 8] {
        let regs = MockXhciDevice::new();
        let mut controller = Controller::bring_up(&regs, &MockPool, || None, spin).unwrap();
        let mut input = InputDriver::discover(&mut controller, &regs, &MockPool, || None, spin);
        controller
            .set_protocol(&regs, &mut input.sources[0].device, true, 0, || None, spin)
            .unwrap();
        regs.fail_hid_request(1, 1, 0, 6);
        let request = crate::device::Setup {
            request_type: 0xa1,
            request: 1,
            value: 0,
            index: 0,
        };
        assert_eq!(
            controller.control(
                &regs,
                &mut input.sources[0].device,
                request,
                length,
                || None,
                spin
            ),
            Err(Error::Completion(6))
        );
        let mut events = [InterruptEvent::default(); 4];
        assert_eq!(
            controller.drain_interrupts(&regs, &mut events),
            0,
            "control error consumed even when its pointer names Data instead of Status"
        );
    }
}
