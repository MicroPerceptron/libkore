use super::usb_admission_tests::KEYBOARD_MOUSE;
use super::*;
use crate::mock::{MockPool, MockXhciDevice};
use kore_abi::pointer::Record;

fn spin() {}

fn ready() -> (
    MockXhciDevice,
    Controller<MockXhciDevice, MockPool>,
    InputDriver<MockPool>,
) {
    let regs = MockXhciDevice::with_hid_config(&KEYBOARD_MOUSE);
    let pool = MockPool;
    let mut controller = Controller::bring_up(&regs, &pool, || None, spin).unwrap();
    let input = InputDriver::discover(&mut controller, &regs, &pool, || None, spin);
    (regs, controller, input)
}

#[test]
fn tagged_composite_interfaces_have_independent_buffers_and_rearms() {
    let (regs, mut controller, mut input) = ready();
    assert_eq!(
        input.sources.len(),
        1,
        "Device ownership cannot be duplicated"
    );
    assert!(
        input.sources[0].device.int_ep().is_none(),
        "legacy helper must refuse ambiguity"
    );
    let mut identities = input.sources();
    assert_eq!(
        identities.next(),
        Some(HidSource {
            port: 1,
            interface: 7
        })
    );
    assert_eq!(
        identities.next(),
        Some(HidSource {
            port: 1,
            interface: 42
        })
    );
    assert_eq!(identities.next(), None);
    drop(identities);

    for dx in [5, 0, -12] {
        regs.inject_report_at(1, 3, &[0, 0, 4, 0, 0, 0, 0, 0]);
        regs.inject_report_at(1, 5, &[1, dx as u8, 0xfd, 1]);
        let mut keyboard = 0;
        let mut mouse = 0;
        let out = input.drain_with_sources(
            &mut controller,
            &regs,
            || None,
            spin,
            &mut |identity, report| {
                assert_eq!(identity.port, 1);
                assert_eq!(report.source, 0, "router supplies the wire namespace");
                match identity.interface {
                    7 => {
                        assert_eq!(report, InputReport::keyboard(0, 0, [4, 0, 0, 0, 0, 0]));
                        keyboard += 1;
                    }
                    42 => {
                        assert_eq!(report, InputReport::mouse(0, 1, dx, -3, 1));
                        mouse += 1;
                    }
                    _ => panic!("interface number must not be an ordinal or DCI"),
                }
                true
            },
        );
        assert_eq!((keyboard, mouse, out.errors), (1, 1, 0));
    }
}

#[test]
fn untagged_drain_cannot_merge_a_composite_under_one_port() {
    let (regs, mut controller, mut input) = ready();
    regs.inject_report_at(1, 3, &[0, 0, 4, 0, 0, 0, 0, 0]);
    regs.inject_report_at(1, 5, &[0, 5, 0, 0]);
    let out = input.drain(&mut controller, &regs, || None, spin, &mut |_| {
        panic!("legacy port-only source namespace is ambiguous")
    });
    assert_eq!((out.events, out.delivered, out.errors), (2, 0, 2));
}

#[test]
fn failed_mouse_resets_only_its_tag_and_keeps_the_keyboard_running() {
    let (regs, mut controller, mut input) = ready();
    regs.fail_interrupt_at(1, 5, 6);
    regs.inject_report_at(1, 3, &[0, 0, 4, 0, 0, 0, 0, 0]);
    let mut reset = 0;
    let mut key = 0;
    let out = input.drain_with_sources(
        &mut controller,
        &regs,
        || None,
        spin,
        &mut |identity, report| {
            match identity.interface {
                42 => {
                    assert_eq!(Record::decode(report), Ok(Record::Reset { all: false }));
                    reset += 1;
                }
                7 => {
                    assert_eq!(report, InputReport::keyboard(0, 0, [4, 0, 0, 0, 0, 0]));
                    key += 1;
                }
                _ => panic!("unknown interface"),
            }
            true
        },
    );
    assert_eq!((out.errors, out.halts_recovered, reset, key), (1, 1, 1, 1));
    regs.inject_report_at(1, 5, &[0, 9, 0, 0]);
    let out = input.drain_with_sources(
        &mut controller,
        &regs,
        || None,
        spin,
        &mut |identity, report| {
            assert_eq!(identity.interface, 42);
            assert_eq!(report, InputReport::mouse(0, 0, 9, 0, 0));
            true
        },
    );
    assert_eq!((out.delivered, out.errors), (1, 0));
}

#[test]
fn failed_endpoint_recovery_does_not_rearm_or_remove_its_sibling() {
    let (regs, mut controller, mut input) = ready();
    regs.fail_interrupt_at(1, 5, 6);
    regs.fail_command(14, 17);
    let mut resets = 0;
    let out = input.drain_with_sources(
        &mut controller,
        &regs,
        || None,
        spin,
        &mut |identity, report| {
            assert_eq!(identity.interface, 42);
            assert_eq!(Record::decode(report), Ok(Record::Reset { all: false }));
            resets += 1;
            true
        },
    );
    assert_eq!((out.halts_recovered, resets), (0, 1));
    assert_eq!(input.source_count(), 1);
    assert_eq!(
        input.sources().next(),
        Some(HidSource {
            port: 1,
            interface: 7
        })
    );
    regs.inject_report_at(1, 3, &[0; 8]);
    let out =
        input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |identity, _| {
            assert_eq!(identity.interface, 7);
            true
        });
    assert_eq!((out.delivered, out.errors), (1, 0));
    regs.set_port_connected(1, false);
    let mut retired = 0;
    input.note_source_changes_with(&controller, &regs, &mut |_| retired += 1);
    assert_eq!(
        retired, 2,
        "even a failed interface must retire its router identity"
    );
}

#[test]
fn reset_backpressure_cannot_publish_fresh_state_before_the_reset_lands() {
    let (regs, mut controller, mut input) = ready();
    regs.fail_interrupt_at(1, 5, 6);
    input.drain_with_sources(
        &mut controller,
        &regs,
        || None,
        spin,
        &mut |identity, report| {
            assert_eq!(identity.interface, 42);
            assert_eq!(Record::decode(report), Ok(Record::Reset { all: false }));
            false
        },
    );
    regs.inject_report_at(1, 5, &[1, 3, 0, 0]);
    let mut attempts = 0;
    let out = input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, report| {
        assert_eq!(Record::decode(report), Ok(Record::Reset { all: false }));
        attempts += 1;
        false
    });
    assert_eq!(out.delivered, 0);
    assert_eq!(attempts, 1, "no mouse state can bypass a pending reset");
    regs.inject_report_at(1, 5, &[0, 9, 0, 0]);
    let mut step = 0;
    let out = input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, report| {
        if step == 0 {
            assert_eq!(Record::decode(report), Ok(Record::Reset { all: false }));
        } else {
            assert_eq!(report, InputReport::mouse(0, 0, 9, 0, 0));
        }
        step += 1;
        true
    });
    assert_eq!((out.delivered, out.errors, step), (1, 0, 2));
}

#[test]
fn old_trb_completion_cannot_consume_the_current_interface_arm() {
    let (regs, mut controller, mut input) = ready();
    let old = input.sources[0].interfaces[1].unwrap().armed.unwrap();
    regs.inject_report_at(1, 5, &[0, 1, 0, 0]);
    input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| true);
    let current = input.sources[0].interfaces[1].unwrap().armed.unwrap();
    assert_ne!(old, current);
    for code in [1, 6] {
        regs.inject_completion_at(1, 5, old, code, 0);
        let out = input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| {
            panic!("stale completion must not publish data or reset")
        });
        assert_eq!((out.events, out.delivered, out.errors), (1, 0, 1));
        assert_eq!(input.sources[0].interfaces[1].unwrap().armed, Some(current));
    }
    regs.inject_report_at(1, 5, &[0, 4, 0, 0]);
    let out = input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, report| {
        assert_eq!(report, InputReport::mouse(0, 0, 4, 0, 0));
        true
    });
    assert_eq!(out.delivered, 1);
}

#[test]
fn fast_replug_retires_all_interface_tags_before_reusing_the_port() {
    let (regs, mut controller, mut input) = ready();
    regs.inject_report_at(1, 5, &[1, 5, 0, 0]); // still queued at unplug
    regs.set_port_connected(1, false);
    regs.set_port_connected(1, true); // one observed edge, CCS stays true
    let mut retired = [None; 2];
    let mut count = 0;
    assert!(
        input.note_source_changes_with(&controller, &regs, &mut |source| {
            retired[count] = Some(source);
            count += 1;
        })
    );
    assert_eq!(
        retired,
        [
            Some(HidSource {
                port: 1,
                interface: 7
            }),
            Some(HidSource {
                port: 1,
                interface: 42
            })
        ]
    );
    assert_eq!(input.source_count(), 0);
    assert!(
        !input.note_source_changes_with(&controller, &regs, &mut |_| panic!("duplicate retire"))
    );
    input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| {
        panic!("departed device report")
    });
    input.discover_after_interrupt(&mut controller, &regs, &MockPool, || None, spin);
    assert_eq!(input.sources.len(), 1);
    assert_eq!(input.source_count(), 2);
    assert_eq!(regs.last_slot(), 2);
    regs.inject_report_at(2, 5, &[0, 2, 0, 0]);
    assert_eq!(
        input
            .drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| true)
            .delivered,
        1
    );
}

#[test]
fn failed_disable_retains_affine_dma_and_blocks_same_port_readdress() {
    let (regs, mut controller, mut input) = ready();
    regs.set_port_connected(1, true);
    input.note_source_changes_with(&controller, &regs, &mut |_| {});
    regs.fail_command(10, 17);
    input.discover_after_interrupt(&mut controller, &regs, &MockPool, || None, spin);
    assert_eq!(input.sources.len(), 1, "DMA retained until hardware stops");
    assert_eq!(input.source_count(), 0);
    assert_eq!(
        regs.last_slot(),
        1,
        "cannot duplicate the slot on an uncertain port"
    );
    regs.clear_command_failure();
    input.discover_after_interrupt(&mut controller, &regs, &MockPool, || None, spin);
    assert_eq!(input.sources.len(), 1);
    assert_eq!(input.source_count(), 2);
    assert_eq!(regs.last_slot(), 2);
}

#[test]
fn removing_an_earlier_source_does_not_skip_arming_a_new_one() {
    let regs = MockXhciDevice::new();
    let mut controller = Controller::bring_up(&regs, &MockPool, || None, spin).unwrap();
    let mut input = InputDriver::discover(&mut controller, &regs, &MockPool, || None, spin);
    regs.set_port_connected(1, false);
    regs.set_port_connected(2, true);
    input.note_source_changes_with(&controller, &regs, &mut |_| {});
    input.discover_after_interrupt(&mut controller, &regs, &MockPool, || None, spin);
    assert_eq!(input.source_count(), 1);
    regs.inject_report_at(3, 3, &[0, 5, 0, 0]);
    assert_eq!(
        input
            .drain_with_sources(&mut controller, &regs, || None, spin, &mut |identity, _| {
                assert_eq!(
                    identity,
                    HidSource {
                        port: 2,
                        interface: 0
                    }
                );
                true
            })
            .delivered,
        1
    );
}

#[test]
fn admission_checks_inactive_alternates_duplicate_identity_and_truncation() {
    let mut alternate = KEYBOARD_MOUSE;
    alternate[37] = 1; // mouse alternate setting is not enabled
    let selected = hid_candidates(&alternate, Speed::Full).unwrap();
    assert_eq!(selected.iter().flatten().count(), 1);
    let mut duplicate_interface = KEYBOARD_MOUSE;
    duplicate_interface[36] = 7;
    assert!(hid_candidates(&duplicate_interface, Speed::Full).is_err());
    let mut duplicate_endpoint = KEYBOARD_MOUSE;
    duplicate_endpoint[54] = 0x81;
    assert!(hid_candidates(&duplicate_endpoint, Speed::Full).is_err());
    assert!(hid_candidates(&KEYBOARD_MOUSE[..58], Speed::Full).is_err());
    let mut short_interface = KEYBOARD_MOUSE;
    short_interface[34] = 8;
    assert!(hid_candidates(&short_interface, Speed::Full).is_err());
    for bad_mps in [0, 2, 65, 0xff] {
        let mut invalid = KEYBOARD_MOUSE;
        invalid[56] = bad_mps;
        assert!(hid_candidates(&invalid, Speed::Full).is_err());
    }
}

#[test]
fn interface_capacity_is_a_refusal_not_aliasing_or_partial_selection() {
    let mut configuration = [0u8; 9 + 5 * 25];
    configuration[..9].copy_from_slice(&KEYBOARD_MOUSE[..9]);
    configuration[2] = configuration.len() as u8;
    configuration[4] = 5;
    for i in 0..5 {
        let start = 9 + i * 25;
        configuration[start..start + 25].copy_from_slice(&KEYBOARD_MOUSE[9..34]);
        configuration[start + 2] = i as u8;
        configuration[start + 20] = 0x81 + i as u8;
    }
    assert!(hid_candidates(&configuration, Speed::Full).is_err());
}

#[test]
fn endpoint_context_cannot_overlap_the_packed_output_context() {
    let regs = MockXhciDevice::new_csz64();
    let mut controller = Controller::bring_up(&regs, &MockPool, || None, spin).unwrap();
    controller.reset_port(&regs, 1, || None, spin).unwrap();
    let port = controller.port(&regs, 1);
    let mut device = controller
        .address_device(&regs, &MockPool, &port, || None, spin)
        .unwrap();
    let endpoint = EndpointDescriptor {
        address: 0x8f,
        attributes: 3,
        max_packet_size: 8,
        interval: 10,
    };
    assert_eq!(
        controller.configure_endpoint(&regs, &MockPool, &mut device, &endpoint, || None, spin),
        Err(Error::Unsupported)
    );
    assert!(device.endpoint(31).is_none());
}

#[test]
fn intermediate_control_progress_cannot_refund_the_wait_budget() {
    use core::{cell::Cell, ops::ControlFlow};
    let regs = MockXhciDevice::new();
    let mut controller = Controller::bring_up(&regs, &MockPool, || None, spin).unwrap();
    let pauses = Cell::new(0u32);
    let progress = Cell::new(0u32);
    let result: Option<()> = controller.wait_event_consuming(
        &regs,
        || None,
        &mut || {
            let count = pauses.get() + 1;
            pauses.set(count);
            if count == 100 {
                regs.inject_completion_at(1, 1, 0x1000, 13, 1);
            }
        },
        |raw| {
            let Ok(xhci::ring::trb::event::Allowed::TransferEvent(event)) =
                xhci::ring::trb::event::Allowed::try_from(raw)
            else {
                return None;
            };
            if event.trb_pointer() != 0x1000 {
                return None;
            }
            progress.set(progress.get() + 1);
            Some(ControlFlow::Continue(())) // Status never arrives
        },
    );
    assert_eq!(result, None);
    assert_eq!(progress.get(), 1);
    assert_eq!(
        pauses.get(),
        crate::controller::MAX_POLLS,
        "the 100 pre-Data pauses count against the same deadline"
    );
    let mut remaining = [InterruptEvent::default(); 1];
    assert_eq!(
        controller.drain_interrupts(&regs, &mut remaining),
        0,
        "owned intermediate event must not enter the foreign-event stash"
    );
}
