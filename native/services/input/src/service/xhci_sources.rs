//! Router-owned USB identities, independent of root-port and interface values.
//!
//! A composite device can have several HID interfaces on one port; a second
//! controller can use exactly the same tuple. Neither tuple is a wire source
//! byte. This bounded table assigns identities and retires each one with a
//! source-local reset before it may be reused. No capability lives here.

use kore_abi::keyboard::Locks;
use kore_abi::{pointer::Record, svc::InputReport};
use kore_xhci::HidSource;

/// Diagnostic/source namespace budget, not a worker or CPU count.
pub const MAX_CONTROLLERS: usize = 4;
pub const SOURCES_PER_CONTROLLER: usize = 31;
/// The production service currently owns one xHC. Taking the rest requires a
/// USB child-device ownership/handoff protocol: the network service also owns
/// xHCs, and the broker cannot safely reclaim an index-only returned claim.
pub const MANAGED_CONTROLLER_CAPACITY: usize = 1;
/// A diagnostic slot that this service does not manage, not absent hardware.
pub const STATE_NOT_MANAGED: u64 = super::xhci_bind::pack(6, 1);
pub const UNKNOWN_IDENTITY: u32 = u32::MAX;

/// `kore_abi::pointer` class 0: a physical mouse, whose motion the
/// compositor's pointer policy may accelerate.
pub const MOUSE_CLASS: u8 = 0;
/// `kore_abi::pointer` class 4: a virtual source. A hypervisor's emulated
/// mouse is one: it carries its host's pointer, which the host has already
/// accelerated, so accelerating it again would be double acceleration.
pub const VIRTUAL_CLASS: u8 = 4;

/// USB HID mice that are a hypervisor's emulation, by the `idVendor` and
/// `idProduct` their device descriptor declares. QEMU's `usb-mouse` and
/// `usb-tablet` declare 0627:0001 (`hw/usb/dev-hid.c`, unchanged from
/// v8.2.2, which the gates run, through master 11.1.50). An unlisted
/// emulation is read as a physical mouse.
const EMULATED_MICE: [kore_xhci::UsbIdentity; 1] = [kore_xhci::UsbIdentity {
    vendor: 0x0627,
    product: 0x0001,
}];

/// The pointer class an admitted USB HID mouse's frames carry. A device
/// that declared no identity is physical: a mouse is accelerated unless it
/// is known not to need it.
pub fn mouse_class(identity: Option<kore_xhci::UsbIdentity>) -> u8 {
    match identity {
        Some(identity) if EMULATED_MICE.contains(&identity) => VIRTUAL_CLASS,
        _ => MOUSE_CLASS,
    }
}

/// Stable read-only claim provenance. The lower 24 bits are the broker's
/// inventory index, bits 24..30 its root ordinal, bit 31 the platform species.
/// Values outside this representation remain explicitly unknown, never aliased.
pub fn controller_identity(root: usize, index: u32, platform: bool) -> u32 {
    if root >= kore_abi::device::MAX_ROOT_COMPLEXES || root >= 128 || index >= (1 << 24) {
        return UNKNOWN_IDENTITY;
    }
    index | ((root as u32) << 24) | (u32::from(platform) << 31)
}

/// Read-only controller testimony. `reports` counts tagged driver records;
/// `dropped` counts records the router could not publish, including generated
/// resets/releases. Neither IRQ delivery nor report decoding proves that the
/// compositor has consumed a report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControllerBooks {
    pub state: u64,
    pub identity: u32,
    pub sources: u32,
    pub discovery_errors: u32,
    pub irq_wakes: u64,
    pub irq_delivered: u64,
    pub irq_events: u64,
    pub irq_errors: u64,
    pub reports: u64,
    pub dropped: u64,
    /// Failed namespace assignments, not a count of distinct devices.
    pub source_exhausted: u32,
}

#[derive(Clone, Copy)]
struct Entry {
    source: HidSource,
    /// The interface is a boot keyboard — the only class with indicators.
    /// Learned at [`SourceMap::admit`], from the driver's own discovery.
    class_keyboard: bool,
    /// The keyboard lane has accepted held keys/modifiers for this source.
    keyboard: bool,
    reset: Option<ResetPending>,
    retiring: bool,
}

#[derive(Clone, Copy)]
struct ResetPending {
    pointer: bool,
    keyboard: bool,
}

/// Fixed namespace excluding zero, the PS/2 source (64), and I2C (128..255).
/// Every controller's band remains disjoint even when ports and interface
/// numbers coincide. The controller index is an owned router slot, not a
/// device-provided number.
pub fn source_id(controller: usize, slot: usize) -> Option<u8> {
    if controller >= MAX_CONTROLLERS || slot >= SOURCES_PER_CONTROLLER {
        return None;
    }
    let ordinal = controller * SOURCES_PER_CONTROLLER + slot + 1;
    Some((ordinal + usize::from(ordinal >= usize::from(super::SOURCE_LEGACY))) as u8)
}

pub struct SourceMap {
    controller: usize,
    entries: [Option<Entry>; SOURCES_PER_CONTROLLER],
    /// The lock state the keymap owner last asked each wire identity's
    /// keyboard to show. Keyed by the identity, not the entry, and kept
    /// across retirement on purpose: the identity is what the keymap owner
    /// assigned to a seat, so a keyboard replugged under the same identity
    /// is a keyboard of the same seat and is owed the same state — which
    /// the owner, having already told this identity, will not send again.
    indicators: [Option<Locks>; SOURCES_PER_CONTROLLER],
    reports: u64,
    dropped: u64,
    exhausted: u32,
}

impl SourceMap {
    pub const fn new(controller: usize) -> Self {
        assert!(controller < MAX_CONTROLLERS);
        Self {
            controller,
            entries: [None; SOURCES_PER_CONTROLLER],
            indicators: [None; SOURCES_PER_CONTROLLER],
            reports: 0,
            dropped: 0,
            exhausted: 0,
        }
    }

    fn slot(&self, source: HidSource) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.is_some_and(|entry| entry.source == source))
    }

    /// Register a live interface once. Exhaustion refuses without replacing a
    /// live entry; raw port/interface numbers never become fallback wire IDs.
    pub fn register(&mut self, source: HidSource) -> Option<u8> {
        let slot = match self.slot(source) {
            Some(slot) if self.entries[slot].is_some_and(|entry| !entry.retiring) => slot,
            // A replug with the same tuple is not the old device. Until its
            // cancellation is accepted, neither this tuple nor another one
            // may acquire the tombstoned wire identity.
            Some(_) => return None,
            None => match self.entries.iter().position(Option::is_none) {
                Some(slot) => {
                    self.entries[slot] = Some(Entry {
                        source,
                        class_keyboard: false,
                        keyboard: false,
                        reset: None,
                        retiring: false,
                    });
                    slot
                }
                None => {
                    self.exhausted = self.exhausted.saturating_add(1);
                    return None;
                }
            },
        };
        source_id(self.controller, slot)
    }

    /// Overwrite the source byte for *every* record, including error resets.
    /// A driver reset is confined to this interface. A keyboard's reset also
    /// releases its held keys on the keyboard lane, without affecting siblings.
    pub fn route(
        &mut self,
        source: HidSource,
        mut report: InputReport,
        sink: &mut impl FnMut(InputReport) -> bool,
    ) -> bool {
        self.reports = self.reports.saturating_add(1);
        let Some(id) = self.register(source) else {
            self.dropped = self.dropped.saturating_add(1);
            return false;
        };
        let slot = self.slot(source).expect("registered source");
        let reset = matches!(Record::decode(report), Ok(Record::Reset { .. }));
        if reset {
            self.begin_reset(slot);
            // A successful pointer reset alone is insufficient: the driver
            // retains its retry obligation until the keyboard release lands
            // too. Previously accepted lanes are not replayed on each retry.
            return self.finish_reset(slot, sink);
        }
        if self.entries[slot].is_some_and(|entry| entry.reset.is_some()) {
            self.dropped = self.dropped.saturating_add(1);
            return false;
        }
        report.source = id;
        let landed = sink(report);
        if !landed {
            self.dropped = self.dropped.saturating_add(1);
        } else if report.kind == InputReport::KIND_KEYBOARD {
            let entry = self.entries[slot].as_mut().expect("registered source");
            entry.keyboard = report.state != 0 || report.data.iter().any(|&key| key != 0);
            // An interface delivering keyboard snapshots is a keyboard,
            // whether or not a discovery pass has admitted it yet.
            entry.class_keyboard = true;
        }
        landed
    }

    /// Called by the driver's per-interface detach callback *before* port
    /// identity reuse. A full sink leaves a tombstone, not a reusable ID.
    /// [`retry_retired`](Self::retry_retired) attempts the remaining lanes on
    /// subsequent real service wakes without blocking healthy sources.
    pub fn retire(&mut self, source: HidSource, sink: &mut impl FnMut(InputReport) -> bool) {
        let Some(slot) = self.slot(source) else {
            return;
        };
        self.entries[slot].as_mut().expect("known source").retiring = true;
        self.begin_reset(slot);
        if self.finish_reset(slot, sink) {
            self.entries[slot] = None;
        }
    }

    /// Retry each retiring source at most once per actual root IRQ/control
    /// wake. This is not a pending-work bit: a full or absent consumer must
    /// never make the service's executor spin or prevent unrelated input.
    pub fn retry_retired(&mut self, sink: &mut impl FnMut(InputReport) -> bool) {
        for slot in 0..self.entries.len() {
            if self.entries[slot].is_some_and(|entry| entry.retiring)
                && self.finish_reset(slot, sink)
            {
                self.entries[slot] = None;
            }
        }
    }

    fn begin_reset(&mut self, slot: usize) {
        let entry = self.entries[slot].as_mut().expect("known source");
        if entry.reset.is_none() {
            entry.reset = Some(ResetPending {
                pointer: true,
                keyboard: entry.keyboard,
            });
        }
    }

    fn finish_reset(&mut self, slot: usize, sink: &mut impl FnMut(InputReport) -> bool) -> bool {
        let entry = self.entries[slot].as_mut().expect("known source");
        let Some(mut pending) = entry.reset else {
            return true;
        };
        let id = source_id(self.controller, slot).expect("bounded source slot");
        if pending.pointer {
            pending.pointer = !sink(Record::Reset { all: false }.encode(id));
            if pending.pointer {
                self.dropped = self.dropped.saturating_add(1);
            }
        }
        if pending.keyboard {
            pending.keyboard = !sink(InputReport::keyboard(id, 0, [0; 6]));
            if pending.keyboard {
                self.dropped = self.dropped.saturating_add(1);
            } else {
                entry.keyboard = false;
            }
        }
        let complete = !pending.pointer && !pending.keyboard;
        entry.reset = (!complete).then_some(pending);
        complete
    }

    /// Register an armed interface with its boot class: its wire identity,
    /// and whether this call created it.
    ///
    /// A new keyboard is what a caller announces with an all-released
    /// snapshot — nothing a keyboard at rest would not say, and how the
    /// keymap owner learns the keyboard exists before its first keystroke,
    /// so one plugged in while Caps Lock is on lights at once.
    pub fn admit(&mut self, source: HidSource, keyboard: bool) -> Option<(u8, bool)> {
        let fresh = self.slot(source).is_none();
        let id = self.register(source)?;
        let slot = self.slot(source).expect("registered source");
        self.entries[slot]
            .as_mut()
            .expect("registered source")
            .class_keyboard = keyboard;
        Some((id, fresh))
    }

    /// Record the lock state `id`'s keyboard should show. `false` when the
    /// identity is not in this table's band.
    ///
    /// Recorded whether or not a keyboard holds the identity right now, and
    /// that is the point of keying the record by identity: the keymap owner
    /// keeps telling every identity its seat has seen, so a keyboard
    /// unplugged meanwhile and replugged under the same identity is lit with
    /// the seat's *current* state, not the one it showed when it left.
    pub fn set_indicators(&mut self, id: u8, locks: Locks) -> bool {
        let Some(slot) =
            (0..SOURCES_PER_CONTROLLER).find(|&slot| source_id(self.controller, slot) == Some(id))
        else {
            return false;
        };
        self.indicators[slot] = Some(locks);
        true
    }

    /// Every live interface with a lock state to show.
    pub fn indicators(&self) -> impl Iterator<Item = (HidSource, Locks)> + '_ {
        self.entries
            .iter()
            .zip(&self.indicators)
            .filter_map(|(entry, locks)| match (entry, locks) {
                (Some(entry), Some(locks)) if !entry.retiring && entry.class_keyboard => {
                    Some((entry.source, *locks))
                }
                _ => None,
            })
    }

    pub fn reports(&self) -> u64 {
        self.reports
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn exhausted(&self) -> u32 {
        self.exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_is_disjoint_across_interfaces_controllers_ps2_and_i2c() {
        let mut used = [false; 256];
        for controller in 0..MAX_CONTROLLERS {
            let mut map = SourceMap::new(controller);
            for interface in 0..SOURCES_PER_CONTROLLER {
                let local = HidSource {
                    port: 255,
                    interface: interface as u8,
                };
                let id = map.register(local).unwrap();
                assert!(id != 0 && id != super::super::SOURCE_LEGACY && id < 128);
                assert!(
                    !used[id as usize],
                    "a controller/interface identity collided"
                );
                used[id as usize] = true;
                assert_eq!(map.register(local), Some(id));
            }
        }
        assert_eq!(used.into_iter().filter(|used| *used).count(), 124);
        assert_eq!(source_id(MAX_CONTROLLERS, 0), None);
        assert_eq!(source_id(0, SOURCES_PER_CONTROLLER), None);
    }

    #[test]
    fn full_namespace_refuses_without_replacing_an_existing_interface() {
        let mut map = SourceMap::new(0);
        for interface in 0..SOURCES_PER_CONTROLLER {
            assert!(
                map.register(HidSource {
                    port: 1,
                    interface: interface as u8
                })
                .is_some()
            );
        }
        let overflow = HidSource {
            port: 1,
            interface: 255,
        };
        assert!(
            !map.route(overflow, InputReport::mouse(64, 0, 1, 1, 0), &mut |_| {
                panic!("a refused source must not reach the router")
            })
        );
        assert_eq!((map.reports(), map.dropped(), map.exhausted()), (1, 1, 1));
        for interface in 0..SOURCES_PER_CONTROLLER {
            assert_eq!(
                map.register(HidSource {
                    port: 1,
                    interface: interface as u8
                }),
                source_id(0, interface),
            );
        }
    }

    #[test]
    fn retirement_resets_and_releases_only_that_interface_before_reuse() {
        let mut map = SourceMap::new(2);
        let keyboard = HidSource {
            port: 1,
            interface: 3,
        };
        let mouse = HidSource {
            port: 1,
            interface: 9,
        };
        let keyboard_id = map.register(keyboard).unwrap();
        let mouse_id = map.register(mouse).unwrap();
        assert!(map.route(
            keyboard,
            InputReport::keyboard(0, 2, [4, 0, 0, 0, 0, 0]),
            &mut |_| true
        ));
        let mut records = [None; 2];
        let mut n = 0;
        map.retire(keyboard, &mut |report| {
            records[n] = Some(report);
            n += 1;
            true
        });
        assert_eq!(n, 2);
        assert_eq!(
            records[0],
            Some(Record::Reset { all: false }.encode(keyboard_id))
        );
        assert_eq!(
            records[1],
            Some(InputReport::keyboard(keyboard_id, 0, [0; 6]))
        );
        assert_eq!(map.register(mouse), Some(mouse_id));
        assert_eq!(
            map.register(HidSource {
                port: 200,
                interface: 255
            }),
            Some(keyboard_id)
        );
    }

    #[test]
    fn pointer_reset_acceptance_cannot_hide_a_full_keyboard_release_lane() {
        let mut map = SourceMap::new(0);
        let source = HidSource {
            port: 1,
            interface: 0,
        };
        let id = map.register(source).unwrap();
        let pressed = InputReport::keyboard(0, 2, [4, 0, 0, 0, 0, 0]);
        assert!(map.route(source, pressed, &mut |_| true));
        let reset = Record::Reset { all: false }.encode(0);
        let mut pointer_resets = 0;
        let mut releases = 0;
        assert!(!map.route(source, reset, &mut |report| {
            if report.kind == InputReport::KIND_KEYBOARD {
                releases += 1;
                assert_eq!(report, InputReport::keyboard(id, 0, [0; 6]));
                false
            } else {
                pointer_resets += 1;
                assert_eq!(report, Record::Reset { all: false }.encode(id));
                true
            }
        }));
        assert_eq!((pointer_resets, releases), (1, 1));
        assert!(!map.route(source, pressed, &mut |_| {
            panic!("fresh state must wait for both reset lanes")
        }));
        assert!(map.route(source, reset, &mut |report| {
            assert_eq!(report, InputReport::keyboard(id, 0, [0; 6]));
            releases += 1;
            true
        }));
        assert_eq!((pointer_resets, releases), (1, 2));
        assert!(map.route(source, pressed, &mut |_| true));
    }

    #[test]
    fn detach_retains_tombstone_until_a_later_wake_accepts_the_reset() {
        let mut map = SourceMap::new(0);
        let detached = HidSource {
            port: 1,
            interface: 0,
        };
        let sibling = HidSource {
            port: 1,
            interface: 1,
        };
        let id = map.register(detached).unwrap();
        let sibling_id = map.register(sibling).unwrap();
        map.retire(detached, &mut |_| false);
        let replacement = HidSource {
            port: 2,
            interface: 0,
        };
        assert_ne!(
            map.register(replacement),
            Some(id),
            "a tombstone is not vacant"
        );
        assert!(
            map.route(sibling, InputReport::mouse(0, 0, 1, 0, 0), &mut |report| {
                assert_eq!(report.source, sibling_id);
                true
            }),
            "a blocked reset must not block healthy interfaces"
        );
        let mut attempts = 0;
        map.retry_retired(&mut |report| {
            attempts += 1;
            assert_eq!(report, Record::Reset { all: false }.encode(id));
            false
        });
        assert_eq!(
            attempts, 1,
            "one attempt per pending lane, not a retry loop"
        );
        map.retry_retired(&mut |report| {
            assert_eq!(report, Record::Reset { all: false }.encode(id));
            true
        });
        assert_eq!(
            map.register(HidSource {
                port: 3,
                interface: 0
            }),
            Some(id)
        );
        map.retry_retired(&mut |_| panic!("completed tombstone must be gone"));
    }

    #[test]
    fn same_tuple_replug_cannot_alias_a_pending_retirement() {
        let mut map = SourceMap::new(0);
        let tuple = HidSource {
            port: 255,
            interface: 17,
        };
        let id = map.register(tuple).unwrap();
        map.retire(tuple, &mut |_| false);
        assert_eq!(map.register(tuple), None);
        assert!(
            !map.route(tuple, InputReport::mouse(0, 1, 8, 0, 0), &mut |_| {
                panic!("the replacement must not inherit the old source")
            })
        );
        assert!(
            !map.route(tuple, Record::Reset { all: false }.encode(0), &mut |_| {
                panic!("a new driver's reset must not bypass the retiring owner")
            })
        );
        map.retry_retired(&mut |_| true);
        assert_eq!(map.register(tuple), Some(id));
        assert!(
            map.route(tuple, InputReport::mouse(0, 0, 8, 0, 0), &mut |report| {
                assert_eq!(report.source, id);
                true
            })
        );
    }

    #[test]
    fn tagged_error_reset_cannot_reset_other_devices_or_retain_raw_identity() {
        let mut map = SourceMap::new(0);
        let local = HidSource {
            port: 64,
            interface: 128,
        };
        let id = map.register(local).unwrap();
        assert!(map.route(
            local,
            Record::Reset { all: true }.encode(128),
            &mut |report| {
                assert_eq!(report, Record::Reset { all: false }.encode(id));
                true
            }
        ));
        assert!(
            !map.route(local, InputReport::mouse(64, 1, 2, -3, 0), &mut |report| {
                assert_eq!(report.source, id);
                false
            })
        );
        assert_eq!((map.reports(), map.dropped(), map.exhausted()), (2, 1, 0));
    }

    #[test]
    fn indicators_follow_the_identity_across_a_replug_and_reach_only_live_keyboards() {
        let mut map = SourceMap::new(1);
        let keyboard = HidSource {
            port: 2,
            interface: 0,
        };
        let mouse = HidSource {
            port: 3,
            interface: 0,
        };
        let (id, fresh) = map.admit(keyboard, true).unwrap();
        assert!(fresh, "a new keyboard is the caller's to announce");
        assert_eq!(map.admit(keyboard, true), Some((id, false)), "once");
        let (mouse_id, _) = map.admit(mouse, false).unwrap();

        assert!(!map.set_indicators(0x40, Locks::CAPS), "not this table's band");
        assert!(map.set_indicators(mouse_id, Locks::CAPS), "recorded for the identity");
        assert!(map.set_indicators(id, Locks::CAPS));
        assert_eq!(
            map.indicators().collect::<std::vec::Vec<_>>(),
            [(keyboard, Locks::CAPS)],
            "a mouse has no indicators to light"
        );

        // A source first met through its reports is classed when the
        // discovery pass that armed it admits it...
        let late = HidSource {
            port: 5,
            interface: 1,
        };
        let late_id = map.register(late).unwrap();
        assert!(map.set_indicators(late_id, Locks::NUM));
        assert!(!map.indicators().any(|(source, _)| source == late), "class not yet known");
        assert_eq!(map.admit(late, true), Some((late_id, false)));
        assert!(map.indicators().any(|entry| entry == (late, Locks::NUM)));
        // ...or by the first keyboard snapshot it delivers.
        let typed = HidSource {
            port: 6,
            interface: 0,
        };
        let typed_id = map.register(typed).unwrap();
        assert!(map.set_indicators(typed_id, Locks::NONE));
        assert!(map.route(typed, InputReport::keyboard(0, 0, [0; 6]), &mut |_| true));
        assert!(map.indicators().any(|entry| entry == (typed, Locks::NONE)));
        map.retire(typed, &mut |_| true);

        // Unplugged: nothing is lit through a dead interface, but the seat's
        // state keeps landing on the identity while it is gone...
        map.retire(keyboard, &mut |_| true);
        assert!(!map.indicators().any(|(source, _)| source == keyboard));
        assert!(map.set_indicators(id, Locks::CAPS.toggled(Locks::SCROLL)));

        // ...so the replug under the same identity shows the seat's current
        // state, not the one it left with.
        let replugged = HidSource {
            port: 4,
            interface: 0,
        };
        assert_eq!(map.admit(replugged, true), Some((id, true)));
        assert!(
            map.indicators()
                .any(|entry| entry == (replugged, Locks::CAPS.toggled(Locks::SCROLL)))
        );
    }

    #[test]
    fn claim_provenance_is_separate_from_source_and_bounded_without_aliasing() {
        assert_eq!(controller_identity(2, 17, false), 0x0200_0011);
        assert_eq!(controller_identity(2, 17, true), 0x8200_0011);
        assert_eq!(
            controller_identity(kore_abi::device::MAX_ROOT_COMPLEXES, 0, false),
            UNKNOWN_IDENTITY
        );
        assert_eq!(controller_identity(0, 1 << 24, false), UNKNOWN_IDENTITY);
        assert_eq!(MANAGED_CONTROLLER_CAPACITY, 1);
        assert_ne!(STATE_NOT_MANAGED, 0);
    }
}
