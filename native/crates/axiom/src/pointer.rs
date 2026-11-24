//! Kore's framed transport adapter. Recognition and cursor policy are Axiom
//! owned, replaceable, and invoked only after complete envelope validation.
mod frame;
#[cfg(test)]
mod tests;
pub mod touchpad;
pub use axinput::acceleration::{AccelerationConfig, AccelerationError, Profile};
pub use axinput::{
    ButtonState, DeviceId, InputTimestamp, MotionScale, PointerPosition, ScrollUnit, SeatEvent,
    SeatEventKind, Viewport,
};
use axinput::{PointerInputKind, PointerSource, Seat, SeatId, frame::*};
use kore_abi::{
    pointer::{self, Record},
    svc::InputReport,
};

/// A scoped raw frame and its derived outputs share one dispatch seam.
/// Applications do not receive privileged global input without routing policy.
pub enum Dispatch<'a> {
    Frame(PointerFrame<'a>),
    Pointer(SeatEvent),
    Gesture(GestureEvent<'a>),
    Cancelled {
        device: DeviceId,
        reason: CancelReason,
    },
}

/// Device policies held at once: one per pointer device in use. A device
/// beyond them takes the binding used least recently, whose device is
/// cancelled first (so nothing it held stays pressed) and starts a fresh
/// policy with its next frame.
pub const POLICY_SLOTS: usize = 8;

/// What the seat keeps for every source byte: the stream's continuity and
/// the buttons its policy holds. Small, so every source has one.
struct Source {
    history: FrameHistory,
    buttons: u64,
    legacy_sequence: u32,
    /// The policy slot bound to this source.
    slot: Option<u8>,
}

/// One policy and the source it serves. Policies carry per-device gesture
/// state (a touchpad's is kilobytes), so the seat holds a few, bound on a
/// device's first frame, rather than one for each of the 256 source bytes.
struct Binding<P> {
    policy: P,
    source: Option<u8>,
    /// The seat's frame count when this binding last took a frame.
    used: u64,
}

/// One compositor seat. The runtime can supply per-device policies backed by
/// explicit allocators/retained graphs. The default is pointer compatibility,
/// not a gesture engine. Wire scratch is reused; the hot path allocates nothing.
pub struct PointerSeat<P = PointerFallback> {
    seat: Seat,
    sources: [Source; 256],
    bindings: [Binding<P>; POLICY_SLOTS],
    /// The policy a newly bound device starts from.
    fresh: P,
    frames: u64,
    button_owners: [u16; 64],
    assembly: frame::Assembly,
    pub rejected_frames: u64,
}

impl PointerSeat<PointerFallback> {
    pub fn new(viewport: Viewport) -> Self {
        Self::with_policy(viewport, PointerFallback::default())
    }
}

impl<P: FramePolicy + Clone> PointerSeat<P> {
    /// A seat whose every device starts from `policy`.
    pub fn with_policy(viewport: Viewport, policy: P) -> Self {
        Self {
            seat: Seat::new(
                SeatId(0),
                viewport,
                PointerPosition::new(
                    (viewport.width() / 2) as i32,
                    (viewport.height() / 2) as i32,
                ),
                MotionScale::IDENTITY,
            ),
            sources: core::array::from_fn(|_| Source {
                history: FrameHistory::default(),
                buttons: 0,
                legacy_sequence: 0,
                slot: None,
            }),
            bindings: core::array::from_fn(|_| Binding {
                policy: policy.clone(),
                source: None,
                used: 0,
            }),
            fresh: policy,
            frames: 0,
            button_owners: [0; 64],
            assembly: frame::Assembly::default(),
            rejected_frames: 0,
        }
    }

    pub fn position(&self) -> PointerPosition {
        self.seat.position()
    }
    /// Convert baseline relative motion to target pixels after device policy.
    /// Absolute touch/pen coordinates, buttons and scroll retain their units.
    pub fn set_motion_scale(&mut self, scale: MotionScale) {
        self.seat.set_motion_scale(scale);
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.seat.set_viewport(viewport);
    }

    /// Reconfigure every device's policy in place, and the policy a newly
    /// bound device starts from, so a setting reaches devices bound before
    /// and after it alike.
    pub fn configure(&mut self, mut configure: impl FnMut(&mut P)) {
        configure(&mut self.fresh);
        for binding in &mut self.bindings {
            configure(&mut binding.policy);
        }
    }

    pub fn reset(&mut self, mut emit: impl FnMut(SeatEvent)) {
        self.reset_with(&mut |event| {
            if let Dispatch::Pointer(event) = event {
                emit(event);
            }
        });
    }
    pub fn reset_with(&mut self, emit: &mut impl FnMut(Dispatch<'_>)) {
        self.assembly.pending = false;
        for index in 0..256 {
            self.cancel_source(index, CancelReason::StreamLost, emit);
        }
    }

    pub fn cancel_device(
        &mut self,
        device: DeviceId,
        reason: CancelReason,
        emit: &mut impl FnMut(Dispatch<'_>),
    ) {
        if let Some(index) = usize::try_from(device.0)
            .ok()
            .filter(|&i| i < self.sources.len())
        {
            if self.assembly.source as usize == index {
                self.assembly.pending = false;
            }
            self.cancel_source(index, reason, emit);
        }
    }

    fn cancel_source(
        &mut self,
        index: usize,
        reason: CancelReason,
        emit: &mut impl FnMut(Dispatch<'_>),
    ) {
        let source = &mut self.sources[index];
        let gone = matches!(
            reason,
            CancelReason::DeviceRemoved | CancelReason::DeviceReset
        );
        if gone {
            source.history.retire();
        } else {
            source.history.interrupt();
        }
        emit(Dispatch::Cancelled {
            device: DeviceId(index as u32),
            reason,
        });
        let policy = source
            .slot
            .map(|slot| &mut self.bindings[usize::from(slot)].policy);
        cancel_policy(
            &mut self.seat,
            &mut self.button_owners,
            source,
            policy,
            index,
            reason,
            emit,
        );
        // A device that is gone frees its policy for the next one.
        if gone && let Some(slot) = source.slot.take() {
            let binding = &mut self.bindings[usize::from(slot)];
            binding.source = None;
            binding.policy = self.fresh.clone();
        }
    }

    /// The policy slot serving source `index`, binding one on the device's
    /// first frame: a free slot, else the one used least recently, whose
    /// device is cancelled before it is reused.
    fn bind(&mut self, index: usize, emit: &mut impl FnMut(Dispatch<'_>)) -> usize {
        self.frames = self.frames.wrapping_add(1);
        let slot = match self.sources[index].slot {
            Some(slot) => usize::from(slot),
            None => {
                let slot = self
                    .bindings
                    .iter()
                    .position(|b| b.source.is_none())
                    .unwrap_or_else(|| {
                        (0..POLICY_SLOTS)
                            .min_by_key(|&slot| self.bindings[slot].used)
                            .unwrap_or(0)
                    });
                if let Some(evicted) = self.bindings[slot].source {
                    let evicted = usize::from(evicted);
                    emit(Dispatch::Cancelled {
                        device: DeviceId(evicted as u32),
                        reason: CancelReason::StreamLost,
                    });
                    let source = &mut self.sources[evicted];
                    cancel_policy(
                        &mut self.seat,
                        &mut self.button_owners,
                        source,
                        Some(&mut self.bindings[slot].policy),
                        evicted,
                        CancelReason::StreamLost,
                        emit,
                    );
                    source.slot = None;
                }
                let binding = &mut self.bindings[slot];
                binding.policy = self.fresh.clone();
                binding.source = Some(index as u8);
                let source = &mut self.sources[index];
                source.slot = Some(slot as u8);
                if source.history.descriptor().is_some() {
                    // A device returning to a fresh policy resumes after a
                    // gap: a button still down must be seen up before it
                    // presses again, exactly as after a lost stream.
                    cancel_policy(
                        &mut self.seat,
                        &mut self.button_owners,
                        source,
                        Some(&mut binding.policy),
                        index,
                        CancelReason::StreamLost,
                        emit,
                    );
                }
                slot
            }
        };
        self.bindings[slot].used = self.frames;
        slot
    }

    /// The policy bound to `device`, if it has one.
    #[cfg(test)]
    fn policy(&self, device: DeviceId) -> Option<&P> {
        let source = self.sources.get(usize::try_from(device.0).ok()?)?;
        source
            .slot
            .map(|slot| &self.bindings[usize::from(slot)].policy)
    }

    /// Compatibility consumer. `feed_with` exposes raw frames and gestures to
    /// a richer compositor without changing transport or recognition code.
    pub fn feed(&mut self, report: InputReport, mut emit: impl FnMut(SeatEvent)) {
        self.feed_with(report, &mut |event| {
            if let Dispatch::Pointer(event) = event {
                emit(event);
            }
        });
    }

    pub fn feed_with(&mut self, report: InputReport, emit: &mut impl FnMut(Dispatch<'_>)) {
        let index = usize::from(report.source);
        if !pointer::is_pointer_kind(report.kind) {
            return;
        }
        if report.kind < pointer::FIRST_KIND {
            // Legacy contacts cannot prove frame boundaries. They are not
            // silently upgraded to complete snapshots. Producers must use the
            // framed path to enable touchpad/touchscreen compatibility policy.
            if report.kind != InputReport::KIND_MOUSE
                || report.reserved != [0; 7]
                || report.data[3..] != [0; 3]
            {
                self.reject(index, emit);
                return;
            }
            if self.assembly.pending {
                let old = usize::from(self.assembly.source);
                self.reject(old, emit);
            }
            let descriptor = DeviceDescriptor::new(
                SeatId(0),
                DeviceId(index as u32),
                DeviceGeneration(0),
                PointerSource::Mouse,
            );
            let slot = self.bind(index, emit);
            let source = &mut self.sources[index];
            source.legacy_sequence = source.legacy_sequence.wrapping_add(1);
            let mut frame = PointerFrame::new(&descriptor, source.legacy_sequence, &[]);
            frame.relative = [
                i32::from(report.data[0] as i8),
                i32::from(report.data[1] as i8),
            ];
            frame.buttons = u64::from(report.state);
            frame.scroll = Some(ScrollSample {
                horizontal: 0.0,
                vertical: f64::from(report.data[2] as i8),
                unit: axinput::ScrollUnit::Detent,
            });
            accept(
                &mut self.seat,
                &mut self.button_owners,
                source,
                &mut self.bindings[slot].policy,
                index,
                frame,
                &mut self.rejected_frames,
                emit,
            );
            return;
        }
        let Ok(record) = Record::decode(report) else {
            self.reject(index, emit);
            return;
        };
        if let Record::Reset { all } = record {
            if all {
                self.reset_with(emit);
            } else {
                self.cancel_device(DeviceId(index as u32), CancelReason::StreamLost, emit);
            }
            return;
        }
        if matches!(record, Record::Begin { .. }) && self.assembly.pending {
            let old = usize::from(self.assembly.source);
            self.reject(old, emit);
        }
        match self.assembly.push(report.source, record) {
            Ok(true) => {
                let slot = self.bind(index, emit);
                let frame = self.assembly.view();
                accept(
                    &mut self.seat,
                    &mut self.button_owners,
                    &mut self.sources[index],
                    &mut self.bindings[slot].policy,
                    index,
                    frame,
                    &mut self.rejected_frames,
                    emit,
                );
            }
            Ok(false) => {}
            Err(_) => self.reject(index, emit),
        }
    }

    fn reject(&mut self, index: usize, emit: &mut impl FnMut(Dispatch<'_>)) {
        self.rejected_frames = self.rejected_frames.saturating_add(1);
        if self.assembly.pending && usize::from(self.assembly.source) != index {
            self.cancel_source(
                usize::from(self.assembly.source),
                CancelReason::MalformedFrame,
                emit,
            );
        }
        self.assembly.pending = false;
        self.cancel_source(index, CancelReason::MalformedFrame, emit);
    }

    /// Runtime services this deadline using its own clock and wait facilities.
    pub fn next_deadline(&self) -> Option<InputTimestamp> {
        self.bindings
            .iter()
            .filter(|b| b.source.is_some())
            .filter_map(|b| b.policy.next_deadline())
            .min()
    }
    pub fn advance(&mut self, now: InputTimestamp, emit: &mut impl FnMut(Dispatch<'_>)) {
        for binding in &mut self.bindings {
            let Some(index) = binding.source.map(usize::from) else {
                continue;
            };
            if binding
                .policy
                .next_deadline()
                .is_some_and(|deadline| deadline <= now)
            {
                let source = &mut self.sources[index];
                binding.policy.advance(now, &mut |event| {
                    dispatch_policy(
                        &mut self.seat,
                        &mut self.button_owners,
                        &mut source.buttons,
                        index,
                        event,
                        emit,
                    )
                });
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accept<P: FramePolicy>(
    seat: &mut Seat,
    owners: &mut [u16; 64],
    source: &mut Source,
    policy: &mut P,
    index: usize,
    frame: PointerFrame<'_>,
    rejected: &mut u64,
    emit: &mut impl FnMut(Dispatch<'_>),
) {
    match source.history.accept(frame) {
        Ok(cancel) => {
            if let Some(reason) = cancel {
                emit(Dispatch::Cancelled {
                    device: frame.descriptor.device,
                    reason,
                });
                cancel_policy(seat, owners, source, Some(&mut *policy), index, reason, emit);
            }
            emit(Dispatch::Frame(frame));
            policy.frame(frame, &mut |event| {
                dispatch_policy(seat, owners, &mut source.buttons, index, event, emit)
            });
        }
        Err(_) => {
            *rejected = rejected.saturating_add(1);
            source.history.interrupt();
            emit(Dispatch::Cancelled {
                device: frame.descriptor.device,
                reason: CancelReason::MalformedFrame,
            });
            cancel_policy(
                seat,
                owners,
                source,
                Some(policy),
                index,
                CancelReason::MalformedFrame,
                emit,
            );
        }
    }
}

/// Cancel the source's policy, when it has one, and release whatever it
/// still holds.
fn cancel_policy<P: FramePolicy>(
    seat: &mut Seat,
    owners: &mut [u16; 64],
    source: &mut Source,
    policy: Option<&mut P>,
    index: usize,
    reason: CancelReason,
    emit: &mut impl FnMut(Dispatch<'_>),
) {
    if let Some(policy) = policy {
        policy.cancel(reason, &mut |event| {
            dispatch_policy(seat, owners, &mut source.buttons, index, event, emit)
        });
    }
    // The seat owns aggregate buttons. Even an incomplete replacement policy
    // must not strand its ownership after a reset, removal or lost stream.
    while source.buttons != 0 {
        let button = source.buttons.trailing_zeros() as u16;
        let input = axinput::PointerInput {
            seat: seat.id(),
            device: DeviceId(index as u32),
            source: source
                .history
                .descriptor()
                .map_or(PointerSource::Virtual, |d| d.source),
            timestamp: None,
            kind: PointerInputKind::Button {
                button: axinput::PointerButton(button),
                state: ButtonState::Released,
            },
        };
        dispatch_policy(
            seat,
            owners,
            &mut source.buttons,
            index,
            PolicyEvent::Pointer(input),
            emit,
        );
    }
}

fn dispatch_policy(
    seat: &mut Seat,
    owners: &mut [u16; 64],
    buttons: &mut u64,
    index: usize,
    event: PolicyEvent<'_>,
    emit: &mut impl FnMut(Dispatch<'_>),
) {
    match event {
        PolicyEvent::Gesture(event) => {
            if event.device == DeviceId(index as u32) {
                emit(Dispatch::Gesture(event));
            }
        }
        PolicyEvent::Pointer(input) => {
            if input.device != DeviceId(index as u32) || input.seat != seat.id() {
                return;
            }
            if let PointerInputKind::Button { button, state } = input.kind {
                if button.0 >= 64 {
                    return;
                }
                let bit = 1u64 << button.0;
                let held = *buttons & bit != 0;
                let pressed = state == ButtonState::Pressed;
                if held == pressed {
                    return;
                }
                let count = &mut owners[usize::from(button.0)];
                let before = *count;
                if pressed {
                    *buttons |= bit;
                    *count += 1;
                } else {
                    *buttons &= !bit;
                    *count -= 1;
                }
                if (before == 0) == (*count == 0) {
                    return;
                }
            }
            if let Ok(event) = seat.feed(input) {
                emit(Dispatch::Pointer(event));
            }
        }
        _ => {}
    }
}
