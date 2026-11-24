//! Indexed, inert observations from the existing input owners. No register
//! probes, grants, waits, or serial output occur while serving a status page.

use kore_abi::input_status::{
    ADMISSION_STATUS_BASE, I2C_STATUS_ROWS, INPUT_STATUS_ROWS, InputDiagnostic,
    POINTER_STATUS_BASE, USB_STATUS_BASE, USB_STATUS_PARTS,
};

use super::{InputService, PointerLane, admission, i2c_relay};

fn count(value: u64) -> u32 {
    value.min(u64::from(u32::MAX)) as u32
}

impl InputService {
    pub fn diagnostic(&self, row: u8) -> Option<InputDiagnostic> {
        let (state, words) = match row {
            0..I2C_STATUS_ROWS => return i2c_relay::diagnostic(row).map(InputDiagnostic::from_i2c),
            USB_STATUS_BASE..POINTER_STATUS_BASE => {
                let offset = row - USB_STATUS_BASE;
                let b = self.xhci_books(usize::from(offset / USB_STATUS_PARTS))?;
                let words = match offset % USB_STATUS_PARTS {
                    0 => [
                        b.state as u32,
                        (b.state >> 32) as u32,
                        b.identity,
                        b.sources,
                        b.discovery_errors,
                    ],
                    1 => [
                        count(b.irq_wakes),
                        count(b.irq_delivered),
                        count(b.irq_events),
                        count(b.irq_errors),
                        count(b.reports),
                    ],
                    _ => [count(b.dropped), b.source_exhausted, 0, 0, 0],
                };
                (0, words)
            }
            POINTER_STATUS_BASE..ADMISSION_STATUS_BASE => return self.pointer.diagnostic(row),
            ADMISSION_STATUS_BASE..INPUT_STATUS_ROWS => {
                let page = admission::diagnostic(row - ADMISSION_STATUS_BASE)?;
                (
                    u8::from(admission::available()),
                    [page.words[0], page.words[1], 0, 0, 0],
                )
            }
            _ => return None,
        };
        InputDiagnostic::from_parts(state, u32::from(row), words)
    }
}

impl PointerLane<'_> {
    fn diagnostic(&self, row: u8) -> Option<InputDiagnostic> {
        let (state, words) = match row {
            POINTER_STATUS_BASE => {
                let dropped = self.queue.as_ref().map_or(0, |queue| queue.dropped());
                (
                    u8::from(self.queue.is_some()),
                    [
                        count(self.observed),
                        self.publications as u32,
                        (self.publications >> 32) as u32,
                        dropped as u32,
                        (dropped >> 32) as u32,
                    ],
                )
            }
            row if row == POINTER_STATUS_BASE + 1 => {
                let (queued, capacity) = self.queue.as_ref().map_or((0, 0), |queue| {
                    (
                        queue.capacity().saturating_sub(queue.capacity_left()),
                        queue.capacity(),
                    )
                });
                (
                    0,
                    [
                        queued,
                        capacity,
                        count(self.rejected),
                        count(self.resets),
                        0,
                    ],
                )
            }
            _ => return None,
        };
        InputDiagnostic::from_parts(state, u32::from(row), words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::{input_status::InputSnapshot, svc::InputReport};

    fn snapshot(pointer: &PointerLane<'_>) -> InputSnapshot {
        let mut snapshot = InputSnapshot::default();
        for row in POINTER_STATUS_BASE..ADMISSION_STATUS_BASE {
            assert!(snapshot.insert_diagnostic(pointer.diagnostic(row).unwrap()));
        }
        snapshot
    }

    #[test]
    fn pointer_books_distinguish_missing_attachment_backlog_consumption_and_loss() {
        #[repr(align(64))]
        struct Window([u8; 4096]);
        let mut bytes = Window([0; 4096]);
        let (producer, mut consumer) = kore_ring::event::split::<
            kore_abi::pointer::PointerState,
            InputReport,
        >(&mut bytes.0, 8)
        .unwrap();
        let mut pointer = PointerLane::default();
        assert_eq!(
            pointer.route(InputReport::mouse(1, 0, 3, 2, 0)),
            Some(false)
        );
        assert_eq!(snapshot(&pointer).pointer.attached.get(), 0);
        pointer.attach(producer);
        assert_eq!(pointer.route(InputReport::mouse(1, 0, 3, 2, 0)), Some(true));
        let books = snapshot(&pointer).pointer;
        assert_eq!(
            (
                books.publications.get(),
                books.queued.get(),
                books.consumed()
            ),
            (5, 5, 0)
        );
        assert_eq!(books.rejected.get(), 1);
        assert_eq!(
            pointer.route(InputReport::mouse(1, 0, 3, 2, 0)),
            Some(false)
        );
        let books = snapshot(&pointer).pointer;
        assert_eq!(
            (
                books.publications.get(),
                books.dropped.get(),
                books.consumed()
            ),
            (10, 5, 0)
        );
        for _ in 0..5 {
            assert!(consumer.pop().is_some());
        }
        assert!(consumer.pop().is_none());
        let books = snapshot(&pointer).pointer;
        assert_eq!((books.queued.get(), books.consumed()), (0, 5));
        assert_eq!(books.reports.get(), 3);
    }
}
