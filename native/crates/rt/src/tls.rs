//! Immutable linker TLS templates and bounded per-thread placement.
use kore_abi::task::layout::Window;

/// Process-lifetime initialization bytes, including an empty runtime TCB.
/// Construction does not confer mapping authority; installation is unsafe.
#[derive(Clone, Copy, Debug)]
pub struct Template {
    pub(crate) base: u64,
    pub(crate) bytes: u64,
    pub(crate) tp_offset: u64,
    pub(crate) align: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    pub mapping: Window,
    pub remaining: Window,
    pub tp: u64,
}

impl Template {
    pub fn new(base: u64, bytes: u64, tp_offset: u64, align: u64) -> Option<Self> {
        if base == 0
            || !align.is_power_of_two()
            || align < 16
            || base % align != 0
            || bytes > isize::MAX as u64
            || base.checked_add(bytes).is_none()
            || tp_offset % 16 != 0
            || tp_offset.checked_add(kore_abi::image::RUNTIME_TCB_BYTES)? > bytes
        {
            return None;
        }
        Some(Self {
            base,
            bytes,
            tp_offset,
            align,
        })
    }

    pub(crate) fn fits_band(self, bytes: u64) -> bool {
        self.align <= bytes
            && self
                .bytes
                .checked_add(4095)
                .is_some_and(|n| (n & !4095) < bytes)
    }

    /// Carve from this child's receiver-map band, never from a sibling's TCB.
    pub(crate) fn place(self, window: Window) -> Option<Placement> {
        let align = self.align.max(4096);
        let base = window.base().checked_add(align - 1)? & !(align - 1);
        let bytes = self.bytes.checked_add(4095)? & !4095;
        let mapping = Window::try_new(base, bytes)?;
        if !window.contains(&mapping) {
            return None;
        }
        let remaining = Window::try_new(mapping.end(), window.end().checked_sub(mapping.end())?)?;
        Some(Placement {
            mapping,
            remaining,
            tp: base.checked_add(self.tp_offset)?,
        })
    }

    /// # Safety
    /// The template bytes are immutable and live; destination is a disjoint,
    /// exclusively writable mapping of at least bytes bytes at its alignment.
    pub(crate) unsafe fn initialize(self, destination: u64) {
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.base as *const u8,
                destination as *mut u8,
                self.bytes as usize,
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn alignment_bounds_and_independent_bands() {
        let template = Template::new(0x10000, 0x3010, 0x2010, 16).unwrap();
        let a = template.place(Window::new(0x80000000, 0x1000000)).unwrap();
        let b = template.place(Window::new(0x81000000, 0x1000000)).unwrap();
        assert!(template.fits_band(0x1000000));
        assert!(!template.fits_band(0x4000));
        assert_eq!(a.tp, 0x80002010);
        assert_eq!(a.remaining.base(), 0x80004000);
        assert!(a.mapping.precedes(&b.mapping));
        assert!(template.place(Window::new(0x80000000, 0x3000)).is_none());
        assert!(Template::new(0x10000, 4096, 16, 16).is_none());
        assert!(Template::new(u64::MAX - 15, 8192, 0, 16).is_none());
    }
}
