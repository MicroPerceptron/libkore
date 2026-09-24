//! Shared init/fini array admission for startup and late linking.
//! Validation performs no calls; callers snapshot admitted addresses before
//! entering code that could mutate the original writable array.

use kore_abi::image::{ModuleDescriptor, seg};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionError {
    /// The array is not composed of complete ELF64 pointers.
    Shape,
    /// A non-sentinel entry does not name loaded executable text.
    Target(usize),
}

/// Does an absolute address name file-backed text in this placed module?
/// Read-only data, writable data, padding beyond filesz and overflowing
/// placement geometry never count as executable targets.
pub fn contains_text(descriptor: &ModuleDescriptor, base: u64, address: u64) -> bool {
    descriptor.live_segs().iter().any(|segment| {
        if segment.kind != seg::KIND_TEXT {
            return false;
        }
        let Some(start) = base.checked_add(segment.vaddr.get()) else {
            return false;
        };
        let Some(end) = start.checked_add(segment.filesz.get()) else {
            return false;
        };
        address >= start && address < end
    })
}

/// Validate the complete array before exposing any address to the caller.
/// The caller supplies the actual loaded executable scope. Zero and all-ones
/// retain the established crt sentinel meaning and are omitted.
pub fn function_addresses(
    bytes: &[u8],
    is_executable: impl Fn(u64) -> bool,
) -> Result<impl Iterator<Item = u64> + '_, FunctionError> {
    if bytes.len() % 8 != 0 {
        return Err(FunctionError::Shape);
    }
    let words = || {
        bytes.chunks_exact(8).map(|word| {
            u64::from_le_bytes(word.try_into().expect("chunks_exact yields eight bytes"))
        })
    };
    for (index, address) in words().enumerate() {
        if address != 0 && address != u64::MAX && !is_executable(address) {
            return Err(FunctionError::Target(index));
        }
    }
    Ok(words().filter(|address| *address != 0 && *address != u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_the_entire_array_before_exposing_an_entry() {
        let mut bytes = [0u8; 32];
        bytes[8..16].copy_from_slice(&0x1000u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        bytes[24..].copy_from_slice(&0x2000u64.to_le_bytes());
        assert!(matches!(
            function_addresses(&bytes, |p| p == 0x1000),
            Err(FunctionError::Target(3))
        ));
        let mut addresses = function_addresses(&bytes, |p| p == 0x1000 || p == 0x2000).unwrap();
        assert_eq!(addresses.next(), Some(0x1000));
        assert_eq!(addresses.next(), Some(0x2000));
        assert_eq!(addresses.next(), None);
        assert!(matches!(
            function_addresses(&bytes[..31], |_| true),
            Err(FunctionError::Shape)
        ));
    }

    #[test]
    fn text_membership_rejects_padding_other_segments_and_overflow() {
        let mut descriptor = ModuleDescriptor::empty();
        descriptor.seg_count = 1;
        descriptor.segs[0].kind = seg::KIND_TEXT;
        descriptor.segs[0].vaddr.set(0x1000);
        descriptor.segs[0].filesz.set(16);
        descriptor.segs[0].memsz.set(4096);
        assert!(contains_text(&descriptor, 0x4000, 0x5000));
        assert!(contains_text(&descriptor, 0x4000, 0x500f));
        assert!(!contains_text(&descriptor, 0x4000, 0x5010));
        assert!(!contains_text(&descriptor, u64::MAX - 8, 0x1000));
        descriptor.segs[0].kind = seg::KIND_RODATA;
        assert!(!contains_text(&descriptor, 0x4000, 0x5000));
    }
}
