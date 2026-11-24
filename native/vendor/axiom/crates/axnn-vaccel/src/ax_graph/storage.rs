//! Compatibility facade over portable tensor storage and pitched outputs.
use super::AxCompileError;
pub use ax_exec::AxOutput;
pub(super) use ax_exec::storage::RuntimeValue;
#[cfg(test)]
pub(super) use ax_exec::storage::{NativeRef, NativeSlot};
use core::alloc::Allocator;
/// A completed graph tensor retaining its backing, without an obligatory byte
/// vector allocation. Input passthroughs can still borrow invocation inputs.
/// On little-endian hosts native F32/I32 results expose their aligned allocation
/// as bytes. Provider transfers and precision conversions remain explicit.
/// Drop this value only after all consumers of its borrowed view have finished.
pub struct AxTensor<'a, A: Allocator> {
    pub(super) value: RuntimeValue<'a, A>,
}
impl<A: Allocator> AxTensor<'_, A> {
    pub fn as_bytes(&self) -> &[u8] {
        self.value.as_slice()
    }
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }
    pub fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }
    /// Materialize the legacy byte-vector ABI, moving existing byte storage
    /// when possible. Typed F32/I32 or borrowed backing requires one copy here.
    /// `allocator` is used only for new storage; moved storage keeps its allocator.
    pub fn into_bytes_in(self, allocator: A) -> Result<Vec<u8, A>, AxCompileError> {
        self.value.into_bytes(allocator).map_err(Into::into)
    }
}
impl<A: Allocator> core::fmt::Debug for AxTensor<'_, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AxTensor")
            .field("bytes", &self.len())
            .finish_non_exhaustive()
    }
}
