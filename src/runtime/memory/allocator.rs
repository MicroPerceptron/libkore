use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;

pub struct MemoryAllocator;

unsafe impl Allocator for MemoryAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let _ = layout;
        todo!("Implement allocation logic");
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let _ = (ptr, layout);
        todo!("Implement deallocation logic");
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_allocate() {
        let allocator = MemoryAllocator;
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let result = allocator.allocate(layout);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deallocate() {
        let allocator = MemoryAllocator;
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let ptr = allocator.allocate(layout).unwrap().as_non_null_ptr();
        unsafe {
            allocator.deallocate(ptr, layout);
        }
    }
}
