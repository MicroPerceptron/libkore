use crate::context::MemoryContext;

pub trait MemoryAddress: Copy + Ord {
    type Context: MemoryContext;
    const WIDTH: usize;
}
