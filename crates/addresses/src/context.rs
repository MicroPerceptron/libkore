use crate::unsigned::Unsigned;

pub trait MemoryContext {
    type Address: Unsigned;
}
