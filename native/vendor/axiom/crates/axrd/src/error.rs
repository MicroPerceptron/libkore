/// Compositor errors, generic over the backend's error type.
///
/// `E` is the backend's `Backend::Error` — callers see
/// `RdError<CpuError>`, `RdError<GpuError>`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdError<E> {
    /// Forwarded backend failure (OOM, unsupported dtype, etc.).
    Backend(E),
    /// Scene capacity exceeded — cannot add more nodes.
    Full,
    /// Parent index does not refer to an existing node.
    BadNode(u16),
    /// Dependency cycle detected during topo sort.
    Cycle,
    /// A [`NodeId`](crate::NodeId) whose slot was freed (and possibly
    /// reused) since it was minted.
    Stale,
    /// A reparent that would make a node its own ancestor.
    WouldCycle,
}

impl<E> From<E> for RdError<E> {
    fn from(e: E) -> Self {
        RdError::Backend(e)
    }
}
