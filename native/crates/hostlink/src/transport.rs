//! The seam implemented by serial and shared-memory adapters.

use crate::LinkEpoch;

/// HostLink's independent traffic classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Plane {
    Control,
    Bulk,
}

/// A transport failure before HostLink message interpretation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortError {
    Backpressure,
    Disconnected,
    Fault,
}

/// One connected HostLink transport.
///
/// Four distinct methods are intentional: an implementation must make the
/// control and bulk queue ownership visible instead of hiding both behind one
/// queue selected by model-controlled data.
pub trait HostLinkTransport {
    /// Synchronize transport-local reset state before the service drains any
    /// bulk records for a newly accepted control-plane epoch. Stream-only
    /// transports own no such state.
    fn synchronize_link_epoch(&mut self, _epoch: Option<LinkEpoch>) -> Result<(), PortError> {
        Ok(())
    }

    fn try_send_control(&mut self, frame: &[u8]) -> Result<(), PortError>;
    fn try_recv_control(&mut self, output: &mut [u8]) -> Result<usize, PortError>;
    fn try_send_bulk(&mut self, frame: &[u8]) -> Result<(), PortError>;
    fn try_recv_bulk(&mut self, output: &mut [u8]) -> Result<usize, PortError>;
}
