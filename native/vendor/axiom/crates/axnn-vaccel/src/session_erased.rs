//! Erase the session control plane, never provider buffer or event handles.
//! One box is allocated when a target is attached. Submissions still reach
//! the concrete Session<P>, with its original binding and lifetime checks.
use crate::session::{
    Invocation, InvocationStatus, InvocationSubmitError, RepeatedExecution, Session,
};
use core::alloc::Allocator;
use virtio_accel_core::{Accelerator, BackendError, ByteSink};

pub(crate) trait OutputReader {
    fn sizes(&self) -> &[usize];
    fn read(&self, index: usize, destination: &mut dyn ByteSink) -> Result<(), BackendError>;
}
pub(crate) struct Outputs<'a, P: Accelerator> {
    pub provider: &'a P,
    pub buffers: &'a [P::Buffer],
    pub sizes: &'a [usize],
}
impl<P: Accelerator> OutputReader for Outputs<'_, P> {
    fn sizes(&self) -> &[usize] {
        self.sizes
    }
    fn read(&self, index: usize, destination: &mut dyn ByteSink) -> Result<(), BackendError> {
        let buffer = self
            .buffers
            .get(index)
            .ok_or(BackendError::InvalidArgument)?;
        self.provider.read_buffer(buffer, 0, destination)
    }
}
type ReadCallback<'a> = dyn FnMut(&dyn OutputReader) -> Result<(), BackendError> + 'a;
pub(crate) trait GraphSession {
    fn install(&mut self, key: u64, bytes: Vec<u8>) -> Result<(), BackendError>;
    fn evict_generic(&mut self, key: u64) -> Result<(), BackendError>;
    fn submit(
        &mut self,
        key: u64,
        inputs: &[&[u8]],
        outputs: &[usize],
    ) -> Result<Invocation, InvocationSubmitError>;
    fn poll(
        &mut self,
        token: Invocation,
        read: &mut ReadCallback<'_>,
    ) -> Result<InvocationStatus<()>, BackendError>;
    fn poll_many_into(
        &mut self,
        token: Invocation,
        sinks: &mut [&mut dyn ByteSink],
    ) -> Result<InvocationStatus<()>, BackendError>;
    fn is_pending(&self, token: Invocation) -> bool;
    fn cancel(&self, token: Invocation) -> Result<(), BackendError>;
    fn repeated(
        &mut self,
        key: u64,
        inputs: &[&[u8]],
        outputs: &[usize],
        repeated: RepeatedExecution<'_>,
        read: &mut ReadCallback<'_>,
    ) -> Result<(), BackendError>;
}
impl<P: Accelerator> GraphSession for Session<P> {
    fn install(&mut self, key: u64, bytes: Vec<u8>) -> Result<(), BackendError> {
        self.ensure(key, || bytes)
    }
    fn evict_generic(&mut self, key: u64) -> Result<(), BackendError> {
        self.evict_generic(key)
    }
    fn submit(
        &mut self,
        key: u64,
        inputs: &[&[u8]],
        outputs: &[usize],
    ) -> Result<Invocation, InvocationSubmitError> {
        self.submit_many(
            key,
            || unreachable!("compiled graph program missing"),
            inputs,
            outputs,
        )
    }
    fn poll(
        &mut self,
        token: Invocation,
        read: &mut ReadCallback<'_>,
    ) -> Result<InvocationStatus<()>, BackendError> {
        self.poll_many_with(token, |provider, buffers, sizes| {
            read(&Outputs {
                provider,
                buffers,
                sizes,
            })
        })
    }
    fn poll_many_into(
        &mut self,
        token: Invocation,
        sinks: &mut [&mut dyn ByteSink],
    ) -> Result<InvocationStatus<()>, BackendError> {
        self.poll_many_into(token, sinks)
    }
    fn is_pending(&self, token: Invocation) -> bool {
        self.is_pending(token)
    }
    fn cancel(&self, token: Invocation) -> Result<(), BackendError> {
        self.cancel(token)
    }
    fn repeated(
        &mut self,
        key: u64,
        inputs: &[&[u8]],
        outputs: &[usize],
        repeated: RepeatedExecution<'_>,
        read: &mut ReadCallback<'_>,
    ) -> Result<(), BackendError> {
        self.run_many_repeated_with(
            key,
            || unreachable!("compiled graph program missing"),
            inputs,
            outputs,
            repeated,
            |provider, buffers, sizes| {
                read(&Outputs {
                    provider,
                    buffers,
                    sizes,
                })
            },
        )
    }
}
impl dyn GraphSession + '_ {
    pub fn poll_with<T>(
        &mut self,
        token: Invocation,
        read: impl FnOnce(&dyn OutputReader) -> Result<T, BackendError>,
    ) -> Result<InvocationStatus<T>, BackendError> {
        let mut read = Some(read);
        let mut output = None;
        let status = self.poll(token, &mut |reader| {
            output = Some(read.take().ok_or(BackendError::InvalidArgument)?(reader)?);
            Ok(())
        })?;
        Ok(match status {
            InvocationStatus::Complete(()) => {
                InvocationStatus::Complete(output.ok_or(BackendError::InvalidArgument)?)
            }
            InvocationStatus::Pending => InvocationStatus::Pending,
            InvocationStatus::Failed(e) => InvocationStatus::Failed(e),
            InvocationStatus::Cancelled => InvocationStatus::Cancelled,
        })
    }
    pub fn run_repeated_in<A: Allocator + Clone>(
        &mut self,
        key: u64,
        inputs: &[&[u8]],
        sizes: &[usize],
        repeated: RepeatedExecution<'_>,
        allocator: A,
    ) -> Result<Vec<Vec<u8, A>, A>, BackendError> {
        let mut output = None;
        self.repeated(key, inputs, sizes, repeated, &mut |reader| {
            let mut values = Vec::new_in(allocator.clone());
            values
                .try_reserve_exact(reader.sizes().len())
                .map_err(|_| BackendError::OutOfMemory)?;
            for (index, &size) in reader.sizes().iter().enumerate() {
                let mut bytes = Vec::new_in(allocator.clone());
                bytes
                    .try_reserve_exact(size)
                    .map_err(|_| BackendError::OutOfMemory)?;
                bytes.resize(size, 0);
                reader.read(index, &mut crate::AxOutput::packed(&mut bytes))?;
                values.push(bytes);
            }
            output = Some(values);
            Ok(())
        })?;
        output.ok_or(BackendError::InvalidArgument)
    }
}
