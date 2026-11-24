//! Bounded, caller-polled work over separately compiled executable slots.
use super::*;

/// Admission failure returns ownership of every input allocation for retry.
pub struct AxStreamSubmitError<R: Allocator> {
    pub error: AxCompileError,
    pub inputs: Vec<Vec<u8, R>, R>,
}

/// One round-robin step. A failed observation can retain a live provider event;
/// the slot and its input charge remain occupied until that event is retired.
pub enum AxStreamProgress<R: Allocator> {
    Idle,
    Progress {
        slot: usize,
        progress: AxPartitionProgress,
    },
    Complete {
        slot: usize,
        outputs: Vec<AxTensor<'static, R>, R>,
    },
    Cancelled {
        slot: usize,
    },
    Failed {
        slot: usize,
        error: AxCompileError,
        retained: bool,
    },
}

/// Fixed-capacity ingress and cooperative overlap for independent graph jobs.
///
/// Each slot has a distinct compiled executable, so providers retain separate
/// I/O and event namespaces. `poll` performs one existing executor step and
/// rotates even when a device remains pending. No thread, busy loop, implicit
/// queue, backend capability override, or copy of admitted input planes is used.
///
/// `input_byte_limit` bounds owned input allocation capacities while jobs are
/// live. It excludes graph/program residency, intermediates and returned
/// outputs; it is not a claim about physical device working-set usage. Providers
/// and the caller's allocators remain responsible for those resource limits.
#[must_use = "live stream slots must be polled or cancelled to retire provider events"]
pub struct AxStream<'e, M: Allocator, R: Allocator + Clone, const N: usize> {
    executables: [&'e AxExecutable<M>; N],
    active: [Option<AxInvocation<'e, 'static, M, R>>; N],
    charges: [usize; N],
    input_byte_limit: usize,
    input_bytes: usize,
    cursor: usize,
    allocator: R,
}

impl<'e, M: Allocator, R: Allocator + Clone, const N: usize> AxStream<'e, M, R, N> {
    pub fn new(
        executables: [&'e AxExecutable<M>; N],
        input_byte_limit: usize,
        allocator: R,
    ) -> Result<Self, AxCompileError> {
        if N == 0 {
            return Err(BackendError::InvalidArgument.into());
        }
        for (i, executable) in executables.iter().enumerate() {
            if executable.owner != executables[0].owner {
                return Err(AxCompileError::WrongExecutor);
            }
            if executables[..i]
                .iter()
                .any(|prior| core::ptr::eq(*prior, *executable))
            {
                return Err(BackendError::InvalidArgument.into());
            }
            validate_executable_resources(executable, &[])?;
        }
        Ok(Self {
            executables,
            active: core::array::from_fn(|_| None),
            charges: [0; N],
            input_byte_limit,
            input_bytes: 0,
            cursor: 0,
            allocator,
        })
    }
    pub const fn input_bytes(&self) -> usize {
        self.input_bytes
    }
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|s| s.is_some()).count()
    }
    pub fn is_available(&self, slot: usize) -> bool {
        self.active.get(slot).is_some_and(Option::is_none)
    }
    pub fn executable(&self, slot: usize) -> Option<&AxExecutable<M>> {
        self.executables.get(slot).copied()
    }

    /// Transfer input ownership after count, shape, slot and byte admission.
    /// Rejections return the original allocations; successful admission moves
    /// them directly into logical tensor slots without materializing new planes.
    pub fn try_submit<C: Accelerator, A: Accelerator>(
        &mut self,
        runtime: &AxHetero<C, A>,
        slot: usize,
        inputs: Vec<Vec<u8, R>, R>,
    ) -> Result<(), AxStreamSubmitError<R>> {
        let prepared = (|| {
            let executable = *self
                .executables
                .get(slot)
                .ok_or(BackendError::InvalidArgument)?;
            if executable.owner != runtime.owner {
                return Err(AxCompileError::WrongExecutor);
            }
            if self.active[slot].is_some() {
                return Err(BackendError::Busy.into());
            }
            if inputs.len() != executable.inputs.len() {
                return Err(AxCompileError::InvalidInputCount {
                    expected: executable.inputs.len(),
                    actual: inputs.len(),
                });
            }
            let graph = executable.graph();
            for (&id, bytes) in executable.inputs.iter().zip(&inputs) {
                let tensor = graph
                    .tensor(id)
                    .map_err(|_| AxCompileError::MissingTensor(id))?;
                let expected =
                    tensor_storage_bytes(tensor).ok_or(AxCompileError::MissingTensor(id))?;
                if bytes.len() != expected {
                    return Err(AxCompileError::InvalidInputSize {
                        tensor: id,
                        expected,
                        actual: bytes.len(),
                    });
                }
            }
            let charge = inputs
                .iter()
                .try_fold(0usize, |sum, v| sum.checked_add(v.capacity()))
                .ok_or(BackendError::ResourceLimit)?;
            if self
                .input_bytes
                .checked_add(charge)
                .is_none_or(|n| n > self.input_byte_limit)
            {
                return Err(BackendError::ResourceLimit.into());
            }
            let mut values = Vec::new_in(self.allocator.clone());
            reserve(&mut values, executable.value_slots)?;
            values.resize_with(executable.value_slots, || None);
            Ok((executable, values, charge))
        })();
        let (executable, mut values, charge) = match prepared {
            Ok(v) => v,
            Err(error) => return Err(AxStreamSubmitError { error, inputs }),
        };
        for (&id, bytes) in executable.inputs.iter().zip(inputs) {
            values[id.0 as usize] = Some(RuntimeValue::Owned(bytes));
        }
        self.active[slot] = Some(AxInvocation {
            owner: runtime.owner,
            executable,
            values,
            resources: &[],
            allocator: self.allocator.clone(),
            next_partition: 0,
            host_copies: AxHostCopyStats::default(),
            pending: None,
            finished: false,
        });
        self.charges[slot] = charge;
        self.input_bytes += charge;
        Ok(())
    }
    fn retire(&mut self, slot: usize) {
        self.active[slot] = None;
        self.input_bytes -= self.charges[slot];
        self.charges[slot] = 0;
    }
    pub fn poll<C: Accelerator, A: Accelerator>(
        &mut self,
        runtime: &mut AxHetero<C, A>,
    ) -> AxStreamProgress<R> {
        let Some(slot) = (0..N)
            .map(|off| (self.cursor + off) % N)
            .find(|&i| self.active[i].is_some())
        else {
            return AxStreamProgress::Idle;
        };
        self.cursor = (slot + 1) % N;
        let invocation = self.active[slot].as_mut().unwrap();
        match runtime.advance_retained_in(invocation) {
            Ok(AxInvocationStatus::Progress(progress)) => {
                AxStreamProgress::Progress { slot, progress }
            }
            Ok(AxInvocationStatus::Complete(outputs)) => {
                self.retire(slot);
                AxStreamProgress::Complete { slot, outputs }
            }
            Ok(AxInvocationStatus::Cancelled(_)) => {
                self.retire(slot);
                AxStreamProgress::Cancelled { slot }
            }
            Err(error) => {
                let retained = !invocation.finished;
                if !retained {
                    self.retire(slot);
                }
                AxStreamProgress::Failed {
                    slot,
                    error,
                    retained,
                }
            }
        }
    }
    /// Cancellation is a request, not proof of terminal completion. Pending
    /// slots retain their charge until polling observes and retires the event.
    pub fn cancel<C: Accelerator, A: Accelerator>(
        &mut self,
        runtime: &AxHetero<C, A>,
        slot: usize,
    ) -> Result<(), AxCompileError> {
        let invocation = self
            .active
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or(BackendError::InvalidArgument)?;
        if invocation.owner != runtime.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        if invocation.pending.is_some() {
            runtime.cancel_in(invocation)
        } else {
            self.retire(slot);
            Ok(())
        }
    }
}
