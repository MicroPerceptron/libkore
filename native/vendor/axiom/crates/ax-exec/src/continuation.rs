//! Cooperative prepared execution over the native instruction engine.
use crate::{Error, ForkJoin, NativeRef, NativeResourceBinding, PreparedNative};
use axnn::lowering::KernelAccuracy;
use axnn_cpu::CpuKernelPlan;
use core::alloc::Allocator;

/// Outcome of one call into a runtime-owned kernel executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelProgress {
    /// No work was admitted and output was untouched; use the native kernel.
    /// Once Pending has been returned for this instruction this is an error.
    Unsupported,
    /// Retain the exact admission and wait for runtime/device progress.
    Pending,
    /// Output is available and no external access to call-scoped slices remains.
    Complete,
}

/// Most instructions one invocation holds admitted at once
/// ([`PreparedKernelExecutor::admission_window`] is clamped to it): the
/// ordered-submission ceiling a device queue accepts in one event.
pub const MAX_ADMISSION_WINDOW: usize = 32;

/// One operand of an instruction offered through
/// [`PreparedKernelExecutor::advance_chained`].
#[derive(Clone, Copy)]
pub enum ChainedInput<'a> {
    /// Final operand bytes, exactly what [`PreparedKernelExecutor::advance`]
    /// receives.
    Host(NativeRef<'a>),
    /// `elements` values starting `offset` values into the output of
    /// `instruction`, an earlier instruction this executor admitted and has
    /// not completed: F32, or the I32 token of a resident KV append. The
    /// host slot does not hold them yet: the executor must read them where
    /// that admission produces them (or refuse).
    Pending {
        instruction: usize,
        offset: usize,
        elements: usize,
    },
}

impl<'a> ChainedInput<'a> {
    /// The host operand, when this is not a pending producer's output.
    pub fn host(self) -> Option<NativeRef<'a>> {
        match self {
            Self::Host(value) => Some(value),
            Self::Pending { .. } => None,
        }
    }
}

/// The recurrent state a [`PreparedKernelExecutor::advance_recurrent`]
/// offer carries, exactly as the native kernels take it: the state before
/// the run, and the buffer its successor states go to, either one slot
/// (updated in place, the state after the last token) or one slot per token
/// of the run (slot `t` the state after token `t`).
pub struct RecurrentState<'a> {
    pub source: &'a [f32],
    /// Tree traversal when verifying branches; None keeps linear semantics.
    pub tree: Option<axnn::lowering::TokenTree<'a>>,
    /// An exclusive resource view, without creating a CPU slice on retries
    /// while a device may still be writing. This view alone does not permit
    /// retaining DMA pointers: the executor must independently own/pin any
    /// storage bound asynchronously, including after cancellation. Otherwise
    /// it stages results and accesses `values()` only when publishing them.
    pub target: axnn_cpu::StateTargetRef<'a>,
}

/// Identity of a KV cache allocated and retained by an external executor.
/// This is inert metadata, not memory authority. `owner` identifies the
/// runtime session/epoch and `allocation` its cache within that epoch; the
/// executor must validate both on every admission. Neither is a host pointer.
///
/// The owner keeps storage and in-flight work alive independently of graph
/// futures. Reusing either identity while an old binding can be used is not
/// permitted. CPU kernels cannot access this cache or silently replace it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentKvCache {
    owner: core::num::NonZeroU64,
    allocation: core::num::NonZeroU64,
    params: axnn::lowering::KvCacheParams,
}

impl ResidentKvCache {
    pub fn new(
        owner: core::num::NonZeroU64,
        allocation: core::num::NonZeroU64,
        params: axnn::lowering::KvCacheParams,
    ) -> Result<Self, Error> {
        if !params.is_valid() || params.storage_bytes().is_none() {
            return Err(crate::PlanError::InvalidPartition.into());
        }
        Ok(Self {
            owner,
            allocation,
            params,
        })
    }

    pub const fn owner(self) -> core::num::NonZeroU64 {
        self.owner
    }
    pub const fn allocation(self) -> core::num::NonZeroU64 {
        self.allocation
    }
    pub const fn params(self) -> axnn::lowering::KvCacheParams {
        self.params
    }
}

/// The append dependency token or attention output of a resident KV operation.
/// Borrowed only for one `advance_kv` call, including when it returns Pending.
pub enum KvOutput<'a> {
    Appended(&'a mut [i32]),
    Attention(&'a mut [f32]),
}

/// A run of instructions an executor may take as one fused kernel. The
/// engine offers a run only when each intermediate value is read by the next
/// instruction of the run and by nothing else (and is not a graph output), so
/// only the run's last output is ever published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedRun {
    /// A projection with no epilogue, then `RELU`, then `MUL(r, r)`: the
    /// projection's squared ReLU.
    SquaredRelu,
    /// A projection with no epilogue, then `ADD` of it and another value (in
    /// either operand order): the projection accumulated onto that `base`.
    Accumulate,
    /// An element-wise ADD or MUL followed by weighted RMS normalization.
    /// `plan` is the binary operation, `inputs` are its operands, and `base`
    /// is the normalization weight. Normalize each consecutive `width`
    /// values, then multiply by the weight cyclically across the output.
    /// The binary result is rounded to F32 before reduction, as in the
    /// unfused graph. A reshape covering the whole intermediate is allowed.
    RmsNorm {
        width: usize,
        epsilon: axnn::lowering::F32Bits,
    },
}

/// Optional execution of pure, single-F32-result instructions and loaded MoE
/// up projections. Routing, residency, sparse down rows, other resource effects
/// and prepared fusions stay in the existing native engine.
///
/// Every tensor borrow is scoped to `advance`, including when it returns
/// Pending. An asynchronous provider must own its device allocations/leases
/// independently; it may stage from these views and copy back a completed
/// result, but must not retain pointers for DMA into borrowed slices. Retained
/// shared-memory routes require a separate owning runtime contract.
///
/// The same instruction number is retried until complete. The executor must
/// retain admission/cancellation/retirement state outside observation futures,
/// reject kernels below `minimum_accuracy`, and report failures rather than
/// returning Unsupported after publication. Dropping a graph continuation does
/// not cancel independently owned device work or authorize buffer reuse.
pub trait PreparedKernelExecutor {
    /// Execute a KV append or attention read against runtime-owned storage.
    /// Only resident bindings reach this hook. The plan supplies geometry,
    /// causal position and the original storage/accuracy contract; `inputs`
    /// have the same order as the corresponding native operation.
    ///
    /// A cache remains resident across invocations. Append overwrites exactly
    /// the supplied positions; attention reads only the causal prefix, never
    /// rejected speculative suffixes. Commit/rollback changes the model's
    /// next position, not cache ownership, and later appends overwrite rejected
    /// rows. No host cache exists to fall back to: Unsupported is an error.
    /// Pending retains owned device work, never these slices. Complete means
    /// the operation retired and its output is published. Cancellation does
    /// not revoke the owner's outstanding DMA or permit allocation reuse.
    fn advance_kv(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        cache: ResidentKvCache,
        inputs: &[Option<NativeRef<'_>>; 3],
        output: KvOutput<'_>,
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        let _ = (instruction, plan, cache, inputs, output, minimum_accuracy);
        Ok(KernelProgress::Unsupported)
    }

    /// Offer a resident KV append or attention read to the admission window,
    /// under the window rules of [`advance_chained`](Self::advance_chained):
    /// retries repeat these arguments, there is no Complete while an earlier
    /// admission is pending, and an operand an earlier admission produces
    /// arrives [`ChainedInput::Pending`]. That covers the keys, values and
    /// queries, and an attention read's token from its append, which the
    /// window admits ahead of it. The cache contract is
    /// [`advance_kv`](Self::advance_kv)'s. A decline behind pending work
    /// drains the window and offers again; a decline with nothing admitted
    /// takes `advance_kv` with host operands, where Unsupported is an error.
    fn advance_kv_chained(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        cache: ResidentKvCache,
        inputs: &[Option<ChainedInput<'_>>; 3],
        output: KvOutput<'_>,
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        let _ = (instruction, plan, cache, inputs, output, minimum_accuracy);
        Ok(KernelProgress::Unsupported)
    }

    /// Read-only sizing of the next expert wave before its source loads any
    /// weights. See [`axnn_cpu::ResumableMoeDelegate::wave_capacity`]. The
    /// source may further reduce this answer to its residency budget. Returning
    /// a larger capacity does not admit work or permit partial publication.
    fn moe_wave_capacity(
        &self,
        params: axnn::lowering::MoeParams,
        tokens: usize,
        routes: &[axnn_cpu::MoeBatchRow],
    ) -> usize {
        let _ = (params, tokens, routes);
        axnn_cpu::DEFAULT_WAVE_EXPERTS
    }

    /// Run one wave of a MoE instruction's expert routes: the contract is
    /// [`axnn_cpu::ResumableMoeDelegate::experts`]'s. Every route of `routes`
    /// (grouped by expert, ascending) adds its weighted expert output to its
    /// token's row of `output`, in route order, onto what earlier waves left
    /// there; the wave's `n`th distinct expert is `experts[n]`, whose weights
    /// stay readable and unmoved until the wave completes. `(instruction,
    /// wave)` is stable across Pending calls. Admission must account for
    /// `minimum_accuracy`; Unsupported (before any Pending) runs the wave,
    /// and every later one of the instruction, natively.
    #[allow(clippy::too_many_arguments)]
    fn advance_moe_experts(
        &mut self,
        instruction: usize,
        wave: usize,
        plan: &CpuKernelPlan,
        params: axnn::lowering::MoeParams,
        hidden: &[f32],
        routes: &[axnn_cpu::MoeBatchRow],
        experts: &[axnn_cpu::ExpertWeights<'_>],
        output: &mut [f32],
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        let _ = (
            instruction,
            wave,
            plan,
            params,
            hidden,
            routes,
            experts,
            output,
            minimum_accuracy,
        );
        Ok(KernelProgress::Unsupported)
    }

    /// Offer a routed-expert instruction (`MoeExperts`) whole, under the
    /// window rules of [`advance_chained`](Self::advance_chained): retries
    /// repeat these arguments, there is no Complete while an earlier
    /// admission is pending, and an operand an earlier admission produces
    /// (the hidden rows `inputs[0]` or the router scores `inputs[1]`) arrives
    /// [`ChainedInput::Pending`]. `inputs[2]` is the correction bias. The
    /// result is the native routed-expert kernel's for these operands: each
    /// token routed by its scores, its picks run from `store` and mixed into
    /// its row of `output`, which is written whole.
    ///
    /// An executor that admits reads expert weights only where the store
    /// holds them in place for good ([`axnn_cpu::ExpertStoreRef::resident`]):
    /// no load happens for a routed admission, and the device may still read
    /// the weights after later instructions ran. The default declines; the
    /// engine then drains the window and runs the instruction's resumable
    /// phases, routing on the host and offering its waves through
    /// [`advance_moe_experts`](Self::advance_moe_experts).
    fn advance_moe_routed(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        store: axnn_cpu::ExpertStoreRef<'_>,
        output: &mut [f32],
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        let _ = (instruction, plan, inputs, store, output, minimum_accuracy);
        Ok(KernelProgress::Unsupported)
    }

    /// Whether the admitted final MoE wave exposes the instruction's complete
    /// output to `ChainedInput::Pending` consumers. Called only after the
    /// invocation establishes that this is its final, delegated Pending wave.
    /// False preserves serial completion. True keeps the invocation and its
    /// expert leases alive while later device consumers are admitted; it does
    /// not permit host reads or source reuse before normal wave retirement.
    fn moe_output_pending(&self, instruction: usize, wave: usize) -> bool {
        let _ = (instruction, wave);
        false
    }

    /// Retire a final MoE wave whose output has no remaining host observer.
    /// Like `retire_unobserved`, this preserves storage used by admitted
    /// consumers. Complete confirms device retirement; the engine then releases
    /// the invocation's expert leases. Unsupported requests normal publication.
    fn retire_moe_unobserved(
        &mut self,
        instruction: usize,
        wave: usize,
        routes: usize,
    ) -> Result<KernelProgress, Error> {
        let _ = (instruction, wave, routes);
        Ok(KernelProgress::Unsupported)
    }

    /// Start a graph invocation. Refuse if an earlier abandoned invocation
    /// still owns an admission or unretired event. Instruction numbers restart
    /// at zero only after that prior work has been drained or contained.
    fn begin(&mut self) -> Result<(), Error>;

    /// Drive retained service work or await a completion notification after
    /// Pending. The owner retains protocol phases and allocations if this
    /// future is dropped. Implementations supply bounded observation and
    /// lost-wakeup handling; this is not permission to spin or block a worker.
    /// No graph tensor is borrowed by this future.
    fn wait(&mut self) -> impl core::future::Future<Output = Result<(), Error>>
    where
        Self: Sized;

    fn advance(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<NativeRef<'_>>; 3],
        output: &mut [f32],
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error>;

    /// How many pure instructions this executor can hold admitted at once.
    /// One, the default, keeps strict one-at-a-time execution: an
    /// instruction completes before the next is offered, through `advance`.
    /// A larger window (clamped to [`MAX_ADMISSION_WINDOW`]) lets the engine
    /// offer the following pure instructions while earlier ones are still
    /// Pending, through [`advance_chained`](Self::advance_chained), so a
    /// device can run a run of them as one submission.
    fn admission_window(&self) -> usize {
        1
    }

    /// Retire an already admitted pure instruction whose output has no
    /// remaining host observer. Every consumer has already been admitted
    /// with a device dependency; the output is not a graph output and native
    /// reference checking is disabled. A fused admission uses its last index.
    ///
    /// Complete retires the admission without publishing its host output.
    /// Storage still needed by admitted consumers must remain alive until
    /// those consumers finish. Pending retains exactly that admission.
    /// Unsupported leaves it unchanged and requests ordinary copy-back.
    /// No resource update may be omitted through this hook.
    fn retire_unobserved(&mut self, instruction: usize) -> Result<KernelProgress, Error> {
        let _ = instruction;
        Ok(KernelProgress::Unsupported)
    }

    /// Offer `instruction` while earlier admissions of this invocation may
    /// still be pending. Used only when [`admission_window`] exceeds one.
    ///
    /// - An operand produced by a pending admission arrives as
    ///   [`ChainedInput::Pending`]; every other operand is final host data,
    ///   borrowed for this call only, exactly as in `advance`.
    /// - While any earlier admission is pending the executor answers Pending
    ///   (admitted) or Unsupported (nothing admitted), never Complete: outputs
    ///   are published strictly in program order.
    /// - Unsupported ends the lookahead. The engine drains the earlier
    ///   admissions, then offers this instruction again with host operands or
    ///   runs it natively.
    /// - The engine retries admissions strictly in admission order with the
    ///   same operand descriptions. A retry answers Pending until that
    ///   output is written, then Complete. The executor must therefore
    ///   complete admissions in the order it accepted them.
    ///
    /// The default forwards host-only offers to `advance` and refuses any
    /// pending operand.
    ///
    /// A windowed engine may first offer a [`FusedRun`] starting at the same
    /// instruction through [`advance_fused`](Self::advance_fused).
    ///
    /// [`admission_window`]: Self::admission_window
    fn advance_chained(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        output: &mut [f32],
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        let mut host = [None; 3];
        for (host, input) in host.iter_mut().zip(inputs) {
            match input {
                None => {}
                Some(ChainedInput::Host(value)) => *host = Some(*value),
                Some(ChainedInput::Pending { .. }) => return Ok(KernelProgress::Unsupported),
            }
        }
        self.advance(instruction, plan, &host, output, minimum_accuracy)
    }

    /// Offer a state-carrying instruction (a Mamba causal convolution or
    /// state-space scan, `UPDATES_RESOURCE`) under the window rules of
    /// `advance_chained`: retries repeat these arguments, and there is no
    /// Complete while an earlier admission is pending. On Complete both
    /// `output` and `state.target` hold the instruction's results. The
    /// default declines, and the instruction runs natively once the window
    /// has drained.
    fn advance_recurrent(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        state: RecurrentState<'_>,
        output: &mut [f32],
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        let _ = (instruction, plan, inputs, state, output, minimum_accuracy);
        Ok(KernelProgress::Unsupported)
    }

    /// Offer a fused run whose last instruction is `last`, under the same
    /// window rules as `advance_chained` (retries repeat these arguments;
    /// no Complete while an earlier admission is pending). `plan` and
    /// `inputs` are the run's first operation; `base` is the residual for
    /// [`FusedRun::Accumulate`], the weight for [`FusedRun::RmsNorm`], or
    /// `None` for [`FusedRun::SquaredRelu`]. `output` is the last instruction's output, the only one
    /// published. An operand produced by a pending admission, `base`
    /// included, arrives as [`ChainedInput::Pending`]. The default declines,
    /// and the engine then offers the run's instructions one by one.
    #[allow(clippy::too_many_arguments)]
    fn advance_fused(
        &mut self,
        last: usize,
        run: FusedRun,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        base: Option<ChainedInput<'_>>,
        output: &mut [f32],
        minimum_accuracy: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        let _ = (last, run, plan, inputs, base, output, minimum_accuracy);
        Ok(KernelProgress::Unsupported)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedProgress {
    /// The runtime should wait for its executor's completion notification.
    Pending,
    /// The instruction budget was consumed; more graph work remains.
    Progress {
        instructions: usize,
    },
    Complete,
}

/// An admitted operand's producer: `elements` values at `offset` into the
/// output of pending instruction `producer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChainSource {
    pub(crate) producer: usize,
    pub(crate) offset: usize,
    pub(crate) elements: usize,
}

/// One instruction held admitted by the executor, with the operand sources
/// it was offered with (a retry repeats them exactly).
#[derive(Clone, Copy)]
struct Admitted {
    /// The instruction whose output this admission publishes: for a fused
    /// run, its last one.
    instruction: usize,
    /// A fused run's first instruction and kind; `None` for one instruction.
    fused: Option<(usize, FusedRun)>,
    /// Final MoE wave; retry through its retained invocation, not a pure offer.
    moe: bool,
    sources: [Option<ChainSource>; 3],
    /// An accumulate run's base, when a pending admission produces it.
    base: Option<ChainSource>,
    started: Option<u64>,
}

impl Admitted {
    const EMPTY: Self = Self {
        instruction: usize::MAX,
        fused: None,
        moe: false,
        sources: [None; 3],
        base: None,
        started: None,
    };

    /// Instructions this admission retires.
    fn span(&self) -> usize {
        self.fused
            .map_or(1, |(first, _)| self.instruction + 1 - first)
    }
}

/// Borrows the reusable arena and ingress for the whole graph run. Construction
/// validates all resource bindings before any instruction executes. No payload
/// allocation, hidden executor, blocking wait, or device discovery is added.
#[must_use = "advance to completion before observing graph outputs"]
pub struct PreparedInvocation<'a, A: Allocator, E: PreparedKernelExecutor> {
    arena: &'a mut PreparedNative<A>,
    inputs: &'a [NativeRef<'a>],
    resources: &'a [NativeResourceBinding<'a>],
    parallel: &'a dyn ForkJoin,
    executor: &'a mut E,
    minimum_accuracy: KernelAccuracy,
    cursor: usize,
    pending: bool,
    failed: bool,
    started: Option<u64>,
    moe: Option<axnn_cpu::MoeInvocation<'a>>,
    /// The executor's admission window, clamped; one keeps the strict
    /// one-instruction path below unchanged.
    window: usize,
    /// Admitted instructions (including a final MoE wave) awaiting completion. All
    /// precede `cursor`; completion (and retirement) follows this order.
    admitted: [Admitted; MAX_ADMISSION_WINDOW],
    admitted_len: usize,
}

impl<A: Allocator> PreparedNative<A> {
    /// Begin cooperative execution with an explicit offload accuracy policy.
    /// Unsupported instructions retain their originally selected CPU plans.
    pub fn begin_with_executor<'a, E: PreparedKernelExecutor>(
        &'a mut self,
        inputs: &'a [NativeRef<'a>],
        resources: &'a [NativeResourceBinding<'a>],
        parallel: &'a dyn ForkJoin,
        executor: &'a mut E,
        minimum_accuracy: KernelAccuracy,
    ) -> Result<PreparedInvocation<'a, A, E>, Error> {
        self.inner.begin_native(inputs, resources)?;
        executor.begin()?;
        let window = executor.admission_window().clamp(1, MAX_ADMISSION_WINDOW);
        Ok(PreparedInvocation {
            arena: self,
            inputs,
            resources,
            parallel,
            executor,
            minimum_accuracy,
            cursor: 0,
            pending: false,
            failed: false,
            started: None,
            moe: None,
            window,
            admitted: [Admitted::EMPTY; MAX_ADMISSION_WINDOW],
            admitted_len: 0,
        })
    }
}

impl<A: Allocator, E: PreparedKernelExecutor> PreparedInvocation<'_, A, E> {
    /// Allow the device owner to progress an admitted instruction. Cancellation
    /// leaves both this cursor and the executor's retained protocol state live.
    pub async fn wait(&mut self) -> Result<(), Error> {
        if self.failed || (!self.pending && self.admitted_len == 0) {
            return Err(Error::InvocationFinished);
        }
        if let Err(error) = self.executor.wait().await {
            self.failed = true;
            return Err(error);
        }
        Ok(())
    }

    /// Execute at most `budget` instructions. Pending performs one provider
    /// observation and returns immediately; this method never retries a wait.
    /// Errors invalidate graph outputs and prohibit resuming partial state.
    pub fn advance(&mut self, budget: usize) -> Result<PreparedProgress, Error> {
        if self.failed {
            return Err(Error::InvocationFinished);
        }
        if budget == 0 {
            return Err(Error::InvalidStepBudget);
        }
        if self.window > 1 {
            let result = self.advance_windowed(budget);
            if result.is_err() {
                self.failed = true;
            }
            return result;
        }
        let mut executed = 0;
        while self.cursor < self.arena.inner.nodes.len() && executed < budget {
            if !self.pending {
                self.started = self.arena.inner.profile.as_ref().map(|(clock, _)| clock());
            }
            let progress = self.arena.inner.run_native_node(
                self.cursor,
                self.inputs,
                self.resources,
                self.parallel,
                &mut None,
                Some(&mut *self.executor),
                self.pending,
                self.minimum_accuracy,
                &mut self.moe,
                crate::native::CpuExecution::Planned,
            );
            match progress {
                Ok(KernelProgress::Pending) => {
                    self.pending = true;
                    return Ok(PreparedProgress::Pending);
                }
                Ok(KernelProgress::Complete) => {
                    self.arena
                        .inner
                        .profile_node(self.cursor, self.started.take());
                    self.pending = false;
                    self.cursor += 1;
                    executed += 1;
                }
                result => {
                    self.failed = true;
                    return Err(result
                        .err()
                        .unwrap_or(Error::Plan(crate::PlanError::InvalidPartition)));
                }
            }
        }
        if self.cursor == self.arena.inner.nodes.len() {
            self.arena.inner.complete = true;
            Ok(PreparedProgress::Complete)
        } else {
            Ok(PreparedProgress::Progress {
                instructions: executed,
            })
        }
    }

    /// The windowed engine (`window > 1`). Each pass retires the oldest
    /// admission if it completed, admits the next pure instruction while the
    /// window has room, and runs any other instruction natively only when
    /// nothing is admitted. Every host-visible effect keeps program order:
    /// outputs publish in admission order, operands produced by a pending
    /// admission are offered as references to it (never as stale host
    /// bytes), and no native instruction runs while an admission is pending.
    fn advance_windowed(&mut self, budget: usize) -> Result<PreparedProgress, Error> {
        let nodes = self.arena.inner.nodes.len();
        let mut executed = 0;
        loop {
            if self.admitted_len > 0 {
                let front = self.admitted[0];
                let unpublished = if !front.moe
                    && self
                        .arena
                        .inner
                        .can_retire_unobserved(front.instruction, self.cursor)
                {
                    self.executor.retire_unobserved(front.instruction)?
                } else {
                    KernelProgress::Unsupported
                };
                let moe_unpublished = if front.moe
                    && self
                        .arena
                        .inner
                        .can_retire_moe_unobserved(front.instruction, self.cursor)
                {
                    let moe = self.moe.as_ref().ok_or(Error::InvocationFinished)?;
                    let wave = moe.pending_final_wave().ok_or(Error::InvocationFinished)?;
                    self.executor.retire_moe_unobserved(
                        front.instruction,
                        wave,
                        moe.pending_wave_routes(),
                    )?
                } else {
                    KernelProgress::Unsupported
                };
                let progress = if moe_unpublished != KernelProgress::Unsupported {
                    if moe_unpublished == KernelProgress::Complete {
                        self.arena
                            .inner
                            .retire_moe_unobserved(front.instruction, &mut self.moe)?;
                    }
                    moe_unpublished
                } else if front.moe {
                    self.arena.inner.run_native_node(
                        front.instruction,
                        self.inputs,
                        self.resources,
                        self.parallel,
                        &mut None,
                        Some(&mut *self.executor),
                        true,
                        self.minimum_accuracy,
                        &mut self.moe,
                        crate::native::CpuExecution::Planned,
                    )?
                } else if unpublished != KernelProgress::Unsupported {
                    unpublished
                } else {
                    match front.fused {
                        Some((first, run)) => self.arena.inner.offer_fused(
                            first,
                            front.instruction,
                            run,
                            self.inputs,
                            &mut *self.executor,
                            &front.sources,
                            front.base,
                            self.minimum_accuracy,
                        )?,
                        None => self.arena.inner.offer_chained(
                            front.instruction,
                            self.inputs,
                            self.resources,
                            &mut *self.executor,
                            &front.sources,
                            self.minimum_accuracy,
                        )?,
                    }
                };
                match progress {
                    KernelProgress::Complete => {
                        let first = front.fused.map_or(front.instruction, |(first, _)| first);
                        // Native MoE retirement already publishes/checks its
                        // output and retires inputs. Pure admissions need that
                        // bookkeeping here.
                        if !front.moe {
                            self.arena.inner.complete_delegated(
                                first,
                                front.instruction,
                                self.inputs,
                                self.resources,
                                self.parallel,
                                self.minimum_accuracy,
                            )?;
                        }
                        self.arena
                            .inner
                            .profile_node(front.instruction, front.started);
                        self.admitted.copy_within(1..self.admitted_len, 0);
                        self.admitted_len -= 1;
                        executed += front.span();
                        if executed >= budget {
                            return Ok(self.windowed_progress(executed));
                        }
                        continue;
                    }
                    KernelProgress::Pending => {}
                    KernelProgress::Unsupported => {
                        return Err(crate::PlanError::InvalidPartition.into());
                    }
                }
            }
            if !self.pending
                && self.cursor < nodes
                && self.admitted_len < self.window
                && let Some((run, last, base_slot)) = self.arena.inner.fused_run(self.cursor)
            {
                let admitted = self.admitted[..self.admitted_len]
                    .iter()
                    .map(|admitted| admitted.instruction);
                let sources = self
                    .arena
                    .inner
                    .chain_sources(self.cursor, admitted.clone());
                let base =
                    base_slot.and_then(|slot| self.arena.inner.chain_source(last, slot, admitted));
                let started = self.arena.inner.profile.as_ref().map(|(clock, _)| clock());
                match self.arena.inner.offer_fused(
                    self.cursor,
                    last,
                    run,
                    self.inputs,
                    &mut *self.executor,
                    &sources,
                    base,
                    self.minimum_accuracy,
                )? {
                    KernelProgress::Pending => {
                        self.admitted[self.admitted_len] = Admitted {
                            instruction: last,
                            fused: Some((self.cursor, run)),
                            moe: false,
                            sources,
                            base,
                            started,
                        };
                        self.admitted_len += 1;
                        self.cursor = last + 1;
                        continue;
                    }
                    KernelProgress::Complete if self.admitted_len == 0 => {
                        self.arena.inner.complete_delegated(
                            self.cursor,
                            last,
                            self.inputs,
                            self.resources,
                            self.parallel,
                            self.minimum_accuracy,
                        )?;
                        self.arena.inner.profile_node(last, started);
                        executed += last + 1 - self.cursor;
                        self.cursor = last + 1;
                        if executed >= budget {
                            return Ok(self.windowed_progress(executed));
                        }
                        continue;
                    }
                    KernelProgress::Complete => {
                        return Err(crate::PlanError::InvalidPartition.into());
                    }
                    // Declined: offer the run's first instruction alone.
                    KernelProgress::Unsupported => {}
                }
            }
            // A routed-expert instruction, or a KV append or read over a
            // resident cache, offered whole: admitted, its operands may stay
            // on the device. Declined behind pending work, the window drains
            // and it is offered again; declined with nothing admitted, it
            // takes the strict path below (MoE's resumable phases routing on
            // the host, a resident cache's `advance_kv`).
            if !self.pending
                && self.cursor < nodes
                && self.admitted_len < self.window
                && (self.arena.inner.routable_moe(self.cursor)
                    || self.arena.inner.routable_kv(self.cursor, self.resources))
            {
                let sources = self.arena.inner.chain_sources(
                    self.cursor,
                    self.admitted[..self.admitted_len]
                        .iter()
                        .map(|admitted| admitted.instruction),
                );
                let started = self.arena.inner.profile.as_ref().map(|(clock, _)| clock());
                match self.arena.inner.offer_chained(
                    self.cursor,
                    self.inputs,
                    self.resources,
                    &mut *self.executor,
                    &sources,
                    self.minimum_accuracy,
                )? {
                    KernelProgress::Pending => {
                        self.admitted[self.admitted_len] = Admitted {
                            instruction: self.cursor,
                            fused: None,
                            moe: false,
                            sources,
                            base: None,
                            started,
                        };
                        self.admitted_len += 1;
                        self.cursor += 1;
                        continue;
                    }
                    KernelProgress::Complete if self.admitted_len == 0 => {
                        self.arena.inner.complete_delegated(
                            self.cursor,
                            self.cursor,
                            self.inputs,
                            self.resources,
                            self.parallel,
                            self.minimum_accuracy,
                        )?;
                        self.arena.inner.profile_node(self.cursor, started);
                        self.cursor += 1;
                        executed += 1;
                        if executed >= budget {
                            return Ok(self.windowed_progress(executed));
                        }
                        continue;
                    }
                    KernelProgress::Complete => {
                        return Err(crate::PlanError::InvalidPartition.into());
                    }
                    KernelProgress::Unsupported if self.admitted_len > 0 => {
                        return Ok(PreparedProgress::Pending);
                    }
                    KernelProgress::Unsupported => {}
                }
            }
            if !self.pending
                && self.cursor < nodes
                && self.admitted_len < self.window
                && self.arena.inner.admissible(self.cursor)
            {
                let sources = self.arena.inner.chain_sources(
                    self.cursor,
                    self.admitted[..self.admitted_len]
                        .iter()
                        .map(|admitted| admitted.instruction),
                );
                let started = self.arena.inner.profile.as_ref().map(|(clock, _)| clock());
                match self.arena.inner.offer_chained(
                    self.cursor,
                    self.inputs,
                    self.resources,
                    &mut *self.executor,
                    &sources,
                    self.minimum_accuracy,
                )? {
                    KernelProgress::Pending => {
                        self.admitted[self.admitted_len] = Admitted {
                            instruction: self.cursor,
                            fused: None,
                            moe: false,
                            sources,
                            base: None,
                            started,
                        };
                        self.admitted_len += 1;
                        self.cursor += 1;
                        continue;
                    }
                    // Synchronous completion is in order only with nothing
                    // admitted ahead of it.
                    KernelProgress::Complete if self.admitted_len == 0 => {
                        self.arena.inner.complete_delegated(
                            self.cursor,
                            self.cursor,
                            self.inputs,
                            self.resources,
                            self.parallel,
                            self.minimum_accuracy,
                        )?;
                        self.arena.inner.profile_node(self.cursor, started);
                        self.cursor += 1;
                        executed += 1;
                        if executed >= budget {
                            return Ok(self.windowed_progress(executed));
                        }
                        continue;
                    }
                    KernelProgress::Complete => {
                        return Err(crate::PlanError::InvalidPartition.into());
                    }
                    // Declined behind pending work: drain it first, then this
                    // instruction is offered again with final host operands.
                    KernelProgress::Unsupported if self.admitted_len > 0 => {
                        return Ok(PreparedProgress::Pending);
                    }
                    KernelProgress::Unsupported => {
                        self.arena.inner.run_declined(
                            self.cursor,
                            self.inputs,
                            self.resources,
                            self.parallel,
                            self.minimum_accuracy,
                            crate::native::CpuExecution::Fallback,
                        )?;
                        self.arena.inner.profile_node(self.cursor, started);
                        self.cursor += 1;
                        executed += 1;
                        if executed >= budget {
                            return Ok(self.windowed_progress(executed));
                        }
                        continue;
                    }
                }
            }
            if self.admitted_len > 0 {
                return Ok(PreparedProgress::Pending);
            }
            if self.cursor == nodes {
                self.arena.inner.complete = true;
                return Ok(PreparedProgress::Complete);
            }
            // Nothing admitted and the next instruction cannot join a window
            // (a resource effect, MoE, or several results): the strict path,
            // including MoE's own resumable phases.
            if !self.pending {
                self.started = self.arena.inner.profile.as_ref().map(|(clock, _)| clock());
            }
            match self.arena.inner.run_native_node(
                self.cursor,
                self.inputs,
                self.resources,
                self.parallel,
                &mut None,
                Some(&mut *self.executor),
                self.pending,
                self.minimum_accuracy,
                &mut self.moe,
                crate::native::CpuExecution::Planned,
            )? {
                KernelProgress::Pending => {
                    if let Some(wave) = self.moe.as_ref().and_then(|moe| moe.pending_final_wave())
                        && self.executor.moe_output_pending(self.cursor, wave)
                    {
                        // The serial path above is entered only with no older
                        // admissions. Hold its resource invocation until this
                        // front admission retires; no subsequent native/source
                        // operation runs while any admission remains pending.
                        self.admitted[0] = Admitted {
                            instruction: self.cursor,
                            moe: true,
                            started: self.started.take(),
                            ..Admitted::EMPTY
                        };
                        self.admitted_len = 1;
                        self.pending = false;
                        self.cursor += 1;
                        continue;
                    }
                    self.pending = true;
                    return Ok(PreparedProgress::Pending);
                }
                KernelProgress::Complete => {
                    self.arena
                        .inner
                        .profile_node(self.cursor, self.started.take());
                    self.pending = false;
                    self.cursor += 1;
                    executed += 1;
                    if executed >= budget {
                        return Ok(self.windowed_progress(executed));
                    }
                }
                KernelProgress::Unsupported => {
                    return Err(crate::PlanError::InvalidPartition.into());
                }
            }
        }
    }

    /// The windowed engine's answer after its budget ran out: complete only
    /// when every instruction ran and nothing is still admitted.
    fn windowed_progress(&mut self, executed: usize) -> PreparedProgress {
        if self.cursor == self.arena.inner.nodes.len() && self.admitted_len == 0 {
            self.arena.inner.complete = true;
            PreparedProgress::Complete
        } else {
            PreparedProgress::Progress {
                instructions: executed,
            }
        }
    }
}
