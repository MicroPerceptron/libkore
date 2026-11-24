//! Resumable routing and expert residency over the native MoE kernel.
//!
//! An invocation routes every token once, groups the routes by expert, and
//! runs them in waves: each wave's distinct experts are loaded together and
//! offered to the delegate as one call, which projects every route through
//! its expert and adds the weighted results to the output. A delegate that
//! declines a wave leaves it, and every later one, to the native kernel. Each
//! token accumulates its experts in ascending order either way.
use super::*;

/// One selected token row, ordered by expert then token in grouped execution.
/// The fields are private because the native router owns selection and mixing.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeBatchRow {
    expert: u32,
    token: u32,
    mixture: f32,
}

impl MoeBatchRow {
    /// One route: `expert` for `token`, weighted `mixture`.
    pub const fn new(expert: u32, token: u32, mixture: f32) -> Self {
        Self {
            expert,
            token,
            mixture,
        }
    }

    pub fn token(self) -> usize {
        self.token as usize
    }

    /// The routed expert.
    pub fn expert(self) -> u32 {
        self.expert
    }

    /// The route's mixture weight.
    pub fn mixture(self) -> f32 {
        self.mixture
    }
}

// Route records occupy the same named F32 scratch arena as native math.
// These representation checks justify the exclusive scratch view below.
const ROUTE_WORDS: usize = size_of::<MoeBatchRow>() / size_of::<f32>();
const _: () = assert!(size_of::<MoeBatchRow>() == 3 * size_of::<f32>());
const _: () = assert!(align_of::<MoeBatchRow>() == align_of::<f32>());

/// Compatibility policy for delegates that do not advertise a wave capacity.
pub const DEFAULT_WAVE_EXPERTS: usize = 32;

/// Bounded inline metadata capacity, not a GPU thread count. At 128 entries
/// the retained weight views occupy 9 KiB on a 64-bit host. Larger layers are
/// split into waves without allocating in the execution path. A delegate must
/// opt in beyond [`DEFAULT_WAVE_EXPERTS`]; the source's residency budget and
/// the delegate's scratch/command budget can both lower the actual wave size.
pub const MAX_WAVE_EXPERTS: usize = 128;

/// Scratch an invocation needs: every route's record, one token's
/// contribution vector, and the native kernel's scratch for one route.
pub fn moe_prepared_scratch_elements(params: MoeParams, tokens: usize) -> Option<usize> {
    if !params.is_valid() || tokens == 0 || tokens > u32::MAX as usize {
        return None;
    }
    let routes = tokens.checked_mul(params.top_k as usize)?;
    let elements = routes
        .checked_mul(ROUTE_WORDS)?
        .checked_add(params.hidden as usize)?
        .checked_add(moe_scratch_elements(params))?;
    (elements.checked_mul(size_of::<f32>())? <= isize::MAX as usize).then_some(elements)
}

/// Result of a call-scoped expert delegate observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeProgress {
    /// No work or output was admitted. Native execution takes this wave.
    Unsupported,
    /// The wave is running; offer the same wave again later.
    Pending,
    /// The wave's contributions are in the output.
    Complete,
}

/// Where a wave of expert routes may run instead of the native kernel.
///
/// Every slice is borrowed only during the call, even on `Pending`: a device
/// executor retains its own allocations and keeps no pointers into them,
/// except the loaded expert weights, which the invocation keeps readable and
/// unmoved (no next `load`) until the wave completes. `Unsupported` is legal
/// only before the wave first returns `Pending`. Errors end the invocation;
/// the runtime still owns device retirement.
pub trait ResumableMoeDelegate<E> {
    /// Maximum distinct experts to load for the next wave. `routes` contains
    /// the remaining routes in expert/token order, before any weights are
    /// loaded. Use their geometry to budget device scratch and submissions.
    /// The engine clamps the answer to `1..=MAX_WAVE_EXPERTS` and the source's
    /// `max_picks`. This is a read-only hint, not admission: it must not retain
    /// tensor views or modify output. Called only before a new load, never
    /// while a wave is Pending; the admitted wave stays unchanged on retries.
    fn wave_capacity(&self, params: MoeParams, tokens: usize, routes: &[MoeBatchRow]) -> usize {
        let _ = (params, tokens, routes);
        DEFAULT_WAVE_EXPERTS
    }

    /// Run wave `wave` (its first route's index: stable across retries):
    /// for every route of `routes` (grouped by expert, ascending, tokens
    /// ascending within an expert), add
    /// `mixture * down_scale * down(sqrelu(up_scale * up(hidden[token])))`
    /// of its expert to `output[token]`. The wave's `n`th distinct expert is
    /// `experts[n]`. Each token's contributions are added in route order
    /// onto what the output already holds (earlier waves' routes).
    /// `Unsupported`, the default, runs the wave natively.
    #[allow(clippy::too_many_arguments)]
    fn experts(
        &mut self,
        wave: usize,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        routes: &[MoeBatchRow],
        experts: &[ExpertWeights<'_>],
        output: &mut [f32],
    ) -> Result<MoeProgress, E> {
        let _ = (wave, plan, params, hidden, routes, experts, output);
        Ok(MoeProgress::Unsupported)
    }
}

/// Where an invocation's host time went, when a clock is set
/// ([`MoeInvocation::with_clock`]): nanoseconds by phase, in
/// [`MOE_PHASE_NAMES`] order. `load` includes the source's own reads (a
/// streaming source's demand reads); `device` is the time inside delegate
/// calls, not the device's run, which falls between `advance` calls and is
/// no phase's; `native` is the waves the native kernel ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoePhases {
    pub ns: [u64; MOE_PHASES],
    /// Successfully completed waves and routes, counted once at completion,
    /// never for repeated Pending observations or declined offers.
    pub delegated_waves: u64,
    pub delegated_routes: u64,
    pub native_waves: u64,
    pub native_routes: u64,
    /// Invocations that switched to native waves after a delegated wave
    /// completed. A wave already Pending can never make this transition.
    pub mixed_invocations: u64,
}

pub const MOE_PHASES: usize = 4;
pub const MOE_PHASE_NAMES: [&str; MOE_PHASES] = ["route", "load", "device", "native"];

impl MoePhases {
    pub fn merge(&mut self, other: &Self) {
        for (total, ns) in self.ns.iter_mut().zip(other.ns) {
            *total += ns;
        }
        self.delegated_waves += other.delegated_waves;
        self.delegated_routes += other.delegated_routes;
        self.native_waves += other.native_waves;
        self.native_routes += other.native_routes;
        self.mixed_invocations += other.mixed_invocations;
    }

    pub fn since(&self, before: &Self) -> Self {
        Self {
            ns: core::array::from_fn(|i| self.ns[i].saturating_sub(before.ns[i])),
            delegated_waves: self.delegated_waves.saturating_sub(before.delegated_waves),
            delegated_routes: self
                .delegated_routes
                .saturating_sub(before.delegated_routes),
            native_waves: self.native_waves.saturating_sub(before.native_waves),
            native_routes: self.native_routes.saturating_sub(before.native_routes),
            mixed_invocations: self
                .mixed_invocations
                .saturating_sub(before.mixed_invocations),
        }
    }
}

#[derive(Clone, Copy)]
enum MoePhase {
    Route,
    Load,
    Device,
    Native,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Route every token and group the routes by expert.
    Route,
    /// Load the next wave's experts.
    Load,
    /// Offer the loaded wave to the delegate, or run it natively.
    Run,
    Finished,
    Failed,
}

/// One graph invocation's MoE cursor. No tensor pointers or heap allocations
/// are retained. Weight views come from the invocation's borrowed expert
/// store and are discarded before its next load.
///
/// The caller must keep tensor values and scratch unchanged between advances,
/// except for the delegate's output writes. It must not invoke another kernel
/// using the same stateful source until completion or abandonment. Abandonment
/// does not cancel a runtime's independently owned accelerator work.
#[must_use = "drive to completion before observing expert output"]
pub struct MoeInvocation<'s> {
    plan: CpuKernelPlan,
    params: MoeParams,
    store: ExpertStoreRef<'s>,
    tokens: usize,
    stage: Stage,
    /// Routes `start..end` form the current wave, of `picks` experts.
    start: usize,
    end: usize,
    picks: usize,
    /// The current wave returned `Pending`: it may no longer decline.
    pending: bool,
    /// The delegate declined a wave: every later wave runs natively.
    native: bool,
    wave_chosen: [ExpertWeights<'s>; MAX_WAVE_EXPERTS],
    clock: Option<fn() -> u64>,
    phases: MoePhases,
}

impl<'s> MoeInvocation<'s> {
    /// A delegated final wave whose weights are still held by this invocation.
    /// Its output may be offered as a device dependency only if the delegate
    /// explicitly supports that handoff. This is not completion: no source
    /// load, scratch reuse, or host observation is permitted until retirement.
    pub fn pending_final_wave(&self) -> Option<usize> {
        (self.stage == Stage::Run
            && self.pending
            && !self.native
            && self.end == self.tokens * self.params.top_k as usize)
            .then_some(self.start)
    }

    /// Retire the final delegated wave after its owner reports completion
    /// without host publication. The caller must have admitted every remaining
    /// output consumer on the device; source leases are released only here.
    pub fn retire_final_wave(&mut self) -> Result<(), CpuError> {
        if self.pending_final_wave().is_none() {
            return Err(CpuError::ResourceUnavailable {
                op: "no final pending expert wave",
            });
        }
        self.phases.delegated_waves += 1;
        self.phases.delegated_routes += (self.end - self.start) as u64;
        self.wave_chosen.fill(ExpertWeights::EMPTY);
        self.stage = Stage::Finished;
        Ok(())
    }

    /// Routes held by the current wave (metadata, not permission to read output).
    pub fn pending_wave_routes(&self) -> usize {
        self.end - self.start
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plan: &CpuKernelPlan,
        store: ExpertStoreRef<'s>,
        hidden: &[f32],
        scores: &[f32],
        correction_bias: &[f32],
        output: &[f32],
        scratch: &[f32],
    ) -> Result<Self, CpuError> {
        let params = validate_invocation(
            plan,
            store,
            hidden,
            scores,
            correction_bias,
            output,
            scratch,
        )?;
        let tokens = hidden.len() / params.hidden as usize;
        if scratch.len()
            < moe_prepared_scratch_elements(params, tokens).ok_or(CpuError::ShapeMismatch {
                op: "expert wave scratch",
            })?
        {
            return Err(CpuError::ShapeMismatch {
                op: "expert wave scratch",
            });
        }
        Ok(Self {
            plan: *plan,
            params,
            store,
            tokens,
            stage: Stage::Route,
            start: 0,
            end: 0,
            picks: 0,
            pending: false,
            native: false,
            wave_chosen: [ExpertWeights::EMPTY; MAX_WAVE_EXPERTS],
            clock: None,
            phases: MoePhases::default(),
        })
    }

    /// Account this invocation's host time by phase ([`MoePhases`]), read
    /// from `clock` (nanoseconds, monotonic) at each phase boundary.
    pub fn with_clock(mut self, clock: fn() -> u64) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Host time (zero without a clock) and completed-wave placement counts.
    pub fn phases(&self) -> MoePhases {
        self.phases
    }

    fn now(&self) -> Option<u64> {
        self.clock.map(|clock| clock())
    }

    fn charge(&mut self, phase: MoePhase, start: Option<u64>) {
        if let (Some(clock), Some(start)) = (self.clock, start) {
            self.phases.ns[phase as usize] += clock().saturating_sub(start);
        }
    }

    /// Advance until completion or one pending wave. Never waits or retries a
    /// pending delegate within this call. Errors are terminal, so a changed
    /// argument or device refusal cannot replay output.
    #[allow(clippy::too_many_arguments)]
    pub fn advance<E: From<CpuError>>(
        &mut self,
        hidden: &[f32],
        scores: &[f32],
        correction_bias: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
        parallel: &dyn ForkJoin,
        delegate: &mut dyn ResumableMoeDelegate<E>,
    ) -> Result<MoeProgress, E> {
        let result = self.advance_inner(
            hidden,
            scores,
            correction_bias,
            output,
            scratch,
            parallel,
            delegate,
        );
        if result.is_err() {
            self.stage = Stage::Failed;
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_inner<E: From<CpuError>>(
        &mut self,
        hidden: &[f32],
        scores: &[f32],
        correction_bias: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
        parallel: &dyn ForkJoin,
        delegate: &mut dyn ResumableMoeDelegate<E>,
    ) -> Result<MoeProgress, E> {
        let invalid = || CpuError::ShapeMismatch {
            op: "expert wave dispatch",
        };
        let params = self.params;
        let (width, k) = (params.hidden as usize, params.top_k as usize);
        let experts = params.experts as usize;
        if matches!(self.stage, Stage::Finished | Stage::Failed) {
            return Err(CpuError::ResourceUnavailable {
                op: "finished expert invocation",
            }
            .into());
        }
        let count = self.tokens * k;
        let required = moe_prepared_scratch_elements(params, self.tokens).ok_or_else(invalid)?;
        if hidden.len() != self.tokens * width
            || output.len() != hidden.len()
            || scores.len() != self.tokens * experts
            || correction_bias.len() != experts
            || scratch.len() < required
        {
            return Err(invalid().into());
        }
        let (route_words, scratch) = scratch.split_at_mut(count * ROUTE_WORDS);
        // SAFETY: exclusive, initialized F32 storage has identical alignment
        // and exactly enough bytes. u32/f32 fields admit every bit pattern;
        // no references to the F32 region coexist with this temporary view.
        // No route pointer survives this call or is offered to device DMA.
        let routes = unsafe {
            core::slice::from_raw_parts_mut(route_words.as_mut_ptr().cast::<MoeBatchRow>(), count)
        };
        let (contribution, native) = scratch.split_at_mut(width);
        let source_capacity = match self.store.experts {
            Experts::Table(_) => MAX_WAVE_EXPERTS,
            Experts::Source(source) => source.max_picks().unwrap_or(k).clamp(1, MAX_WAVE_EXPERTS),
        };
        loop {
            match self.stage {
                Stage::Route => {
                    let start = self.now();
                    for (token, row) in scores.chunks_exact(experts).enumerate() {
                        let route = Route::new(row, correction_bias, k, params.normalize);
                        for (entry, &slot) in routes[token * k..(token + 1) * k]
                            .iter_mut()
                            .zip(&route.order[..k])
                        {
                            *entry = MoeBatchRow {
                                expert: route.picked[slot] as u32,
                                token: token as u32,
                                mixture: route.weights[slot] * params.routed_scale.to_f32(),
                            };
                        }
                    }
                    // In-place O(R log R), R = tokens * top_k. Expert order
                    // keeps every token's accumulation order ascending.
                    routes.sort_unstable_by_key(|r| (r.expert, r.token));
                    if let Experts::Source(source) = self.store.experts {
                        let mut previous = None;
                        for row in routes.iter() {
                            if previous != Some(row.expert) {
                                source.will_load(&[row.expert]);
                                previous = Some(row.expert);
                            }
                        }
                    }
                    output.fill(0.0);
                    self.charge(MoePhase::Route, start);
                    self.stage = Stage::Load;
                }
                Stage::Load => {
                    // Discard every source borrow before the next load may evict.
                    self.wave_chosen.fill(ExpertWeights::EMPTY);
                    self.start = self.end;
                    if self.start == count {
                        self.stage = Stage::Finished;
                        return Ok(MoeProgress::Complete);
                    }
                    let capacity = source_capacity.min(if self.native {
                        DEFAULT_WAVE_EXPERTS
                    } else {
                        delegate
                            .wave_capacity(params, self.tokens, &routes[self.start..])
                            .clamp(1, MAX_WAVE_EXPERTS)
                    });
                    let mut picked = [0u32; MAX_WAVE_EXPERTS];
                    let (mut end, mut picks) = (self.start, 0);
                    while end < count && picks < capacity {
                        let expert = routes[end].expert;
                        picked[picks] = expert;
                        end += routes[end..].partition_point(|r| r.expert == expert);
                        picks += 1;
                    }
                    let start = self.now();
                    match self.store.experts {
                        Experts::Table(table) => {
                            for (weights, &expert) in
                                self.wave_chosen.iter_mut().zip(&picked[..picks])
                            {
                                *weights = table[expert as usize];
                            }
                        }
                        Experts::Source(source) => {
                            let first = routes[self.start].token();
                            source.load(
                                &picked[..picks],
                                &hidden[first * width..(first + 1) * width],
                                &mut self.wave_chosen[..picks],
                            )?;
                        }
                    }
                    if self.wave_chosen[..picks]
                        .iter()
                        .any(|expert| !expert.fits(params))
                    {
                        return Err(invalid().into());
                    }
                    if let Some(residency) = self.store.residency {
                        for expert in &self.wave_chosen[..picks] {
                            residency.will_read(expert.up);
                            residency.will_read(expert.down);
                        }
                    }
                    self.charge(MoePhase::Load, start);
                    self.end = end;
                    self.picks = picks;
                    self.pending = false;
                    self.stage = Stage::Run;
                }
                Stage::Run => {
                    let wave = &routes[self.start..self.end];
                    if !self.native {
                        let start = self.now();
                        let progress = delegate.experts(
                            self.start,
                            &self.plan,
                            params,
                            hidden,
                            wave,
                            &self.wave_chosen[..self.picks],
                            output,
                        )?;
                        self.charge(MoePhase::Device, start);
                        match progress {
                            MoeProgress::Pending => {
                                self.pending = true;
                                return Ok(MoeProgress::Pending);
                            }
                            MoeProgress::Complete => {
                                self.phases.delegated_waves += 1;
                                self.phases.delegated_routes += wave.len() as u64;
                                self.stage = Stage::Load;
                                continue;
                            }
                            MoeProgress::Unsupported if self.pending => {
                                return Err(invalid().into());
                            }
                            MoeProgress::Unsupported => {
                                self.native = true;
                                self.phases.mixed_invocations =
                                    u64::from(self.phases.delegated_waves != 0);
                            }
                        }
                    }
                    let start = self.now();
                    native_wave(
                        &self.plan,
                        params,
                        hidden,
                        wave,
                        &self.wave_chosen[..self.picks],
                        output,
                        contribution,
                        native,
                        parallel,
                    )?;
                    self.charge(MoePhase::Native, start);
                    self.phases.native_waves += 1;
                    self.phases.native_routes += wave.len() as u64;
                    self.stage = Stage::Load;
                }
                Stage::Finished | Stage::Failed => unreachable!("checked above"),
            }
        }
    }
}

/// The native kernel over one wave, as [`ResumableMoeDelegate::experts`]
/// describes it: each route's weighted expert output added to its token's
/// row of `output`, in route order. `contribution` holds `params.hidden`
/// values and `scratch` [`moe_scratch_elements`]; a delegate's checker
/// recomputes a wave with it.
#[allow(clippy::too_many_arguments)]
pub fn native_wave(
    plan: &CpuKernelPlan,
    params: MoeParams,
    hidden: &[f32],
    wave: &[MoeBatchRow],
    chosen: &[ExpertWeights<'_>],
    output: &mut [f32],
    contribution: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let width = params.hidden as usize;
    let mut pick = 0;
    for (index, route) in wave.iter().enumerate() {
        if index > 0 && route.expert != wave[index - 1].expert {
            pick += 1;
        }
        let token = route.token();
        invoke_loaded_experts(
            plan,
            params,
            &hidden[token * width..(token + 1) * width],
            &chosen[pick..pick + 1],
            &[route.mixture],
            contribution,
            scratch,
            parallel,
            false,
        )?;
        for (out, &value) in output[token * width..(token + 1) * width]
            .iter_mut()
            .zip(contribution.iter())
        {
            *out += value;
        }
    }
    Ok(())
}
