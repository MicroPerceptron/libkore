//! `axtosa` — a portable TOSA 1.0 graph executor over the `axnn-cpu`
//! kernel set.
//!
//! The CPU backend of the virtio-accel / kore accelerator ecosystem:
//! artifacts are verified and planned by `virtio-accel-tosa` (untrusted
//! FlatBuffers → validated, topologically-ordered execution plan), and
//! each operator dispatches onto axnn-cpu kernels. `no_std + alloc`,
//! so the same executor can serve host adapters and kore itself.
//!
//! ## Support boundary
//!
//! Static, single-block TOSA 1.0 graphs. Floating-point operators are
//! computed in an f32 plane: FP16/F8 operands widen exactly, results
//! narrow with round-to-nearest-even, so single-op cases are correctly
//! rounded. **B8/F8 (OCP E4M3/E5M2) are first-class**: identity, casts to
//! FP16/FP32 through the SIMD widen kernels, and MATMUL over F8
//! operands (accumulated in f32 — a precision superset of the spec's
//! FP16 accumulator, within the TOSA FP profile's accuracy bounds).
//! INT8 covers identity, saturating RNE casts from FP16/FP32, and
//! zero-point-aware MATMUL. Unsupported operators are rejected at load, never
//! silently approximated.

#![no_std]
#![feature(allocator_api)]
#![feature(f16)]

extern crate alloc;

mod ax_lower;

pub use ax_lower::{AxLowerError, lower_ax_in};

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use virtio_accel_tosa::{
    AnalyzedValueKind, DType, ExtensionSet, Level, NanPropagationMode, Op, OpAttributes,
    OperatorId, ProfileSet, Target, ValueId as TosaValueId, Version, parse,
};

/// Executor failure.
#[derive(Debug)]
pub enum ExecError {
    /// Artifact failed structural parsing.
    Parse(virtio_accel_tosa::Error),
    /// Artifact failed semantic validation / analysis for the target.
    Analysis(virtio_accel_tosa::AnalysisError),
    /// Graph shape the executor does not handle (multi-region, control
    /// flow, variables, dynamic shapes).
    Shape(&'static str),
    /// Operator (or dtype combination) outside the support boundary.
    Unsupported(Op),
    /// Input binding count or byte length mismatch.
    Binding {
        slot: usize,
        expected: usize,
        actual: usize,
    },
}

impl From<virtio_accel_tosa::Error> for ExecError {
    fn from(e: virtio_accel_tosa::Error) -> Self {
        ExecError::Parse(e)
    }
}
impl From<virtio_accel_tosa::AnalysisError> for ExecError {
    fn from(e: virtio_accel_tosa::AnalysisError) -> Self {
        ExecError::Analysis(e)
    }
}

/// The target this executor admits.
pub fn target() -> Target {
    Target::new(
        Version::TOSA_1_0,
        ProfileSet::ALL,
        Level::Level8K,
        ExtensionSet::FP8E4M3
            .union(ExtensionSet::FP8E5M2)
            .union(ExtensionSet::INT4),
    )
}

/// Provider-owned execution plan for one validated static TOSA block.
///
/// Compilation copies the small amount of metadata the executor needs out of
/// the borrowed FlatBuffer view. The artifact itself is therefore no longer
/// parsed, semantically analyzed, or searched on warm submissions. Constants,
/// shapes, use counts, operator operands, and the topological order are all
/// immutable program state; invocation-local tensor storage remains separate.
#[derive(Debug)]
pub struct Program {
    values: Vec<ValuePlan>,
    operators: Vec<OperatorPlan>,
    order: Vec<Instruction>,
    inputs: Vec<PlanValue>,
    outputs: Vec<PlanValue>,
    takes: Vec<u32>,
    constant_folds: usize,
}

/// Static optimizer decisions recorded in a compiled [`Program`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramStats {
    /// Source TOSA operators admitted by the program.
    pub operators: usize,
    /// Warm-path instructions after safe graph fusion.
    pub instructions: usize,
    /// Scalar `MUL` followed by single-consumer `ADD` regions fused into one
    /// strict SIMD kernel.
    pub scale_add_fusions: usize,
    /// Full-plane `MUL` followed by single-consumer `ADD` regions fused into
    /// one strict SIMD kernel. This is the common premultiplied-compositor
    /// path.
    pub mul_add_fusions: usize,
    /// Scalar-target `SUB` → `MUL` → `ADD` interpolation regions fused into
    /// one strict SIMD kernel.
    pub lerp_fusions: usize,
    /// Two-axis linearized translation followed by lower/upper clamp regions
    /// fused into one strict SIMD plane pass.
    pub linearized_clamp_fusions: usize,
    /// `MAXIMUM(x, floor)` → `RSQRT` → `x * inverse` guarded square-root
    /// regions evaluated in one plane pass.
    pub guarded_sqrt_fusions: usize,
    /// Source operators removed from warm execution after their constant
    /// subgraphs were evaluated at provider load time.
    pub constant_folds: usize,
}

#[derive(Clone, Copy, Debug)]
enum Instruction {
    Operator(OperatorId),
    /// A scalar multiplication consumed exactly once by an ADD. The
    /// intermediate has no observation boundary, so the executor can keep it
    /// in SIMD registers while preserving separate multiply/add rounding.
    ScaleAdd {
        input: PlanValue,
        scale: PlanValue,
        addend: PlanValue,
        output: PlanValue,
        order: axnn_cpu::elementwise::ScaleAddOrder,
    },
    /// A full-plane multiplication consumed exactly once by an ADD. The
    /// product stays in SIMD registers instead of occupying a framebuffer-
    /// sized temporary.
    MulAdd {
        left: PlanValue,
        right: PlanValue,
        addend: PlanValue,
        output: PlanValue,
        order: axnn_cpu::elementwise::MulAddOrder,
    },
    /// `current + (target - current) * weight`, with scalar `target` and
    /// full-size current/weight planes.
    Lerp {
        current: PlanValue,
        target: PlanValue,
        weight: PlanValue,
        output: PlanValue,
    },
    /// `clamp(value + derivative_x * offset_x + derivative_y * offset_y,
    /// lower, upper)` with scalar offsets and clamp limits.
    LinearizedClamp {
        value: PlanValue,
        derivative_x: PlanValue,
        offset_x: PlanValue,
        derivative_y: PlanValue,
        offset_y: PlanValue,
        lower: PlanValue,
        upper: PlanValue,
        output: PlanValue,
    },
    /// `input * rsqrt(maximum(input, floor))` with a scalar floor.
    GuardedSqrt {
        input: PlanValue,
        floor: PlanValue,
        output: PlanValue,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PlanValue(u32);

impl From<TosaValueId> for PlanValue {
    fn from(value: TosaValueId) -> Self {
        Self(value.get())
    }
}

#[derive(Debug)]
struct ValuePlan {
    kind: ValueKind,
    byte_size: usize,
    constant: Option<ConstantValue>,
}

/// Provider-resident constant in its last observed representation.
///
/// Folded FP16 operations stay on the F32 compute plane until a byte-typed
/// consumer or block output observes them, matching warm execution's
/// deliberate accumulator-precision contract.
#[derive(Clone, Debug)]
enum ConstantValue {
    Bytes(Vec<u8>),
    Plane(Vec<f32>),
}

#[derive(Debug)]
enum ValueKind {
    Tensor { dtype: DType, dims: ShapeDims },
    Shape,
}

const MAX_LEVEL_8K_RANK: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ShapeDims {
    values: [usize; MAX_LEVEL_8K_RANK],
    rank: u8,
}

impl ShapeDims {
    fn copy_from(dimensions: impl Iterator<Item = i32>) -> Result<Self, ExecError> {
        let mut values = [0; MAX_LEVEL_8K_RANK];
        let mut rank = 0;
        for dimension in dimensions {
            let Some(slot) = values.get_mut(rank) else {
                return Err(ExecError::Shape("tensor rank exceeds Level 8K"));
            };
            *slot = dimension.max(0) as usize;
            rank += 1;
        }
        Ok(Self {
            values,
            rank: rank as u8,
        })
    }

    fn as_slice(&self) -> &[usize] {
        &self.values[..self.rank as usize]
    }

    fn elements(self) -> usize {
        self.as_slice().iter().product()
    }
}

impl Deref for ShapeDims {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[derive(Debug)]
struct OperatorPlan {
    op: Op,
    inputs: [PlanValue; 4],
    input_count: u8,
    output: PlanValue,
    attributes: OwnedAttributes,
}

#[derive(Debug)]
enum OwnedAttributes {
    None,
    Clamp {
        min_val: Vec<u8>,
        max_val: Vec<u8>,
    },
    MaxPool2d {
        kernel: Vec<i32>,
        stride: Vec<i32>,
        pad: Vec<i32>,
    },
    Reduce {
        axis: i32,
    },
    ArgMax {
        axis: i32,
    },
    NanMode(NanPropagationMode),
}

impl OwnedAttributes {
    fn copy_from(attributes: OpAttributes<'_>) -> Self {
        match attributes {
            OpAttributes::Clamp {
                min_val, max_val, ..
            } => Self::Clamp {
                min_val: min_val.to_vec(),
                max_val: max_val.to_vec(),
            },
            OpAttributes::MaxPool2d {
                kernel,
                stride,
                pad,
                ..
            } => Self::MaxPool2d {
                kernel: kernel.iter().collect(),
                stride: stride.iter().collect(),
                pad: pad.iter().collect(),
            },
            OpAttributes::ReduceSum { axis } | OpAttributes::ReduceMax { axis, .. } => {
                Self::Reduce { axis }
            }
            OpAttributes::ArgMax { axis, .. } => Self::ArgMax { axis },
            OpAttributes::Maximum { nan_mode } | OpAttributes::Minimum { nan_mode } => {
                Self::NanMode(nan_mode)
            }
            _ => Self::None,
        }
    }
}

fn visit_instruction_inputs(
    instruction: Instruction,
    operators: &[OperatorPlan],
    mut visit: impl FnMut(PlanValue, u32) -> Result<(), ExecError>,
) -> Result<(), ExecError> {
    match instruction {
        Instruction::Operator(id) => {
            let operator = operators
                .get(id.get() as usize)
                .ok_or(ExecError::Shape("operator id outside compiled program"))?;
            for &input in &operator.inputs[..operator.input_count as usize] {
                visit(input, 1)?;
            }
        }
        Instruction::ScaleAdd {
            input,
            scale,
            addend,
            ..
        } => {
            visit(input, 1)?;
            visit(scale, 1)?;
            visit(addend, 1)?;
        }
        Instruction::MulAdd {
            left,
            right,
            addend,
            ..
        } => {
            visit(left, 1)?;
            visit(right, 1)?;
            visit(addend, 1)?;
        }
        Instruction::Lerp {
            current,
            target,
            weight,
            ..
        } => {
            // The source SUB and ADD both observe `current`.
            visit(current, 2)?;
            visit(target, 1)?;
            visit(weight, 1)?;
        }
        Instruction::LinearizedClamp {
            value,
            derivative_x,
            offset_x,
            derivative_y,
            offset_y,
            lower,
            upper,
            ..
        } => {
            visit(value, 1)?;
            visit(derivative_x, 1)?;
            visit(offset_x, 1)?;
            visit(derivative_y, 1)?;
            visit(offset_y, 1)?;
            visit(lower, 1)?;
            visit(upper, 1)?;
        }
        Instruction::GuardedSqrt { input, floor, .. } => {
            // MAXIMUM and the final MUL both observe `input`.
            visit(input, 2)?;
            visit(floor, 1)?;
        }
    }
    Ok(())
}

fn instruction_output(
    instruction: Instruction,
    operators: &[OperatorPlan],
) -> Result<PlanValue, ExecError> {
    match instruction {
        Instruction::Operator(id) => operators
            .get(id.get() as usize)
            .map(|operator| operator.output)
            .ok_or(ExecError::Shape("operator id outside compiled program")),
        Instruction::ScaleAdd { output, .. }
        | Instruction::MulAdd { output, .. }
        | Instruction::Lerp { output, .. }
        | Instruction::LinearizedClamp { output, .. }
        | Instruction::GuardedSqrt { output, .. } => Ok(output),
    }
}

const fn instruction_source_operators(instruction: Instruction) -> usize {
    match instruction {
        Instruction::Operator(_) => 1,
        Instruction::ScaleAdd { .. } | Instruction::MulAdd { .. } => 2,
        Instruction::Lerp { .. } => 3,
        Instruction::LinearizedClamp { .. } => 6,
        Instruction::GuardedSqrt { .. } => 3,
    }
}

impl Program {
    /// Validate and compile a static single-block artifact once.
    pub fn compile(artifact: &[u8]) -> Result<Self, ExecError> {
        let model = parse(artifact)?;
        let analysis = model.analyze_for(target())?;

        if analysis.regions().len() != 1 || analysis.blocks().len() != 1 {
            return Err(ExecError::Shape("multi-region/multi-block graphs"));
        }
        let block = analysis.blocks()[0].id();

        let mut values = Vec::with_capacity(analysis.values().len());
        for value in analysis.values() {
            let kind = match value.kind() {
                AnalyzedValueKind::Tensor(tensor) => ValueKind::Tensor {
                    dtype: tensor.dtype(),
                    dims: ShapeDims::copy_from(tensor.dimensions())?,
                },
                AnalyzedValueKind::Shape(_) => ValueKind::Shape,
            };
            let byte_size = value
                .byte_size()
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or(ExecError::Shape("dynamic or unsized value"))?;
            values.push(ValuePlan {
                kind,
                byte_size,
                constant: analysis
                    .serialized_constant(value.id())
                    .map(<[u8]>::to_vec)
                    .map(ConstantValue::Bytes),
            });
        }

        let mut operators = Vec::with_capacity(analysis.operators().len());
        for operator in analysis.operators() {
            let op = operator.op();
            if !supported(op) {
                return Err(ExecError::Unsupported(op));
            }
            let source_inputs = analysis.operator_inputs(operator.id());
            let source_outputs = analysis.operator_outputs(operator.id());
            if source_inputs.len() > 4 || source_outputs.len() != 1 {
                return Err(ExecError::Shape("unsupported operator arity"));
            }
            let mut inputs = [PlanValue::default(); 4];
            for (target, &source) in inputs.iter_mut().zip(source_inputs) {
                *target = source.into();
            }
            operators.push(OperatorPlan {
                op,
                inputs,
                input_count: source_inputs.len() as u8,
                output: source_outputs[0].into(),
                attributes: OwnedAttributes::copy_from(operator.source().attributes()),
            });
        }

        let mut takes = vec![0u32; values.len()];
        for &operator in analysis.execution_order(block) {
            for &value in analysis.operator_inputs(operator) {
                takes[value.get() as usize] += 1;
            }
        }
        for &value in analysis.block_outputs(block) {
            takes[value.get() as usize] += 1;
        }

        let source_order = analysis.execution_order(block);

        // Discover fusions from SSA producer/consumer relationships rather
        // than adjacency in one valid topological order. Independent nodes
        // are routinely interleaved between a MUL and its sole ADD consumer;
        // requiring adjacency materialized an otherwise unobservable tensor.
        // Delaying the MUL read to the ADD is safe because `takes` still keeps
        // every producer input live until the fused instruction consumes it.
        let mut producers = vec![None; values.len()];
        for &operator_id in source_order {
            let operator = &operators[operator_id.get() as usize];
            producers[operator.output.0 as usize] = Some(operator_id);
        }
        let mut fusion_at = vec![None; operators.len()];
        let mut fused_producer = vec![false; operators.len()];
        for &consumer_id in source_order {
            let consumer = &operators[consumer_id.get() as usize];
            if let Some((instruction, fused)) =
                linearized_clamp_instruction(consumer_id, &operators, &values, &takes, &producers)
            {
                fusion_at[consumer_id.get() as usize] = Some(instruction);
                for producer in fused {
                    fused_producer[producer.get() as usize] = true;
                }
                continue;
            }
            if let Some((instruction, maximum_id, rsqrt_id)) =
                guarded_sqrt_instruction(consumer_id, &operators, &values, &takes, &producers)
            {
                fusion_at[consumer_id.get() as usize] = Some(instruction);
                fused_producer[maximum_id.get() as usize] = true;
                fused_producer[rsqrt_id.get() as usize] = true;
                continue;
            }
            if consumer.op != Op::ADD {
                continue;
            }
            if let Some((instruction, sub_id, mul_id)) =
                lerp_instruction(consumer_id, &operators, &values, &takes, &producers)
            {
                fusion_at[consumer_id.get() as usize] = Some(instruction);
                fused_producer[sub_id.get() as usize] = true;
                fused_producer[mul_id.get() as usize] = true;
                continue;
            }
            for &input in &consumer.inputs[..consumer.input_count as usize] {
                let Some(producer_id) = producers[input.0 as usize] else {
                    continue;
                };
                if let Some(instruction) =
                    scale_add_instruction(producer_id, consumer_id, &operators, &values, &takes)
                {
                    fusion_at[consumer_id.get() as usize] = Some(instruction);
                    fused_producer[producer_id.get() as usize] = true;
                    break;
                }
                if let Some(instruction) =
                    mul_add_instruction(producer_id, consumer_id, &operators, &values, &takes)
                {
                    fusion_at[consumer_id.get() as usize] = Some(instruction);
                    fused_producer[producer_id.get() as usize] = true;
                    break;
                }
            }
        }

        let mut order = Vec::with_capacity(source_order.len());
        for &operator_id in source_order {
            let index = operator_id.get() as usize;
            if fused_producer[index] {
                continue;
            }
            order.push(fusion_at[index].unwrap_or(Instruction::Operator(operator_id)));
        }

        let mut program = Self {
            values,
            operators,
            order,
            inputs: analysis
                .block_inputs(block)
                .iter()
                .copied()
                .map(PlanValue::from)
                .collect(),
            outputs: analysis
                .block_outputs(block)
                .iter()
                .copied()
                .map(PlanValue::from)
                .collect(),
            takes,
            constant_folds: 0,
        };
        program.fold_constants()?;
        Ok(program)
    }

    /// Evaluate input-independent instructions once, retain only constants
    /// that cross into warm execution, and rebuild live-use counts for the
    /// resulting program. This is the provider analogue of exact-resolution
    /// static scene preparation, but it applies to every admitted TOSA graph.
    fn fold_constants(&mut self) -> Result<(), ExecError> {
        let source_order = core::mem::take(&mut self.order);
        let mut retained = Vec::with_capacity(source_order.len());
        let mut constant_ready: Vec<bool> = self
            .values
            .iter()
            .map(|value| value.constant.is_some())
            .collect();
        // One streaming execution state covers the whole constant region.
        // Keeping folded intermediates in its value slots lets the usual
        // consume-aware executor move them directly into their sole consumer;
        // the old per-instruction state cloned the full value/take tables and
        // then cloned every framebuffer-sized intermediate back out again.
        let mut fold_bufs = vec![None; self.values.len()];
        let mut fold_takes = self.takes.clone();
        for instruction in source_order {
            let mut is_constant = true;
            visit_instruction_inputs(instruction, &self.operators, |input, _| {
                let ready = constant_ready
                    .get(input.0 as usize)
                    .ok_or(ExecError::Shape("value id outside compiled program"))?;
                is_constant &= *ready;
                Ok(())
            })?;
            if !is_constant {
                retained.push(instruction);
                continue;
            }

            let output = instruction_output(instruction, &self.operators)?;
            let output_index = output.0 as usize;
            let ready = constant_ready
                .get(output_index)
                .copied()
                .ok_or(ExecError::Shape("value id outside compiled program"))?;
            if !ready {
                let mut execution = Exec {
                    program: self,
                    bufs: fold_bufs,
                    takes: fold_takes,
                };
                execution.dispatch(instruction)?;
                if execution.bufs[output_index].is_none() {
                    return Err(ExecError::Shape("constant fold produced no value"));
                }
                fold_bufs = execution.bufs;
                fold_takes = execution.takes;
                constant_ready[output_index] = true;
            }
            self.constant_folds = self
                .constant_folds
                .saturating_add(instruction_source_operators(instruction));
        }

        // Results that still have a warm instruction or block-output use are
        // present in the streaming slots because the original use counts did
        // not allow their last constant consumer to move them away. Promote
        // only those survivors into provider-resident constants; dead folded
        // intermediates have already been consumed and need no second copy.
        for (index, slot) in fold_bufs.iter_mut().enumerate() {
            if self.values[index].constant.is_some() {
                continue;
            }
            let Some(value) = slot.take() else {
                continue;
            };
            self.values[index].constant = Some(match value {
                Val::Plane(values) => ConstantValue::Plane(values),
                Val::Bytes(bytes) => ConstantValue::Bytes(bytes),
                Val::Borrowed(bytes) => ConstantValue::Bytes(bytes.to_vec()),
            });
        }
        self.order = retained;
        self.rebuild_takes()?;
        self.prune_dead_constants()?;
        Ok(())
    }

    fn rebuild_takes(&mut self) -> Result<(), ExecError> {
        self.takes.fill(0);
        for &instruction in &self.order {
            visit_instruction_inputs(instruction, &self.operators, |input, uses| {
                let take = self
                    .takes
                    .get_mut(input.0 as usize)
                    .ok_or(ExecError::Shape("value id outside compiled program"))?;
                *take = take
                    .checked_add(uses)
                    .ok_or(ExecError::Shape("value use count overflow"))?;
                Ok(())
            })?;
        }
        for &output in &self.outputs {
            let take = self
                .takes
                .get_mut(output.0 as usize)
                .ok_or(ExecError::Shape("value id outside compiled program"))?;
            *take = take
                .checked_add(1)
                .ok_or(ExecError::Shape("value use count overflow"))?;
        }
        Ok(())
    }

    fn prune_dead_constants(&mut self) -> Result<(), ExecError> {
        let mut needed = vec![false; self.values.len()];
        for &instruction in &self.order {
            visit_instruction_inputs(instruction, &self.operators, |input, _| {
                let slot = needed
                    .get_mut(input.0 as usize)
                    .ok_or(ExecError::Shape("value id outside compiled program"))?;
                *slot = true;
                Ok(())
            })?;
        }
        for &output in &self.outputs {
            let slot = needed
                .get_mut(output.0 as usize)
                .ok_or(ExecError::Shape("value id outside compiled program"))?;
            *slot = true;
        }
        for (value, needed) in self.values.iter_mut().zip(needed) {
            if !needed {
                value.constant = None;
            }
        }
        Ok(())
    }

    /// Number of block inputs in binding order.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Number of block outputs in binding order.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Return load-time optimizer telemetry without exposing plan internals.
    pub fn stats(&self) -> ProgramStats {
        ProgramStats {
            operators: self.operators.len(),
            instructions: self.order.len(),
            scale_add_fusions: self
                .order
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::ScaleAdd { .. }))
                .count(),
            mul_add_fusions: self
                .order
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::MulAdd { .. }))
                .count(),
            lerp_fusions: self
                .order
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::Lerp { .. }))
                .count(),
            linearized_clamp_fusions: self
                .order
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::LinearizedClamp { .. }))
                .count(),
            guarded_sqrt_fusions: self
                .order
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::GuardedSqrt { .. }))
                .count(),
            constant_folds: self.constant_folds,
        }
    }

    /// Execute this already-compiled plan.
    pub fn execute(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, ExecError> {
        let mut exec = self.bind(inputs)?;

        for &instruction in &self.order {
            exec.dispatch(instruction)?;
        }

        let mut outputs = Vec::with_capacity(self.outputs.len());
        for &value in &self.outputs {
            outputs.push(exec.take(value)?);
        }
        Ok(outputs)
    }

    /// Execute directly against caller-provided output bindings.
    ///
    /// Inputs remain borrowed for the invocation and outputs are written into
    /// their final backing. This is the provider-facing path: it avoids an
    /// input clone on admission and avoids allocating or returning a second
    /// set of output byte planes.
    pub fn execute_into(
        &self,
        inputs: &[&[u8]],
        outputs: &mut [&mut [u8]],
    ) -> Result<(), ExecError> {
        if outputs.len() != self.outputs.len() {
            return Err(ExecError::Binding {
                slot: usize::MAX,
                expected: self.outputs.len(),
                actual: outputs.len(),
            });
        }
        let mut exec = self.bind(inputs)?;
        for &instruction in &self.order {
            exec.dispatch(instruction)?;
        }
        for (slot, (&value, output)) in self.outputs.iter().zip(outputs).enumerate() {
            let expected = exec.byte_size(value)?;
            if output.len() != expected {
                return Err(ExecError::Binding {
                    slot: self.inputs.len() + slot,
                    expected,
                    actual: output.len(),
                });
            }
            exec.write_bytes(value, output)?;
        }
        Ok(())
    }

    fn bind<'program, 'input>(
        &'program self,
        inputs: &[&'input [u8]],
    ) -> Result<Exec<'program, 'input>, ExecError> {
        if self.inputs.len() != inputs.len() {
            return Err(ExecError::Binding {
                slot: usize::MAX,
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }

        let mut exec = Exec {
            program: self,
            bufs: vec![None; self.values.len()],
            takes: self.takes.clone(),
        };
        for (slot, (&value, bytes)) in self.inputs.iter().zip(inputs).enumerate() {
            let expected = exec.byte_size(value)?;
            if bytes.len() != expected {
                return Err(ExecError::Binding {
                    slot,
                    expected,
                    actual: bytes.len(),
                });
            }
            exec.bufs[value.0 as usize] = Some(Val::Borrowed(bytes));
        }
        Ok(exec)
    }
}

fn tensor_info(value: &ValuePlan) -> Option<(DType, ShapeDims)> {
    match value.kind {
        ValueKind::Tensor { dtype, dims } => Some((dtype, dims)),
        ValueKind::Shape => None,
    }
}

fn linearized_clamp_instruction(
    minimum_id: OperatorId,
    operators: &[OperatorPlan],
    values: &[ValuePlan],
    takes: &[u32],
    producers: &[Option<OperatorId>],
) -> Option<(Instruction, [OperatorId; 5])> {
    let minimum = operators.get(minimum_id.get() as usize)?;
    if minimum.op != Op::MINIMUM
        || minimum.input_count != 2
        || !matches!(
            &minimum.attributes,
            OwnedAttributes::NanMode(mode) if *mode == NanPropagationMode::PROPAGATE
        )
    {
        return None;
    }
    // The graphics DSL emits exact left-to-right arithmetic so fusion keeps
    // every IEEE rounding and NaN boundary: MIN(MAX((value + dx*ox) +
    // dy*oy, lower), upper).
    let lower_clamped = minimum.inputs[0];
    let upper = minimum.inputs[1];
    let maximum_id = producers.get(lower_clamped.0 as usize).copied().flatten()?;
    let maximum = operators.get(maximum_id.get() as usize)?;
    if maximum.op != Op::MAXIMUM
        || maximum.input_count != 2
        || takes.get(lower_clamped.0 as usize).copied()? != 1
        || !matches!(
            &maximum.attributes,
            OwnedAttributes::NanMode(mode) if *mode == NanPropagationMode::PROPAGATE
        )
    {
        return None;
    }
    let translated = maximum.inputs[0];
    let lower = maximum.inputs[1];
    let add_y_id = producers.get(translated.0 as usize).copied().flatten()?;
    let add_y = operators.get(add_y_id.get() as usize)?;
    if add_y.op != Op::ADD
        || add_y.input_count != 2
        || takes.get(translated.0 as usize).copied()? != 1
    {
        return None;
    }
    let translated_x = add_y.inputs[0];
    let y_product = add_y.inputs[1];
    let add_x_id = producers.get(translated_x.0 as usize).copied().flatten()?;
    let mul_y_id = producers.get(y_product.0 as usize).copied().flatten()?;
    let add_x = operators.get(add_x_id.get() as usize)?;
    let mul_y = operators.get(mul_y_id.get() as usize)?;
    if add_x.op != Op::ADD
        || add_x.input_count != 2
        || mul_y.op != Op::MUL
        || mul_y.input_count < 2
        || takes.get(translated_x.0 as usize).copied()? != 1
        || takes.get(y_product.0 as usize).copied()? != 1
    {
        return None;
    }
    let value = add_x.inputs[0];
    let x_product = add_x.inputs[1];
    let mul_x_id = producers.get(x_product.0 as usize).copied().flatten()?;
    let mul_x = operators.get(mul_x_id.get() as usize)?;
    if mul_x.op != Op::MUL
        || mul_x.input_count < 2
        || takes.get(x_product.0 as usize).copied()? != 1
    {
        return None;
    }
    let derivative_x = mul_x.inputs[0];
    let offset_x = mul_x.inputs[1];
    let derivative_y = mul_y.inputs[0];
    let offset_y = mul_y.inputs[1];
    let (output_dtype, output_dims) = tensor_info(values.get(minimum.output.0 as usize)?)?;
    if !matches!(output_dtype, DType::FP16 | DType::FP32) {
        return None;
    }
    for full in [
        value,
        derivative_x,
        derivative_y,
        x_product,
        translated_x,
        y_product,
        translated,
        lower_clamped,
    ] {
        let (dtype, dims) = tensor_info(values.get(full.0 as usize)?)?;
        if dtype != output_dtype || dims != output_dims {
            return None;
        }
    }
    for scalar in [offset_x, offset_y, lower, upper] {
        let (dtype, dims) = tensor_info(values.get(scalar.0 as usize)?)?;
        if dtype != output_dtype || dims.elements() != 1 {
            return None;
        }
    }
    Some((
        Instruction::LinearizedClamp {
            value,
            derivative_x,
            offset_x,
            derivative_y,
            offset_y,
            lower,
            upper,
            output: minimum.output,
        },
        [maximum_id, add_y_id, add_x_id, mul_y_id, mul_x_id],
    ))
}

fn guarded_sqrt_instruction(
    mul_id: OperatorId,
    operators: &[OperatorPlan],
    values: &[ValuePlan],
    takes: &[u32],
    producers: &[Option<OperatorId>],
) -> Option<(Instruction, OperatorId, OperatorId)> {
    let mul = operators.get(mul_id.get() as usize)?;
    if mul.op != Op::MUL || mul.input_count < 2 {
        return None;
    }
    for inverse_slot in 0..2 {
        let inverse = mul.inputs[inverse_slot];
        let input = mul.inputs[1 - inverse_slot];
        let Some(rsqrt_id) = producers.get(inverse.0 as usize).copied().flatten() else {
            continue;
        };
        let Some(rsqrt) = operators.get(rsqrt_id.get() as usize) else {
            continue;
        };
        if rsqrt.op != Op::RSQRT
            || rsqrt.input_count != 1
            || takes.get(inverse.0 as usize).copied()? != 1
        {
            continue;
        }
        let guarded = rsqrt.inputs[0];
        let Some(maximum_id) = producers.get(guarded.0 as usize).copied().flatten() else {
            continue;
        };
        let Some(maximum) = operators.get(maximum_id.get() as usize) else {
            continue;
        };
        if maximum.op != Op::MAXIMUM
            || maximum.input_count != 2
            || takes.get(guarded.0 as usize).copied()? != 1
            || !matches!(
                &maximum.attributes,
                OwnedAttributes::NanMode(mode) if *mode == NanPropagationMode::PROPAGATE
            )
        {
            continue;
        }
        let floor = if maximum.inputs[0] == input {
            maximum.inputs[1]
        } else if maximum.inputs[1] == input {
            maximum.inputs[0]
        } else {
            continue;
        };
        let (input_dtype, input_dims) = tensor_info(values.get(input.0 as usize)?)?;
        let (floor_dtype, floor_dims) = tensor_info(values.get(floor.0 as usize)?)?;
        let (guarded_dtype, guarded_dims) = tensor_info(values.get(guarded.0 as usize)?)?;
        let (inverse_dtype, inverse_dims) = tensor_info(values.get(inverse.0 as usize)?)?;
        let (output_dtype, output_dims) = tensor_info(values.get(mul.output.0 as usize)?)?;
        if !matches!(input_dtype, DType::FP16 | DType::FP32)
            || floor_dtype != input_dtype
            || guarded_dtype != input_dtype
            || inverse_dtype != input_dtype
            || output_dtype != input_dtype
            || floor_dims.elements() != 1
            || guarded_dims != input_dims
            || inverse_dims != input_dims
            || output_dims != input_dims
        {
            continue;
        }
        return Some((
            Instruction::GuardedSqrt {
                input,
                floor,
                output: mul.output,
            },
            maximum_id,
            rsqrt_id,
        ));
    }
    None
}

fn scale_add_instruction(
    mul_id: OperatorId,
    add_id: OperatorId,
    operators: &[OperatorPlan],
    values: &[ValuePlan],
    takes: &[u32],
) -> Option<Instruction> {
    let mul = operators.get(mul_id.get() as usize)?;
    let add = operators.get(add_id.get() as usize)?;
    if mul.op != Op::MUL || add.op != Op::ADD || mul.input_count < 2 || add.input_count != 2 {
        return None;
    }
    let intermediate = mul.output;
    if takes.get(intermediate.0 as usize).copied()? != 1 {
        return None;
    }
    let product_left = if add.inputs[0] == intermediate {
        true
    } else if add.inputs[1] == intermediate {
        false
    } else {
        return None;
    };
    let addend = if product_left {
        add.inputs[1]
    } else {
        add.inputs[0]
    };
    let (mul_dtype, mul_dims) = tensor_info(values.get(intermediate.0 as usize)?)?;
    let (out_dtype, out_dims) = tensor_info(values.get(add.output.0 as usize)?)?;
    if !matches!(mul_dtype, DType::FP16 | DType::FP32)
        || !matches!(out_dtype, DType::FP16 | DType::FP32)
        || mul_dims != out_dims
    {
        return None;
    }

    let lhs = mul.inputs[0];
    let rhs = mul.inputs[1];
    let (_, lhs_dims) = tensor_info(values.get(lhs.0 as usize)?)?;
    let (_, rhs_dims) = tensor_info(values.get(rhs.0 as usize)?)?;
    let (input, scale, scale_left) = if lhs_dims == out_dims && rhs_dims.elements() == 1 {
        (lhs, rhs, false)
    } else if rhs_dims == out_dims && lhs_dims.elements() == 1 {
        (rhs, lhs, true)
    } else {
        return None;
    };
    let (_, addend_dims) = tensor_info(values.get(addend.0 as usize)?)?;
    if addend_dims != out_dims && addend_dims.elements() != 1 {
        return None;
    }

    let order = match (product_left, scale_left) {
        (true, false) => axnn_cpu::elementwise::ScaleAddOrder::InputScaleAddend,
        (true, true) => axnn_cpu::elementwise::ScaleAddOrder::ScaleInputAddend,
        (false, false) => axnn_cpu::elementwise::ScaleAddOrder::AddendInputScale,
        (false, true) => axnn_cpu::elementwise::ScaleAddOrder::AddendScaleInput,
    };
    Some(Instruction::ScaleAdd {
        input,
        scale,
        addend,
        output: add.output,
        order,
    })
}

fn mul_add_instruction(
    mul_id: OperatorId,
    add_id: OperatorId,
    operators: &[OperatorPlan],
    values: &[ValuePlan],
    takes: &[u32],
) -> Option<Instruction> {
    let mul = operators.get(mul_id.get() as usize)?;
    let add = operators.get(add_id.get() as usize)?;
    // TOSA 1.0 carries the validated integer shift as a third MUL input even
    // for floating-point arithmetic. The executor, like the existing scalar
    // scale/add fusion above, consumes the two numeric operands here.
    if mul.op != Op::MUL || add.op != Op::ADD || mul.input_count < 2 || add.input_count != 2 {
        return None;
    }
    let intermediate = mul.output;
    if takes.get(intermediate.0 as usize).copied()? != 1 {
        return None;
    }
    let (addend, order) = if add.inputs[0] == intermediate {
        (
            add.inputs[1],
            axnn_cpu::elementwise::MulAddOrder::ProductAddend,
        )
    } else if add.inputs[1] == intermediate {
        (
            add.inputs[0],
            axnn_cpu::elementwise::MulAddOrder::AddendProduct,
        )
    } else {
        return None;
    };
    let left = mul.inputs[0];
    let right = mul.inputs[1];
    // Keeping the warm executor's ownership accounting simple is worth more
    // than fusing the unusual aliased form. Ordinary compositor regions use
    // three distinct planes.
    if left == right || left == addend || right == addend {
        return None;
    }
    let (output_dtype, output_dims) = tensor_info(values.get(add.output.0 as usize)?)?;
    if !matches!(output_dtype, DType::FP16 | DType::FP32) {
        return None;
    }
    for value in [left, right, addend, intermediate] {
        let (dtype, dims) = tensor_info(values.get(value.0 as usize)?)?;
        if dtype != output_dtype || dims != output_dims {
            return None;
        }
    }
    Some(Instruction::MulAdd {
        left,
        right,
        addend,
        output: add.output,
        order,
    })
}

fn lerp_instruction(
    add_id: OperatorId,
    operators: &[OperatorPlan],
    values: &[ValuePlan],
    takes: &[u32],
    producers: &[Option<OperatorId>],
) -> Option<(Instruction, OperatorId, OperatorId)> {
    let add = operators.get(add_id.get() as usize)?;
    if add.op != Op::ADD || add.input_count != 2 {
        return None;
    }
    for product_slot in 0..2 {
        let product = add.inputs[product_slot];
        let current = add.inputs[1 - product_slot];
        let mul_id = match producers.get(product.0 as usize).copied().flatten() {
            Some(id) => id,
            None => continue,
        };
        let mul = operators.get(mul_id.get() as usize)?;
        if mul.op != Op::MUL || mul.input_count < 2 || takes.get(product.0 as usize).copied()? != 1
        {
            continue;
        }
        for difference_slot in 0..2 {
            let difference = mul.inputs[difference_slot];
            let weight = mul.inputs[1 - difference_slot];
            let sub_id = match producers.get(difference.0 as usize).copied().flatten() {
                Some(id) => id,
                None => continue,
            };
            let sub = operators.get(sub_id.get() as usize)?;
            if sub.op != Op::SUB
                || sub.input_count != 2
                || sub.inputs[1] != current
                || takes.get(difference.0 as usize).copied()? != 1
                || weight == current
            {
                continue;
            }
            let target = sub.inputs[0];
            let (output_dtype, output_dims) = tensor_info(values.get(add.output.0 as usize)?)?;
            let (current_dtype, current_dims) = tensor_info(values.get(current.0 as usize)?)?;
            let (target_dtype, target_dims) = tensor_info(values.get(target.0 as usize)?)?;
            let (weight_dtype, weight_dims) = tensor_info(values.get(weight.0 as usize)?)?;
            if !matches!(output_dtype, DType::FP16 | DType::FP32)
                || current_dtype != output_dtype
                || target_dtype != output_dtype
                || weight_dtype != output_dtype
                || current_dims != output_dims
                || weight_dims != output_dims
                || target_dims.elements() != 1
            {
                continue;
            }
            return Some((
                Instruction::Lerp {
                    current,
                    target,
                    weight,
                    output: add.output,
                },
                sub_id,
                mul_id,
            ));
        }
    }
    None
}

/// Execute a static single-block TOSA artifact.
///
/// `inputs` are raw little-endian tensor bytes in the block's declared
/// input order; outputs are returned in declared output order.
pub fn execute(artifact: &[u8], inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, ExecError> {
    Program::compile(artifact)?.execute(inputs)
}

fn supported(op: Op) -> bool {
    matches!(
        op,
        Op::CONST
            | Op::CONST_SHAPE
            | Op::IDENTITY
            | Op::CAST
            | Op::ADD
            | Op::SUB
            | Op::MUL
            | Op::POW
            | Op::MAXIMUM
            | Op::MINIMUM
            | Op::ABS
            | Op::CEIL
            | Op::COS
            | Op::EXP
            | Op::FLOOR
            | Op::LOG
            | Op::NEGATE
            | Op::RECIPROCAL
            | Op::RSQRT
            | Op::SIN
            | Op::EQUAL
            | Op::GREATER
            | Op::GREATER_EQUAL
            | Op::LOGICAL_AND
            | Op::LOGICAL_OR
            | Op::LOGICAL_XOR
            | Op::LOGICAL_NOT
            | Op::SELECT
            | Op::TANH
            | Op::SIGMOID
            | Op::CLAMP
            | Op::MATMUL
            | Op::MAX_POOL2D
            | Op::REDUCE_SUM
            | Op::REDUCE_MAX
            | Op::RESHAPE
            | Op::ARGMAX
    )
}

/// A live value: raw storage bytes, or the f32 compute plane with narrowing
/// deferred until a byte observation or explicit CAST boundary.
#[derive(Clone)]
enum Val<'a> {
    Borrowed(&'a [u8]),
    Bytes(Vec<u8>),
    Plane(Vec<f32>),
}

struct Exec<'program, 'input> {
    program: &'program Program,
    bufs: Vec<Option<Val<'input>>>,
    /// Remaining reads per value (all operator-input appearances plus
    /// block-output listings). A consuming read that drops the count to
    /// zero moves the buffer out instead of cloning; reads that don't
    /// decrement simply leave a surplus, which only ever costs a copy.
    takes: Vec<u32>,
}

impl Exec<'_, '_> {
    /// Analysis value IDs are dense indexes in the compiled program.
    fn index(&self, id: PlanValue) -> usize {
        id.0 as usize
    }

    fn value(&self, id: PlanValue) -> Result<&ValuePlan, ExecError> {
        self.program
            .values
            .get(self.index(id))
            .ok_or(ExecError::Shape("value id outside compiled program"))
    }

    fn dtype(&self, id: PlanValue) -> Result<DType, ExecError> {
        match &self.value(id)?.kind {
            ValueKind::Tensor { dtype, .. } => Ok(*dtype),
            ValueKind::Shape => Err(ExecError::Shape("shape operand where tensor expected")),
        }
    }

    fn dims(&self, id: PlanValue) -> Result<ShapeDims, ExecError> {
        match &self.value(id)?.kind {
            ValueKind::Tensor { dims, .. } => Ok(*dims),
            ValueKind::Shape => Err(ExecError::Shape("shape operand where tensor expected")),
        }
    }

    fn elements(&self, id: PlanValue) -> Result<usize, ExecError> {
        Ok(self.dims(id)?.elements())
    }

    fn byte_size(&self, id: PlanValue) -> Result<usize, ExecError> {
        Ok(self.value(id)?.byte_size)
    }

    /// Write a value into its final provider binding without manufacturing an
    /// intermediate byte vector. Deferred float planes narrow directly into
    /// the caller's fixed buffer; already-byte-backed values copy once.
    fn write_bytes(&self, id: PlanValue, output: &mut [u8]) -> Result<(), ExecError> {
        if output.len() != self.byte_size(id)? {
            return Err(ExecError::Shape("direct output binding size"));
        }
        let slot = self.index(id);
        match self.bufs[slot].as_ref() {
            Some(Val::Plane(values)) => narrow_from_f32_into(self.dtype(id)?, values, output),
            Some(Val::Borrowed(bytes)) => {
                output.copy_from_slice(bytes);
                Ok(())
            }
            Some(Val::Bytes(bytes)) => {
                output.copy_from_slice(bytes);
                Ok(())
            }
            None => match self
                .value(id)?
                .constant
                .as_ref()
                .ok_or(ExecError::Shape("read of unwritten value"))?
            {
                ConstantValue::Plane(values) => {
                    narrow_from_f32_into(self.dtype(id)?, values, output)
                }
                ConstantValue::Bytes(bytes) => {
                    output.copy_from_slice(bytes);
                    Ok(())
                }
            },
        }
    }

    /// Borrow a value's bytes, materializing serialized constants and
    /// narrowing deferred planes in place.
    fn bytes(&mut self, id: PlanValue) -> Result<&[u8], ExecError> {
        let slot = self.index(id);
        if self.bufs[slot].is_none() {
            self.bufs[slot] = Some(
                match self
                    .value(id)?
                    .constant
                    .clone()
                    .ok_or(ExecError::Shape("read of unwritten value"))?
                {
                    ConstantValue::Bytes(bytes) => Val::Bytes(bytes),
                    ConstantValue::Plane(values) => Val::Plane(values),
                },
            );
        }
        if matches!(self.bufs[slot], Some(Val::Plane(_))) {
            let dtype = self.dtype(id)?;
            let Some(Val::Plane(plane)) = self.bufs[slot].take() else {
                unreachable!()
            };
            self.bufs[slot] = Some(Val::Bytes(narrow_from_f32(dtype, &plane)?));
        }
        match self.bufs[slot].as_ref().unwrap() {
            Val::Borrowed(bytes) => Ok(bytes),
            Val::Bytes(bytes) => Ok(bytes),
            Val::Plane(_) => unreachable!(),
        }
    }

    fn take(&mut self, id: PlanValue) -> Result<Vec<u8>, ExecError> {
        self.bytes(id)?;
        let slot = self.index(id);
        let t = self.takes[slot];
        self.takes[slot] = t.saturating_sub(1);
        match (&self.bufs[slot], t) {
            (Some(Val::Borrowed(bytes)), _) => Ok(bytes.to_vec()),
            (Some(Val::Bytes(bytes)), 2..) => Ok(bytes.clone()),
            _ => match self.bufs[slot].take() {
                Some(Val::Bytes(bytes)) => Ok(bytes),
                _ => unreachable!(),
            },
        }
    }

    /// Widen a value into the f32 compute plane without consuming it
    /// (exact for all supported source dtypes).
    fn as_f32(&mut self, id: PlanValue) -> Result<Vec<f32>, ExecError> {
        let dtype = self.dtype(id)?;
        let n = self.elements(id)?;
        let slot = self.index(id);
        if let Some(Val::Plane(plane)) = &self.bufs[slot] {
            return Ok(plane.clone());
        }
        if self.bufs[slot].is_none()
            && let Some(ConstantValue::Plane(plane)) = &self.value(id)?.constant
        {
            return Ok(plane.clone());
        }
        let bytes = self.bytes(id)?;
        widen_to_f32(dtype, bytes, n)
    }

    /// Consume-aware widen: the last counted read moves the plane out
    /// (or widens the bytes once) with no copy; earlier reads clone.
    fn take_f32(&mut self, id: PlanValue) -> Result<Vec<f32>, ExecError> {
        let slot = self.index(id);
        let t = self.takes[slot];
        self.takes[slot] = t.saturating_sub(1);
        if t == 1 {
            match self.bufs[slot].take() {
                Some(Val::Plane(plane)) => return Ok(plane),
                Some(Val::Borrowed(bytes)) => {
                    let dtype = self.dtype(id)?;
                    let n = self.elements(id)?;
                    return widen_to_f32(dtype, bytes, n);
                }
                Some(Val::Bytes(bytes)) => {
                    let dtype = self.dtype(id)?;
                    let n = self.elements(id)?;
                    return widen_to_f32(dtype, &bytes, n);
                }
                // Unmaterialized constant: restore the count and fall
                // through to the borrow path.
                None => self.takes[slot] += 1,
            }
        }
        self.as_f32(id)
    }

    /// Consume several reads of one f32 value at once. Multi-operator
    /// instructions can then move a plane when every remaining use belongs
    /// to the fused region instead of cloning it for each source operator.
    fn take_f32_uses(&mut self, id: PlanValue, uses: u32) -> Result<Vec<f32>, ExecError> {
        let slot = self.index(id);
        let remaining = self.takes[slot];
        if uses == 0 || remaining < uses {
            return Err(ExecError::Shape("invalid fused value use count"));
        }
        self.takes[slot] = remaining - uses;
        if remaining == uses {
            match self.bufs[slot].take() {
                Some(Val::Plane(plane)) => return Ok(plane),
                Some(Val::Borrowed(bytes)) => {
                    let dtype = self.dtype(id)?;
                    let elements = self.elements(id)?;
                    return widen_to_f32(dtype, bytes, elements);
                }
                Some(Val::Bytes(bytes)) => {
                    let dtype = self.dtype(id)?;
                    let elements = self.elements(id)?;
                    return widen_to_f32(dtype, &bytes, elements);
                }
                None => {}
            }
        }
        self.as_f32(id)
    }

    /// Consume one scalar float read without allocating a one-element plane.
    /// Static geometry graphs use scalar constants pervasively; keeping them
    /// in the compiled constant table makes warm execution allocation-free at
    /// this boundary.
    fn take_scalar_f32(&mut self, id: PlanValue) -> Result<f32, ExecError> {
        if self.elements(id)? != 1 {
            return Err(ExecError::Shape("scalar float operand shape"));
        }
        let dtype = self.dtype(id)?;
        let slot = self.index(id);
        let value = match self.bufs[slot].as_ref() {
            Some(Val::Plane(plane)) => plane[0],
            Some(Val::Borrowed(bytes)) => decode_scalar_f32(dtype, bytes)?,
            Some(Val::Bytes(bytes)) => decode_scalar_f32(dtype, bytes)?,
            None => {
                let constant = self
                    .value(id)?
                    .constant
                    .as_ref()
                    .ok_or(ExecError::Shape("read of unwritten scalar value"))?;
                match constant {
                    ConstantValue::Bytes(bytes) => decode_scalar_f32(dtype, bytes)?,
                    ConstantValue::Plane(values) => values[0],
                }
            }
        };
        let remaining = self.takes[slot];
        self.takes[slot] = remaining.saturating_sub(1);
        if remaining <= 1 {
            self.bufs[slot] = None;
        }
        Ok(value)
    }

    /// Store an op result on the f32 compute plane. Narrowing to the
    /// value's storage dtype is deferred until bytes are observed
    /// (block outputs, CAST, integer/F8 consumers) — chained float ops
    /// see full f32 precision, exactly as NPU providers keep fused
    /// intermediates at accumulator precision. TOSA's numeric model
    /// permits higher-precision intermediates; RNE applies at the
    /// observation boundary.
    fn store_plane(&mut self, id: PlanValue, values: Vec<f32>) -> Result<(), ExecError> {
        match self.dtype(id)? {
            DType::FP32 | DType::FP16 => {}
            _ => return Err(ExecError::Shape("unsupported store dtype")),
        }
        let slot = self.index(id);
        self.bufs[slot] = Some(Val::Plane(values));
        Ok(())
    }

    fn operator(&self, id: OperatorId) -> Result<&OperatorPlan, ExecError> {
        self.program
            .operators
            .get(id.get() as usize)
            .ok_or(ExecError::Shape("operator id outside compiled program"))
    }

    fn dispatch(&mut self, instruction: Instruction) -> Result<(), ExecError> {
        let op_id = match instruction {
            Instruction::Operator(op_id) => op_id,
            Instruction::ScaleAdd {
                input,
                scale,
                addend,
                output,
                order,
            } => return self.scale_add(input, scale, addend, output, order),
            Instruction::MulAdd {
                left,
                right,
                addend,
                output,
                order,
            } => return self.mul_add(left, right, addend, output, order),
            Instruction::Lerp {
                current,
                target,
                weight,
                output,
            } => return self.lerp(current, target, weight, output),
            Instruction::LinearizedClamp {
                value,
                derivative_x,
                offset_x,
                derivative_y,
                offset_y,
                lower,
                upper,
                output,
            } => {
                return self.linearized_clamp(
                    value,
                    derivative_x,
                    offset_x,
                    derivative_y,
                    offset_y,
                    lower,
                    upper,
                    output,
                );
            }
            Instruction::GuardedSqrt {
                input,
                floor,
                output,
            } => return self.guarded_sqrt(input, floor, output),
        };
        let operator = self.operator(op_id)?;
        let op = operator.op;
        let inputs = operator.inputs;
        let input_count = operator.input_count;
        let out = operator.output;
        let ins = &inputs[..input_count as usize];

        match op {
            // Constants materialize lazily on first read.
            Op::CONST | Op::CONST_SHAPE => Ok(()),
            Op::IDENTITY | Op::RESHAPE => {
                // Same byte layout: move or clone the value whole, in
                // whichever representation it currently is.
                let slot_in = self.index(ins[0]);
                let t = self.takes[slot_in];
                self.takes[slot_in] = t.saturating_sub(1);
                let val = match (&self.bufs[slot_in], t) {
                    (Some(_), 0 | 1) => self.bufs[slot_in].take().unwrap(),
                    (Some(Val::Borrowed(bytes)), _) => Val::Borrowed(bytes),
                    (Some(Val::Bytes(b)), _) => Val::Bytes(b.clone()),
                    (Some(Val::Plane(p)), _) => Val::Plane(p.clone()),
                    (None, _) => {
                        self.takes[slot_in] = t;
                        Val::Bytes(self.bytes(ins[0])?.to_vec())
                    }
                };
                let slot = self.index(out);
                self.bufs[slot] = Some(val);
                Ok(())
            }
            Op::CAST => self.cast(ins[0], out),
            Op::ADD => self.binary(ins, out, |a, b| a + b),
            Op::SUB => self.binary(ins, out, |a, b| a - b),
            Op::MUL => self.binary(ins, out, |a, b| a * b),
            Op::POW => self.binary(ins, out, powf),
            Op::MAXIMUM | Op::MINIMUM => self.extrema(op_id, ins, out),
            Op::ABS => self.unary(ins[0], out, axmath::abs_f32),
            Op::CEIL => self.unary(ins[0], out, axmath::ceil_f32),
            Op::COS => self.unary(ins[0], out, axmath::cos_f32),
            Op::EXP => self.unary(ins[0], out, axmath::exp_f32),
            Op::FLOOR => self.unary(ins[0], out, axmath::floor_f32),
            Op::LOG => self.unary(ins[0], out, axmath::log_f32),
            Op::NEGATE => self.unary(ins[0], out, |v| -v),
            Op::RECIPROCAL => self.unary(ins[0], out, |v| 1.0 / v),
            Op::RSQRT => self.unary(ins[0], out, |v| 1.0 / axmath::sqrt_f32(v)),
            Op::SIN => self.unary(ins[0], out, axmath::sin_f32),
            Op::EQUAL => self.compare(ins, out, |a, b| a == b),
            Op::GREATER => self.compare(ins, out, |a, b| a > b),
            Op::GREATER_EQUAL => self.compare(ins, out, |a, b| a >= b),
            Op::LOGICAL_AND => self.logical_binary(ins, out, |a, b| a && b),
            Op::LOGICAL_OR => self.logical_binary(ins, out, |a, b| a || b),
            Op::LOGICAL_XOR => self.logical_binary(ins, out, |a, b| a != b),
            Op::LOGICAL_NOT => self.logical_not(ins[0], out),
            Op::SELECT => self.select(ins, out),
            Op::TANH => self.unary_plane(ins[0], out, axnn_cpu::activation::tanh_f32_inplace),
            Op::SIGMOID => self.unary_plane(ins[0], out, axnn_cpu::activation::sigmoid_f32_inplace),
            Op::CLAMP => self.clamp(op_id, ins[0], out),
            Op::MATMUL => self.matmul(ins, out),
            Op::MAX_POOL2D => self.max_pool2d(op_id, ins[0], out),
            Op::REDUCE_SUM => self.reduce(op_id, ins[0], out, 0.0, |acc, v| acc + v),
            Op::REDUCE_MAX => self.reduce(op_id, ins[0], out, f32::NEG_INFINITY, f32::max),
            Op::ARGMAX => self.argmax(op_id, ins[0], out),
            other => Err(ExecError::Unsupported(other)),
        }
    }

    /// Execute MAXIMUM/MINIMUM with the NaN policy encoded by the validated
    /// TOSA operator. Keeping the policy in the compiled plan prevents a warm
    /// dispatch from silently substituting one of TOSA's two legal modes.
    fn extrema(
        &mut self,
        op_id: OperatorId,
        ins: &[PlanValue],
        out: PlanValue,
    ) -> Result<(), ExecError> {
        let operator = self.operator(op_id)?;
        let op = operator.op;
        let mode = match &operator.attributes {
            OwnedAttributes::NanMode(mode) => *mode,
            _ => return Err(ExecError::Unsupported(op)),
        };
        // Keep the policy branch outside the element loop. Each arm
        // monomorphizes `binary` with its concrete operation, preserving the
        // optimizer's ability to inline/vectorize these geometry hot paths.
        match (op, mode) {
            (Op::MAXIMUM, NanPropagationMode::PROPAGATE) => {
                self.binary(ins, out, maximum_propagate_nan)
            }
            (Op::MAXIMUM, NanPropagationMode::IGNORE) => self.binary(ins, out, f32::max),
            (Op::MINIMUM, NanPropagationMode::PROPAGATE) => {
                self.binary(ins, out, minimum_propagate_nan)
            }
            (Op::MINIMUM, NanPropagationMode::IGNORE) => self.binary(ins, out, f32::min),
            _ => Err(ExecError::Unsupported(op)),
        }
    }

    /// Elementwise unary op via an in-place plane kernel.
    fn unary_plane(
        &mut self,
        input: PlanValue,
        out: PlanValue,
        f: impl Fn(&mut [f32]),
    ) -> Result<(), ExecError> {
        let mut plane = self.take_f32(input)?;
        f(&mut plane);
        self.store_plane(out, plane)
    }

    /// Elementwise unary op over the f32 compute plane.
    fn unary(
        &mut self,
        input: PlanValue,
        out: PlanValue,
        f: impl Fn(f32) -> f32,
    ) -> Result<(), ExecError> {
        let mut plane = self.take_f32(input)?;
        for v in plane.iter_mut() {
            *v = f(*v);
        }
        self.store_plane(out, plane)
    }

    /// Elementwise floating comparison with TOSA right-aligned broadcasting.
    fn compare(
        &mut self,
        ins: &[PlanValue],
        out: PlanValue,
        f: impl Fn(f32, f32) -> bool,
    ) -> Result<(), ExecError> {
        let (da, db) = (self.dims(ins[0])?, self.dims(ins[1])?);
        let dout = self.dims(out)?;
        let n: usize = dout.iter().product();
        let mut res = vec![0u8; n];
        if da.elements() == 1 && db == dout {
            let a = self.take_scalar_f32(ins[0])?;
            let b = self.take_f32(ins[1])?;
            for (result, &value) in res.iter_mut().zip(&b) {
                *result = u8::from(f(a, value));
            }
            return self.store_bool(out, res);
        }
        if db.elements() == 1 && da == dout {
            let a = self.take_f32(ins[0])?;
            let b = self.take_scalar_f32(ins[1])?;
            for (result, &value) in res.iter_mut().zip(&a) {
                *result = u8::from(f(value, b));
            }
            return self.store_bool(out, res);
        }
        let a = self.take_f32(ins[0])?;
        let b = self.take_f32(ins[1])?;
        if da == dout && db == dout {
            for ((result, &a), &b) in res.iter_mut().zip(&a).zip(&b) {
                *result = u8::from(f(a, b));
            }
            return self.store_bool(out, res);
        }
        if da.len() == 2 && db.len() == 2 && dout.len() == 2 {
            for row in 0..dout[0] {
                for column in 0..dout[1] {
                    let output = row * dout[1] + column;
                    let ia = broadcast_index_2d(da, row, column);
                    let ib = broadcast_index_2d(db, row, column);
                    res[output] = u8::from(f(a[ia], b[ib]));
                }
            }
            return self.store_bool(out, res);
        }
        let (sa, sb) = (broadcast_strides(&da, &dout), broadcast_strides(&db, &dout));
        let mut coords = [0usize; MAX_LEVEL_8K_RANK];
        for (i, value) in res.iter_mut().enumerate() {
            linear_to_coords(i, &dout, &mut coords[..dout.len()]);
            let ia: usize = coords[..dout.len()]
                .iter()
                .zip(&sa)
                .map(|(c, s)| c * s)
                .sum();
            let ib: usize = coords[..dout.len()]
                .iter()
                .zip(&sb)
                .map(|(c, s)| c * s)
                .sum();
            *value = u8::from(f(a[ia], b[ib]));
        }
        self.store_bool(out, res)
    }

    /// Elementwise boolean binary with TOSA right-aligned broadcasting.
    fn logical_binary(
        &mut self,
        ins: &[PlanValue],
        out: PlanValue,
        f: impl Fn(bool, bool) -> bool,
    ) -> Result<(), ExecError> {
        let (da, db) = (self.dims(ins[0])?, self.dims(ins[1])?);
        let dout = self.dims(out)?;
        let a = self.take_bool(ins[0])?;
        let b = self.take_bool(ins[1])?;
        let n: usize = dout.iter().product();
        let mut res = vec![0u8; n];
        if da.elements() == 1 && db == dout {
            for (result, &value) in res.iter_mut().zip(&b) {
                *result = u8::from(f(a[0] != 0, value != 0));
            }
            return self.store_bool(out, res);
        }
        if db.elements() == 1 && da == dout {
            for (result, &value) in res.iter_mut().zip(&a) {
                *result = u8::from(f(value != 0, b[0] != 0));
            }
            return self.store_bool(out, res);
        }
        if da == dout && db == dout {
            for ((result, &a), &b) in res.iter_mut().zip(&a).zip(&b) {
                *result = u8::from(f(a != 0, b != 0));
            }
            return self.store_bool(out, res);
        }
        if da.len() == 2 && db.len() == 2 && dout.len() == 2 {
            for row in 0..dout[0] {
                for column in 0..dout[1] {
                    let output = row * dout[1] + column;
                    let ia = broadcast_index_2d(da, row, column);
                    let ib = broadcast_index_2d(db, row, column);
                    res[output] = u8::from(f(a[ia] != 0, b[ib] != 0));
                }
            }
            return self.store_bool(out, res);
        }
        let (sa, sb) = (broadcast_strides(&da, &dout), broadcast_strides(&db, &dout));
        let mut coords = [0usize; MAX_LEVEL_8K_RANK];
        for (i, value) in res.iter_mut().enumerate() {
            linear_to_coords(i, &dout, &mut coords[..dout.len()]);
            let ia: usize = coords[..dout.len()]
                .iter()
                .zip(&sa)
                .map(|(c, s)| c * s)
                .sum();
            let ib: usize = coords[..dout.len()]
                .iter()
                .zip(&sb)
                .map(|(c, s)| c * s)
                .sum();
            *value = u8::from(f(a[ia] != 0, b[ib] != 0));
        }
        self.store_bool(out, res)
    }

    fn logical_not(&mut self, input: PlanValue, out: PlanValue) -> Result<(), ExecError> {
        let mut values = self.take_bool(input)?;
        for value in &mut values {
            *value = u8::from(*value == 0);
        }
        self.store_bool(out, values)
    }

    /// Select floating values using a broadcast boolean mask.
    fn select(&mut self, ins: &[PlanValue], out: PlanValue) -> Result<(), ExecError> {
        let (dm, da, db) = (self.dims(ins[0])?, self.dims(ins[1])?, self.dims(ins[2])?);
        let dout = self.dims(out)?;
        let mask = self.take_bool(ins[0])?;
        if dm.elements() == 1 && da == dout && db == dout {
            let a = self.take_f32(ins[1])?;
            let b = self.take_f32(ins[2])?;
            return self.store_plane(out, if mask[0] != 0 { a } else { b });
        }
        if dm == dout && da == dout && db == dout {
            let mut a = self.take_f32(ins[1])?;
            let b = self.take_f32(ins[2])?;
            for ((value, &mask), &alternative) in a.iter_mut().zip(&mask).zip(&b) {
                if mask == 0 {
                    *value = alternative;
                }
            }
            return self.store_plane(out, a);
        }
        if dm == dout && da.elements() == 1 && db == dout {
            let selected = self.take_scalar_f32(ins[1])?;
            let mut result = self.take_f32(ins[2])?;
            for (value, &mask) in result.iter_mut().zip(&mask) {
                if mask != 0 {
                    *value = selected;
                }
            }
            return self.store_plane(out, result);
        }
        if dm == dout && da == dout && db.elements() == 1 {
            let mut a = self.take_f32(ins[1])?;
            let alternative = self.take_scalar_f32(ins[2])?;
            for (value, &mask) in a.iter_mut().zip(&mask) {
                if mask == 0 {
                    *value = alternative;
                }
            }
            return self.store_plane(out, a);
        }
        if dm == dout && da.elements() == 1 && db.elements() == 1 {
            let selected = self.take_scalar_f32(ins[1])?;
            let alternative = self.take_scalar_f32(ins[2])?;
            let result = mask
                .into_iter()
                .map(|mask| if mask != 0 { selected } else { alternative })
                .collect();
            return self.store_plane(out, result);
        }
        let a = self.take_f32(ins[1])?;
        let b = self.take_f32(ins[2])?;
        let (sm, sa, sb) = (
            broadcast_strides(&dm, &dout),
            broadcast_strides(&da, &dout),
            broadcast_strides(&db, &dout),
        );
        let n: usize = dout.iter().product();
        let mut res = vec![0.0f32; n];
        let mut coords = [0usize; MAX_LEVEL_8K_RANK];
        for (i, value) in res.iter_mut().enumerate() {
            linear_to_coords(i, &dout, &mut coords[..dout.len()]);
            let coordinates = &coords[..dout.len()];
            let im: usize = coordinates.iter().zip(&sm).map(|(c, s)| c * s).sum();
            let ia: usize = coordinates.iter().zip(&sa).map(|(c, s)| c * s).sum();
            let ib: usize = coordinates.iter().zip(&sb).map(|(c, s)| c * s).sum();
            *value = if mask[im] != 0 { a[ia] } else { b[ib] };
        }
        self.store_plane(out, res)
    }

    fn take_bool(&mut self, id: PlanValue) -> Result<Vec<u8>, ExecError> {
        if self.dtype(id)? != DType::BOOL {
            return Err(ExecError::Shape("boolean operand dtype"));
        }
        self.take(id)
    }

    fn store_bool(&mut self, id: PlanValue, values: Vec<u8>) -> Result<(), ExecError> {
        if self.dtype(id)? != DType::BOOL || values.len() != self.byte_size(id)? {
            return Err(ExecError::Shape("boolean result shape"));
        }
        let slot = self.index(id);
        self.bufs[slot] = Some(Val::Bytes(values));
        Ok(())
    }

    fn cast(&mut self, src: PlanValue, dst: PlanValue) -> Result<(), ExecError> {
        let (from, to) = (self.dtype(src)?, self.dtype(dst)?);
        if from == to
            && matches!(
                from,
                DType::FP8E4M3 | DType::FP8E5M2 | DType::FP16 | DType::FP32
            )
        {
            // Same-type casts preserve encoded NaN payloads and signed zero.
            // A deferred plane is observed on its own copy/move: narrowing
            // the source in place could change another float consumer.
            let slot = self.index(src);
            let deferred = matches!(self.bufs[slot], Some(Val::Plane(_)))
                || (self.bufs[slot].is_none()
                    && matches!(
                        self.value(src)?.constant.as_ref(),
                        Some(ConstantValue::Plane(_))
                    ));
            let bytes = if deferred {
                let plane = self.take_f32(src)?;
                narrow_from_f32(to, &plane)?
            } else {
                self.take(src)?
            };
            let slot = self.index(dst);
            self.bufs[slot] = Some(Val::Bytes(bytes));
            return Ok(());
        }
        match (from, to) {
            // F8 → FP16 rides the native half-output dequants: no f32
            // hop at all on FP16-native targets, exact on all encodings.
            (DType::FP8E4M3 | DType::FP8E5M2, DType::FP16) => {
                let n = self.elements(src)?;
                let bytes = self.take(src)?;
                let mut out = vec![0u8; self.byte_size(dst)?];
                match from {
                    DType::FP8E4M3 => axnn_cpu::fp8::dequant_e4m3_f16(&bytes, &mut out, n),
                    _ => axnn_cpu::fp8::dequant_e5m2_f16(&bytes, &mut out, n),
                }
                let slot = self.index(dst);
                self.bufs[slot] = Some(Val::Bytes(out));
                Ok(())
            }
            // Explicit narrow casts round now, even when a following CAST
            // widens again before either result crosses a byte boundary.
            (
                DType::FP8E4M3 | DType::FP8E5M2 | DType::FP16 | DType::FP32,
                DType::FP16 | DType::FP32,
            ) => {
                let mut plane = self.take_cast_f32(src)?;
                if to == DType::FP16 {
                    for value in &mut plane {
                        *value = (*value as f16) as f32;
                    }
                }
                self.store_plane(dst, plane)
            }
            (
                DType::FP8E4M3 | DType::FP8E5M2 | DType::FP16 | DType::FP32,
                DType::FP8E4M3 | DType::FP8E5M2,
            ) => {
                let plane = self.take_cast_f32(src)?;
                let bytes = narrow_from_f32(to, &plane)?;
                let slot = self.index(dst);
                self.bufs[slot] = Some(Val::Bytes(bytes));
                Ok(())
            }
            (DType::FP16 | DType::FP32, DType::INT8) => {
                let plane = self.take_cast_f32(src)?;
                let bytes = plane.into_iter().map(cast_f32_to_i8).collect();
                let slot = self.index(dst);
                self.bufs[slot] = Some(Val::Bytes(bytes));
                Ok(())
            }
            _ => Err(ExecError::Shape("unsupported CAST dtype pair")),
        }
    }

    /// CAST observes the declared source encoding, while ordinary FP16
    /// arithmetic may retain its F32 accumulator plane. Consume/clone first
    /// so this observation cannot reduce another live consumer's precision.
    fn take_cast_f32(&mut self, src: PlanValue) -> Result<Vec<f32>, ExecError> {
        let dtype = self.dtype(src)?;
        let mut plane = self.take_f32(src)?;
        if dtype == DType::FP16 {
            for value in &mut plane {
                *value = (*value as f16) as f32;
            }
        }
        Ok(plane)
    }

    /// Elementwise binary with TOSA right-aligned broadcasting. The MUL
    /// shift operand (input 2, integer profile) is ignored for floats.
    fn binary(
        &mut self,
        ins: &[PlanValue],
        out: PlanValue,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<(), ExecError> {
        let (da, db) = (self.dims(ins[0])?, self.dims(ins[1])?);
        let dout = self.dims(out)?;
        // Scalar broadcast is pervasive in geometry graphs. Handle it before
        // the general coordinate mapper so each lane is one SIMD-friendly
        // arithmetic operation instead of rank divisions and modulos.
        if da.elements() == 1 && db == dout {
            let a = self.take_scalar_f32(ins[0])?;
            let mut b = self.take_f32(ins[1])?;
            for value in &mut b {
                *value = f(a, *value);
            }
            return self.store_plane(out, b);
        }
        if db.elements() == 1 && da == dout {
            let mut a = self.take_f32(ins[0])?;
            let b = self.take_scalar_f32(ins[1])?;
            for value in &mut a {
                *value = f(*value, b);
            }
            return self.store_plane(out, a);
        }
        // A value used as both operands is one logical read region. Consume
        // both SSA uses together and map one plane in place instead of
        // cloning it twice and entering the general broadcast mapper.
        if ins[0] == ins[1] && da == dout {
            let mut values = self.take_f32_uses(ins[0], 2)?;
            for value in &mut values {
                *value = f(*value, *value);
            }
            return self.store_plane(out, values);
        }
        // Same-shape fast path: consume the lhs plane, map in place.
        if da == dout && db == dout && ins[0] != ins[1] {
            let mut a = self.take_f32(ins[0])?;
            let b = self.take_f32(ins[1])?;
            for (av, bv) in a.iter_mut().zip(&b) {
                *av = f(*av, *bv);
            }
            return self.store_plane(out, a);
        }
        let a = self.take_f32(ins[0])?;
        let b = self.take_f32(ins[1])?;
        let n: usize = dout.iter().product();
        if da.len() == 2 && db.len() == 2 && dout.len() == 2 {
            let mut result = vec![0.0f32; n];
            for row in 0..dout[0] {
                for column in 0..dout[1] {
                    let output = row * dout[1] + column;
                    let ia = broadcast_index_2d(da, row, column);
                    let ib = broadcast_index_2d(db, row, column);
                    result[output] = f(a[ia], b[ib]);
                }
            }
            return self.store_plane(out, result);
        }
        let (sa, sb) = (broadcast_strides(&da, &dout), broadcast_strides(&db, &dout));
        let mut res = vec![0.0f32; n];
        let mut coords = [0usize; MAX_LEVEL_8K_RANK];
        for (i, r) in res.iter_mut().enumerate() {
            linear_to_coords(i, &dout, &mut coords[..dout.len()]);
            let coordinates = &coords[..dout.len()];
            let ia: usize = coordinates.iter().zip(&sa).map(|(c, s)| c * s).sum();
            let ib: usize = coordinates.iter().zip(&sb).map(|(c, s)| c * s).sum();
            *r = f(a[ia], b[ib]);
        }
        self.store_plane(out, res)
    }

    fn scale_add(
        &mut self,
        input: PlanValue,
        scale: PlanValue,
        addend: PlanValue,
        out: PlanValue,
        order: axnn_cpu::elementwise::ScaleAddOrder,
    ) -> Result<(), ExecError> {
        let mut input = self.take_f32(input)?;
        let scale = self.take_scalar_f32(scale)?;
        let addend_dims = self.dims(addend)?;
        if addend_dims.elements() == 1 {
            let addend = self.take_scalar_f32(addend)?;
            axnn_cpu::elementwise::scale_add_scalar_f32_inplace(&mut input, scale, addend, order);
        } else {
            let addend = self.take_f32(addend)?;
            axnn_cpu::elementwise::scale_add_f32_inplace(&mut input, scale, &addend, order);
        }
        self.store_plane(out, input)
    }

    fn mul_add(
        &mut self,
        left: PlanValue,
        right: PlanValue,
        addend: PlanValue,
        out: PlanValue,
        order: axnn_cpu::elementwise::MulAddOrder,
    ) -> Result<(), ExecError> {
        let mut left = self.take_f32(left)?;
        let right = self.take_f32(right)?;
        let addend = self.take_f32(addend)?;
        axnn_cpu::elementwise::mul_add_f32_inplace(&mut left, &right, &addend, order);
        self.store_plane(out, left)
    }

    fn lerp(
        &mut self,
        current: PlanValue,
        target: PlanValue,
        weight: PlanValue,
        out: PlanValue,
    ) -> Result<(), ExecError> {
        // `current` appears in both SUB and ADD in the source region.
        let mut current = self.take_f32_uses(current, 2)?;
        let target = self.take_scalar_f32(target)?;
        let weight = self.take_f32(weight)?;
        axnn_cpu::elementwise::lerp_scalar_f32_inplace(&mut current, target, &weight);
        self.store_plane(out, current)
    }

    #[allow(clippy::too_many_arguments)]
    fn linearized_clamp(
        &mut self,
        value: PlanValue,
        derivative_x: PlanValue,
        offset_x: PlanValue,
        derivative_y: PlanValue,
        offset_y: PlanValue,
        lower: PlanValue,
        upper: PlanValue,
        out: PlanValue,
    ) -> Result<(), ExecError> {
        let mut value = self.take_f32(value)?;
        let derivative_x = self.take_f32(derivative_x)?;
        let offset_x = self.take_scalar_f32(offset_x)?;
        let derivative_y = self.take_f32(derivative_y)?;
        let offset_y = self.take_scalar_f32(offset_y)?;
        let lower = self.take_scalar_f32(lower)?;
        let upper = self.take_scalar_f32(upper)?;
        axnn_cpu::elementwise::linearized_clamp_f32_inplace(
            &mut value,
            &derivative_x,
            offset_x,
            &derivative_y,
            offset_y,
            lower,
            upper,
        );
        self.store_plane(out, value)
    }

    fn guarded_sqrt(
        &mut self,
        input: PlanValue,
        floor: PlanValue,
        out: PlanValue,
    ) -> Result<(), ExecError> {
        let mut input = self.take_f32_uses(input, 2)?;
        let floor = self.take_scalar_f32(floor)?;
        axnn_cpu::elementwise::guarded_sqrt_f32_inplace(&mut input, floor);
        self.store_plane(out, input)
    }

    fn clamp(&mut self, op: OperatorId, input: PlanValue, out: PlanValue) -> Result<(), ExecError> {
        let OwnedAttributes::Clamp { min_val, max_val } = &self.operator(op)?.attributes else {
            return Err(ExecError::Shape("CLAMP attributes"));
        };
        let dtype = self.dtype(input)?;
        let scalar = |bytes: &[u8]| -> Result<f32, ExecError> {
            Ok(match dtype {
                DType::FP32 => f32::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| ExecError::Shape("clamp bound"))?,
                ),
                DType::FP16 => f16::from_bits(u16::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| ExecError::Shape("clamp bound"))?,
                )) as f32,
                _ => return Err(ExecError::Shape("CLAMP dtype")),
            })
        };
        let (lo, hi) = (scalar(min_val)?, scalar(max_val)?);
        let mut plane = self.take_f32(input)?;
        for v in plane.iter_mut() {
            *v = v.clamp(lo, hi);
        }
        self.store_plane(out, plane)
    }

    /// TOSA MATMUL: `[N, H, C] × [N, C, W] → [N, H, W]`, with rank-1
    /// zero-point operands (inputs 2 and 3).
    fn matmul(&mut self, ins: &[PlanValue], out: PlanValue) -> Result<(), ExecError> {
        let da = self.dims(ins[0])?;
        let db = self.dims(ins[1])?;
        let (batch, h, c, w) = (da[0], da[1], da[2], db[2]);
        let a_dtype = self.dtype(ins[0])?;

        if a_dtype == DType::INT8 {
            // Integer profile: i32 accumulation with zero points.
            let zp = |e: &mut Self, id: PlanValue| -> Result<i32, ExecError> {
                Ok(e.bytes(id)?[0] as i8 as i32)
            };
            let (a_zp, b_zp) = (zp(self, ins[2])?, zp(self, ins[3])?);
            let a = self.bytes(ins[0])?.to_vec();
            let b = self.bytes(ins[1])?.to_vec();
            let mut res = vec![0u8; self.byte_size(out)?];
            for n in 0..batch {
                for i in 0..h {
                    for j in 0..w {
                        let mut acc = 0i32;
                        for p in 0..c {
                            let av = a[n * h * c + i * c + p] as i8 as i32 - a_zp;
                            let bv = b[n * c * w + p * w + j] as i8 as i32 - b_zp;
                            acc += av * bv;
                        }
                        let o = (n * h * w + i * w + j) * 4;
                        res[o..o + 4].copy_from_slice(&acc.to_le_bytes());
                    }
                }
            }
            let slot = self.index(out);
            self.bufs[slot] = Some(Val::Bytes(res));
            return Ok(());
        }

        // F8 × F8: run the native packed GEMM (dequant-block scratch,
        // FMLAL on the f16 tier) straight off the byte planes — no
        // widen/narrow round trip through f32 storage.
        if matches!(a_dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            let a = self.bytes(ins[0])?.to_vec();
            let b = self.bytes(ins[1])?.to_vec();
            let bias = vec![0u8; h * w * 2]; // f16 zero addend matrix
            let mut res = vec![0.0f32; batch * h * w];
            // The packed GEMM takes B in weights layout ([w][c], each
            // output column's reduction contiguous); TOSA hands [c][w].
            let mut bt = vec![0u8; c * w];
            for n in 0..batch {
                let (asl, bsl) = (&a[n * h * c..][..h * c], &b[n * c * w..][..c * w]);
                for p in 0..c {
                    for j in 0..w {
                        bt[j * c + p] = bsl[p * w + j];
                    }
                }
                let dst = &mut res[n * h * w..][..h * w];
                match a_dtype {
                    DType::FP8E4M3 => {
                        axnn_cpu::fp8::gemm_e4m3_e4m3_f16(asl, &bt, &bias, dst, h, w, c)
                    }
                    _ => axnn_cpu::fp8::gemm_e5m2_e5m2_f16(asl, &bt, &bias, dst, h, w, c),
                }
            }
            return self.store_plane(out, res);
        }

        // Float plane (FP32 / FP16): widen once, run the axnn-cpu
        // GEMM per batch with f32 accumulation, narrow once.
        let a = self.take_f32(ins[0])?;
        let b = self.take_f32(ins[1])?;
        let mut res = vec![0.0f32; batch * h * w];
        for n in 0..batch {
            axnn_cpu::linalg::gemm_f32(
                &a[n * h * c..(n + 1) * h * c],
                &b[n * c * w..(n + 1) * c * w],
                &mut res[n * h * w..(n + 1) * h * w],
                h,
                w,
                c,
            );
        }
        self.store_plane(out, res)
    }

    /// NHWC max pooling.
    fn max_pool2d(
        &mut self,
        op: OperatorId,
        input: PlanValue,
        out: PlanValue,
    ) -> Result<(), ExecError> {
        let OwnedAttributes::MaxPool2d {
            kernel,
            stride,
            pad,
        } = &self.operator(op)?.attributes
        else {
            return Err(ExecError::Shape("MAX_POOL2D attributes"));
        };
        let (kh, kw) = (
            kernel.first().copied().unwrap_or(1) as usize,
            kernel.get(1).copied().unwrap_or(1) as usize,
        );
        let (sh, sw) = (
            stride.first().copied().unwrap_or(1) as usize,
            stride.get(1).copied().unwrap_or(1) as usize,
        );
        let (pt, pl) = (
            pad.first().copied().unwrap_or(0) as usize,
            pad.get(2).copied().unwrap_or(0) as usize,
        );

        let din = self.dims(input)?;
        let dout = self.dims(out)?;
        let (n, ih, iw, ch) = (din[0], din[1], din[2], din[3]);
        let (oh, ow) = (dout[1], dout[2]);
        let plane = self.take_f32(input)?;
        let mut res = vec![f32::NEG_INFINITY; n * oh * ow * ch];
        for b in 0..n {
            for oy in 0..oh {
                for ox in 0..ow {
                    for c in 0..ch {
                        let mut best = f32::NEG_INFINITY;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * sh + ky).wrapping_sub(pt);
                                let ix = (ox * sw + kx).wrapping_sub(pl);
                                if iy < ih && ix < iw {
                                    best = best.max(plane[((b * ih + iy) * iw + ix) * ch + c]);
                                }
                            }
                        }
                        res[((b * oh + oy) * ow + ox) * ch + c] = best;
                    }
                }
            }
        }
        self.store_plane(out, res)
    }

    fn reduce(
        &mut self,
        op: OperatorId,
        input: PlanValue,
        out: PlanValue,
        init: f32,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<(), ExecError> {
        let axis = match self.operator(op)?.attributes {
            OwnedAttributes::Reduce { axis } => axis,
            _ => return Err(ExecError::Shape("REDUCE attributes")),
        } as usize;
        let din = self.dims(input)?;
        let plane = self.take_f32(input)?;
        let outer: usize = din[..axis].iter().product();
        let extent = din[axis];
        let inner: usize = din[axis + 1..].iter().product();
        let mut res = vec![init; outer * inner];
        for o in 0..outer {
            for e in 0..extent {
                for i in 0..inner {
                    let r = &mut res[o * inner + i];
                    *r = f(*r, plane[(o * extent + e) * inner + i]);
                }
            }
        }
        self.store_plane(out, res)
    }

    fn argmax(
        &mut self,
        op: OperatorId,
        input: PlanValue,
        out: PlanValue,
    ) -> Result<(), ExecError> {
        let OwnedAttributes::ArgMax { axis } = self.operator(op)?.attributes else {
            return Err(ExecError::Shape("ARGMAX attributes"));
        };
        let axis = axis as usize;
        let din = self.dims(input)?;
        let plane = self.as_f32(input)?;
        let outer: usize = din[..axis].iter().product();
        let extent = din[axis];
        let inner: usize = din[axis + 1..].iter().product();
        let mut res = vec![0u8; outer * inner * 4];
        for o in 0..outer {
            for i in 0..inner {
                let (mut best, mut best_i) = (f32::NEG_INFINITY, 0i32);
                for e in 0..extent {
                    let v = plane[(o * extent + e) * inner + i];
                    if v > best {
                        best = v;
                        best_i = e as i32;
                    }
                }
                let off = (o * inner + i) * 4;
                res[off..off + 4].copy_from_slice(&best_i.to_le_bytes());
            }
        }
        let slot = self.index(out);
        self.bufs[slot] = Some(Val::Bytes(res));
        Ok(())
    }
}

fn decode_scalar_f32(dtype: DType, bytes: &[u8]) -> Result<f32, ExecError> {
    match dtype {
        DType::FP32 => bytes
            .get(..4)
            .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
            .map(f32::from_le_bytes)
            .ok_or(ExecError::Shape("scalar FP32 storage")),
        DType::FP16 => bytes
            .get(..2)
            .and_then(|bytes| <[u8; 2]>::try_from(bytes).ok())
            .map(u16::from_le_bytes)
            .map(f16::from_bits)
            .map(|value| value as f32)
            .ok_or(ExecError::Shape("scalar FP16 storage")),
        DType::FP8E4M3 | DType::FP8E5M2 => widen_to_f32(dtype, bytes, 1).map(|values| values[0]),
        _ => Err(ExecError::Shape("dtype outside the f32 compute plane")),
    }
}

/// Widen storage bytes to the f32 compute plane (exact for all
/// supported source dtypes). Chunked loops so LLVM emits vector
/// conversions on targets with native f16.
fn widen_to_f32(dtype: DType, bytes: &[u8], n: usize) -> Result<Vec<f32>, ExecError> {
    let mut out = vec![0.0f32; n];
    match dtype {
        DType::FP32 => {
            for (o, ch) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*ch);
            }
        }
        DType::FP16 => {
            for (o, ch) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = f16::from_bits(u16::from_le_bytes([ch[0], ch[1]])) as f32;
            }
        }
        DType::FP8E4M3 => axnn_cpu::fp8::dequant_e4m3(bytes, &mut out, n),
        DType::FP8E5M2 => axnn_cpu::fp8::dequant_e5m2(bytes, &mut out, n),
        _ => return Err(ExecError::Shape("dtype outside the f32 compute plane")),
    }
    Ok(out)
}

/// Narrow an f32 plane into storage bytes (RNE).
fn narrow_from_f32(dtype: DType, values: &[f32]) -> Result<Vec<u8>, ExecError> {
    let bytes_per_value = match dtype {
        DType::FP32 => 4,
        DType::FP16 => 2,
        DType::FP8E4M3 | DType::FP8E5M2 => 1,
        _ => return Err(ExecError::Shape("unsupported store dtype")),
    };
    let bytes = values
        .len()
        .checked_mul(bytes_per_value)
        .ok_or(ExecError::Shape("float storage size overflow"))?;
    let mut output = vec![0u8; bytes];
    narrow_from_f32_into(dtype, values, &mut output)?;
    Ok(output)
}

/// Narrow directly into caller-owned storage, avoiding a temporary output
/// plane on provider submissions.
fn narrow_from_f32_into(dtype: DType, values: &[f32], output: &mut [u8]) -> Result<(), ExecError> {
    match dtype {
        DType::FP32 => {
            if values.len().checked_mul(4) != Some(output.len()) {
                return Err(ExecError::Shape("FP32 output storage size"));
            }
            for (ch, v) in output.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                ch.copy_from_slice(&v.to_le_bytes());
            }
            Ok(())
        }
        DType::FP16 => {
            if values.len().checked_mul(2) != Some(output.len()) {
                return Err(ExecError::Shape("FP16 output storage size"));
            }
            for (ch, v) in output.as_chunks_mut::<2>().0.iter_mut().zip(values) {
                ch.copy_from_slice(&(*v as f16).to_bits().to_le_bytes());
            }
            Ok(())
        }
        DType::FP8E4M3 => {
            if output.len() != values.len() {
                return Err(ExecError::Shape("FP8E4M3 output storage size"));
            }
            for (output, &value) in output.iter_mut().zip(values) {
                *output = axnn_cpu::fp8::f32_to_e4m3_rne(value);
            }
            Ok(())
        }
        DType::FP8E5M2 => {
            if output.len() != values.len() {
                return Err(ExecError::Shape("FP8E5M2 output storage size"));
            }
            for (output, &value) in output.iter_mut().zip(values) {
                *output = axnn_cpu::fp8::f32_to_e5m2_rne(value);
            }
            Ok(())
        }
        _ => Err(ExecError::Shape("unsupported store dtype")),
    }
}

#[inline]
fn cast_f32_to_i8(value: f32) -> u8 {
    let clipped = if value.is_nan() {
        // TOSA leaves NaN-to-integer conversion unpredictable; choose the
        // deterministic Rust saturation result used by the other casts.
        0.0
    } else {
        value.clamp(f32::from(i8::MIN), f32::from(i8::MAX))
    };
    let truncated = clipped as i32;
    let fraction = clipped - truncated as f32;
    let rounded = if fraction > 0.5 || (fraction == 0.5 && truncated & 1 != 0) {
        truncated + 1
    } else if fraction < -0.5 || (fraction == -0.5 && truncated & 1 != 0) {
        truncated - 1
    } else {
        truncated
    };
    rounded as i8 as u8
}

/// Output-coordinate strides for one right-aligned row-major broadcast input.
/// Missing leading dimensions and authored size-one dimensions both use zero.
fn broadcast_strides(dims: &[usize], out: &[usize]) -> [usize; MAX_LEVEL_8K_RANK] {
    debug_assert!(dims.len() <= out.len());
    let mut strides = [0usize; MAX_LEVEL_8K_RANK];
    let offset = out.len() - dims.len();
    let mut acc = 1usize;
    for i in (0..dims.len()).rev() {
        strides[offset + i] = if dims[i] == 1 { 0 } else { acc };
        acc *= dims[i];
    }
    strides
}

/// Direct row-major index for a validated rank-2 broadcast operand.
///
/// TOSA has already established that each operand dimension is either one or
/// the corresponding output dimension. Keeping this arithmetic outside the
/// pixel loop's dynamic-rank coordinate decoder matters for physical
/// framebuffer fields and batched GR state planes alike.
#[inline]
fn broadcast_index_2d(dims: ShapeDims, row: usize, column: usize) -> usize {
    let row = if dims[0] == 1 { 0 } else { row };
    let column = if dims[1] == 1 { 0 } else { column };
    row * dims[1] + column
}

fn linear_to_coords(mut i: usize, dims: &[usize], coords: &mut [usize]) {
    for d in (0..dims.len()).rev() {
        coords[d] = i % dims[d];
        i /= dims[d];
    }
}

/// `a^b` for the FP profile: exact integer exponents delegate to
/// powi-style repeated multiplication; general exponents use exp/ln.
fn powf(a: f32, b: f32) -> f32 {
    let bi = b as i32;
    if bi as f32 == b && bi.unsigned_abs() <= 64 {
        let mut r = 1.0f32;
        for _ in 0..bi.unsigned_abs() {
            r *= a;
        }
        if bi < 0 { 1.0 / r } else { r }
    } else {
        axmath_exp(b * axmath_ln(a))
    }
}

fn maximum_propagate_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a.max(b)
    }
}

fn minimum_propagate_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a.min(b)
    }
}

fn axmath_exp(x: f32) -> f32 {
    axmath::exp_f32(x)
}
fn axmath_ln(x: f32) -> f32 {
    axmath::log_f32(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_int8_cast_uses_rne_and_saturation() {
        assert_eq!(cast_f32_to_i8(-129.0), i8::MIN as u8);
        assert_eq!(cast_f32_to_i8(-1.5), (-2_i8) as u8);
        assert_eq!(cast_f32_to_i8(-0.5), 0);
        assert_eq!(cast_f32_to_i8(0.5), 0);
        assert_eq!(cast_f32_to_i8(1.5), 2);
        assert_eq!(cast_f32_to_i8(128.0), i8::MAX as u8);
    }

    fn cast_test_program(dtypes: &[DType], elements: usize) -> Program {
        Program {
            values: dtypes
                .iter()
                .map(|&dtype| ValuePlan {
                    kind: ValueKind::Tensor {
                        dtype,
                        dims: rank2(1, elements),
                    },
                    byte_size: elements
                        * match dtype {
                            DType::FP32 => 4,
                            DType::FP16 | DType::BF16 => 2,
                            DType::FP8E4M3 | DType::FP8E5M2 | DType::INT8 => 1,
                            _ => panic!("unexpected CAST test dtype"),
                        },
                    constant: None,
                })
                .collect(),
            operators: Vec::new(),
            order: Vec::new(),
            inputs: vec![PlanValue(0)],
            outputs: vec![PlanValue(dtypes.len() as u32 - 1)],
            takes: vec![1; dtypes.len()],
            constant_folds: 0,
        }
    }

    #[test]
    fn explicit_fp16_cast_roundtrip_observes_ties_subnormals_and_overflow() {
        let values = [
            0.0_f32,
            -0.0,
            f32::from_bits(0x3f02_b400), // volume-sampling integration witness
            1.0 + 1.0 / 2048.0,          // halfway to even lower half
            1.0 + 3.0 / 2048.0,          // halfway to even upper half
            -1.0 - 1.0 / 2048.0,
            f32::from_bits(0x3300_0000), // 2^-25
            3.0 * f32::from_bits(0x3300_0000),
            f32::from_bits(0x3380_0000), // 2^-24
            65504.0,
            65520.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let mut program = cast_test_program(&[DType::FP32, DType::FP16, DType::FP32], values.len());
        program.takes[0] = 2;
        let mut exec = Exec {
            program: &program,
            bufs: vec![Some(Val::Plane(values.to_vec())), None, None],
            takes: program.takes.clone(),
        };
        exec.cast(PlanValue(0), PlanValue(1)).unwrap();
        exec.cast(PlanValue(1), PlanValue(2)).unwrap();
        let actual = exec.take_f32(PlanValue(2)).unwrap();
        for (&actual, original) in actual.iter().zip(values) {
            let expected = (original as f16) as f32;
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "CAST({original})");
            }
        }
        let sibling = exec.take_f32(PlanValue(0)).unwrap();
        assert!(
            sibling
                .iter()
                .zip(values)
                .all(|(actual, original)| actual.to_bits() == original.to_bits())
        );
    }

    #[test]
    fn casts_observe_deferred_fp16_sources_without_narrowing_sibling_consumers() {
        for (destination, input, expected) in [
            (
                DType::FP16,
                1.0 + 1.0 / 2048.0,
                0x3c00_u16.to_le_bytes().to_vec(),
            ),
            (
                DType::FP32,
                1.0 + 1.0 / 2048.0,
                1.0_f32.to_le_bytes().to_vec(),
            ),
            (DType::FP8E4M3, 1.0626, vec![0x38]), // source rounds to the F8 halfway value
            (DType::FP8E5M2, 1.1251, vec![0x3c]),
            (DType::INT8, 0.5001, vec![0]), // source rounds to 0.5, integer RNE gives zero
        ] {
            for constant in [false, true] {
                let mut program = cast_test_program(&[DType::FP16, destination], 1);
                program.takes[0] = 2;
                if constant {
                    program.values[0].constant = Some(ConstantValue::Plane(vec![input]));
                }
                let mut exec = Exec {
                    program: &program,
                    bufs: vec![(!constant).then(|| Val::Plane(vec![input])), None],
                    takes: program.takes.clone(),
                };
                exec.cast(PlanValue(0), PlanValue(1)).unwrap();
                let mut actual = vec![0_u8; expected.len()];
                exec.write_bytes(PlanValue(1), &mut actual).unwrap();
                assert_eq!(
                    actual, expected,
                    "FP16 -> {destination:?}, constant={constant}"
                );
                assert_eq!(
                    exec.take_f32(PlanValue(0)).unwrap()[0].to_bits(),
                    input.to_bits(),
                    "CAST rounded a shared source plane in place"
                );
            }
        }
    }

    #[test]
    fn same_float_type_casts_preserve_all_storage_bits() {
        for (dtype, bytes) in [
            (
                DType::FP32,
                [
                    0_u32,
                    0x8000_0000,
                    0x3f80_0000,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7f80_1234,
                    0xffc0_5678,
                ]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
            ),
            (
                DType::FP16,
                [0_u16, 0x8000, 0x3c00, 0x7c00, 0xfc00, 0x7c01, 0xfe55]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
            (DType::FP8E4M3, (0_u8..=255).collect()),
            (DType::FP8E5M2, (0_u8..=255).collect()),
        ] {
            let width = match dtype {
                DType::FP32 => 4,
                DType::FP16 => 2,
                _ => 1,
            };
            for backing in 0..3 {
                let mut program = cast_test_program(&[dtype, dtype], bytes.len() / width);
                if backing == 2 {
                    program.values[0].constant = Some(ConstantValue::Bytes(bytes.clone()));
                }
                let input = match backing {
                    0 => Some(Val::Borrowed(bytes.as_slice())),
                    1 => Some(Val::Bytes(bytes.clone())),
                    _ => None,
                };
                let mut exec = Exec {
                    program: &program,
                    bufs: vec![input, None],
                    takes: program.takes.clone(),
                };
                exec.cast(PlanValue(0), PlanValue(1)).unwrap();
                assert_eq!(
                    exec.take(PlanValue(1)).unwrap(),
                    bytes,
                    "{dtype:?}, backing={backing}"
                );
            }
        }
        for (from, to) in [
            (DType::INT8, DType::INT8),
            (DType::INT8, DType::FP32),
            (DType::BF16, DType::FP32),
        ] {
            let program = cast_test_program(&[from, to], 1);
            let mut exec = Exec {
                program: &program,
                bufs: vec![None; 2],
                takes: program.takes.clone(),
            };
            assert!(exec.cast(PlanValue(0), PlanValue(1)).is_err());
        }
    }

    fn rank2(rows: usize, columns: usize) -> ShapeDims {
        shape_dims(&[rows, columns])
    }

    fn shape_dims(dims: &[usize]) -> ShapeDims {
        let mut values = [0; MAX_LEVEL_8K_RANK];
        values[..dims.len()].copy_from_slice(dims);
        ShapeDims {
            values,
            rank: dims.len() as u8,
        }
    }

    #[test]
    fn rank2_broadcast_index_handles_rows_columns_and_planes() {
        assert_eq!(broadcast_index_2d(rank2(1, 3), 1, 2), 2);
        assert_eq!(broadcast_index_2d(rank2(2, 1), 1, 2), 1);
        assert_eq!(broadcast_index_2d(rank2(2, 3), 1, 2), 5);
        assert_eq!(broadcast_index_2d(rank2(1, 1), 1, 2), 0);
    }

    fn broadcast_offsets(dims: &[usize], out: &[usize]) -> Vec<usize> {
        let strides = broadcast_strides(dims, out);
        let elements = out.iter().product();
        let mut coords = [0usize; MAX_LEVEL_8K_RANK];
        (0..elements)
            .map(|index| {
                linear_to_coords(index, out, &mut coords[..out.len()]);
                coords[..out.len()]
                    .iter()
                    .zip(strides)
                    .map(|(coordinate, stride)| coordinate * stride)
                    .sum()
            })
            .collect()
    }

    #[test]
    fn rank1_broadcast_repeats_along_missing_leading_axis() {
        let strides = broadcast_strides(&[3], &[4, 3]);
        assert_eq!(&strides[..2], &[0, 1]);
        assert_eq!(
            broadcast_offsets(&[3], &[4, 3]),
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        );
    }

    #[test]
    fn lower_rank_broadcast_right_aligns_internal_singletons() {
        let strides = broadcast_strides(&[2, 1, 3], &[5, 2, 4, 3]);
        assert_eq!(&strides[..4], &[0, 3, 0, 1]);
        let offsets = broadcast_offsets(&[2, 1, 3], &[5, 2, 4, 3]);
        assert_eq!(&offsets[..12], &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(&offsets[24..36], &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(&offsets[48..60], &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(&offsets[12..24], &[3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]);
    }

    #[test]
    fn rank_mismatched_binary_execution_indexes_trailing_axes() {
        let tensor = |dims: ShapeDims| ValuePlan {
            kind: ValueKind::Tensor {
                dtype: DType::FP32,
                dims,
            },
            byte_size: dims.elements() * size_of::<f32>(),
            constant: None,
        };
        let program = Program {
            values: vec![
                tensor(shape_dims(&[3])),
                tensor(shape_dims(&[4, 3])),
                tensor(shape_dims(&[4, 3])),
            ],
            operators: Vec::new(),
            order: Vec::new(),
            inputs: vec![PlanValue(0), PlanValue(1)],
            outputs: vec![PlanValue(2)],
            takes: vec![1, 1, 1],
            constant_folds: 0,
        };
        let left: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let right: Vec<u8> = (10..22)
            .map(|value| value as f32)
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut exec = program.bind(&[&left, &right]).unwrap();

        exec.binary(&[PlanValue(0), PlanValue(1)], PlanValue(2), |a, b| a + b)
            .unwrap();

        let actual: Vec<f32> = exec
            .take(PlanValue(2))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(
            actual,
            [
                11.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0, 21.0, 20.0, 22.0, 24.0
            ]
        );
    }

    #[test]
    fn folded_fp16_chain_retains_the_f32_compute_plane_until_output() {
        let value = |bits: Option<u16>| ValuePlan {
            kind: ValueKind::Tensor {
                dtype: DType::FP16,
                dims: rank2(1, 1),
            },
            byte_size: 2,
            constant: bits.map(|bits| ConstantValue::Bytes(bits.to_le_bytes().to_vec())),
        };
        let mut program = Program {
            values: vec![
                value(Some(0x3555)), // nearest FP16 to one third
                value(Some(0x4200)), // 3.0
                value(Some(0xbc00)), // -1.0
                value(None),
            ],
            operators: Vec::new(),
            order: vec![Instruction::ScaleAdd {
                input: PlanValue(0),
                scale: PlanValue(1),
                addend: PlanValue(2),
                output: PlanValue(3),
                order: axnn_cpu::elementwise::ScaleAddOrder::InputScaleAddend,
            }],
            inputs: Vec::new(),
            outputs: vec![PlanValue(3)],
            takes: vec![1, 1, 1, 1],
            constant_folds: 0,
        };

        program.fold_constants().unwrap();
        assert!(program.order.is_empty());
        let Some(ConstantValue::Plane(values)) = &program.values[3].constant else {
            panic!("folded FP16 arithmetic narrowed before observation")
        };
        assert_eq!(values, &[-0.000_244_140_63]);
        assert_eq!(program.execute(&[]).unwrap()[0], 0x8c00_u16.to_le_bytes());
        let mut output = [0u8; 2];
        program.execute_into(&[], &mut [&mut output]).unwrap();
        assert_eq!(output, 0x8c00_u16.to_le_bytes());
    }

    #[test]
    fn streaming_constant_fold_promotes_only_the_live_tail() {
        let value = |bits: Option<u16>| ValuePlan {
            kind: ValueKind::Tensor {
                dtype: DType::FP16,
                dims: rank2(1, 1),
            },
            byte_size: 2,
            constant: bits.map(|bits| ConstantValue::Bytes(bits.to_le_bytes().to_vec())),
        };
        let scale_add = |input, output| Instruction::ScaleAdd {
            input: PlanValue(input),
            scale: PlanValue(1),
            addend: PlanValue(2),
            output: PlanValue(output),
            order: axnn_cpu::elementwise::ScaleAddOrder::InputScaleAddend,
        };
        let mut program = Program {
            values: vec![
                value(Some(0x3555)),
                value(Some(0x4200)),
                value(Some(0xbc00)),
                value(None),
                value(None),
            ],
            operators: Vec::new(),
            order: vec![scale_add(0, 3), scale_add(3, 4)],
            inputs: Vec::new(),
            outputs: vec![PlanValue(4)],
            takes: vec![1, 2, 2, 1, 1],
            constant_folds: 0,
        };

        program.fold_constants().unwrap();
        assert!(program.order.is_empty());
        assert!(
            program.values[3].constant.is_none(),
            "the consumed intermediate escaped streaming scratch storage"
        );
        let Some(ConstantValue::Plane(values)) = &program.values[4].constant else {
            panic!("the block-output tail was not promoted")
        };
        assert_eq!(values, &[-1.000_732_4]);
        assert_eq!(program.execute(&[]).unwrap()[0], 0xbc01_u16.to_le_bytes());
    }
}
