//! TOSA frontend for the backend-independent `.ax` language.

use alloc::vec::Vec;
use core::alloc::Allocator;

use ax::{
    DType as AxDType, DYNAMIC_DIM, GraphBuf, NodeFlags, NodeSpec, OpCode, TensorFlags, TensorId,
    TensorSpec,
};
use virtio_accel_tosa::{
    AnalyzedValueKind, ConstantState, DType, NanPropagationMode, Op, OpAttributes, TosaAnalysis,
    ValueId, parse,
};

#[derive(Debug)]
pub enum AxLowerError {
    Parse(virtio_accel_tosa::Error),
    Analysis(virtio_accel_tosa::AnalysisError),
    AllocationFailed,
    MultiBlock,
    UnrankedTensor,
    UnsupportedDType(DType),
    UnsupportedOperator(Op),
    UnsupportedAttributes(Op),
    InvalidDimension(i32),
    Encode(ax::AllocationError),
}

impl From<virtio_accel_tosa::Error> for AxLowerError {
    fn from(value: virtio_accel_tosa::Error) -> Self {
        Self::Parse(value)
    }
}

impl From<virtio_accel_tosa::AnalysisError> for AxLowerError {
    fn from(value: virtio_accel_tosa::AnalysisError) -> Self {
        Self::Analysis(value)
    }
}

impl From<ax::AllocationError> for AxLowerError {
    fn from(value: ax::AllocationError) -> Self {
        Self::Encode(value)
    }
}

struct LowerTensor<'a, A: Allocator> {
    dtype: AxDType,
    flags: TensorFlags,
    dimensions: Vec<u32, A>,
    constant: &'a [u8],
}

struct LowerNode<A: Allocator> {
    op: OpCode,
    inputs: Vec<TensorId, A>,
    outputs: Vec<TensorId, A>,
}

/// Validate a TOSA artifact and lower its single block into canonical `.ax`
/// bytes owned by `allocator`.
///
/// The bridge intentionally lives in `axtosa`: TOSA is a frontend, not an
/// `ax` dependency. The initial operator set covers the attribute-free neural
/// path used by fused MLP/CPPN graphs; richer TOSA attributes will be mapped to
/// canonical `ax` attributes separately.
pub fn lower_ax_in<A: Allocator + Clone>(
    artifact: &[u8],
    allocator: A,
) -> Result<GraphBuf<A>, AxLowerError> {
    let model = parse(artifact)?;
    let analysis = model.analyze_for(super::target())?;
    lower_analysis_in(&analysis, allocator)
}

fn lower_analysis_in<A: Allocator + Clone>(
    analysis: &TosaAnalysis<'_>,
    allocator: A,
) -> Result<GraphBuf<A>, AxLowerError> {
    if analysis.regions().len() != 1 || analysis.blocks().len() != 1 {
        return Err(AxLowerError::MultiBlock);
    }
    let block = analysis.blocks()[0].id();
    let block_inputs = analysis.block_inputs(block);
    let block_outputs = analysis.block_outputs(block);

    let mut tensors = Vec::new_in(allocator.clone());
    reserve(&mut tensors, analysis.values().len())?;
    for value in analysis.values() {
        let mut flags = TensorFlags(0);
        if block_inputs.contains(&value.id()) {
            flags = flags | TensorFlags::INPUT;
        }
        if block_outputs.contains(&value.id()) {
            flags = flags | TensorFlags::OUTPUT;
        }
        let constant = analysis.serialized_constant(value.id()).unwrap_or(&[]);
        if value.constant() == ConstantState::Serialized {
            flags = flags | TensorFlags::CONSTANT;
        }
        let (dtype, dimensions) = match value.kind() {
            AnalyzedValueKind::Tensor(tensor) => {
                let mut dimensions = Vec::new_in(allocator.clone());
                reserve(
                    &mut dimensions,
                    tensor.rank().ok_or(AxLowerError::UnrankedTensor)?,
                )?;
                for dimension in tensor.dimensions() {
                    let dimension = if dimension == -1 {
                        DYNAMIC_DIM
                    } else {
                        u32::try_from(dimension)
                            .map_err(|_| AxLowerError::InvalidDimension(dimension))?
                    };
                    dimensions.push(dimension);
                }
                (map_dtype(tensor.dtype())?, dimensions)
            }
            AnalyzedValueKind::Shape(shape) => {
                let mut dimensions = Vec::new_in(allocator.clone());
                reserve(&mut dimensions, 1)?;
                dimensions.push(shape.rank());
                (AxDType::I64, dimensions)
            }
        };
        tensors.push(LowerTensor {
            dtype,
            flags,
            dimensions,
            constant,
        });
    }

    let mut nodes = Vec::new_in(allocator.clone());
    reserve(&mut nodes, analysis.execution_order(block).len())?;
    for &operator_id in analysis.execution_order(block) {
        let operator = analysis.operator(operator_id);
        let Some(op) = map_op(operator.op())? else {
            continue;
        };
        let attributes = operator.source().attributes();
        let supported_attributes = matches!(attributes, OpAttributes::Empty { .. })
            || matches!(
                attributes,
                OpAttributes::Maximum {
                    nan_mode: NanPropagationMode::PROPAGATE
                } | OpAttributes::Minimum {
                    nan_mode: NanPropagationMode::PROPAGATE
                }
            );
        if !supported_attributes {
            return Err(AxLowerError::UnsupportedAttributes(operator.op()));
        }
        let operator_inputs = analysis.operator_inputs(operator_id);
        let input_count = if operator.op() == Op::MATMUL || operator.op() == Op::MUL {
            let dtype = match analysis.value(operator_inputs[0]).kind() {
                AnalyzedValueKind::Tensor(tensor) => tensor.dtype(),
                AnalyzedValueKind::Shape(_) => {
                    return Err(AxLowerError::UnsupportedOperator(operator.op()));
                }
            };
            if !is_float(dtype) {
                return Err(AxLowerError::UnsupportedOperator(operator.op()));
            }
            2
        } else if operator.op() == Op::NEGATE {
            let dtype = match analysis.value(operator_inputs[0]).kind() {
                AnalyzedValueKind::Tensor(tensor) => tensor.dtype(),
                AnalyzedValueKind::Shape(_) => {
                    return Err(AxLowerError::UnsupportedOperator(operator.op()));
                }
            };
            if !is_float(dtype) {
                return Err(AxLowerError::UnsupportedOperator(operator.op()));
            }
            1
        } else {
            operator_inputs.len()
        };
        let mut inputs = Vec::new_in(allocator.clone());
        reserve(&mut inputs, input_count)?;
        for &value in &operator_inputs[..input_count] {
            inputs.push(tensor_id(value));
        }
        let mut outputs = Vec::new_in(allocator.clone());
        reserve(&mut outputs, analysis.operator_outputs(operator_id).len())?;
        for &value in analysis.operator_outputs(operator_id) {
            outputs.push(tensor_id(value));
        }
        nodes.push(LowerNode {
            op,
            inputs,
            outputs,
        });
    }

    let mut tensor_specs = Vec::new_in(allocator.clone());
    reserve(&mut tensor_specs, tensors.len())?;
    for tensor in &tensors {
        tensor_specs.push(TensorSpec {
            dtype: tensor.dtype,
            flags: tensor.flags,
            dimensions: &tensor.dimensions,
            constant: tensor.constant,
        });
    }
    let mut node_specs = Vec::new_in(allocator.clone());
    reserve(&mut node_specs, nodes.len())?;
    for node in &nodes {
        node_specs.push(NodeSpec {
            op: node.op,
            flags: NodeFlags::default(),
            inputs: &node.inputs,
            outputs: &node.outputs,
            attributes: &[],
        });
    }
    Ok(GraphBuf::encode_in(&tensor_specs, &node_specs, allocator)?)
}

fn map_dtype(dtype: DType) -> Result<AxDType, AxLowerError> {
    Ok(if dtype == DType::BOOL {
        AxDType::U8
    } else if dtype == DType::INT8 {
        AxDType::I8
    } else if dtype == DType::INT16 {
        AxDType::I16
    } else if dtype == DType::INT32 {
        AxDType::I32
    } else if dtype == DType::FP32 {
        AxDType::F32
    } else if dtype == DType::FP16 {
        AxDType::F16
    } else if dtype == DType::BF16 {
        AxDType::B16
    } else if dtype == DType::FP8E4M3 {
        AxDType::B8
    } else if dtype == DType::FP8E5M2 {
        AxDType::F8
    } else {
        return Err(AxLowerError::UnsupportedDType(dtype));
    })
}

fn is_float(dtype: DType) -> bool {
    dtype == DType::FP32
        || dtype == DType::FP16
        || dtype == DType::BF16
        || dtype == DType::FP8E4M3
        || dtype == DType::FP8E5M2
}

fn map_op(op: Op) -> Result<Option<OpCode>, AxLowerError> {
    Ok(Some(if op == Op::CONST || op == Op::CONST_SHAPE {
        return Ok(None);
    } else if op == Op::IDENTITY {
        OpCode::IDENTITY
    } else if op == Op::CAST {
        OpCode::CAST
    } else if op == Op::ADD {
        OpCode::ADD
    } else if op == Op::SUB {
        OpCode::SUB
    } else if op == Op::MUL {
        OpCode::MUL
    } else if op == Op::POW {
        OpCode::POW
    } else if op == Op::MAXIMUM {
        OpCode::MAXIMUM
    } else if op == Op::MINIMUM {
        OpCode::MINIMUM
    } else if op == Op::MATMUL {
        OpCode::MATMUL
    } else if op == Op::ABS {
        OpCode::ABS
    } else if op == Op::CEIL {
        OpCode::CEIL
    } else if op == Op::COS {
        OpCode::COS
    } else if op == Op::EXP {
        OpCode::EXP
    } else if op == Op::FLOOR {
        OpCode::FLOOR
    } else if op == Op::LOG {
        OpCode::LOG
    } else if op == Op::LOGICAL_NOT {
        OpCode::LOGICAL_NOT
    } else if op == Op::NEGATE {
        OpCode::NEGATE
    } else if op == Op::RECIPROCAL {
        OpCode::RECIPROCAL
    } else if op == Op::RSQRT {
        OpCode::RSQRT
    } else if op == Op::SIN {
        OpCode::SIN
    } else if op == Op::SELECT {
        OpCode::SELECT
    } else if op == Op::EQUAL {
        OpCode::EQUAL
    } else if op == Op::GREATER {
        OpCode::GREATER
    } else if op == Op::GREATER_EQUAL {
        OpCode::GREATER_EQUAL
    } else if op == Op::LOGICAL_AND {
        OpCode::LOGICAL_AND
    } else if op == Op::LOGICAL_OR {
        OpCode::LOGICAL_OR
    } else if op == Op::LOGICAL_XOR {
        OpCode::LOGICAL_XOR
    } else if op == Op::TANH {
        OpCode::TANH
    } else if op == Op::SIGMOID {
        OpCode::SIGMOID
    } else if op == Op::RESHAPE {
        OpCode::RESHAPE
    } else {
        return Err(AxLowerError::UnsupportedOperator(op));
    }))
}

const fn tensor_id(value: ValueId) -> TensorId {
    TensorId(value.get())
}

fn reserve<T, A: Allocator>(values: &mut Vec<T, A>, additional: usize) -> Result<(), AxLowerError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| AxLowerError::AllocationFailed)
}
