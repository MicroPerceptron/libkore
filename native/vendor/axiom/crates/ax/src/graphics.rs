//! Reusable graphics regions expressed in canonical `.ax` operators.
//!
//! These helpers are deliberately graph construction APIs rather than CPU
//! raster entry points. Every provider sees the same ordinary operator
//! region, so a portable executor may fuse it into a SIMD kernel while an
//! accelerator may lower it into its native graph or shader representation.

use core::alloc::Allocator;

use crate::{AllocationError, DType, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId};

/// A premultiplied linear-light RGB field and its matching alpha field.
///
/// All four tensors are expected to have the same dimensions and `F32`
/// element type. Keeping the representation explicit prevents a caller from
/// accidentally feeding straight-alpha colour into source-over arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PremultipliedRgba {
    pub rgb: [TensorId; 3],
    pub alpha: TensorId,
}

/// A prepared scalar field and its screen-space translation derivatives.
///
/// `derivative_x` and `derivative_y` encode the change required to move the
/// visible field in the positive screen direction. Keeping preparation
/// separate lets static glyph/shape coverage remain exact-resolution data
/// while a shallow per-frame displacement stays in the portable graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearizedField2d {
    pub value: TensorId,
    pub derivative_x: TensorId,
    pub derivative_y: TensorId,
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Apply a first-order screen-space translation to a prepared F32 field.
    ///
    /// The emitted region is `value + derivative_x * offset_x +
    /// derivative_y * offset_y`. It is intended for restrained subpixel
    /// parallax, not large image warps. All inputs are ordinary tensors, so
    /// providers may fuse the multiply/add chains without a graphics-specific
    /// backend ABI.
    pub fn linearized_translate_2d_f32(
        &mut self,
        field: LinearizedField2d,
        offset: [TensorId; 2],
        dimensions: &[u32],
    ) -> Result<TensorId, AllocationError> {
        let x = graphics_binary(self, OpCode::MUL, field.derivative_x, offset[0], dimensions)?;
        let y = graphics_binary(self, OpCode::MUL, field.derivative_y, offset[1], dimensions)?;
        let shifted = graphics_binary(self, OpCode::ADD, field.value, x, dimensions)?;
        graphics_binary(self, OpCode::ADD, shifted, y, dimensions)
    }

    /// Composite one solid-colour coverage field over opaque RGB.
    ///
    /// Each channel is emitted as `destination + (colour - destination) *
    /// alpha`, the canonical interpolation region used by fills, glyph ink,
    /// and other single-colour graphics layers. Providers see only portable
    /// arithmetic; the portable executor keeps the three chains in its shared
    /// strict SIMD lerp kernel.
    pub fn solid_source_over_opaque_f32(
        &mut self,
        alpha: TensorId,
        colour: [f32; 3],
        destination: [TensorId; 3],
        dimensions: &[u32],
    ) -> Result<[TensorId; 3], AllocationError> {
        let scalar_dimensions = [1_u32; axnn::MAX_RANK];
        let scalar_dimensions =
            scalar_dimensions
                .get(..dimensions.len())
                .ok_or(crate::AxError::RankTooLarge(
                    u8::try_from(dimensions.len()).unwrap_or(u8::MAX),
                ))?;
        let mut output = [TensorId(0); 3];
        for channel in 0..3 {
            let target = self.tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                scalar_dimensions,
                &colour[channel].to_le_bytes(),
            )?;
            let difference =
                graphics_binary(self, OpCode::SUB, target, destination[channel], dimensions)?;
            let weighted = graphics_binary(self, OpCode::MUL, difference, alpha, dimensions)?;
            output[channel] = graphics_binary(
                self,
                OpCode::ADD,
                destination[channel],
                weighted,
                dimensions,
            )?;
        }
        Ok(output)
    }

    /// Composite a premultiplied source over an opaque RGB destination.
    ///
    /// The emitted canonical region is, per channel,
    /// `source + destination * (1 - source_alpha)`. The portable TOSA
    /// executor recognizes each single-consumer `MUL -> ADD` pair and keeps
    /// the product in SIMD registers; other providers retain ordinary,
    /// backend-independent operators that they can lower or fuse themselves.
    pub fn premultiplied_source_over_opaque_f32(
        &mut self,
        source: PremultipliedRgba,
        destination: [TensorId; 3],
        dimensions: &[u32],
    ) -> Result<[TensorId; 3], AllocationError> {
        let scalar_dimensions = [1_u32; axnn::MAX_RANK];
        let scalar_dimensions =
            scalar_dimensions
                .get(..dimensions.len())
                .ok_or(crate::AxError::RankTooLarge(
                    u8::try_from(dimensions.len()).unwrap_or(u8::MAX),
                ))?;
        let one = self.tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            scalar_dimensions,
            &1.0_f32.to_le_bytes(),
        )?;
        let retained_alpha = graphics_binary(self, OpCode::SUB, one, source.alpha, dimensions)?;
        let mut output = [TensorId(0); 3];
        for channel in 0..3 {
            let retained_destination = graphics_binary(
                self,
                OpCode::MUL,
                destination[channel],
                retained_alpha,
                dimensions,
            )?;
            output[channel] = graphics_binary(
                self,
                OpCode::ADD,
                source.rgb[channel],
                retained_destination,
                dimensions,
            )?;
        }
        Ok(output)
    }
}

fn graphics_binary<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    op: OpCode,
    left: TensorId,
    right: TensorId,
    dimensions: &[u32],
) -> Result<TensorId, AllocationError> {
    let output = builder.tensor(DType::F32, TensorFlags::default(), dimensions, &[])?;
    builder.node(op, NodeFlags::default(), &[left, right], &[output], &[])?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn premultiplied_over_opaque_is_a_portable_fusible_region() {
        let mut builder = GraphBuilder::new_in(Global);
        let dimensions = [2, 4];
        let field = |builder: &mut GraphBuilder<Global>| {
            builder
                .tensor(DType::F32, TensorFlags::INPUT, &dimensions, &[])
                .unwrap()
        };
        let source = PremultipliedRgba {
            rgb: core::array::from_fn(|_| field(&mut builder)),
            alpha: field(&mut builder),
        };
        let destination = core::array::from_fn(|_| field(&mut builder));
        let output = builder
            .premultiplied_source_over_opaque_f32(source, destination, &dimensions)
            .unwrap();
        for channel in output {
            let marked = builder
                .tensor(DType::F32, TensorFlags::OUTPUT, &dimensions, &[])
                .unwrap();
            builder
                .node(
                    OpCode::IDENTITY,
                    NodeFlags::default(),
                    &[channel],
                    &[marked],
                    &[],
                )
                .unwrap();
        }

        let encoded = builder.finish().unwrap();
        let graph = encoded.parse().unwrap();
        assert_eq!(graph.node_count(), 10);
        let mut sub = 0;
        let mut mul = 0;
        let mut add = 0;
        for index in 0..graph.node_count() {
            match graph.node(index).unwrap().op {
                OpCode::SUB => sub += 1,
                OpCode::MUL => mul += 1,
                OpCode::ADD => add += 1,
                OpCode::IDENTITY => {}
                op => panic!("unexpected graphics op {op:?}"),
            }
        }
        assert_eq!((sub, mul, add), (1, 3, 3));
    }

    #[test]
    fn linearized_translation_is_canonical_mul_add_arithmetic() {
        let mut builder = GraphBuilder::new_in(Global);
        let dimensions = [2, 4];
        let field = |builder: &mut GraphBuilder<Global>, dimensions: &[u32]| {
            builder
                .tensor(DType::F32, TensorFlags::INPUT, dimensions, &[])
                .unwrap()
        };
        let prepared = LinearizedField2d {
            value: field(&mut builder, &dimensions),
            derivative_x: field(&mut builder, &dimensions),
            derivative_y: field(&mut builder, &dimensions),
        };
        let offset = [field(&mut builder, &[1, 1]), field(&mut builder, &[1, 1])];
        let output = builder
            .linearized_translate_2d_f32(prepared, offset, &dimensions)
            .unwrap();
        let marked = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &dimensions, &[])
            .unwrap();
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[output],
                &[marked],
                &[],
            )
            .unwrap();

        let encoded = builder.finish().unwrap();
        let graph = encoded.parse().unwrap();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.node(0).unwrap().op, OpCode::MUL);
        assert_eq!(graph.node(1).unwrap().op, OpCode::MUL);
        assert_eq!(graph.node(2).unwrap().op, OpCode::ADD);
        assert_eq!(graph.node(3).unwrap().op, OpCode::ADD);
        assert_eq!(graph.node(4).unwrap().op, OpCode::IDENTITY);
    }

    #[test]
    fn solid_source_over_is_three_portable_lerp_regions() {
        let mut builder = GraphBuilder::new_in(Global);
        let dimensions = [2, 4];
        let field = |builder: &mut GraphBuilder<Global>| {
            builder
                .tensor(DType::F32, TensorFlags::INPUT, &dimensions, &[])
                .unwrap()
        };
        let alpha = field(&mut builder);
        let destination = core::array::from_fn(|_| field(&mut builder));
        let output = builder
            .solid_source_over_opaque_f32(alpha, [0.1, 0.2, 0.3], destination, &dimensions)
            .unwrap();
        for channel in output {
            let marked = builder
                .tensor(DType::F32, TensorFlags::OUTPUT, &dimensions, &[])
                .unwrap();
            builder
                .node(
                    OpCode::IDENTITY,
                    NodeFlags::default(),
                    &[channel],
                    &[marked],
                    &[],
                )
                .unwrap();
        }

        let encoded = builder.finish().unwrap();
        let graph = encoded.parse().unwrap();
        assert_eq!(graph.node_count(), 12);
        for channel in 0..3 {
            assert_eq!(graph.node(channel * 3).unwrap().op, OpCode::SUB);
            assert_eq!(graph.node(channel * 3 + 1).unwrap().op, OpCode::MUL);
            assert_eq!(graph.node(channel * 3 + 2).unwrap().op, OpCode::ADD);
        }
    }
}
