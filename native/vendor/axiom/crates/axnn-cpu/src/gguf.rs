//! Import canonical little-endian GGML blocks into Axiom's packed lane order.
//!
//! Q4_K and Q6_K share block sizes with Axiom Q4/Q6 but **not** quant lane
//! order; Q4 scale bits also differ. Import once when admitting a weight page; retain the compact result.
//! This module parses no GGUF metadata and performs no allocation or float
//! conversion. The caller must validate tensor type, dimensions and file range.
use crate::{CpuError, quant};
use axnn::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlBlockType {
    Q8_0,
    Q4K,
    Q6K,
}
impl GgmlBlockType {
    pub const fn dtype(self) -> DType {
        match self {
            Self::Q8_0 => DType::Q8,
            Self::Q4K => DType::Q4,
            Self::Q6K => DType::Q6,
        }
    }
}

/// Convert whole GGML blocks into caller-owned Axiom packed storage. Metadata
/// values (including signed scales and IEEE half bits) are preserved exactly.
/// Size validation happens before writing any destination byte.
pub fn import_ggml_blocks(
    kind: GgmlBlockType,
    source: &[u8],
    destination: &mut [u8],
) -> Result<(), CpuError> {
    let bytes = kind.dtype().storage_bytes(kind.dtype().block_size());
    if source.len() != destination.len() || source.len() % bytes != 0 {
        return Err(CpuError::ShapeMismatch {
            op: "GGML packed import",
        });
    }
    for (src, dst) in source
        .chunks_exact(bytes)
        .zip(destination.chunks_exact_mut(bytes))
    {
        dst.copy_from_slice(src);
        match kind {
            GgmlBlockType::Q8_0 => {}
            GgmlBlockType::Q4K => {
                let sc = &src[4..16];
                let mut scales = [0; 8];
                let mut mins = [0; 8];
                for j in 0..8 {
                    scales[j] = if j < 4 {
                        sc[j] & 63
                    } else {
                        (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4)
                    };
                    mins[j] = if j < 4 {
                        sc[j + 4] & 63
                    } else {
                        (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)
                    };
                }
                quant::pack_q4_scales(&scales, &mins, &mut dst[4..16]);
                dst[quant::Q4_QS_OFFSET..].fill(0);
                for j in 0..256 {
                    let lane = j % 64;
                    let q = (src[quant::Q4_QS_OFFSET + j / 64 * 32 + lane % 32] >> (lane / 32 * 4))
                        & 15;
                    dst[quant::Q4_QS_OFFSET + j / 2] |= q << (j % 2 * 4);
                }
            }
            GgmlBlockType::Q6K => {
                dst[..quant::Q6_SC_OFFSET].fill(0);
                for j in 0..256 {
                    let group = j % 128 / 32;
                    let lane = j % 32;
                    let low = (src[j / 128 * 64 + group % 2 * 32 + lane] >> (group / 2 * 4)) & 15;
                    let high = (src[quant::Q6_QH_OFFSET + j / 128 * 32 + lane] >> (group * 2)) & 3;
                    dst[j / 2] |= low << (j % 2 * 4);
                    dst[quant::Q6_QH_OFFSET + j / 4] |= high << (j % 4 * 2);
                }
            }
        }
    }
    Ok(())
}

/// Import a mutable weight page in place using at most one 210-byte stack
/// block. Q8_0 requires no lane conversion and is left untouched. Callers can
/// retain the page directly, avoiding a second page-sized destination buffer.
pub fn import_ggml_blocks_in_place(kind: GgmlBlockType, page: &mut [u8]) -> Result<(), CpuError> {
    let bytes = kind.dtype().block_byte_size();
    if !page.len().is_multiple_of(bytes) {
        return Err(CpuError::ShapeMismatch {
            op: "GGML packed import",
        });
    }
    if kind == GgmlBlockType::Q8_0 {
        return Ok(());
    }
    let mut scratch = [0u8; quant::Q6_BLOCK_BYTES];
    for block in page.chunks_exact_mut(bytes) {
        scratch[..bytes].copy_from_slice(block);
        import_ggml_blocks(kind, &scratch[..bytes], block)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, invoke_matmul_transposed, plan_cpu_kernel, quant};
    use axnn::lowering::KernelRequest;
    use axnn::{TensorDesc, shape};

    #[test]
    fn canonical_k_quant_lanes_reach_graph_kernel_without_float_repacking() {
        for kind in [GgmlBlockType::Q4K, GgmlBlockType::Q6K] {
            let mut source = [0u8; 420];
            let len = kind.dtype().storage_bytes(512);
            let mut oracle = [0f32; 512];
            for (block, src) in source[..len].chunks_exact_mut(len / 2).enumerate() {
                match kind {
                    GgmlBlockType::Q4K => {
                        src[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                        src[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                        // Exercise both halves of the canonical six-bit metadata.
                        let scales = [1u8, 31, 47, 63, 2, 17, 33, 62];
                        let mins = [2u8, 33, 17, 62, 1, 47, 31, 63];
                        for j in 0..4 {
                            src[4 + j] = scales[j] | ((scales[j + 4] >> 4) << 6);
                            src[8 + j] = mins[j] | ((mins[j + 4] >> 4) << 6);
                            src[12 + j] = (scales[j + 4] & 15) | ((mins[j + 4] & 15) << 4);
                        }
                        for group in 0..4 {
                            for lane in 0..32 {
                                let lo = ((lane * 3 + group + block) % 16) as u8;
                                let hi = ((lane * 7 + group + block + 5) % 16) as u8;
                                src[16 + group * 32 + lane] = lo | (hi << 4);
                                oracle[block * 256 + group * 64 + lane] = lo as f32
                                    * scales[group * 2] as f32
                                    - 0.5 * mins[group * 2] as f32;
                                oracle[block * 256 + group * 64 + 32 + lane] = hi as f32
                                    * scales[group * 2 + 1] as f32
                                    - 0.5 * mins[group * 2 + 1] as f32;
                            }
                        }
                    }
                    GgmlBlockType::Q6K => {
                        src[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
                        for (i, sc) in src[192..208].iter_mut().enumerate() {
                            *sc = (i as i8 - 7) as u8;
                        }
                        for half in 0..2 {
                            for lane in 0..32 {
                                let q: [u8; 4] = core::array::from_fn(|g| {
                                    ((lane * 3 + g * 13 + block * 7) % 64) as u8
                                });
                                src[half * 64 + lane] = q[0] & 15 | (q[2] & 15) << 4;
                                src[half * 64 + 32 + lane] = q[1] & 15 | (q[3] & 15) << 4;
                                src[128 + half * 32 + lane] = q[0] >> 4
                                    | (q[1] >> 4) << 2
                                    | (q[2] >> 4) << 4
                                    | (q[3] >> 4) << 6;
                                for g in 0..4 {
                                    let index = half * 128 + g * 32 + lane;
                                    oracle[block * 256 + index] = (q[g] as i32 - 32) as f32
                                        * (src[192 + index / 16] as i8) as f32;
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            let mut packed = [0u8; 420];
            import_ggml_blocks(kind, &source[..len], &mut packed[..len]).unwrap();
            let mut in_place = source;
            import_ggml_blocks_in_place(kind, &mut in_place[..len]).unwrap();
            assert_eq!(in_place[..len], packed[..len]);
            let mut decoded = [0f32; 512];
            match kind {
                GgmlBlockType::Q4K => quant::dequantize_q4(&packed[..len], &mut decoded),
                _ => quant::dequantize_q6(&packed[..len], &mut decoded),
            }
            assert_eq!(decoded, oracle);
            let mut x = [0u8; 544];
            for (b, block) in x.chunks_exact_mut(34).enumerate() {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                for j in 0..32 {
                    block[2 + j] = (j as i8 - 17 + (b % 3) as i8) as u8;
                }
            }
            let expected: f32 = oracle
                .iter()
                .enumerate()
                .map(|(j, w)| w * (x[j / 32 * 34 + 2 + j % 32] as i8) as f32)
                .sum();
            let w = TensorDesc::new(shape![1, 512], kind.dtype());
            let a = TensorDesc::new(shape![1, 512], DType::Q8);
            let o = TensorDesc::new(shape![1, 1], DType::F32);
            let request = KernelRequest::matmul_transposed(&w, &a, &o).unwrap();
            for caps in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
                let plan = plan_cpu_kernel(&request, caps).unwrap();
                let mut output = [f32::NAN];
                invoke_matmul_transposed(&plan, &packed[..len], &x, &mut output).unwrap();
                assert_eq!(output[0], expected);
                assert!(
                    invoke_matmul_transposed(&plan, &packed[..len - 1], &x, &mut output).is_err()
                );
                assert_eq!(output[0], expected);
            }
        }
    }
    #[test]
    fn malformed_import_preserves_destination() {
        let mut dst = [42; 144];
        assert!(import_ggml_blocks(GgmlBlockType::Q4K, &[0; 143], &mut dst).is_err());
        assert_eq!(dst, [42; 144]);
    }
}
