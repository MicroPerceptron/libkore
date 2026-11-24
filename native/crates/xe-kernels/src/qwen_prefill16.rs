//! Sealed M16 families. This module is compiled only with the explicit
//! `qwen-prefill16-native` feature, so a missing family is a build failure.
//! The Xe provider still checks the selected generation and package envelope.

use super::{QWEN3_5_MOE_HIDDEN, QWEN_HIDDEN, QwenDense, QwenProjection, XeKernelArtifact, artifact};

pub fn qwen_projection_prefill16(kind: QwenProjection) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenProjection::DeltaQkv => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n10240k5120-e0.kxefamily")),
        QwenProjection::DeltaGate => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n6144k5120-e1.kxefamily")),
        QwenProjection::DeltaBeta => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n48k5120-e2.kxefamily")),
        QwenProjection::DeltaStep => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n48k5120-e0.kxefamily")),
        QwenProjection::MixerOut => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n5120k6144-e0.kxefamily")),
        QwenProjection::GatedQuery => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n12288k5120-e0.kxefamily")),
        QwenProjection::KeyValue => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n1024k5120-e0.kxefamily")),
        QwenProjection::MlpGate => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n17408k5120-e1.kxefamily")),
        QwenProjection::MlpUp => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n17408k5120-e0.kxefamily")),
        QwenProjection::MlpDown => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n5120k17408-e0.kxefamily")),
        QwenProjection::MtpFc => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n5120k5120-e0.kxefamily")),
        QwenProjection::MoeMixerIn => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n8192k2048-e0.kxefamily")),
        QwenProjection::MoeDeltaGate => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n4096k2048-e1.kxefamily")),
        QwenProjection::MoeMixerOut => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n2048k4096-e0.kxefamily")),
        QwenProjection::MoeKeyValue => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n512k2048-e0.kxefamily")),
        QwenProjection::MoeMtpFc => artifact(include_bytes!("../../../drivers/xe/kernels/anv-nvfp4-m16n2048k2048-e0.kxefamily")),
        _ => return None,
    })
}

pub fn qwen_lm_head_prefill16(vocab: u32, hidden: u32) -> Option<XeKernelArtifact> {
    match (vocab, hidden) {
        (248320, QWEN_HIDDEN) => Some(artifact(include_bytes!("../../../drivers/xe/kernels/anv-dense-bf16-m16n248320k5120.kxefamily"))),
        (248320, QWEN3_5_MOE_HIDDEN) => Some(artifact(include_bytes!("../../../drivers/xe/kernels/anv-dense-bf16-m16n248320k2048.kxefamily"))),
        _ => None,
    }
}

pub fn qwen_dense_prefill16(kind: QwenDense) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenDense::MoeRouter => artifact(include_bytes!("../../../drivers/xe/kernels/anv-dense-bf16-m16n257k2048-e0.kxefamily")),
        QwenDense::MoeStep => artifact(include_bytes!("../../../drivers/xe/kernels/anv-dense-bf16-m16n32k2048-e0.kxefamily")),
        QwenDense::MoeBeta => artifact(include_bytes!("../../../drivers/xe/kernels/anv-dense-bf16-m16n32k2048-e2.kxefamily")),
    })
}

pub fn f32_row_gather_prefill16(rows: u32, width: u32) -> Option<XeKernelArtifact> {
    match (rows, width) {
        (768, 256) => Some(artifact(include_bytes!("../../../drivers/xe/kernels/anv-gather-f32-v768-h256.kxefamily"))),
        (512, 256) => Some(artifact(include_bytes!("../../../drivers/xe/kernels/anv-gather-f32-v512-h256.kxefamily"))),
        _ => None,
    }
}
