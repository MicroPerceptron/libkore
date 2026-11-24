//! Architecture-level checked Xe kernel artifacts shared by clients and the
//! native provider. A caller names an operation, never a silicon generation.
//! The Xe provider selects its native package using the generation established
//! at founding. These artifacts contain code and metadata, not tensor bytes.

#![no_std]

use kore_driver::accel::FORMAT_XE_NATIVE;
use virtio_accel_core::{ArtifactRef, ByteSource, TargetIdentity};

/// Generation-transparent general vector operations for Xe EUs.
pub mod xve;

#[cfg(feature = "qwen-prefill16-native")]
mod qwen_prefill16;
#[cfg(feature = "qwen-prefill16-native")]
pub use qwen_prefill16::{qwen_dense_prefill16, qwen_lm_head_prefill16, qwen_projection_prefill16, f32_row_gather_prefill16};

/// Dedicated prompt packages are admitted only when all exact M16 family
/// files are present at build time. Source shaders never satisfy this gate.
pub const QWEN_PREFILL16_ROWS: u32 = 16;
pub const QWEN_PREFILL16_NATIVE: bool = cfg!(feature = "qwen-prefill16-native");

#[cfg(not(feature = "qwen-prefill16-native"))]
pub fn qwen_projection_prefill16(_: QwenProjection) -> Option<XeKernelArtifact> { None }
#[cfg(not(feature = "qwen-prefill16-native"))]
pub fn qwen_lm_head_prefill16(_: u32, _: u32) -> Option<XeKernelArtifact> { None }
#[cfg(not(feature = "qwen-prefill16-native"))]
pub fn qwen_dense_prefill16(_: QwenDense) -> Option<XeKernelArtifact> { None }
#[cfg(not(feature = "qwen-prefill16-native"))]
pub fn f32_row_gather_prefill16(_: u32, _: u32) -> Option<XeKernelArtifact> { None }

/// One immutable architecture-level program. The client stages [`Self::bytes`]
/// for Kore's accelerator service; a native `Accelerator` uses
/// [`Self::artifact_ref`]. Neither method allocates or copies model buffers.
#[derive(Clone, Copy, Debug)]
pub struct XeKernelArtifact {
    bytes: &'static [u8],
    source: &'static dyn ByteSource,
}

impl XeKernelArtifact {
    /// The exact measured program payload for one `LoadProgram` request.
    pub const fn bytes(self) -> &'static [u8] {
        self.bytes
    }

    /// The bytes the provider stages per ordered sequence that names this
    /// program: one package's, the largest in a family (the provider
    /// resolves one generation; the lane cannot know which), or the whole
    /// payload of a lone package. Charging a family's every generation
    /// halved the lane's windows on a two-generation catalog.
    pub fn staged_bytes(self) -> usize {
        family_largest_package(self.bytes).unwrap_or(self.bytes.len())
    }

    /// Native accelerator envelope over the same immutable bytes.
    pub fn artifact_ref(self) -> ArtifactRef<'static> {
        ArtifactRef {
            format: FORMAT_XE_NATIVE,
            target: TargetIdentity([0; 12]),
            payload: self.source,
            resident_bytes: self.bytes.len() as u64,
        }
    }

    /// Whether `other` is the same package. Compared by content, never by
    /// address alone: a small catalog function (the sparse down, the RMS
    /// forms, the gather, the XVE maps, the dense projections) is inlined
    /// into each crate that calls it, and each inlined copy carries its own
    /// copy of the package bytes. The program an app loaded and the kernel
    /// an executor selects are then one package at two addresses.
    pub fn same(self, other: Self) -> bool {
        let (a, b) = (self.bytes, other.bytes);
        a.len() == b.len() && (a.as_ptr() == b.as_ptr() || a == b)
    }
}

/// The XMX FP16 matrix witness: A `[8, 16]`, B `[16, 16]`, and F32 C `[8, 16]`.
///
/// Slots 0 and 1 read A and B; slot 2 writes C. The loaded package fixes its
/// local workgroup shape and is submitted with artifact-defined launch. This
/// small kernel establishes the generation-transparent path, not throughput.
pub fn xmx_matmul() -> XeKernelArtifact {
    artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-xmx-matmul.kxefamily"
    ))
}

/// Exact precompiled NVFP4 projection specializations. The checkpoint shapes
/// below carry separately captured Xe2 and Xe3 ISA; missing variants fail at
/// load. Routed experts run as grouped waves instead ([`moe_wave`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Projection {
    /// M=1, N=17, K=32; no activation.
    DecodeTile,
    /// M=8, N=32, K=32; SiLU activation.
    PrefillSiluTile,
    /// Mamba gate, SiLU fused after `[4096,2688]` projection.
    MambaGateSiluDecode,
    MambaGateSiluPrefill,
    /// Mamba X/B/C projection `[6144,2688]`.
    MambaXbcDecode,
    MambaXbcPrefill,
    /// Mamba time-step projection `[64,2688]`.
    MambaDtDecode,
    MambaDtPrefill,
    /// Mamba out or attention out projection `[2688,4096]`.
    HiddenOutDecode,
    HiddenOutPrefill,
    /// Attention query projection `[4096,2688]`.
    AttentionQDecode,
    AttentionQPrefill,
    /// Attention K or V projection `[256,2688]`.
    AttentionKvDecode,
    AttentionKvPrefill,
    /// Shared expert up projection `[3712,2688]` with squared ReLU.
    SharedUpSquaredReluDecode,
    SharedUpSquaredReluPrefill,
    /// Runtime row count for multi-row prompts (Xe2-only).
    SharedUpSquaredReluWide,
    /// Shared expert down projection `[2688,3712]`, added into the routed
    /// output using slot 5 with a route scale of 1.0.
    SharedDownAccumulateDecode,
    SharedDownAccumulatePrefill,
    /// Runtime row count for multi-row prompts (Xe2-only).
    SharedDownAccumulateWide,
    /// Final untied vocabulary projection `[131072,2688]`.
    LmHeadDecode,
    LmHeadPrefill,
    /// Runtime row count for multi-row prompts (Xe2-only).
    LmHeadWide,
}

impl Nvfp4Projection {
    /// Every specialization, for callers that check the whole catalog.
    pub const ALL: [Self; 23] = [
        Self::DecodeTile,
        Self::PrefillSiluTile,
        Self::MambaGateSiluDecode,
        Self::MambaGateSiluPrefill,
        Self::MambaXbcDecode,
        Self::MambaXbcPrefill,
        Self::MambaDtDecode,
        Self::MambaDtPrefill,
        Self::HiddenOutDecode,
        Self::HiddenOutPrefill,
        Self::AttentionQDecode,
        Self::AttentionQPrefill,
        Self::AttentionKvDecode,
        Self::AttentionKvPrefill,
        Self::SharedUpSquaredReluDecode,
        Self::SharedUpSquaredReluPrefill,
        Self::SharedUpSquaredReluWide,
        Self::SharedDownAccumulateDecode,
        Self::SharedDownAccumulatePrefill,
        Self::SharedDownAccumulateWide,
        Self::LmHeadDecode,
        Self::LmHeadPrefill,
        Self::LmHeadWide,
    ];

    /// Invocations per workgroup, which every generation's package in this
    /// kind's family was captured with; the provider refuses any other.
    /// Every kind launches `[ceil(N / 16), ceil(M / 8), 1]` workgroups, one
    /// 8x16 output tile each, whichever kernel it was sealed from:
    /// xe-nvfp4-projection.comp runs a tile on one 32-lane team, and
    /// xe-nvfp4-projection-stream.comp on `SPLIT_K` teams that split K.
    /// A kind sealed with more than one team must be Xe2-only (or have
    /// every generation captured the same way).
    pub const fn group_width(self) -> u32 {
        32
    }

    /// Whether this kind's package takes M from word 1 of its parameter
    /// block (built with `RUNTIME_M=1`), so it serves any M > 1 at its
    /// shape; otherwise it serves only its own M. A prompt program's
    /// projections reach the GPU through these.
    pub const fn runtime_m(self) -> bool {
        matches!(
            self,
            Self::SharedUpSquaredReluWide | Self::SharedDownAccumulateWide | Self::LmHeadWide
        )
    }
}

/// XMX NVFP4 projection over caller-owned activation, weight, scale, and
/// output buffers. Slots 0..4 are F32 activation, packed E2M1 weights, E4M3FN
/// block scales, F32 tensor scale, and F32 output, respectively. The selected
/// variant fixes geometry and epilogue. Expert weights can remain in one
/// imported shared mapping across submissions. The `SharedDownAccumulate*`
/// variants add slot 5, a read-only F32 route scale. Their slot 4 is read/write:
/// initialize it before submission and order later submissions before reading
/// the sum. These variants avoid a separate output copy.
pub fn nvfp4_projection(kind: Nvfp4Projection) -> XeKernelArtifact {
    match kind {
        Nvfp4Projection::DecodeTile => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n17k32-e0.kxefamily"
        )),
        Nvfp4Projection::PrefillSiluTile => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n32k32-e1.kxefamily"
        )),
        Nvfp4Projection::MambaGateSiluDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n4096k2688-e1.kxefamily"
        )),
        Nvfp4Projection::MambaGateSiluPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n4096k2688-e1.kxefamily"
        )),
        Nvfp4Projection::MambaXbcDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n6144k2688-e0.kxefamily"
        )),
        Nvfp4Projection::MambaXbcPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n6144k2688-e0.kxefamily"
        )),
        Nvfp4Projection::MambaDtDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n64k2688-e0.kxefamily"
        )),
        Nvfp4Projection::MambaDtPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n64k2688-e0.kxefamily"
        )),
        Nvfp4Projection::HiddenOutDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n2688k4096-e0.kxefamily"
        )),
        Nvfp4Projection::HiddenOutPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n2688k4096-e0.kxefamily"
        )),
        Nvfp4Projection::AttentionQDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n4096k2688-e0.kxefamily"
        )),
        Nvfp4Projection::AttentionQPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n4096k2688-e0.kxefamily"
        )),
        Nvfp4Projection::AttentionKvDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n256k2688-e0.kxefamily"
        )),
        Nvfp4Projection::AttentionKvPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n256k2688-e0.kxefamily"
        )),
        Nvfp4Projection::SharedUpSquaredReluDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n3712k2688-e3.kxefamily"
        )),
        Nvfp4Projection::SharedUpSquaredReluPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n3712k2688-e3.kxefamily"
        )),
        Nvfp4Projection::SharedUpSquaredReluWide => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-rt-n3712k2688-e3.kxefamily"
        )),
        Nvfp4Projection::SharedDownAccumulateDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n2688k3712-e4.kxefamily"
        )),
        Nvfp4Projection::SharedDownAccumulatePrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n2688k3712-e4.kxefamily"
        )),
        Nvfp4Projection::SharedDownAccumulateWide => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-rt-n2688k3712-e4.kxefamily"
        )),
        Nvfp4Projection::LmHeadDecode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n131072k2688-e0.kxefamily"
        )),
        Nvfp4Projection::LmHeadPrefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n131072k2688-e0.kxefamily"
        )),
        Nvfp4Projection::LmHeadWide => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-rt-n131072k2688-e0.kxefamily"
        )),
    }
}

/// Dense projections of checkpoints that keep some layers out of NVFP4: F32
/// activations `[8, K]` times rows the checkpoint stores unquantized, with a
/// per-tensor scale and an optional epilogue:
/// `out = epilogue(scale * a @ w.T)`. The Nemotron-H NVFP4 release keeps its
/// Mamba in/out projections as E4M3 (`float8_e4m3fn`, one scale a tensor;
/// the in-projection is three row ranges of one weight) and its attention
/// projections as BF16. Slots 0..3 are the F32 activation, the weight rows
/// (read as 32-bit words: the view must be 4-byte aligned), the F32 scale,
/// and the F32 `[8, N]` output. Launch `[ceil(N / 16), 1, 1]` workgroups of
/// [`DenseProjection::group_width`] (xe-dense-projection.comp). Xe2-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseProjection {
    /// E4M3 `[4096, 2688]` (the in-projection's gate rows), SiLU.
    MambaGateSiluE4m3Prefill,
    /// E4M3 `[6144, 2688]` (X, B and C rows).
    MambaXbcE4m3Prefill,
    /// E4M3 `[64, 2688]` (time-step rows).
    MambaDtE4m3Prefill,
    /// E4M3 `[2688, 4096]` (the out-projection).
    MambaOutE4m3Prefill,
    /// BF16 `[4096, 2688]`.
    AttentionQBf16Prefill,
    /// BF16 `[256, 2688]`, K or V.
    AttentionKvBf16Prefill,
    /// BF16 `[2688, 4096]`.
    AttentionOutBf16Prefill,
}

impl DenseProjection {
    /// Every specialization, for callers that check the whole catalog.
    pub const ALL: [Self; 7] = [
        Self::MambaGateSiluE4m3Prefill,
        Self::MambaXbcE4m3Prefill,
        Self::MambaDtE4m3Prefill,
        Self::MambaOutE4m3Prefill,
        Self::AttentionQBf16Prefill,
        Self::AttentionKvBf16Prefill,
        Self::AttentionOutBf16Prefill,
    ];

    /// Invocations per workgroup the package was captured with (32 per
    /// K-splitting team, the fastest build per shape on Gigatron); the
    /// provider refuses any other.
    pub const fn group_width(self) -> u32 {
        match self {
            Self::MambaXbcE4m3Prefill | Self::AttentionQBf16Prefill => 32,
            _ => 32 * 4,
        }
    }
}

/// The captured package of a dense projection (Xe2; res/XE_GEMM_TIMING.md,
/// dense projections). A generation without a capture refuses it at load,
/// and the projection stays on the CPU.
pub fn dense_projection(kind: DenseProjection) -> Option<XeKernelArtifact> {
    Some(match kind {
        DenseProjection::MambaGateSiluE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-m8n4096k2688-e1.kxefamily"
        )),
        DenseProjection::MambaXbcE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-m8n6144k2688-e0.kxefamily"
        )),
        DenseProjection::MambaDtE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-m8n64k2688-e0.kxefamily"
        )),
        DenseProjection::MambaOutE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-m8n2688k4096-e0.kxefamily"
        )),
        DenseProjection::AttentionQBf16Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n4096k2688-e0.kxefamily"
        )),
        DenseProjection::AttentionKvBf16Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n256k2688-e0.kxefamily"
        )),
        DenseProjection::AttentionOutBf16Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n2688k4096-e0.kxefamily"
        )),
    })
}

/// The runtime-M package of a dense projection: the same shape, epilogue
/// and group width, built with `RUNTIME_M=1` so it takes M from word 1 of
/// its parameter block and serves any M > 1 (a prompt program's 256 rows).
/// The fixed M=8 package stays the faster one for 8-row checks (3-20% on
/// Gigatron). Xe2 only.
pub fn dense_projection_wide(kind: DenseProjection) -> Option<XeKernelArtifact> {
    Some(match kind {
        DenseProjection::MambaGateSiluE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-rmn4096k2688-e1.kxefamily"
        )),
        DenseProjection::MambaXbcE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-rmn6144k2688-e0.kxefamily"
        )),
        DenseProjection::MambaDtE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-rmn64k2688-e0.kxefamily"
        )),
        DenseProjection::MambaOutE4m3Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-e4m3-rmn2688k4096-e0.kxefamily"
        )),
        DenseProjection::AttentionQBf16Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-rmn4096k2688-e0.kxefamily"
        )),
        DenseProjection::AttentionKvBf16Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-rmn256k2688-e0.kxefamily"
        )),
        DenseProjection::AttentionOutBf16Prefill => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-rmn2688k4096-e0.kxefamily"
        )),
    })
}

/// Nemotron F32 RMS forms at the published checkpoint geometry. Each row is
/// one workgroup in X; one-token decode dispatches one hidden row or eight
/// Mamba group rows. Multi-token prefill dispatches the corresponding row
/// count. The epsilon is the checkpoint's `1e-5`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmsNorm {
    /// Width 2688: input, weight, normalized output (slots 0..2).
    Hidden,
    /// Width 2688: residual lhs/rhs, weight, residual output, normalized
    /// output (slots 0..4). The published residual is the next layer's state.
    ResidualHidden,
    /// Eight groups of width 512: Mamba scan output, gate, grouped weight,
    /// normalized output (slots 0..3). Eliminates the gated intermediate.
    MambaGateGroup,
}

/// Select a checked Xe RMS artifact without naming the silicon generation.
/// The buffers are caller-owned and may be imported shared DMA mappings;
/// this call stages only immutable program bytes.
pub fn rms_norm(kind: RmsNorm) -> XeKernelArtifact {
    match kind {
        RmsNorm::Hidden => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rms-op0-w2688-g1.kxefamily"
        )),
        RmsNorm::ResidualHidden => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rms-op1-w2688-g1.kxefamily"
        )),
        RmsNorm::MambaGateGroup => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rms-op2-w512-g8.kxefamily"
        )),
    }
}

/// Mamba depthwise causal convolution at the checkpoint's 6144-channel,
/// four-tap geometry. Both forms fuse SiLU and publish successor state into
/// a distinct target mapping, preserving the source for commit or rollback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MambaConv {
    /// One token; launch 24 workgroups in X, 256 lanes each.
    Decode,
    /// Eight consecutive tokens; the same 24-workgroup launch.
    Prefill8,
}

/// Slots 0..5 are input `[tokens,6144]`, taps `[6144,4]`, bias `[6144]`,
/// source state `[6144,3]`, target state `[6144,3]`, and output
/// `[tokens,6144]`, all F32. The state buffers must be distinct.
pub fn mamba_conv(kind: MambaConv) -> XeKernelArtifact {
    match kind {
        MambaConv::Decode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c6144-t1.kxefamily"
        )),
        MambaConv::Prefill8 => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c6144-t8.kxefamily"
        )),
    }
}

/// A runtime-shaped depthwise causal convolution: four taps, the SiLU fused,
/// one invocation per channel (`xe-nemotron-conv.comp` built with
/// `RUNTIME_SHAPE=1`, and `TREE_SHAPE=1` for a candidate tree).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalConv {
    /// The same six slots as [`mamba_conv`], but the F32 source state ends
    /// with a `{tokens, slots}` trailer (two LE u32s). The target holds one
    /// final state or one per token for verification rollback.
    Runtime,
    /// The trailer adds one LE u32 parent per node (the root `u32::MAX`);
    /// the target retains every node's state.
    Tree,
    /// [`Self::Runtime`] with its successor slots [`BANK_SLOT_ROOM`] bytes
    /// apart past each state (`SLOT_STRIDE`), so every slot carries its own
    /// trailer and an accepted successor is the next source where it lies:
    /// a speculative verification's banked state. Qwen3.5's width only.
    Bank,
}

impl CausalConv {
    pub const ALL: [Self; 3] = [Self::Runtime, Self::Tree, Self::Bank];
    /// The channel counts [`causal_conv`] has packages for.
    pub const CHANNELS: [u32; 3] = [6144, 10240, 8192];
}

/// The package of a runtime-shaped convolution at a captured channel count:
/// 6144 (Nemotron-H's Mamba x, B and C), 10240 (the Qwen3.5 27B's query,
/// key and value) and 8192 (the 35B-A3B's; these two Xe3 only, the 8192
/// without a tree package). Launch `[ceil(channels / 256), 1, 1]`
/// workgroups of 256; shape data adds no workgroups. A width without a
/// capture is `None`, and its convolution stays on the CPU
/// (res/XE_INFERENCE_FOLLOWUP.md, work placement).
pub fn causal_conv(kind: CausalConv, channels: u32) -> Option<XeKernelArtifact> {
    Some(match (kind, channels) {
        (CausalConv::Runtime, 6144) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c6144-rt.kxefamily"
        )),
        (CausalConv::Tree, 6144) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c6144-tree.kxefamily"
        )),
        (CausalConv::Runtime, 10240) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c10240-rt.kxefamily"
        )),
        (CausalConv::Tree, 10240) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c10240-tree.kxefamily"
        )),
        (CausalConv::Bank, 10240) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c10240-bank.kxefamily"
        )),
        (CausalConv::Runtime, 8192) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c8192-rt.kxefamily"
        )),
        (CausalConv::Bank, 8192) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-conv-c8192-bank.kxefamily"
        )),
        _ => return None,
    })
}

/// Mamba-2 selective scan for the published 64 heads, 64 channels per head,
/// eight groups, and 128 state elements per channel. Xe2 and Xe3 ISA were
/// captured and validated separately against the scalar-reference witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MambaScan {
    /// One token; launch 16 workgroups of 256 lanes in X.
    Decode,
    /// Eight consecutive tokens; the same 16-workgroup launch.
    Prefill8,
}

/// Slots 0..5 are X/B/C `[tokens,6144]`, dt `[tokens,64]`, head parameters
/// `[3,64]`, read-only source state `[64,64,128]`, read/write target state of
/// the same shape, and output `[tokens,4096]`, all F32. The state mappings
/// must be distinct. A caller commits the target only after completion.
pub fn mamba_scan(kind: MambaScan) -> XeKernelArtifact {
    match kind {
        MambaScan::Decode => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-scan-t1.kxefamily"
        )),
        MambaScan::Prefill8 => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-scan-t8.kxefamily"
        )),
    }
}

/// The runtime-shaped Mamba-2 scan (`xe-nemotron-scan.comp` built with
/// `RUNTIME_SHAPE=1`): the same six slots as [`mamba_scan`], but the F32
/// source state ends with a `{tokens, slots}` trailer (two LE u32s). The
/// target holds one final state or one per token (2 MiB each) for rollback.
/// Launch `[16, 1, 1]` workgroups of 256; shape data adds no workgroups.
/// Sealed for Xe2 only (the Xe3 capture is pending); `None` would
/// keep the scan on the CPU (res/XE_INFERENCE_FOLLOWUP.md, work placement).
pub fn mamba_scan_runtime() -> Option<XeKernelArtifact> {
    Some(artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-scan-rt.kxefamily"
    )))
}

/// Candidate-tree scan with the same parent trailer and per-node state layout.
/// Each invocation owns its state elements across all topologically ordered nodes.
pub fn mamba_scan_tree() -> Option<XeKernelArtifact> {
    Some(artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-scan-tree.kxefamily"
    )))
}

/// The gated delta rule (Gated DeltaNet linear attention) at a captured head
/// geometry: `xe-delta-rule.comp`, one workgroup per value head, each
/// invocation owning a slice of one state column in registers across the
/// run's tokens. The query and key are L2-normalized in the kernel.
///
/// Slots 0..5 are the query/key/value rows `[tokens, 2 * keys + values]`,
/// the log-space decay and the write strength `[tokens, value_heads]`, the
/// read-only source state `[value_heads, key_dim, value_dim]` followed by
/// the runtime-shaped recurrent trailer (`{tokens, slots}`, a tree adds one
/// parent per node), the read/write target state (one slot, or one per
/// token), and the output `[tokens, value_heads * value_dim]`, all F32: the
/// Mamba runtime packages' six-slot contract. Launch `[value_heads, 1, 1]`
/// workgroups of [`DeltaRule::group_width`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaRule {
    /// Linear runs: decode, prompt chunks and verification.
    Runtime,
    /// Candidate trees: each node continues from its parent's state.
    Tree,
    /// Linear runs whose successor slots lie [`BANK_SLOT_ROOM`] bytes apart
    /// past each state (`SLOT_STRIDE`): every slot carries its own trailer,
    /// so a speculative verification's accepted successor is the next run's
    /// source where it lies.
    Bank,
}

impl DeltaRule {
    pub const ALL: [Self; 3] = [Self::Runtime, Self::Tree, Self::Bank];

    /// Invocations per workgroup the package was captured with: the value
    /// width times the column slices (four for linear runs; eight for a
    /// tree, whose parent reloads would otherwise need more than 128 GRFs).
    pub const fn group_width(self, geometry: DeltaRuleGeometry) -> u32 {
        geometry.value_dim
            * match self {
                Self::Runtime | Self::Bank => 4,
                Self::Tree => 8,
            }
    }
}

/// A gated delta rule's head geometry: query/key heads of `key_dim`, and
/// value heads of `value_dim`, each value head reading key head
/// `head / (value_heads / key_heads)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaRuleGeometry {
    pub key_heads: u32,
    pub key_dim: u32,
    pub value_heads: u32,
    pub value_dim: u32,
}

impl DeltaRuleGeometry {
    /// The published Qwen3.5/3.6/3.8 27B linear attention: 16 query/key
    /// heads and 48 value heads, all 128 wide.
    pub const QWEN3_5_27B: Self = Self {
        key_heads: 16,
        key_dim: 128,
        value_heads: 48,
        value_dim: 128,
    };
    /// Qwen3.6-35B-A3B's: the same 16 query/key heads, 32 value heads.
    pub const QWEN3_5_MOE: Self = Self {
        key_heads: 16,
        key_dim: 128,
        value_heads: 32,
        value_dim: 128,
    };
    pub const ALL: [Self; 2] = [Self::QWEN3_5_27B, Self::QWEN3_5_MOE];
}

/// The captured package of a gated delta rule: Xe3 (Biggertron) for the
/// 27B geometry and the 35B-A3B's (linear and banked runs, no tree); the
/// Xe2 capture is pending, so the family's other generation refuses at load
/// and the rule stays on the CPU there.
pub fn delta_rule(kind: DeltaRule, geometry: DeltaRuleGeometry) -> Option<XeKernelArtifact> {
    if geometry == DeltaRuleGeometry::QWEN3_5_MOE {
        return match kind {
            DeltaRule::Runtime => Some(artifact(include_bytes!(
                "../../../drivers/xe/kernels/anv-delta-k16v32d128-rt.kxefamily"
            ))),
            DeltaRule::Bank => Some(artifact(include_bytes!(
                "../../../drivers/xe/kernels/anv-delta-k16v32d128-bank.kxefamily"
            ))),
            DeltaRule::Tree => None,
        };
    }
    if geometry != DeltaRuleGeometry::QWEN3_5_27B {
        return None;
    }
    Some(match kind {
        DeltaRule::Runtime => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-delta-k16v48d128-rt.kxefamily"
        )),
        DeltaRule::Tree => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-delta-k16v48d128-tree.kxefamily"
        )),
        DeltaRule::Bank => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-delta-k16v48d128-bank.kxefamily"
        )),
    })
}

/// The bytes a banked recurrent package ([`DeltaRule::Bank`],
/// [`CausalConv::Bank`]) leaves past each successor state before the next
/// slot: one page, room for the slot's own `{tokens, slots}` trailer, which
/// keeps every slot on a page. The packages bake it (`SLOT_STRIDE` is the
/// state's floats plus 1024); a state region laid out for them must match.
pub const BANK_SLOT_ROOM: u64 = 4096;

/// Fused F32 MoE routing for the published 128-expert, top-six layer. Each
/// workgroup projects one 2688-wide token through the router, applies sigmoid
/// and selection-only correction bias, then writes six expert IDs in ascending
/// order with six normalized, 2.5-scaled unbiased weights. Dispatch one or
/// eight X workgroups for decode or prefill. Slots 0..4 are hidden F32,
/// router F32 `[128,2688]`, correction bias F32 `[128]`, IDs I32 `[rows,6]`,
/// and scales F32 `[rows,6]`. Only the tiny IDs need a CPU read to fetch
/// streamed experts; the hidden and weight mappings stay resident.
///
/// Xe2 and Xe3 ISA are captured independently; callers name the operation.
pub fn moe_route() -> XeKernelArtifact {
    artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-route.kxefamily"
    ))
}

/// BF16 embedding gather from the published `[131072,2688]` vocabulary.
/// Slot 0 is the read-only packed BF16 table, slot 1 is I32 token IDs, and
/// slot 2 is F32 output. Dispatch one X workgroup per token. The caller must
/// validate IDs; invalid IDs leave their output row untouched. This kernel
/// reads only selected table rows through the caller's existing DMA mapping.
/// Xe2 and Xe3 native ISA are captured and selected by the provider.
pub fn embedding_gather_bf16() -> XeKernelArtifact {
    artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-gather.kxefamily"
    ))
}

/// The decode graph's final top-one selection over F32 `[tokens,131072]`
/// logits. Slot 0 is read-only logits, slot 1 is F32 selected values, and
/// slot 2 is I32 vocabulary IDs. Dispatch one X workgroup per token. Ranking
/// follows `f32::total_cmp` with ties going to the lower index; only the
/// selected ID needs to cross to the host after this GPU-resident reduction.
pub fn vocabulary_top1() -> XeKernelArtifact {
    artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-top1.kxefamily"
    ))
}

/// One member of a routed-expert wave over AXEX v3 expert blocks
/// (res/XE_MOE_OVERHAUL.md): each chunk set's grouped up projection
/// (`xe-nvfp4-grouped-up.comp`, squared ReLU fused), its grouped down
/// projection with the routes' mixture (`xe-nvfp4-grouped-down.comp`), then
/// one combine that sums each token's routes in expert order
/// (`xe-moe-combine.comp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeWave {
    Up,
    Down,
    Combine,
}

impl MoeWave {
    pub const ALL: [Self; 3] = [Self::Up, Self::Down, Self::Combine];

    /// Invocations per workgroup the sealed package was captured with.
    pub const fn group_width(self) -> u32 {
        match self {
            Self::Up | Self::Down => 32,
            Self::Combine => 64,
        }
    }

    /// Storage slots: five expert-cache chunks and the wave region for the
    /// grouped projections, the wave region alone for the combine.
    pub const fn bindings(self) -> usize {
        match self {
            Self::Up | Self::Down => 6,
            Self::Combine => 1,
        }
    }
}

/// The package of one wave member (Xe2, captured on Gigatron: the grouped
/// projections at SPLIT_K=1 and WEIGHT_DEPTH=1, the combine at 64 lanes;
/// res/XE_MOE_OVERHAUL.md). A generation without a capture refuses it at
/// load, and the routed experts run on the CPU.
pub fn moe_wave(part: MoeWave) -> Option<XeKernelArtifact> {
    Some(match part {
        MoeWave::Up => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-moe-grouped-up.kxefamily"
        )),
        MoeWave::Down => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-moe-grouped-down.kxefamily"
        )),
        MoeWave::Combine => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-moe-combine.kxefamily"
        )),
    })
}

/// One member of a resident KV layer: the paged E4M3 append
/// (`xe-kv-append.comp`: 0 keys, 1 values, 2 the cache, 3 control; one group
/// of a lane per column per (token, KV head)), the grouped-query attention
/// read (`xe-kv-attention.comp`: 0 queries, 1 the cache, 2 control, 3
/// partials, 4 output; one SIMD32 subgroup per query row, Y the causal
/// splits) and the split reduction (`REDUCE=1`: 0 control, 1 partials, 2
/// output). The cache's byte layout is axnn-cpu's (`kv_cache`), so the CPU
/// kernels are the oracle (`tools/xe-kv-fixture`,
/// `tools/qwen-xe-validate/kv-fixture`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKernel {
    Append,
    Attention,
    Reduce,
}

/// The geometry a resident KV package bakes: the cache's KV heads and head
/// width, the attention read's query heads, and how many of those query rows
/// one attention (and reduction) workgroup owns. A workgroup's rows share one
/// KV head's decoded tile, so `rows` divides `query_heads / kv_heads`; the
/// workgroup itself is the next power of two of SIMD32 subgroups, because ANV
/// moves a 96- or 192-lane workgroup's cross-thread inline data out of the
/// register the v3 package's push load reads (the extra subgroups only stage
/// the shared tile).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeometry {
    pub kv_heads: u32,
    pub head_dim: u32,
    pub query_heads: u32,
    pub rows: u32,
}

impl KvGeometry {
    /// Nemotron: 32 query heads over two KV heads of 128 (Xe2 and Xe3).
    pub const NEMOTRON: Self = Self {
        kv_heads: 2,
        head_dim: 128,
        query_heads: 32,
        rows: 8,
    };
    /// Qwen3-32B: GQA 8:1, 64 query heads over eight KV heads of 128 (Xe3).
    pub const QWEN3: Self = Self {
        kv_heads: 8,
        head_dim: 128,
        query_heads: 64,
        rows: 8,
    };
    /// Qwen3.8-27B's gated attention: 24 query heads over four KV heads of
    /// 256, six rows (one KV head's group) per eight-subgroup workgroup (Xe3).
    pub const QWEN3_5: Self = Self {
        kv_heads: 4,
        head_dim: 256,
        query_heads: 24,
        rows: 6,
    };
    /// Qwen3.6-35B-A3B's gated attention: GQA 8:1, 16 query heads over two
    /// KV heads of 256, eight rows (one KV head's group) per workgroup (Xe3).
    pub const QWEN3_5_MOE: Self = Self {
        kv_heads: 2,
        head_dim: 256,
        query_heads: 16,
        rows: 8,
    };

    pub const ALL: [Self; 4] = [Self::NEMOTRON, Self::QWEN3, Self::QWEN3_5, Self::QWEN3_5_MOE];

    /// The captured geometry of a cache of `kv_heads` heads `head_dim` wide.
    pub fn of_cache(kv_heads: u32, head_dim: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|geometry| (geometry.kv_heads, geometry.head_dim) == (kv_heads, head_dim))
    }

    /// SIMD32 subgroups in an attention or reduction workgroup: `rows`
    /// rounded up to a power of two.
    pub const fn subgroups(self) -> u32 {
        self.rows.next_power_of_two()
    }
}

impl KvKernel {
    pub const ALL: [Self; 3] = [Self::Append, Self::Attention, Self::Reduce];

    /// Invocations per workgroup the sealed package was captured with: the
    /// append a lane per column, the read and reduction 32 per subgroup.
    pub const fn group_width(self, geometry: KvGeometry) -> u32 {
        match self {
            Self::Append => geometry.head_dim,
            Self::Attention | Self::Reduce => 32 * geometry.subgroups(),
        }
    }

    /// Storage slots, in binding order.
    pub const fn bindings(self) -> usize {
        match self {
            Self::Append => 4,
            Self::Attention => 5,
            Self::Reduce => 3,
        }
    }
}

/// The package of one resident KV kernel at a baked geometry
/// ([`KvGeometry`]): Nemotron's two-KV-head capture (Xe2 Lunar Lake and
/// Xe3), Qwen3's `g8d128`, Qwen3.5's `g4d256` and the 35B-A3B's `g2d256`
/// (Xe3 only, standalone RoPE). A geometry or generation without a capture
/// refuses it at load, and attention stays on the CPU there.
pub fn kv_kernel(kind: KvKernel, geometry: KvGeometry) -> Option<XeKernelArtifact> {
    let nemotron = geometry == KvGeometry::NEMOTRON;
    let qwen3 = geometry == KvGeometry::QWEN3;
    if geometry == KvGeometry::QWEN3_5_MOE {
        return Some(match kind {
            KvKernel::Append => artifact(include_bytes!(
                "../../../drivers/xe/kernels/anv-kv-append-g2d256.kxefamily"
            )),
            KvKernel::Attention => artifact(include_bytes!(
                "../../../drivers/xe/kernels/anv-kv-attention-g2d256.kxefamily"
            )),
            KvKernel::Reduce => artifact(include_bytes!(
                "../../../drivers/xe/kernels/anv-kv-reduce-g2d256.kxefamily"
            )),
        });
    }
    if !nemotron && !qwen3 && geometry != KvGeometry::QWEN3_5 {
        return None;
    }
    Some(match kind {
        KvKernel::Append if nemotron => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-append.kxefamily"
        )),
        KvKernel::Append if qwen3 => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-append-g8d128.kxefamily"
        )),
        KvKernel::Append => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-append-g4d256.kxefamily"
        )),
        KvKernel::Attention if nemotron => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-attention.kxefamily"
        )),
        KvKernel::Attention if qwen3 => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-attention-g8d128.kxefamily"
        )),
        KvKernel::Attention => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-attention-g4d256.kxefamily"
        )),
        KvKernel::Reduce if nemotron => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-reduce.kxefamily"
        )),
        KvKernel::Reduce if qwen3 => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-reduce-g8d128.kxefamily"
        )),
        KvKernel::Reduce => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-kv-reduce-g4d256.kxefamily"
        )),
    })
}

/// The largest package in a family payload (`KXEFAM`: a 16-byte header
/// with the variant count at byte 10, then 16-byte entries carrying each
/// package's offset at byte 4 and size at byte 8), or `None` for a payload
/// that is not a canonical family. The driver's package reader is the
/// authority on the format; this reads only what the lane's accounting
/// needs.
fn family_largest_package(bytes: &[u8]) -> Option<usize> {
    const MAGIC: &[u8; 8] = b"KXEFAM\0\0";
    const HEADER: usize = 16;
    const ENTRY: usize = 16;
    if bytes.len() < HEADER || &bytes[..8] != MAGIC {
        return None;
    }
    let count = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
    let directory = HEADER.checked_add(count.checked_mul(ENTRY)?)?;
    if count == 0 || bytes.len() < directory {
        return None;
    }
    let mut largest = 0usize;
    for index in 0..count {
        let entry = &bytes[HEADER + index * ENTRY..HEADER + (index + 1) * ENTRY];
        let offset = u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]) as usize;
        let size = u32::from_le_bytes([entry[8], entry[9], entry[10], entry[11]]) as usize;
        if offset.checked_add(size)? > bytes.len() {
            return None;
        }
        largest = largest.max(size);
    }
    Some(largest)
}

/// One kernel of a mixture-of-experts layer whose experts are graph
/// weights: expert tables in `ax::routed`'s record layout (axiom's
/// `MOE_ROUTE`, `MOE_GATE_UP_NVFP4` and `MOE_DOWN_NVFP4`), every expert of
/// the layer resident, so the route never leaves the device. Unlike the
/// [`MoeWave`] members, nothing is grouped by expert: each token's slots
/// read their own experts' records, which a decode's one row (or a
/// verification's few) wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutedKernel {
    /// `xe-moe-route.comp`: slot 0 the router's F32 logits `[T, experts +
    /// shared]`, slot 1 the I32 routes `[T, slots, 2]` (each an expert and
    /// its weight's F32 bits); one workgroup a token.
    Route,
    /// `xe-nvfp4-routed.comp` MODE 0: slot 0 F32 `x [T, hidden]`, slot 1 the
    /// gate/up table, slot 2 the routes, slot 3 F32 `[T, slots, inner]`:
    /// `silu(gate . x) * (up . x)`, each under its tensor scale; a
    /// workgroup writes eight of one slot's outputs.
    GateUp,
    /// `xe-nvfp4-routed.comp` MODE 1: slot 0 F32 `h [T, slots, inner]`,
    /// slot 1 the down table, slot 2 the routes, slot 3 F32 `[T, hidden]`:
    /// the slots' projections, weighted and summed in slot order; a
    /// workgroup writes sixteen rows of one token.
    Down,
}

impl RoutedKernel {
    pub const ALL: [Self; 3] = [Self::Route, Self::GateUp, Self::Down];

    /// Invocations per workgroup the sealed package was captured with: the
    /// route one subgroup's worth, gate/up four teams and down two
    /// (`tools/xe_qwen_capture.py`).
    pub const fn group_width(self) -> u32 {
        match self {
            Self::Route => 32,
            Self::GateUp => 128,
            Self::Down => 64,
        }
    }

    /// Storage slots of the package.
    pub const fn bindings(self) -> usize {
        match self {
            Self::Route => 2,
            Self::GateUp | Self::Down => 4,
        }
    }

    /// Storage slots of the pooled build ([`routed_pool_kernel`]): the
    /// table's, and the control block between the routes and the output.
    /// The route has none.
    pub const fn pool_bindings(self) -> Option<usize> {
        match self {
            Self::Route => None,
            Self::GateUp | Self::Down => Some(5),
        }
    }

    /// Workgroups for `tokens` rows: a token's route; each slot's eight
    /// gate/up outputs; a token's sixteen down rows.
    pub fn groups(self, geometry: RoutedGeometry, tokens: u32) -> Option<[u32; 3]> {
        if tokens == 0 {
            return None;
        }
        Some(match self {
            Self::Route => [tokens, 1, 1],
            Self::GateUp => [geometry.inner / 8, tokens.checked_mul(geometry.slots())?, 1],
            Self::Down => [geometry.hidden / 16, tokens, 1],
        })
    }
}

/// A routed-expert geometry: `experts` routed experts picking `top_k` a
/// token beside `shared` always-on ones, over a `hidden`-wide state, every
/// expert's SwiGLU `inner` wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedGeometry {
    pub experts: u32,
    pub top_k: u32,
    pub shared: u32,
    pub hidden: u32,
    pub inner: u32,
}

impl RoutedGeometry {
    /// Qwen3.6-35B-A3B (`qwen3_5_moe`): 256 experts picking 8 and one
    /// shared, over a 2048-wide state, every expert 512 wide.
    pub const QWEN3_6_35B_A3B: Self = Self {
        experts: 256,
        top_k: 8,
        shared: 1,
        hidden: 2048,
        inner: 512,
    };
    pub const ALL: [Self; 1] = [Self::QWEN3_6_35B_A3B];

    /// Records of a table: the routed experts, then the shared ones.
    pub const fn records(self) -> u32 {
        self.experts + self.shared
    }

    /// Routes of a token: the picks, then the shared experts.
    pub const fn slots(self) -> u32 {
        self.top_k + self.shared
    }

    /// Bytes of one expert's gate/up record (`ax::routed::ExpertRecord`:
    /// sixteen header bytes, `2 * inner` rows of `hidden` E2M1 codes, their
    /// E4M3 block scales, padded to 64).
    pub const fn gate_up_record_bytes(self) -> u64 {
        expert_record_bytes(2 * self.inner as u64, self.hidden as u64)
    }

    /// Bytes of one expert's down record: `hidden` rows of `inner`.
    pub const fn down_record_bytes(self) -> u64 {
        expert_record_bytes(self.hidden as u64, self.inner as u64)
    }

    /// Where a pool slot's down record starts: past its gate/up record,
    /// padded to a page ([`routed_pool_kernel`]).
    pub const fn pool_down_offset(self) -> u64 {
        self.gate_up_record_bytes().div_ceil(POOL_PAGE) * POOL_PAGE
    }

    /// Bytes of one pool slot: the gate/up record and then the down record,
    /// each padded to a page, so one page-aligned storage read fills it
    /// (`xe-nvfp4-routed.comp`'s `STRIDE` at POOL 1).
    pub const fn pool_slot_bytes(self) -> u64 {
        self.pool_down_offset() + self.down_record_bytes().div_ceil(POOL_PAGE) * POOL_PAGE
    }
}

/// The page a pool slot's records are padded to: one storage read a slot.
pub const POOL_PAGE: u64 = 4096;

/// `ax::routed::ExpertRecord::bytes` for `rows` NVFP4 rows of `cols`.
const fn expert_record_bytes(rows: u64, cols: u64) -> u64 {
    (16 + rows * cols / 2 + rows * cols / 16).div_ceil(64) * 64
}

const _: () = {
    // The 35B's records (inference's tests36) and the pooled packages'
    // sealed STRIDE.
    assert!(RoutedGeometry::QWEN3_6_35B_A3B.gate_up_record_bytes() == 1_179_712);
    assert!(RoutedGeometry::QWEN3_6_35B_A3B.down_record_bytes() == 589_888);
    assert!(RoutedGeometry::QWEN3_6_35B_A3B.pool_down_offset() == 1_183_744);
    assert!(RoutedGeometry::QWEN3_6_35B_A3B.pool_slot_bytes() == 1_777_664);
};

/// The package of `kind` at a captured `geometry`. Each family carries Xe3
/// and Xe2 natives (`tools/qwen-xe-validate/seal.py`); both generations'
/// packages match the CPU reference (`run_all.py`, Biggertron and Gigatron).
pub fn routed_kernel(kind: RoutedKernel, geometry: RoutedGeometry) -> Option<XeKernelArtifact> {
    if geometry != RoutedGeometry::QWEN3_6_35B_A3B {
        return None;
    }
    Some(match kind {
        RoutedKernel::Route => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-moe-route-e256k8s1.kxefamily"
        )),
        RoutedKernel::GateUp => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-routed-up-h2048i512e257s9.kxefamily"
        )),
        RoutedKernel::Down => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-routed-down-h2048i512e257s9.kxefamily"
        )),
    })
}

/// The pooled build of `kind` at a captured `geometry`
/// (`xe-nvfp4-routed.comp` POOL 1, `res/QWEN_EXPERT_STREAMING.md`): gate/up
/// or down over a slot pool of [`RoutedGeometry::pool_slot_bytes`] a slot
/// (the down projection bound from the first slot's down record), routes
/// that name slots, and the pool's record count in a control block. `None`
/// for the route, which reads no expert. Xe3 and Xe2 natives, each
/// matching the CPU reference over pools the host rewrote the routes into.
pub fn routed_pool_kernel(kind: RoutedKernel, geometry: RoutedGeometry) -> Option<XeKernelArtifact> {
    if geometry != RoutedGeometry::QWEN3_6_35B_A3B {
        return None;
    }
    match kind {
        RoutedKernel::Route => None,
        RoutedKernel::GateUp => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-routed-pool-up-h2048i512s9.kxefamily"
        ))),
        RoutedKernel::Down => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-routed-pool-down-h2048i512s9.kxefamily"
        ))),
    }
}

/// Lanes of the wave plan's one workgroup (`xe-moe-plan.comp`).
pub const MOE_PLAN_LANES: u32 = 256;

/// The device's own wave plan for a resident layer (`xe-moe-plan.comp`):
/// routing from the router's scores and the records the grouped kernels
/// read. One 256-lane workgroup over three slots: the scores, the hidden
/// rows and the wave region. Xe2 (Gigatron) and Xe3 (Biggertron) are each
/// captured from the same SPIR-V (`anv-moe-plan.kxefamily`).
pub fn moe_plan() -> Option<XeKernelArtifact> {
    Some(artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-moe-plan.kxefamily"
    )))
}

/// A Qwen dense-model decode projection: an M=1 NVFP4 stream package
/// (`xe-nvfp4-projection-stream.comp`) captured at one (N, K, epilogue) with
/// SPLIT_K teams per output tile (`res/QWEN_XE_CATALOG.md`,
/// `tools/xe_qwen_capture.py`). Qwen3-32B's five come first, then
/// Qwen3.8-27B's (the qwen3_5 architecture: Gated DeltaNet mixers and gated
/// attention), whose k/v projection is Qwen3's `[1024, 5120]` package. The
/// shape metadata (N, K, epilogue, slots) is what the capture driver seals
/// against, so it lives here, in the one catalog, rather than in two
/// guesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenProjection {
    /// q `[8192, 5120]`, identity (e0), stream SPLIT_K=4.
    Query,
    /// k and v `[1024, 5120]`, identity (e0), SPLIT_K=4: Qwen3's eight KV
    /// heads of 128 and Qwen3.5's four of 256, one shape, one package.
    KeyValue,
    /// O `[5120, 8192]`, identity (e0), SPLIT_K=8 (the residual add and the
    /// attention alone stay graph-level ops, not a fused epilogue).
    AttentionO,
    /// FFN gate and up `[25600, 5120]`, identity (e0), SPLIT_K=4 — the gate's
    /// SiLU and the gate·up multiply are separate graph ops (the fusion is
    /// neutral at M=1), so gate and up share this one package.
    Ffn,
    /// FFN down `[5120, 25600]`, identity (e0), SPLIT_K=8.
    FfnDown,
    /// Qwen3.5's Gated DeltaNet `in_proj_qkv` `[10240, 5120]` (query and key
    /// 2048 each, value 6144), identity (e0), SPLIT_K=4.
    DeltaQkv,
    /// `in_proj_z` `[6144, 5120]` with SiLU fused (e1): the output gate of
    /// the delta rule's normalized heads. SPLIT_K=4.
    DeltaGate,
    /// `in_proj_b` `[48, 5120]` with sigmoid fused (e2): the write strength.
    /// SPLIT_K=8: three output tiles leave the GPU latency-bound, and eight
    /// teams cut each tile's chain (108 to 75 us on the Xe3 sweep).
    DeltaBeta,
    /// `in_proj_a` `[48, 5120]`, identity (e0), SPLIT_K=8: the step's
    /// `softplus(a + dt_bias)` is an F32 map.
    DeltaStep,
    /// `[5120, 6144]`, identity (e0), SPLIT_K=4: the delta rule's `out_proj`
    /// and the gated attention's `o_proj` (one shape, one package).
    MixerOut,
    /// The gated attention's `q_proj` `[12288, 5120]`, identity (e0),
    /// SPLIT_K=4: each head's 256 query rows, then its 256 gate rows.
    GatedQuery,
    /// MLP gate `[17408, 5120]` with SiLU fused (e1), SPLIT_K=4.
    MlpGate,
    /// MLP up `[17408, 5120]`, identity (e0), SPLIT_K=4.
    MlpUp,
    /// MLP down `[5120, 17408]`, identity (e0), SPLIT_K=8 (K-heavy).
    MlpDown,
    /// The prediction head's `fc` halves `[5120, 5120]` (its concatenation
    /// of the embedding and the hidden row, split by the pack into
    /// `fc_embedding` and `fc_hidden`), identity (e0), SPLIT_K=4.
    MtpFc,
    /// The prediction head's draft lm_head over the vocabulary's first
    /// [`QWEN3_5_DRAFT_VOCAB`] tokens, `[98304, 5120]`: rows of the target's
    /// BF16 head quantized to NVFP4 at pack time (drafts need only their
    /// argmax; verification keeps the exact head), identity, SPLIT_K=4.
    DraftHead,
    /// Qwen3.6-35B-A3B's `[8192, 2048]`, identity, SPLIT_K=2: its Gated
    /// DeltaNet `in_proj_qkv` (query and key 2048 each, value 4096) and its
    /// gated attention's `q_proj` (16 heads' 256 query rows, then their gate
    /// rows), one shape, one package.
    MoeMixerIn,
    /// The 35B-A3B's `in_proj_z` `[4096, 2048]` with SiLU fused (e1),
    /// SPLIT_K=2.
    MoeDeltaGate,
    /// The 35B-A3B's `[2048, 4096]`, identity, SPLIT_K=2: the delta rule's
    /// `out_proj` and the gated attention's `o_proj`.
    MoeMixerOut,
    /// The 35B-A3B's k and v `[512, 2048]` (two KV heads of 256), identity,
    /// SPLIT_K=2.
    MoeKeyValue,
    /// The 35B-A3B prediction head's `fc` halves `[2048, 2048]`, identity,
    /// SPLIT_K=2.
    MoeMtpFc,
    /// The 35B-A3B's draft lm_head over the vocabulary's first
    /// [`QWEN3_5_DRAFT_VOCAB`] tokens, `[98304, 2048]`, identity, SPLIT_K=4.
    MoeDraftHead,
}

impl QwenProjection {
    pub const ALL: [Self; 22] = [
        Self::Query,
        Self::KeyValue,
        Self::AttentionO,
        Self::Ffn,
        Self::FfnDown,
        Self::DeltaQkv,
        Self::DeltaGate,
        Self::DeltaBeta,
        Self::DeltaStep,
        Self::MixerOut,
        Self::GatedQuery,
        Self::MlpGate,
        Self::MlpUp,
        Self::MlpDown,
        Self::MtpFc,
        Self::DraftHead,
        Self::MoeMixerIn,
        Self::MoeDeltaGate,
        Self::MoeMixerOut,
        Self::MoeKeyValue,
        Self::MoeMtpFc,
        Self::MoeDraftHead,
    ];

    /// Qwen3-32B's decode projections.
    pub const QWEN3: [Self; 5] = [
        Self::Query,
        Self::KeyValue,
        Self::AttentionO,
        Self::Ffn,
        Self::FfnDown,
    ];

    /// Qwen3.8-27B's decode projections.
    pub const QWEN3_5: [Self; 10] = [
        Self::DeltaQkv,
        Self::DeltaGate,
        Self::DeltaBeta,
        Self::DeltaStep,
        Self::MixerOut,
        Self::GatedQuery,
        Self::KeyValue,
        Self::MlpGate,
        Self::MlpUp,
        Self::MlpDown,
    ];

    /// Qwen3.8-27B's prediction head (`mtp.*`, NVFP4 as the pack quantizes
    /// it), beyond the decode's own packages its layer shares: its `fc`
    /// halves and its draft lm_head.
    pub const QWEN3_5_MTP: [Self; 2] = [Self::MtpFc, Self::DraftHead];

    /// Qwen3.6-35B-A3B's decode projections (the routed experts are
    /// [`RoutedKernel`]s; the router and the delta rule's gate inputs stay
    /// BF16, [`QwenDense`]).
    pub const QWEN3_5_MOE: [Self; 4] = [
        Self::MoeMixerIn,
        Self::MoeDeltaGate,
        Self::MoeMixerOut,
        Self::MoeKeyValue,
    ];

    /// Qwen3.6-35B-A3B's prediction head beyond the layer packages it
    /// shares with the decode.
    pub const QWEN3_5_MOE_MTP: [Self; 2] = [Self::MoeMtpFc, Self::MoeDraftHead];

    /// The projections with a multi-row package ([`qwen_projection_rows`]):
    /// every 27B layer linear, which a verification or a prompt chunk runs
    /// over up to [`QWEN_ROWS`] tokens, and the prediction head's `fc`
    /// halves, which its first pass of a step runs over the same rows.
    pub const ROWS: [Self; 16] = [
        Self::DeltaQkv,
        Self::DeltaGate,
        Self::DeltaBeta,
        Self::DeltaStep,
        Self::MixerOut,
        Self::GatedQuery,
        Self::KeyValue,
        Self::MlpGate,
        Self::MlpUp,
        Self::MlpDown,
        Self::MtpFc,
        Self::MoeMixerIn,
        Self::MoeDeltaGate,
        Self::MoeMixerOut,
        Self::MoeKeyValue,
        Self::MoeMtpFc,
    ];

    /// Output columns.
    pub const fn n(self) -> u32 {
        match self {
            Self::Query => 8192,
            Self::KeyValue => 1024,
            Self::AttentionO | Self::FfnDown | Self::MixerOut | Self::MlpDown | Self::MtpFc => 5120,
            Self::Ffn => 25600,
            Self::DeltaQkv => 10240,
            Self::DeltaGate => 6144,
            Self::DeltaBeta | Self::DeltaStep => 48,
            Self::GatedQuery => 12288,
            Self::MlpGate | Self::MlpUp => 17408,
            Self::DraftHead | Self::MoeDraftHead => QWEN3_5_DRAFT_VOCAB,
            Self::MoeMixerIn => 8192,
            Self::MoeDeltaGate => 4096,
            Self::MoeMixerOut | Self::MoeMtpFc => 2048,
            Self::MoeKeyValue => 512,
        }
    }

    /// Input (weight) columns; a multiple of 16 for the NVFP4 block scale.
    pub const fn k(self) -> u32 {
        match self {
            Self::AttentionO => 8192,
            Self::FfnDown => 25600,
            Self::MixerOut => 6144,
            Self::MlpDown => 17408,
            Self::MoeMixerOut => 4096,
            Self::MoeMixerIn | Self::MoeDeltaGate | Self::MoeKeyValue | Self::MoeMtpFc | Self::MoeDraftHead => {
                QWEN3_5_MOE_HIDDEN
            }
            _ => 5120,
        }
    }

    /// The epilogue fused into the package (its `EPILOGUE` build define):
    /// 0 identity, 1 SiLU, 2 sigmoid.
    pub const fn epilogue(self) -> u32 {
        match self {
            Self::DeltaGate | Self::MlpGate | Self::MoeDeltaGate => 1,
            Self::DeltaBeta => 2,
            _ => 0,
        }
    }

    /// Bound storage slots: five NVFP4 (the graph keeps the residual add
    /// separate, so no route-scale slot).
    pub const fn slots(self) -> u8 {
        5
    }

    /// Workgroup lanes of the stream form: 32 x SPLIT_K (4 for N-heavy, 8 for
    /// the K-heavy down/O projections and the 48-row gates, 2 for the
    /// 35B-A3B's 2048- and 4096-wide inputs but its draft head, per the Xe3
    /// sweeps).
    pub const fn group_width(self) -> u32 {
        match self {
            Self::AttentionO
            | Self::FfnDown
            | Self::MlpDown
            | Self::DeltaBeta
            | Self::DeltaStep => 32 * 8,
            Self::MoeMixerIn | Self::MoeDeltaGate | Self::MoeMixerOut | Self::MoeKeyValue | Self::MoeMtpFc => 32 * 2,
            _ => 32 * 4,
        }
    }

    /// The multi-row package's workgroup ([`qwen_projection_rows`], the
    /// stream kernel's `XMX_STREAM`): `(tiles, split)`, its SIMD16
    /// subgroups along N (sixteen columns each) and its teams along K. From
    /// the Xe3 steady-state sweep (medians of five interleaved runs at eight
    /// rows, back-to-back dispatches, 300 MB of weights in rotation):
    /// sixteen by two for most shapes (110 GB/s at N 17408 K 5120, 105 at N
    /// 5120 K 17408), eight by two for in_proj_qkv, eight by four for the
    /// 1024-column k/v, one by sixteen for the latency-bound 48-column gates
    /// (three workgroups). The 35B-A3B's at K 2048 and 4096: sixteen by two
    /// for in_proj_qkv/q_proj and out_proj/o_proj (96 and 89 GB/s), eight by
    /// two for in_proj_z, four by eight for the 512-column k/v, four by four
    /// for the head's fc halves.
    pub const fn rows_geometry(self) -> (u32, u32) {
        match self {
            Self::DeltaQkv => (8, 2),
            Self::KeyValue => (8, 4),
            Self::DeltaBeta | Self::DeltaStep => (1, 16),
            Self::MoeDeltaGate => (8, 2),
            Self::MoeKeyValue => (4, 8),
            Self::MoeMtpFc => (4, 4),
            _ => (16, 2),
        }
    }

    /// Workgroup lanes of the multi-row package: 16 x tiles x split.
    pub const fn rows_group_width(self) -> u32 {
        let (tiles, split) = self.rows_geometry();
        16 * tiles * split
    }

    /// Output columns one multi-row workgroup covers: launch
    /// `[ceil(N / rows_group_columns), 1, 1]` workgroups.
    pub const fn rows_group_columns(self) -> u32 {
        16 * self.rows_geometry().0
    }

    /// Xe3 M16 captures retain each shape's N tile count and use two K
    /// teams. This geometry applies only to the dedicated M16 packages.
    pub const fn prefill16_geometry(self) -> (u32, u32) {
        (self.rows_geometry().0, 2)
    }

    pub const fn prefill16_group_width(self) -> u32 {
        let (tiles, split) = self.prefill16_geometry();
        16 * tiles * split
    }

    pub const fn prefill16_group_columns(self) -> u32 {
        16 * self.prefill16_geometry().0
    }
}

/// The most token rows of a Qwen verification or prompt-chunk pass: the
/// multi-row packages bake it ([`qwen_projection_rows`],
/// [`qwen_lm_head_rows`]) and take a pass's own count, 1 to it, from their
/// parameter block; the pass's q/gate split ([`F32_ROW_GATHERS`]) is
/// captured at it.
pub const QWEN_ROWS: u32 = 8;

/// The captured package of a Qwen projection (Xe3; the Xe2 capture is pending,
/// so the family's second generation refuses cleanly at load).
pub fn qwen_projection(kind: QwenProjection) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenProjection::Query => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n8192k5120-e0.kxefamily"
        )),
        QwenProjection::KeyValue => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n1024k5120-e0.kxefamily"
        )),
        QwenProjection::AttentionO => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n5120k8192-e0.kxefamily"
        )),
        QwenProjection::Ffn => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n25600k5120-e0.kxefamily"
        )),
        QwenProjection::FfnDown => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n5120k25600-e0.kxefamily"
        )),
        QwenProjection::DeltaQkv => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n10240k5120-e0.kxefamily"
        )),
        QwenProjection::DeltaGate => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n6144k5120-e1.kxefamily"
        )),
        QwenProjection::DeltaBeta => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n48k5120-e2.kxefamily"
        )),
        QwenProjection::DeltaStep => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n48k5120-e0.kxefamily"
        )),
        QwenProjection::MixerOut => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n5120k6144-e0.kxefamily"
        )),
        QwenProjection::GatedQuery => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n12288k5120-e0.kxefamily"
        )),
        QwenProjection::MlpGate => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n17408k5120-e1.kxefamily"
        )),
        QwenProjection::MlpUp => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n17408k5120-e0.kxefamily"
        )),
        QwenProjection::MlpDown => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n5120k17408-e0.kxefamily"
        )),
        QwenProjection::MtpFc => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n5120k5120-e0.kxefamily"
        )),
        QwenProjection::DraftHead => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n98304k5120-e0.kxefamily"
        )),
        QwenProjection::MoeMixerIn => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n8192k2048-e0.kxefamily"
        )),
        QwenProjection::MoeDeltaGate => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n4096k2048-e1.kxefamily"
        )),
        QwenProjection::MoeMixerOut => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n2048k4096-e0.kxefamily"
        )),
        QwenProjection::MoeKeyValue => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n512k2048-e0.kxefamily"
        )),
        QwenProjection::MoeMtpFc => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n2048k2048-e0.kxefamily"
        )),
        QwenProjection::MoeDraftHead => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m1n98304k2048-e0.kxefamily"
        )),
    })
}

/// The multi-row package of a Qwen projection ([`QwenProjection::ROWS`]):
/// `xe-nvfp4-projection-stream.comp` with `XMX_STREAM=1 M=8 RUNTIME_M=1`,
/// sealed at SIMD16 (Xe3). `[m, K]` activations to `[m, N]` for any `m` up
/// to [`QWEN_ROWS`], the parameter block's word 1; launch
/// `[ceil(N / rows_group_columns), 1, 1]` workgroups of
/// [`QwenProjection::rows_group_width`]. `None` for a kind without one.
pub fn qwen_projection_rows(kind: QwenProjection) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenProjection::DeltaQkv => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n10240k5120-e0.kxefamily"
        )),
        QwenProjection::DeltaGate => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n6144k5120-e1.kxefamily"
        )),
        QwenProjection::DeltaBeta => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n48k5120-e2.kxefamily"
        )),
        QwenProjection::DeltaStep => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n48k5120-e0.kxefamily"
        )),
        QwenProjection::MixerOut => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n5120k6144-e0.kxefamily"
        )),
        QwenProjection::GatedQuery => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n12288k5120-e0.kxefamily"
        )),
        QwenProjection::KeyValue => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n1024k5120-e0.kxefamily"
        )),
        QwenProjection::MlpGate => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n17408k5120-e1.kxefamily"
        )),
        QwenProjection::MlpUp => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n17408k5120-e0.kxefamily"
        )),
        QwenProjection::MlpDown => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n5120k17408-e0.kxefamily"
        )),
        QwenProjection::MtpFc => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n5120k5120-e0.kxefamily"
        )),
        QwenProjection::MoeMixerIn => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n8192k2048-e0.kxefamily"
        )),
        QwenProjection::MoeDeltaGate => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n4096k2048-e1.kxefamily"
        )),
        QwenProjection::MoeMixerOut => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n2048k4096-e0.kxefamily"
        )),
        QwenProjection::MoeKeyValue => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n512k2048-e0.kxefamily"
        )),
        QwenProjection::MoeMtpFc => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-nvfp4-m8n2048k2048-e0.kxefamily"
        )),
        _ => return None,
    })
}

/// The captured Qwen vocabularies: Qwen3's 151936 rows and Qwen3.5's
/// 248320 ([`QWEN_HEADS`] names the hidden widths each is captured over).
pub const QWEN_VOCABULARIES: [u32; 2] = [151936, 248320];

/// The captured untied BF16 heads, and the embedding gathers beside them,
/// by (vocabulary, hidden width): Qwen3-32B's and Qwen3.8-27B's over 5120,
/// Qwen3.6-35B-A3B's over 2048.
pub const QWEN_HEADS: [(u32, u32); 3] = [(151936, QWEN_HIDDEN), (248320, QWEN_HIDDEN), (248320, QWEN3_5_MOE_HIDDEN)];

/// The tokens Qwen3.5's draft head scores: the vocabulary's first 98304.
/// Its ids follow the BPE merges, so a prefix keeps the frequent tokens:
/// below 94208 they are English and code, a Chinese block follows, and the
/// multilingual extension and the specials (the chat's end, `</think>`)
/// lie past 151643. Over 29M tokens of local text (Rust's books and std
/// docs, man pages, Markdown, Python and Rust sources) the prefix holds
/// 99.5% of them and 99.9% of the English and code (the rest is the books'
/// Chinese, Japanese and Korean translations), against 97.4% at 65536; the
/// head reads 0.28 GB a draft instead of 0.72. A draft is only a proposal,
/// so the prefix costs acceptance, never correctness.
pub const QWEN3_5_DRAFT_VOCAB: u32 = 98304;

/// The hidden width of Qwen3-32B and Qwen3.8-27B, which their vocabulary
/// and layer-norm packages bake.
pub const QWEN_HIDDEN: u32 = 5120;

/// Qwen3.6-35B-A3B's hidden width (`qwen3_5_moe`).
pub const QWEN3_5_MOE_HIDDEN: u32 = 2048;

/// The Qwen dense (unquantized) projection: the untied BF16 lm_head
/// `[vocab, hidden]` of a captured head ([`QWEN_HEADS`]) for one row, on
/// the vector engines (`xe-dense-projection.comp` with `VECTOR_M=1`: 105
/// GB/s on Xe3 against XMX's 62), [`qwen_lm_head_lanes`] lanes. Launch
/// `[ceil(vocab / 16), 1, 1]` workgroups.
pub fn qwen_lm_head(vocab: u32, hidden: u32) -> Option<XeKernelArtifact> {
    Some(match (vocab, hidden) {
        (151936, QWEN_HIDDEN) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m1n151936k5120.kxefamily"
        )),
        (248320, QWEN_HIDDEN) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m1n248320k5120.kxefamily"
        )),
        (248320, QWEN3_5_MOE_HIDDEN) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m1n248320k2048.kxefamily"
        )),
        _ => return None,
    })
}

/// Lanes of a [`qwen_lm_head`] workgroup, 32 x SPLIT_K: four teams over
/// 5120, eight over 2048 (where ANV unrolls fewer teams' short loop past
/// 128 GRFs).
pub const fn qwen_lm_head_lanes(hidden: u32) -> u32 {
    if hidden == QWEN3_5_MOE_HIDDEN { 32 * 8 } else { 32 * 4 }
}

/// The BF16 lm_head over up to [`QWEN_ROWS`] rows (a verification's): XMX
/// on BF16 as stored (`xe-dense-projection.comp` with `XMX_STREAM=1 M=8
/// RUNTIME_M=1`, SIMD16), the pass's row count from the parameter block;
/// [`qwen_lm_head_rows_geometry`]'s lanes and vocabulary rows a
/// workgroup. Qwen3.5's vocabulary only.
pub fn qwen_lm_head_rows(vocab: u32, hidden: u32) -> Option<XeKernelArtifact> {
    match (vocab, hidden) {
        (248320, QWEN_HIDDEN) => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n248320k5120.kxefamily"
        ))),
        (248320, QWEN3_5_MOE_HIDDEN) => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n248320k2048.kxefamily"
        ))),
        _ => None,
    }
}

/// A [`qwen_lm_head_rows`] workgroup: its lanes (sixteen per SIMD16
/// subgroup, TILES along the vocabulary times SPLIT_K teams along K) and
/// the vocabulary rows it covers (sixteen a tile). Sixteen tiles of two
/// teams over 5120; eight of four over 2048 (111 against 107 GB/s).
pub const fn qwen_lm_head_rows_geometry(hidden: u32) -> (u32, u32) {
    let (tiles, split) = if hidden == QWEN3_5_MOE_HIDDEN { (8, 4) } else { (16, 2) };
    (16 * tiles * split, 16 * tiles)
}

/// The Xe3 M16 captures use two K teams, with four N tiles at K2048 and
/// eight at K5120.
pub const fn qwen_lm_head_prefill16_geometry(hidden: u32) -> (u32, u32) {
    if hidden == QWEN3_5_MOE_HIDDEN { (128, 64) } else { (256, 128) }
}

/// The Qwen top-1 selection over F32 `[tokens, vocab]` logits at a captured
/// vocabulary, and the embed gather over a BF16 `[vocab, hidden]` table of
/// [`QWEN_HEADS`]. Both are tiny work per token; captured so the first
/// decode's graph can reach the GPU rather than fall back silently.
pub fn qwen_top1(vocab: u32) -> Option<XeKernelArtifact> {
    Some(match vocab {
        151936 => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-top1-v151936.kxefamily"
        )),
        248320 => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-top1-v248320.kxefamily"
        )),
        QWEN3_5_DRAFT_VOCAB => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-top1-v98304.kxefamily"
        )),
        _ => return None,
    })
}

pub fn qwen_gather(vocab: u32, hidden: u32) -> Option<XeKernelArtifact> {
    Some(match (vocab, hidden) {
        (151936, QWEN_HIDDEN) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-gather-v151936-h5120.kxefamily"
        )),
        (248320, QWEN_HIDDEN) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-gather-v248320-h5120.kxefamily"
        )),
        (248320, QWEN3_5_MOE_HIDDEN) => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-gather-v248320-h2048.kxefamily"
        )),
        _ => return None,
    })
}

/// Qwen3.6-35B-A3B's small BF16 projections, kept as published
/// (`xe-dense-projection.comp`, `FORMAT=2`): its router (the 256 experts'
/// rows, then the shared expert's gate row) and its delta rule's gate
/// inputs. Latency-bound: one row on the vector engines at a required
/// SIMD16 (four teams, 128 lanes: 7 us against eight teams' 12 at SIMD32),
/// a verification's on XMX as BF16 (one tile of eight teams, 128 lanes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenDense {
    /// `mlp.router.weight` `[257, 2048]`, identity.
    MoeRouter,
    /// `in_proj_a` `[32, 2048]`, identity: the step's `softplus(a +
    /// dt_bias)` is an F32 map.
    MoeStep,
    /// `in_proj_b` `[32, 2048]` with sigmoid fused (e2): the write strength.
    MoeBeta,
}

impl QwenDense {
    pub const ALL: [Self; 3] = [Self::MoeRouter, Self::MoeStep, Self::MoeBeta];

    pub const fn n(self) -> u32 {
        match self {
            Self::MoeRouter => 257,
            Self::MoeStep | Self::MoeBeta => 32,
        }
    }

    pub const fn k(self) -> u32 {
        QWEN3_5_MOE_HIDDEN
    }

    /// The fused epilogue: 0 identity, 2 sigmoid.
    pub const fn epilogue(self) -> u32 {
        match self {
            Self::MoeBeta => 2,
            _ => 0,
        }
    }

    /// Lanes of the one-row and the multi-row workgroup alike: four teams of
    /// 32, and one tile of eight SIMD16 teams; each covers sixteen columns.
    pub const fn group_width(self) -> u32 {
        128
    }

    /// The Xe3 M16 BF16 captures use one N tile and two K teams.
    pub const fn prefill16_group_width(self) -> u32 {
        32
    }
}

/// The one-row package of a [`QwenDense`] projection: launch
/// `[ceil(n / 16), 1, 1]` workgroups.
pub fn qwen_dense(kind: QwenDense) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenDense::MoeRouter => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m1n257k2048-e0.kxefamily"
        )),
        QwenDense::MoeStep => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m1n32k2048-e0.kxefamily"
        )),
        QwenDense::MoeBeta => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m1n32k2048-e2.kxefamily"
        )),
    })
}

/// The multi-row package of a [`QwenDense`] projection, up to
/// [`QWEN_ROWS`] rows from the parameter block: launch `[ceil(n / 16), 1,
/// 1]` workgroups.
pub fn qwen_dense_rows(kind: QwenDense) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenDense::MoeRouter => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n257k2048-e0.kxefamily"
        )),
        QwenDense::MoeStep => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n32k2048-e0.kxefamily"
        )),
        QwenDense::MoeBeta => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-dense-bf16-m8n32k2048-e2.kxefamily"
        )),
    })
}

/// Lanes of a [`fill_u32`] workgroup: one word each.
pub const FILL_LANES: u32 = 256;

/// A fill of 32-bit words with one value (`xe-fill-u32.comp`): the control
/// block `{count, value}`, then the words, written; a plain store, so any
/// bit pattern lands exactly. Launch `ceil(count / FILL_LANES)` workgroups.
/// For a region the process cannot map: the frozen decode writes its
/// recurrent-state trailers with it. Xe3 only; Xe2 refuses it at load.
pub fn fill_u32() -> XeKernelArtifact {
    artifact(include_bytes!(
        "../../../drivers/xe/kernels/anv-fill-u32.kxefamily"
    ))
}

/// The `[rows, width]` F32 tables an [`f32_row_gather`] package is captured
/// for: `[48, 256]`, Qwen3.8-27B's gated attention `q_proj` output of one
/// token, each of its 24 heads' 256 query values followed by its 256 gate
/// values; `[384, 256]`, the same of [`QWEN_ROWS`] tokens; `[32, 256]` and
/// `[256, 256]`, Qwen3.6-35B-A3B's 16 heads at one token and at eight.
pub const F32_ROW_GATHERS: [(u32, u32); 4] = [(48, 256), (384, 256), (32, 256), (256, 256)];

/// An F32 row gather (`xe-nemotron-gather.comp` with `FORMAT=1`) over a
/// `[rows, width]` table of [`F32_ROW_GATHERS`], one 256-lane workgroup per
/// gathered row: one token's q/gate split into queries then gates. Xe3
/// only.
pub fn f32_row_gather(rows: u32, width: u32) -> Option<XeKernelArtifact> {
    match (rows, width) {
        (48, 256) => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-gather-f32-v48-h256.kxefamily"
        ))),
        (384, 256) => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-gather-f32-v384-h256.kxefamily"
        ))),
        (32, 256) => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-gather-f32-v32-h256.kxefamily"
        ))),
        (256, 256) => Some(artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-gather-f32-v256-h256.kxefamily"
        ))),
        _ => None,
    }
}

/// Qwen RMS norms: the 5120-wide layer norms and the per-head QK-norms, all
/// eps 1e-6. A QK-norm weight is a single vector shared across heads, so the
/// heads are a dispatch count, not a kernel grouping. Qwen3.5 stores its norms
/// as `w` for `1 + w`, which the graph bakes into the weight, so its layer
/// norms are [`Self::Layer`] and its delta rule's 128-wide head norm (a plain
/// weight) is [`Self::HeadQk`]'s package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenRms {
    /// input / post-attention / final norm, width 5120.
    Layer,
    /// q_norm / k_norm, width 128 (Qwen3; Qwen3.5's delta-rule head norm).
    HeadQk,
    /// Qwen3.5's gated attention q_norm / k_norm, width 256.
    GatedHeadQk,
    /// Qwen3.6-35B-A3B's input / post-attention / final norm, width 2048.
    MoeLayer,
}

impl QwenRms {
    pub const ALL: [Self; 4] = [Self::Layer, Self::HeadQk, Self::GatedHeadQk, Self::MoeLayer];

    /// Qwen3-32B's norms.
    pub const QWEN3: [Self; 2] = [Self::Layer, Self::HeadQk];

    /// Qwen3.8-27B's norms.
    pub const QWEN3_5: [Self; 3] = [Self::Layer, Self::HeadQk, Self::GatedHeadQk];

    /// Qwen3.6-35B-A3B's norms.
    pub const QWEN3_5_MOE: [Self; 3] = [Self::MoeLayer, Self::HeadQk, Self::GatedHeadQk];

    pub const fn width(self) -> u32 {
        match self {
            Self::Layer => 5120,
            Self::HeadQk => 128,
            Self::GatedHeadQk => 256,
            Self::MoeLayer => QWEN3_5_MOE_HIDDEN,
        }
    }

    /// The norm package at a captured `width`.
    pub const fn of_width(width: u32) -> Option<Self> {
        match width {
            5120 => Some(Self::Layer),
            128 => Some(Self::HeadQk),
            256 => Some(Self::GatedHeadQk),
            QWEN3_5_MOE_HIDDEN => Some(Self::MoeLayer),
            _ => None,
        }
    }

    /// Whether this is a model's layer norm (over its hidden state) rather
    /// than a per-head one.
    pub const fn is_layer(self) -> bool {
        matches!(self, Self::Layer | Self::MoeLayer)
    }
}

/// The Qwen RMS (and QK-norm) package, captured on Xe3 (`op0`, eps 1e-6).
pub fn qwen_rms(kind: QwenRms) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenRms::Layer => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rms-op0-w5120-g1.kxefamily"
        )),
        QwenRms::HeadQk => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rms-op0-w128-g1.kxefamily"
        )),
        QwenRms::GatedHeadQk => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rms-op0-w256-g1.kxefamily"
        )),
        QwenRms::MoeLayer => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rms-op0-w2048-g1.kxefamily"
        )),
    })
}

/// Qwen NeoX split-half RoPE (standalone `xe-rope.comp`, 64 lanes per
/// (token, head)): Qwen3's Q with 64 heads and K with 8 (GQA 8:1), each
/// 128-wide head rotated whole; Qwen3.5's gated attention Q with 24 heads and
/// K with 4, each 256 wide with its leading 64 rotated and the rest passed
/// through (`partial_rotary_factor` 0.25).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenRope {
    Query,
    Key,
    GatedQuery,
    GatedKey,
    /// Qwen3.6-35B-A3B's 16 query heads (the leading 64 of 256 rotate).
    MoeQuery,
    /// Its two KV heads.
    MoeKey,
}

impl QwenRope {
    pub const ALL: [Self; 6] = [
        Self::Query,
        Self::Key,
        Self::GatedQuery,
        Self::GatedKey,
        Self::MoeQuery,
        Self::MoeKey,
    ];

    /// Qwen3-32B's rotations.
    pub const QWEN3: [Self; 2] = [Self::Query, Self::Key];

    /// Qwen3.8-27B's rotations.
    pub const QWEN3_5: [Self; 2] = [Self::GatedQuery, Self::GatedKey];

    /// Qwen3.6-35B-A3B's rotations.
    pub const QWEN3_5_MOE: [Self; 2] = [Self::MoeQuery, Self::MoeKey];

    pub const fn heads(self) -> u32 {
        match self {
            Self::Query => 64,
            Self::Key => 8,
            Self::GatedQuery => 24,
            Self::GatedKey => 4,
            Self::MoeQuery => 16,
            Self::MoeKey => 2,
        }
    }

    pub const fn head_dim(self) -> u32 {
        match self {
            Self::Query | Self::Key => 128,
            Self::GatedQuery | Self::GatedKey | Self::MoeQuery | Self::MoeKey => 256,
        }
    }

    /// The leading columns of each head that rotate.
    pub const fn rotary(self) -> u32 {
        match self {
            Self::Query | Self::Key => 128,
            Self::GatedQuery | Self::GatedKey | Self::MoeQuery | Self::MoeKey => 64,
        }
    }

    /// The captured rotation of `heads` heads `head_dim` wide whose leading
    /// `rotary` columns rotate.
    pub fn of(heads: u32, head_dim: u32, rotary: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| (kind.heads(), kind.head_dim(), kind.rotary()) == (heads, head_dim, rotary))
    }
}

/// The Qwen RoPE package, captured on Xe3 (standalone NeoX split-half).
pub fn qwen_rope(kind: QwenRope) -> Option<XeKernelArtifact> {
    Some(match kind {
        QwenRope::Query => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rope-neox-h64d128.kxefamily"
        )),
        QwenRope::Key => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rope-neox-h8d128.kxefamily"
        )),
        QwenRope::GatedQuery => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rope-neox-h24d256r64.kxefamily"
        )),
        QwenRope::GatedKey => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rope-neox-h4d256r64.kxefamily"
        )),
        QwenRope::MoeQuery => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rope-neox-h16d256r64.kxefamily"
        )),
        QwenRope::MoeKey => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-rope-neox-h2d256r64.kxefamily"
        )),
    })
}

fn artifact<const N: usize>(bytes: &'static [u8; N]) -> XeKernelArtifact {
    XeKernelArtifact {
        bytes,
        source: bytes,
    }
}

#[cfg(test)]
mod tests {
    /// A two-generation family stages one generation's package: about half
    /// the payload, never less than a package's ISA; a lone package stages
    /// all of itself.
    #[test]
    fn staged_bytes_charge_one_generation_of_a_family() {
        let read = super::kv_kernel(super::KvKernel::Attention, super::KvGeometry::NEMOTRON).unwrap();
        let staged = read.staged_bytes();
        assert!(staged < read.bytes().len(), "{staged} < {}", read.bytes().len());
        assert!(staged * 2 >= read.bytes().len() - 64, "{staged} vs {}", read.bytes().len());
        assert!(staged > 16 * 1024);
        let plan = super::moe_plan().unwrap();
        assert!(plan.staged_bytes() < plan.bytes().len());
        assert_eq!(super::family_largest_package(b"not a family"), None);
    }

    use super::*;

    #[test]
    fn m16_catalog_requires_the_explicit_sealed_families() {
        assert_eq!(QWEN_PREFILL16_ROWS, 16);
        assert_eq!(QwenProjection::DeltaBeta.prefill16_geometry(), (1, 2));
        assert_eq!(QwenProjection::MoeMixerIn.prefill16_geometry(), (16, 2));
        assert_eq!(qwen_lm_head_prefill16_geometry(QWEN_HIDDEN), (256, 128));
        assert_eq!(qwen_lm_head_prefill16_geometry(QWEN3_5_MOE_HIDDEN), (128, 64));
        assert_eq!(QwenDense::MoeRouter.prefill16_group_width(), 32);
        if !QWEN_PREFILL16_NATIVE {
            for kind in QwenProjection::ROWS {
                assert!(qwen_projection_prefill16(kind).is_none());
            }
            for kind in QwenDense::ALL {
                assert!(qwen_dense_prefill16(kind).is_none());
            }
            assert!(qwen_lm_head_prefill16(248320, QWEN_HIDDEN).is_none());
            assert!(f32_row_gather_prefill16(768, 256).is_none());
        }
    }

    static FIRST: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    static COPY: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    static OTHER: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 9];

    /// Two statics are two addresses; equal bytes are one package.
    #[test]
    fn a_package_is_the_same_at_any_address() {
        let (first, copy, other) = (artifact(&FIRST), artifact(&COPY), artifact(&OTHER));
        assert_ne!(first.bytes().as_ptr(), copy.bytes().as_ptr());
        assert!(first.same(copy) && copy.same(first) && first.same(first));
        assert!(!first.same(other));
        assert!(!first.same(artifact(&[1, 2, 3, 4])));
        let norm = rms_norm(RmsNorm::Hidden);
        assert!(norm.same(rms_norm(RmsNorm::Hidden)));
        assert!(!norm.same(rms_norm(RmsNorm::MambaGateGroup)));
    }

    /// The Qwen catalogue's shape metadata stays in one place: the values the
    /// capture driver sealed (res/QWEN_XE_CATALOG.md), asserted here, and every
    /// captured family resolves for Xe3 with its sealed slot count.
    #[test]
    fn qwen_catalog_metadata_matches_the_capture_plan() {
        let shape = |p: QwenProjection| (p.n(), p.k(), p.slots());
        assert_eq!(shape(QwenProjection::Query), (8192, 5120, 5));
        assert_eq!(shape(QwenProjection::KeyValue), (1024, 5120, 5));
        assert_eq!(shape(QwenProjection::AttentionO), (5120, 8192, 5));
        assert_eq!(shape(QwenProjection::Ffn), (25600, 5120, 5));
        assert_eq!(shape(QwenProjection::FfnDown), (5120, 25600, 5));
        for projection in QwenProjection::ALL {
            assert_eq!(projection.k() % 16, 0, "{projection:?} K is block-aligned");
            let package = qwen_projection(projection).expect("captured");
            assert!(package.bytes().len() > 64, "{projection:?} has a sealed payload");
        }
        assert_eq!(QwenRms::Layer.width(), 5120);
        assert_eq!(QwenRms::HeadQk.width(), 128);
        assert_eq!(QwenRope::Query.heads(), 64);
        assert_eq!(QwenRope::Key.heads(), 8);
        for (vocab, hidden) in QWEN_HEADS {
            assert!(qwen_lm_head(vocab, hidden).is_some() && qwen_top1(vocab).is_some());
            assert!(qwen_gather(vocab, hidden).is_some());
        }
        assert!(qwen_lm_head(131072, QWEN_HIDDEN).is_none() && qwen_gather(QWEN_HIDDEN, QWEN_HIDDEN).is_none());
        assert!(qwen_lm_head(151936, QWEN3_5_MOE_HIDDEN).is_none());
        assert!(qwen_rms(QwenRms::Layer).is_some() && qwen_rope(QwenRope::Query).is_some());
    }

    /// Qwen3.8-27B's decode packages: every projection shape and epilogue
    /// from the QUASAR checkpoint's linears, the partial rotations, the
    /// 256-wide norm, the q/gate split and the four-KV-head cache.
    #[test]
    fn qwen3_5_catalog_matches_the_27b_layer() {
        let shape = |p: QwenProjection| (p.n(), p.k(), p.epilogue(), p.group_width());
        assert_eq!(shape(QwenProjection::DeltaQkv), (10240, 5120, 0, 128));
        assert_eq!(shape(QwenProjection::DeltaGate), (6144, 5120, 1, 128));
        assert_eq!(shape(QwenProjection::DeltaBeta), (48, 5120, 2, 256));
        assert_eq!(shape(QwenProjection::DeltaStep), (48, 5120, 0, 256));
        assert_eq!(shape(QwenProjection::MixerOut), (5120, 6144, 0, 128));
        assert_eq!(shape(QwenProjection::GatedQuery), (12288, 5120, 0, 128));
        assert_eq!(shape(QwenProjection::KeyValue), (1024, 5120, 0, 128));
        assert_eq!(shape(QwenProjection::MlpGate), (17408, 5120, 1, 128));
        assert_eq!(shape(QwenProjection::MlpUp), (17408, 5120, 0, 128));
        assert_eq!(shape(QwenProjection::MlpDown), (5120, 17408, 0, 256));
        assert_eq!(shape(QwenProjection::MtpFc), (5120, 5120, 0, 128));
        assert_eq!(shape(QwenProjection::DraftHead), (QWEN3_5_DRAFT_VOCAB, 5120, 0, 128));
        assert!(qwen_top1(QWEN3_5_DRAFT_VOCAB).is_some());
        // The multi-row packages: every 27B layer linear and the head's fc,
        // sealed; nothing else has one. A workgroup covers whole sixteen-
        // column tiles and at most 64 subgroups.
        for kind in QwenProjection::ALL {
            assert_eq!(qwen_projection_rows(kind).is_some(), QwenProjection::ROWS.contains(&kind), "{kind:?}");
            let (tiles, split) = kind.rows_geometry();
            assert!(tiles * split <= 64 && kind.rows_group_columns() % 16 == 0, "{kind:?}");
        }
        for kind in QwenProjection::QWEN3_5 {
            assert!(QwenProjection::ROWS.contains(&kind), "{kind:?} verifies at eight rows");
        }
        assert_eq!(QwenProjection::MlpDown.rows_group_width(), 512);
        assert_eq!(QwenProjection::DeltaBeta.rows_group_columns(), 16);
        assert!(qwen_lm_head_rows(248320, QWEN_HIDDEN).is_some() && qwen_lm_head_rows(151936, QWEN_HIDDEN).is_none());
        assert_eq!(qwen_lm_head_rows_geometry(QWEN_HIDDEN), (512, 256));
        assert!(f32_row_gather(48 * QWEN_ROWS, 256).is_some() && f32_row_gather(192, 256).is_none());
        assert!(f32_row_gather(32 * QWEN_ROWS, 256).is_some());
        for (kind, channels) in [(CausalConv::Bank, 10240), (CausalConv::Bank, 6144)] {
            assert_eq!(causal_conv(kind, channels).is_some(), channels == 10240);
        }
        assert!(delta_rule(DeltaRule::Bank, DeltaRuleGeometry::QWEN3_5_27B).is_some());
        // Each model's set is part of the catalog, and no two members of one
        // set are the same (N, K, epilogue).
        for set in [
            &QwenProjection::QWEN3[..],
            &QwenProjection::QWEN3_5[..],
            &QwenProjection::QWEN3_5_MTP[..],
            &QwenProjection::QWEN3_5_MOE[..],
            &QwenProjection::QWEN3_5_MOE_MTP[..],
        ] {
            for (i, a) in set.iter().enumerate() {
                assert!(QwenProjection::ALL.contains(a));
                for b in &set[i + 1..] {
                    assert_ne!((a.n(), a.k(), a.epilogue()), (b.n(), b.k(), b.epilogue()));
                }
            }
        }
        assert_eq!(QwenRms::of_width(256), Some(QwenRms::GatedHeadQk));
        assert_eq!(QwenRms::of_width(128), Some(QwenRms::HeadQk));
        assert_eq!(QwenRms::of_width(2688), None);
        assert_eq!(QwenRms::of_width(2048), Some(QwenRms::MoeLayer));
        assert_eq!(QwenRope::of(24, 256, 64), Some(QwenRope::GatedQuery));
        assert_eq!(QwenRope::of(4, 256, 64), Some(QwenRope::GatedKey));
        assert_eq!(QwenRope::of(64, 128, 128), Some(QwenRope::Query));
        assert_eq!(QwenRope::of(24, 256, 256), None, "a whole-head 256 rotation is not captured");
        for rope in QwenRope::QWEN3_5 {
            assert!(qwen_rope(rope).is_some());
        }
        assert!(f32_row_gather(48, 256).is_some() && f32_row_gather(96, 256).is_none());
        let qwen3_5 = KvGeometry::QWEN3_5;
        assert_eq!(KvGeometry::of_cache(4, 256), Some(qwen3_5));
        assert_eq!(KvGeometry::of_cache(4, 128), None);
        assert_eq!(qwen3_5.subgroups(), 8);
        assert_eq!(KvGeometry::QWEN3.subgroups(), 8);
        for geometry in KvGeometry::ALL {
            assert_eq!((geometry.query_heads / geometry.kv_heads) % geometry.rows, 0);
            assert!(geometry.subgroups().is_power_of_two());
            for kind in KvKernel::ALL {
                assert!(kv_kernel(kind, geometry).is_some(), "{geometry:?} {kind:?}");
            }
        }
        assert_eq!(KvKernel::Append.group_width(qwen3_5), 256);
        assert_eq!(KvKernel::Attention.group_width(qwen3_5), 256);
        assert_eq!(KvKernel::Append.group_width(KvGeometry::QWEN3), 128);
        let mismatched = KvGeometry {
            rows: 8,
            ..KvGeometry::QWEN3_5
        };
        assert!(kv_kernel(KvKernel::Attention, mismatched).is_none());
    }

    /// Qwen3.6-35B-A3B's decode packages: every projection shape and
    /// epilogue of its layers, its prediction head's, its head and gather
    /// over 2048, the router and gate inputs as BF16, its norms, rotations,
    /// two-KV-head cache, convolution and delta rule.
    #[test]
    fn qwen3_5_moe_catalog_matches_the_35b_layer() {
        let shape = |p: QwenProjection| (p.n(), p.k(), p.epilogue(), p.group_width());
        assert_eq!(shape(QwenProjection::MoeMixerIn), (8192, 2048, 0, 64));
        assert_eq!(shape(QwenProjection::MoeDeltaGate), (4096, 2048, 1, 64));
        assert_eq!(shape(QwenProjection::MoeMixerOut), (2048, 4096, 0, 64));
        assert_eq!(shape(QwenProjection::MoeKeyValue), (512, 2048, 0, 64));
        assert_eq!(shape(QwenProjection::MoeMtpFc), (2048, 2048, 0, 64));
        assert_eq!(shape(QwenProjection::MoeDraftHead), (QWEN3_5_DRAFT_VOCAB, 2048, 0, 128));
        for kind in QwenProjection::QWEN3_5_MOE.into_iter().chain([QwenProjection::MoeMtpFc]) {
            assert!(qwen_projection(kind).is_some() && qwen_projection_rows(kind).is_some(), "{kind:?}");
        }
        assert_eq!(QwenProjection::MoeKeyValue.rows_group_width(), 512);
        assert_eq!(QwenProjection::MoeMtpFc.rows_group_columns(), 64);
        assert!(qwen_projection(QwenProjection::MoeDraftHead).is_some());
        assert!(qwen_lm_head(248320, QWEN3_5_MOE_HIDDEN).is_some() && qwen_lm_head_rows(248320, QWEN3_5_MOE_HIDDEN).is_some());
        assert_eq!(qwen_lm_head_lanes(QWEN3_5_MOE_HIDDEN), 256);
        assert_eq!(qwen_lm_head_rows_geometry(QWEN3_5_MOE_HIDDEN), (512, 128));
        assert!(qwen_gather(248320, QWEN3_5_MOE_HIDDEN).is_some());
        let dense: [(u32, u32, u32); 3] = QwenDense::ALL.map(|kind| (kind.n(), kind.k(), kind.epilogue()));
        assert_eq!(dense, [(257, 2048, 0), (32, 2048, 0), (32, 2048, 2)]);
        for (i, kind) in QwenDense::ALL.into_iter().enumerate() {
            let (one, rows) = (qwen_dense(kind).unwrap(), qwen_dense_rows(kind).unwrap());
            assert!(!one.same(rows));
            for other in &QwenDense::ALL[i + 1..] {
                assert!(!one.same(qwen_dense(*other).unwrap()));
            }
        }
        assert!(QwenRms::QWEN3_5_MOE.into_iter().all(|kind| qwen_rms(kind).is_some()));
        assert!(QwenRms::MoeLayer.is_layer() && !QwenRms::GatedHeadQk.is_layer());
        assert_eq!(QwenRope::of(16, 256, 64), Some(QwenRope::MoeQuery));
        assert_eq!(QwenRope::of(2, 256, 64), Some(QwenRope::MoeKey));
        assert!(QwenRope::QWEN3_5_MOE.into_iter().all(|kind| qwen_rope(kind).is_some()));
        let kv = KvGeometry::QWEN3_5_MOE;
        assert_eq!(KvGeometry::of_cache(2, 256), Some(kv));
        assert_eq!(KvGeometry::of_cache(2, 128), Some(KvGeometry::NEMOTRON));
        for kind in KvKernel::ALL {
            assert!(kv_kernel(kind, kv).is_some(), "{kind:?}");
        }
        assert_eq!((KvKernel::Append.group_width(kv), KvKernel::Attention.group_width(kv)), (256, 256));
        for kind in [CausalConv::Runtime, CausalConv::Bank] {
            assert!(causal_conv(kind, 8192).is_some(), "{kind:?}");
        }
        assert!(causal_conv(CausalConv::Tree, 8192).is_none());
        let rule = DeltaRuleGeometry::QWEN3_5_MOE;
        assert!(delta_rule(DeltaRule::Runtime, rule).is_some() && delta_rule(DeltaRule::Bank, rule).is_some());
        assert!(delta_rule(DeltaRule::Tree, rule).is_none());
        assert_eq!(DeltaRule::Runtime.group_width(rule), 512);
        for (rows, width) in [(32, 256), (256, 256)] {
            assert!(f32_row_gather(rows, width).is_some());
        }
    }

    /// Qwen3.6-35B-A3B's routed experts: every kernel sealed at its
    /// geometry and nowhere else, each a distinct package, launched over
    /// whole tiles of its outputs.
    #[test]
    fn routed_kernels_match_the_capture_plan() {
        let qwen = RoutedGeometry::QWEN3_6_35B_A3B;
        assert_eq!((qwen.records(), qwen.slots()), (257, 9));
        for (i, kind) in RoutedKernel::ALL.into_iter().enumerate() {
            let package = routed_kernel(kind, qwen).expect("captured");
            assert!(package.bytes().len() > 64, "{kind:?}");
            for other in &RoutedKernel::ALL[i + 1..] {
                assert!(!package.same(routed_kernel(*other, qwen).unwrap()), "{kind:?} {other:?}");
            }
            assert!(routed_kernel(kind, RoutedGeometry { top_k: 6, ..qwen }).is_none());
            assert!(routed_kernel(kind, RoutedGeometry { inner: 768, ..qwen }).is_none());
            assert_eq!(kind.groups(qwen, 0), None);
        }
        assert_eq!(RoutedKernel::Route.groups(qwen, 8), Some([8, 1, 1]));
        assert_eq!(RoutedKernel::GateUp.groups(qwen, 1), Some([64, 9, 1]));
        assert_eq!(RoutedKernel::GateUp.groups(qwen, 8), Some([64, 72, 1]));
        assert_eq!(RoutedKernel::Down.groups(qwen, 8), Some([128, 8, 1]));
        assert_eq!(
            RoutedKernel::ALL.map(|kind| (kind.group_width(), kind.bindings())),
            [(32, 2), (128, 4), (64, 4)]
        );
    }
}
