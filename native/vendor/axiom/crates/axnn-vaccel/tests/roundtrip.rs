//! Builder → hardened parser → provider → kernels, closed loop.
//!
//! Every artifact the builder emits must survive `virtio-accel-tosa`'s
//! parse + semantic analysis for the axtosa target (the same validator
//! all NPU providers run), and executing it through the full
//! `Accelerator` contract must reproduce an independent scalar matmul.

use axnn_vaccel::{AxTosaAccelerator, Hetero, InvocationStatus, MatMulDType, Route, Session};
use virtio_accel_core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ContextDesc, MemoryDomain, QueueDesc, Timeout,
};

const DTS: [MatMulDType; 4] = [
    MatMulDType::F32,
    MatMulDType::F16,
    MatMulDType::B8,
    MatMulDType::F8,
];

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let magnitude = if exp == 0 {
        // Subnormal: mant × 2⁻²⁴.
        mant as f32 * f32::from_bits(0x3380_0000)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        f32::from_bits(((exp + 112) << 23) | (mant << 13))
    };
    f32::from_bits(sign | magnitude.to_bits())
}

fn f32_to_f16_bits(v: f32) -> u16 {
    // Round-to-nearest-even via the f32 halfway trick: exact for the
    // value range these tests generate (no overflow/underflow cases).
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let av = v.abs();
    if av == 0.0 {
        return sign;
    }
    // Scale into u32 mantissa space, round, reassemble.
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    if e >= -14 {
        let mant = (bits & 0x7f_ffff) | 0x80_0000;
        let shift = 13;
        let half = 1u32 << (shift - 1);
        let mut m = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        let mut exp16 = (e + 15) as u32;
        if m >= 0x800 {
            // Mantissa carry: 0x800 pattern means exponent bump.
            if m >= 0x1000 {
                exp16 += 1;
                m >>= 1;
            }
        }
        (sign as u32 | (exp16 << 10) | (m & 0x3ff)) as u16
    } else {
        // Subnormal f16.
        let scaled = av * f32::from_bits(0x4b80_0000); // × 2²⁴
        let m = scaled.round_ties_even() as u32;
        sign | (m as u16 & 0x3ff)
    }
}

/// Deterministic pseudo-random byte stream.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Generate an input plane for `dt`: raw little-endian bytes plus the
/// exact f32 value of each element.
fn plane(dt: MatMulDType, len: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let mut bytes = Vec::with_capacity(len * dt.input_size());
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        match dt {
            MatMulDType::F32 => {
                let v = (rng.next() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                bytes.extend_from_slice(&v.to_le_bytes());
                values.push(v);
            }
            MatMulDType::F16 => {
                let v = (rng.next() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                let b = f32_to_f16_bits(v);
                bytes.extend_from_slice(&b.to_le_bytes());
                values.push(f16_bits_to_f32(b));
            }
            MatMulDType::B8 => {
                // Exponent-masked to |v| ≤ 1.875 so k-length dots stay
                // far inside the f16 output range (no NaN encodings).
                let b = {
                    let raw = (rng.next() & 0xff) as u8;
                    (raw & 0x3f) | (raw & 0x80)
                };
                bytes.push(b);
                let mut v = [0.0f32];
                axnn_cpu::fp8::dequant_e4m3(&[b], &mut v, 1);
                values.push(v[0]);
            }
            MatMulDType::F8 => {
                // Exponent-masked to |v| ≤ 1.75 (finite by construction).
                let b = {
                    let raw = (rng.next() & 0xff) as u8;
                    (raw & 0x3f) | (raw & 0x80)
                };
                bytes.push(b);
                let mut v = [0.0f32];
                axnn_cpu::fp8::dequant_e5m2(&[b], &mut v, 1);
                values.push(v[0]);
            }
        }
    }
    (bytes, values)
}

/// Independent oracle: scalar triple loop in f32.
fn scalar_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn decode_output(dt: MatMulDType, bytes: &[u8]) -> Vec<f32> {
    match dt.output_size() {
        4 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    }
}

#[test]
fn artifacts_pass_the_hardened_validator() {
    let target = AxTosaAccelerator::target();
    for dt in DTS {
        let bytes = axnn_vaccel::artifact::matmul(dt, 5, 24, 9).unwrap();
        let model = virtio_accel_tosa::parse(&bytes)
            .unwrap_or_else(|e| panic!("{dt:?}: parse rejected builder output: {e:?}"));
        model
            .analyze_for(target)
            .unwrap_or_else(|e| panic!("{dt:?}: semantic analysis rejected: {e:?}"));
    }
}

#[test]
fn fixed_artifact_authoring_rejects_unrepresentable_or_oversize_shapes() {
    use axnn_vaccel::artifact::ArtifactError;

    // 8192 is not a MATMUL extent limit: it is a Level 8K kernel/stride
    // limit. A small 8193-row tensor is valid for this target.
    assert!(axnn_vaccel::artifact::matmul(MatMulDType::F32, 8193, 1, 1).is_ok());
    assert_eq!(
        axnn_vaccel::artifact::matmul(MatMulDType::F32, 0, 1, 1),
        Err(ArtifactError::ZeroDimension)
    );
    assert_eq!(
        axnn_vaccel::artifact::matmul(MatMulDType::F32, i32::MAX as usize + 1, 1, 1),
        Err(ArtifactError::DimensionTooLarge)
    );
    // This F32 tensor is exactly 2 GiB, one byte above Level 8K's maximum
    // tensor storage. It must remain a recoverable semantic error.
    assert_eq!(
        axnn_vaccel::artifact::matmul(MatMulDType::F32, 536_870_912, 1, 1),
        Err(ArtifactError::AuthoringFailed)
    );
    assert_eq!(
        axnn_vaccel::artifact::mlp_f16(1, &[2, 3], &[vec![0.0; 5]]),
        Err(ArtifactError::InvalidWeightShape { layer: 0 })
    );
}

#[test]
fn malformed_matmul_never_enters_provider_routing_or_program_cache() {
    let mut session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    assert_eq!(
        session.ensure_program(MatMulDType::F32, 536_870_912, 1, 1),
        Err(BackendError::InvalidArgument)
    );
    assert_eq!(
        session.matmul_bytes(MatMulDType::F32, 0, 1, 1, &[], &[]),
        Err(BackendError::InvalidArgument)
    );

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut hetero = Hetero::new(cpu, Some(accel));
    assert_eq!(
        hetero.matmul_bytes(MatMulDType::F32, usize::MAX, 2, 1, &[], &[]),
        Err(BackendError::InvalidArgument)
    );
    assert!(!hetero.is_rejected(MatMulDType::F32));
}

#[test]
fn provider_matmul_matches_scalar_reference() {
    let mut session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    for dt in DTS {
        for (m, k, n) in [(5usize, 24usize, 9usize), (8, 32, 16), (1, 7, 1)] {
            let (a_bytes, a_vals) = plane(dt, m * k, 0x1234 + dt.tag() as u64);
            let (b_bytes, b_vals) = plane(dt, k * n, 0x9876 + dt.tag() as u64);
            let out = session
                .matmul_bytes(dt, m, k, n, &a_bytes, &b_bytes)
                .unwrap_or_else(|e| panic!("{dt:?} {m}x{k}x{n}: {e:?}"));
            let got = decode_output(dt, &out);
            let expected = scalar_matmul(&a_vals, &b_vals, m, k, n);
            // f32 path is exact up to summation order; f16-plane outputs
            // add one RNE narrowing (2⁻¹¹ relative).
            let tol = |e: f32| match dt {
                MatMulDType::F32 => e.abs() * 1e-6 + 1e-6,
                _ => e.abs() * 2.0e-3 + 2.0e-3,
            };
            for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() <= tol(e),
                    "{dt:?} {m}x{k}x{n} elem {i}: got {g}, expected {e}"
                );
            }
        }
    }
}

#[test]
fn f32_output_is_bit_identical_to_the_direct_kernel() {
    let mut session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let (m, k, n) = (7, 40, 11);
    let (a_bytes, a_vals) = plane(MatMulDType::F32, m * k, 42);
    let (b_bytes, b_vals) = plane(MatMulDType::F32, k * n, 43);
    let out = session
        .matmul_bytes(MatMulDType::F32, m, k, n, &a_bytes, &b_bytes)
        .unwrap();
    let mut direct = vec![0.0f32; m * n];
    axnn_cpu::linalg::gemm_f32(&a_vals, &b_vals, &mut direct, m, n, k);
    let got = decode_output(MatMulDType::F32, &out);
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        direct.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "provider plumbing must be byte-faithful to the kernel"
    );
}

#[test]
fn hetero_routes_by_threshold_and_fallback() {
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut hetero = Hetero::new(cpu, Some(accel));
    // 2·8·8·8 = 1024 FLOP; threshold above it pins small work to the CPU.
    hetero.min_accel_flops = 10_000;

    let (a, _) = plane(MatMulDType::F32, 64, 1);
    let (b, _) = plane(MatMulDType::F32, 64, 2);
    let (route, _) = hetero
        .matmul_bytes(MatMulDType::F32, 8, 8, 8, &a, &b)
        .unwrap();
    assert_eq!(route, Route::Cpu);

    let (a, _) = plane(MatMulDType::F32, 32 * 32, 3);
    let (b, _) = plane(MatMulDType::F32, 32 * 32, 4);
    let (route, _) = hetero
        .matmul_bytes(MatMulDType::F32, 32, 32, 32, &a, &b)
        .unwrap();
    assert_eq!(route, Route::Accel);

    // No accelerator session at all → everything routes to the CPU.
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut solo: Hetero<AxTosaAccelerator, AxTosaAccelerator> = Hetero::new(cpu, None);
    let (route, _) = solo
        .matmul_bytes(MatMulDType::F32, 8, 8, 8, &a[..8 * 8 * 4], &b[..8 * 8 * 4])
        .unwrap();
    assert_eq!(route, Route::Cpu);
}

#[test]
fn provider_rejects_missing_bindings_and_foreign_targets() {
    let backend = AxTosaAccelerator::new();
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();

    let artifact = axnn_vaccel::artifact::matmul(MatMulDType::F32, 2, 2, 2).unwrap();
    let model = virtio_accel_tosa::parse(&artifact).unwrap();
    let program = backend
        .load_program(
            &context,
            model
                .artifact_ref(AxTosaAccelerator::target(), u64::MAX)
                .unwrap(),
        )
        .unwrap();

    // Submitting without the output binding must reject cleanly.
    let desc = BufferDesc::new(
        16,
        64,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let (buf_a, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();
    let (buf_b, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();
    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &buf_a,
            range: BufferRange::new(0, 16).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &buf_b,
            range: BufferRange::new(0, 16).unwrap(),
            access: AccessMode::Read,
        },
    ];
    let err = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(
        err,
        virtio_accel_core::SubmitFailure::Rejected(BackendError::InvalidArgument)
    ));

    // An artifact stamped for a different target must not load.
    #[derive(Debug)]
    struct Src<'a>(&'a [u8]);
    impl virtio_accel_core::ByteSource for Src<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
            let start = offset as usize;
            target.copy_from_slice(&self.0[start..start + target.len()]);
            Ok(())
        }
    }
    let foreign = virtio_accel_core::TargetIdentity([0xdead_beef; 12]);
    let payload = Src(&artifact);
    let artifact_ref = virtio_accel_core::ArtifactRef {
        format: virtio_accel_tosa::ARTIFACT_FORMAT,
        target: foreign,
        payload: &payload,
        resident_bytes: u64::MAX,
    };
    assert!(matches!(
        backend.load_program(&context, artifact_ref),
        Err(BackendError::Incompatible)
    ));

    backend.free_buffer(buf_a).unwrap();
    backend.free_buffer(buf_b).unwrap();
    backend.unload_program(program).unwrap();
    backend.destroy_queue(queue).unwrap();
    backend.destroy_context(context).unwrap();
}

#[test]
fn fused_mlp_matches_scalar_reference() {
    let (m, dims) = (37usize, [5usize, 16, 12, 3]);
    let mut rng = Lcg(0xfeed);
    let weights: Vec<Vec<f32>> = dims
        .windows(2)
        .map(|w| {
            (0..w[0] * w[1])
                .map(|_| (rng.next() as f32 / u32::MAX as f32 - 0.5) * 0.8)
                .collect()
        })
        .collect();

    let bytes = axnn_vaccel::artifact::mlp_f32(m, &dims, &weights).unwrap();
    let model = virtio_accel_tosa::parse(&bytes).expect("parse fused mlp");
    model
        .analyze_for(AxTosaAccelerator::target())
        .expect("semantic analysis of fused mlp");

    let (x_bytes, x_vals) = plane(MatMulDType::F32, m * dims[0], 0xf00d);
    let mut session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let out = session
        .run(1, || bytes.clone(), &[&x_bytes], m * 3 * 4)
        .expect("fused mlp run");
    let got = decode_output(MatMulDType::F32, &out);

    // Scalar reference: matmul → tanh, …, matmul → sigmoid.
    let mut act = x_vals;
    for (layer, w) in weights.iter().enumerate() {
        let (k, n) = (dims[layer], dims[layer + 1]);
        let mut next = scalar_matmul(&act, w, m, k, n);
        for v in &mut next {
            *v = if layer + 1 == weights.len() {
                1.0 / (1.0 + (-*v).exp())
            } else {
                v.tanh()
            };
        }
        act = next;
    }
    for (i, (&g, &e)) in got.iter().zip(&act).enumerate() {
        // axtosa's tanh/sigmoid ride axmath approximations.
        assert!(
            (g - e).abs() <= 2e-4,
            "fused mlp elem {i}: got {g}, expected {e}"
        );
    }

    // Second call must hit the program cache (same key, build not run).
    let again = session
        .run(
            1,
            || unreachable!("cache miss on warm key"),
            &[&x_bytes],
            m * 3 * 4,
        )
        .unwrap();
    assert_eq!(out, again);
}

#[test]
fn nonblocking_session_keeps_events_and_fixed_buffers_owned_until_polled() {
    let (m, dims) = (8usize, [4usize, 6, 3]);
    let weights = vec![vec![0.03125; 4 * 6], vec![0.0625; 6 * 3]];
    let artifact = axnn_vaccel::artifact::mlp_f32(m, &dims, &weights).unwrap();
    let (input, _) = plane(MatMulDType::F32, m * dims[0], 0xa51c);
    let expected = axtosa::execute(&artifact, &[&input]).unwrap();
    let output_bytes = [m * 3 * 4];
    let mut session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();

    let invocation = session
        .submit_many(41, || artifact.clone(), &[&input], &output_bytes)
        .unwrap();
    assert_eq!(session.pending_count(), 1);
    let independent = session
        .submit_many(42, || artifact.clone(), &[&input], &output_bytes)
        .unwrap();
    assert_eq!(session.pending_count(), 2);
    let busy = session
        .submit_many(
            41,
            || unreachable!("in-flight key is already compiled"),
            &[&input],
            &output_bytes,
        )
        .unwrap_err();
    assert_eq!(busy.error, BackendError::Busy);
    assert_eq!(busy.invocation, None);

    let output = match session.poll_many(invocation).unwrap() {
        InvocationStatus::Complete(output) => output,
        other => panic!("axtosa provider should complete admission synchronously: {other:?}"),
    };
    assert_eq!(output, expected);
    assert_eq!(session.pending_count(), 1);
    assert_eq!(
        session.poll_many(invocation).unwrap_err(),
        BackendError::InvalidArgument
    );
    assert_eq!(
        match session.poll_many(independent).unwrap() {
            InvocationStatus::Complete(output) => output,
            other => panic!("independent cached program should complete: {other:?}"),
        },
        expected
    );
    assert_eq!(session.pending_count(), 0);

    // Terminal polling releases the fixed I/O signature for a warm submit.
    let warm = session
        .submit_many(
            41,
            || unreachable!("warm async key must hit the program cache"),
            &[&input],
            &output_bytes,
        )
        .unwrap();
    assert!(matches!(
        session.poll_many(warm).unwrap(),
        InvocationStatus::Complete(_)
    ));
    let bound = session
        .submit_many(41, || unreachable!(), &[&input], &output_bytes)
        .unwrap();
    let mut short = [0xa5; 1];
    assert_eq!(
        session
            .poll_many_into(bound, &mut [&mut axnn_vaccel::AxOutput::packed(&mut short)])
            .unwrap_err(),
        BackendError::InvalidArgument
    );
    assert!(session.is_pending(bound));
    assert_eq!(short, [0xa5]);
    let mut backing = vec![0xa5; m * 16 + 8];
    assert!(matches!(
        session
            .poll_many_into(
                bound,
                &mut [&mut axnn_vaccel::AxOutput::pitched(&mut backing, 4, 12, m, 16).unwrap()]
            )
            .unwrap(),
        InvocationStatus::Complete(())
    ));
    let mut expected_backing = vec![0xa5; backing.len()];
    for (row, bytes) in expected[0].as_chunks::<12>().0.iter().enumerate() {
        expected_backing[4 + row * 16..4 + row * 16 + 12].copy_from_slice(bytes);
    }
    assert_eq!(backing, expected_backing);
    assert!(!session.is_pending(bound));

    #[derive(Debug)]
    struct RejectRead(u64);
    impl virtio_accel_core::ByteSink for RejectRead {
        fn len(&self) -> u64 {
            self.0
        }
        fn write_at(&mut self, _: u64, _: &[u8]) -> Result<(), BackendError> {
            Err(BackendError::InvalidArgument)
        }
    }
    let failed = session
        .submit_many(41, || unreachable!(), &[&input], &output_bytes)
        .unwrap();
    assert_eq!(
        session
            .poll_many_into(failed, &mut [&mut RejectRead(output_bytes[0] as u64)])
            .unwrap_err(),
        BackendError::InvalidArgument
    );
    assert!(
        !session.is_pending(failed),
        "terminal event was retired even though readback failed"
    );
}

#[test]
fn fused_mlp_f16_validates_and_runs() {
    let (m, dims) = (24usize, [4usize, 10, 3]);
    let mut rng = Lcg(0xf16);
    let weights: Vec<Vec<f32>> = dims
        .windows(2)
        .map(|w| {
            (0..w[0] * w[1])
                .map(|_| (rng.next() as f32 / u32::MAX as f32 - 0.5) * 0.7)
                .collect()
        })
        .collect();
    let bytes = axnn_vaccel::artifact::mlp_f16(m, &dims, &weights).unwrap();
    let model = virtio_accel_tosa::parse(&bytes).expect("parse f16 mlp");
    model
        .analyze_for(AxTosaAccelerator::target())
        .expect("semantic analysis of f16 mlp");

    let (x_bytes, x_vals) = plane(MatMulDType::F16, m * dims[0], 0xf00d);
    let mut session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let out = session
        .run(2, || bytes.clone(), &[&x_bytes], m * 3 * 2)
        .expect("f16 mlp run");
    let got = decode_output(MatMulDType::F16, &out);

    let mut act = x_vals;
    for (layer, w) in weights.iter().enumerate() {
        let (k, n) = (dims[layer], dims[layer + 1]);
        // Mirror the artifact's f16 weight narrowing in the reference.
        let wq: Vec<f32> = w
            .iter()
            .map(|v| f16_bits_to_f32(f32_to_f16_bits(*v)))
            .collect();
        let mut next = scalar_matmul(&act, &wq, m, k, n);
        for v in &mut next {
            *v = if layer + 1 == weights.len() {
                1.0 / (1.0 + (-*v).exp())
            } else {
                v.tanh()
            };
        }
        act = next;
    }
    for (i, (&g, &e)) in got.iter().zip(&act).enumerate() {
        // f16 storage between every op bounds the error at ~2⁻¹⁰ per hop.
        assert!(
            (g - e).abs() <= 6e-3,
            "f16 mlp elem {i}: got {g}, expected {e}"
        );
    }
}

#[test]
fn fused_mlp_b8_preserves_narrow_hidden_activations() {
    let (m, dims) = (19usize, [4usize, 10, 7, 3]);
    let mut rng = Lcg(0xe4e3);
    let weights: Vec<Vec<u8>> = dims
        .windows(2)
        .map(|shape| {
            (0..shape[0] * shape[1])
                .map(|_| {
                    let value = (rng.next() as f32 / u32::MAX as f32 - 0.5) * 0.7;
                    axnn_cpu::fp8::f32_to_e4m3_rne(value)
                })
                .collect()
        })
        .collect();
    let artifact = axnn_vaccel::artifact::mlp_b8(m, &dims, &weights).unwrap();
    let model = virtio_accel_tosa::parse(&artifact).expect("parse e4m3 mlp");
    model
        .analyze_for(AxTosaAccelerator::target())
        .expect("semantic analysis of e4m3 mlp");

    let input: Vec<u8> = (0..m * dims[0])
        .map(|index| axnn_cpu::fp8::f32_to_e4m3_rne(index as f32 / 64.0 - 0.5))
        .collect();
    let output = axtosa::execute(&artifact, &[&input]).expect("execute e4m3 mlp");
    let got = decode_output(MatMulDType::F16, &output[0]);

    let mut activation: Vec<f32> = input
        .iter()
        .map(|&value| axnn_cpu::fp8::e4m3_to_f32(value))
        .collect();
    for (layer, weight) in weights.iter().enumerate() {
        let decoded: Vec<f32> = weight
            .iter()
            .map(|&value| axnn_cpu::fp8::e4m3_to_f32(value))
            .collect();
        let mut next = scalar_matmul(&activation, &decoded, m, dims[layer], dims[layer + 1]);
        if layer + 1 == weights.len() {
            for value in &mut next {
                *value = 1.0 / (1.0 + (-*value).exp());
            }
            activation = next;
        } else {
            activation = next
                .into_iter()
                .map(|value| {
                    axnn_cpu::fp8::e4m3_to_f32(axnn_cpu::fp8::f32_to_e4m3_rne(value.tanh()))
                })
                .collect();
        }
    }
    for (index, (&actual, &expected)) in got.iter().zip(&activation).enumerate() {
        assert!(
            (actual - expected).abs() <= 6e-3,
            "e4m3 mlp elem {index}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn profiled_execution_uses_the_same_cached_program_path() {
    let (m, k, n) = (2, 3, 4);
    let artifact = axnn_vaccel::artifact::matmul(MatMulDType::F32, m, k, n).unwrap();
    let (a, _) = plane(MatMulDType::F32, m * k, 0x51);
    let (b, _) = plane(MatMulDType::F32, k * n, 0x52);
    let expected = axtosa::execute(&artifact, &[&a, &b]).unwrap();
    let mut session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    session.ensure(991, || artifact.clone()).unwrap();
    let execution = session
        .run_many_profiled(991, || artifact.clone(), &[&a, &b], &[m * n * 4])
        .unwrap();
    assert_eq!(execution.outputs, expected);
    assert_eq!(
        execution.timings.total(),
        execution.timings.preparation
            + execution.timings.upload
            + execution.timings.submit
            + execution.timings.readback
    );
}
