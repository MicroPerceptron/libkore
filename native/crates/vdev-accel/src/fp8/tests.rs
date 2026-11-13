use super::*;
use kore_svc_accel_spmd::{
    FORMAT_REGISTERED, IDENTITY_MAGIC, REGISTERED_MANIFEST, REGISTERED_RESIDENT_BYTES, SpmdBackend,
    SpmdBuffer, SpmdContext,
};
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactFormat, ArtifactRef, BackendError, BindingRef, BufferDesc,
    BufferRange, BufferUsage, ContextDesc, ContextFlags, EventState, MemoryDomain, QueueDesc,
    QueueFlags, TargetIdentity, Timeout,
};

const NAN: u8 = 0x7f;
const LARGEST: u8 = 0x7e;

/// The value of one E4M3 byte; `None` for the NaN.
fn decode(code: u8) -> Option<f64> {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((code >> 3) & 0xf);
    let mantissa = f64::from(code & 0x7);
    if exponent == 0xf && code & 0x7 == 0x7 {
        return None;
    }
    Some(if exponent == 0 {
        sign * mantissa * 2f64.powi(-9)
    } else {
        sign * (8.0 + mantissa) * 2f64.powi(exponent - 10)
    })
}

/// Round a real to E4M3 by the device's rules: nearest, ties to even,
/// saturating, the sign kept on a zero.
fn encode(value: f64) -> u8 {
    if value.is_nan() {
        return NAN;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let magnitude = value.abs();
    if magnitude >= 448.0 {
        return sign | LARGEST;
    }
    let mut best = 0u8;
    for code in 1..=LARGEST {
        let (candidate, held) = (decode(code).unwrap(), decode(best).unwrap());
        let (near, far) = ((candidate - magnitude).abs(), (held - magnitude).abs());
        if near < far || (near == far && code % 2 == 0) {
            best = code;
        }
    }
    sign | best
}

/// What this machine's own math library says the function is, in binary64.
/// The tables were not made with it; agreeing with it is the check.
fn reference(op: Op, x: f64) -> f64 {
    match op {
        Op::Sin => x.sin(),
        Op::Cos => x.cos(),
        Op::Tan => x.tan(),
        Op::Exp => x.exp(),
        Op::Log => x.ln(),
        Op::Sqrt => x.sqrt(),
    }
}

/// The generator's tables against an independent computation of all 1536
/// entries. Its closest rounding decision is 2.5e-5 of a boundary away
/// (`tools/fp8_tables.py --margin`), eleven orders of magnitude past
/// binary64's error, so agreement here is conclusive and not luck.
#[test]
fn every_entry_is_the_correctly_rounded_result() {
    for op in Op::ALL {
        for code in 0..=u8::MAX {
            let want = match decode(code) {
                None => NAN,
                // The limit of log at either zero is negative infinity,
                // which saturates: `ln(-0.0)` is that in binary64 too.
                Some(x) => encode(reference(op, x)),
            };
            assert_eq!(
                op.apply(code),
                want,
                "{}({code:#04x} = {:?})",
                op.name(),
                decode(code)
            );
        }
    }
}

/// The rules a driver writer reads, each on a value they can check by hand.
#[test]
fn the_stated_rules_hold_on_values_one_can_check_by_hand() {
    let code = |value: f64| encode(value);
    let one = code(1.0);
    assert_eq!(one, 0x38);
    assert_eq!(decode(Op::Sin.apply(one)), Some(0.8125), "sin 1 = 0.8414…");
    assert_eq!(decode(Op::Cos.apply(one)), Some(0.5625), "cos 1 = 0.5403…");
    assert_eq!(decode(Op::Tan.apply(one)), Some(1.5), "tan 1 = 1.5574…");
    assert_eq!(decode(Op::Exp.apply(one)), Some(2.75), "e = 2.7182…");
    assert_eq!(Op::Log.apply(one), 0x00, "log 1 is exactly zero");
    assert_eq!(decode(Op::Sqrt.apply(code(4.0))), Some(2.0));
    assert_eq!(Op::Cos.apply(0x00), one);
    assert_eq!(Op::Exp.apply(0x00), one);

    // No infinity: a result past the largest magnitude is that magnitude.
    assert_eq!(decode(LARGEST), Some(448.0));
    assert_eq!(Op::Exp.apply(LARGEST), LARGEST);
    assert_eq!(Op::Exp.apply(code(8.0)), LARGEST, "e^8 = 2980.9…");
    assert_eq!(Op::Log.apply(0x00), 0x80 | LARGEST, "log 0 is -448");
    assert_eq!(Op::Log.apply(0x80), 0x80 | LARGEST, "and so is log -0");
    assert_eq!(Op::Exp.apply(0x80 | LARGEST), 0x00, "e^-448 rounds to zero");

    // Outside the reals, and a NaN operand, give the one canonical NaN.
    for op in Op::ALL {
        assert_eq!(op.apply(0x7f), NAN, "{}", op.name());
        assert_eq!(op.apply(0xff), NAN, "{}", op.name());
    }
    assert_eq!(Op::Log.apply(code(-1.0)), NAN);
    assert_eq!(Op::Sqrt.apply(code(-1.0)), NAN);

    // A zero keeps its sign where IEEE 754 keeps it.
    for op in [Op::Sin, Op::Tan, Op::Sqrt] {
        assert_eq!(op.apply(0x00), 0x00, "{}", op.name());
        assert_eq!(op.apply(0x80), 0x80, "{}", op.name());
    }
    // sin and tan are odd, cos is even, over every value.
    for code in 0..=LARGEST {
        assert_eq!(Op::Sin.apply(code | 0x80), Op::Sin.apply(code) ^ 0x80);
        assert_eq!(Op::Tan.apply(code | 0x80), Op::Tan.apply(code) ^ 0x80);
        assert_eq!(Op::Cos.apply(code | 0x80), Op::Cos.apply(code));
    }
}

#[test]
fn the_registry_names_each_function_once() {
    assert_eq!(KERNELS.len(), Op::ALL.len());
    for (kernel, op) in KERNELS.iter().zip(Op::ALL) {
        assert_eq!(kernel.id, op.kernel_id());
        assert_eq!(kernel.name, op.name());
        assert_eq!((kernel.inputs, kernel.elem_bytes), (1, 1));
        assert_eq!(kernel.identity().0[..2], [IDENTITY_MAGIC, op.kernel_id()]);
    }
    for (index, kernel) in KERNELS.iter().enumerate() {
        assert!(
            !KERNELS[..index].iter().any(|prior| prior.id == kernel.id),
            "duplicate kernel id {:#x}",
            kernel.id
        );
    }
}

/// A span is its own share of the grid: two disjoint spans equal one run.
#[test]
fn a_tiled_run_equals_the_whole_run() {
    let operands: [u8; 256] = core::array::from_fn(|i| i as u8);
    let run = KERNELS[Op::Exp as usize].run;
    let launch = kore_driver::accel::LaunchShape::ArtifactDefined;
    let mut whole = [0u8; 256];
    run(TileCtx {
        tiles: 0..256,
        total: 256,
        launch,
        inputs: &[&operands],
        out: &mut whole,
    });
    assert_eq!(whole, *Op::Exp.table());

    let mut split = [0u8; 256];
    let (low, high) = split.split_at_mut(100);
    run(TileCtx {
        tiles: 0..100,
        total: 256,
        launch,
        inputs: &[&operands],
        out: low,
    });
    run(TileCtx {
        tiles: 100..256,
        total: 256,
        launch,
        inputs: &[&operands],
        out: high,
    });
    assert_eq!(split, whole);
}

const BYTES: u64 = 256;

fn buffer(backend: &SpmdBackend, context: &SpmdContext, usage: BufferUsage) -> SpmdBuffer {
    let usage = usage | BufferUsage::TRANSFER_SOURCE | BufferUsage::TRANSFER_DESTINATION;
    let desc = BufferDesc::new(BYTES, 1, MemoryDomain::Host, usage).expect("descriptor");
    backend
        .allocate_buffer(context, desc)
        .expect("allocated")
        .into_parts()
        .0
}

fn artifact(id: u32) -> ArtifactRef<'static> {
    let mut words = [0u32; 12];
    words[0] = IDENTITY_MAGIC;
    words[1] = id;
    ArtifactRef {
        format: ArtifactFormat::new(FORMAT_REGISTERED).expect("format"),
        target: TargetIdentity(words),
        payload: &REGISTERED_MANIFEST,
        resident_bytes: REGISTERED_RESIDENT_BYTES,
    }
}

/// The device, as the protocol reaches it: every function over every byte,
/// through the CPU backend's context, buffers, program, queue and event.
#[test]
fn the_backend_runs_each_function_over_every_value() {
    let backend = SpmdBackend::new(KERNELS);
    let context = backend
        .create_context(ContextDesc {
            flags: ContextFlags::empty(),
        })
        .expect("context");
    let queue = backend
        .create_queue(
            &context,
            QueueDesc {
                flags: QueueFlags::empty(),
            },
        )
        .expect("queue");
    let mut operands = buffer(&backend, &context, BufferUsage::PROGRAM_INPUT);
    let results = buffer(&backend, &context, BufferUsage::PROGRAM_OUTPUT);
    let every_byte: [u8; 256] = core::array::from_fn(|i| i as u8);
    backend
        .write_buffer(&mut operands, 0, &every_byte)
        .expect("written");
    let whole = BufferRange::new(0, BYTES).expect("range");

    for op in Op::ALL {
        let program = backend
            .load_program(&context, artifact(op.kernel_id()))
            .expect("loaded");
        let event = backend
            .submit(
                &queue,
                &program,
                &[
                    BindingRef {
                        slot: 0,
                        buffer: &operands,
                        range: whole,
                        access: AccessMode::Read,
                    },
                    BindingRef {
                        slot: 1,
                        buffer: &results,
                        range: whole,
                        access: AccessMode::Write,
                    },
                ],
                Timeout::Infinite,
            )
            .map_err(|_| "refused")
            .expect("admitted");
        assert_eq!(backend.poll_event(&event), Ok(EventState::Pending));
        assert!(backend.run_pending(), "the admitted launch runs");
        assert_eq!(backend.poll_event(&event), Ok(EventState::Complete));
        let mut read = [0u8; 256];
        backend.read_buffer(&results, 0, &mut read).expect("read");
        assert_eq!(read, *op.table(), "{}", op.name());
        backend.destroy_event(event).map_err(|_| "event").unwrap();
        backend
            .unload_program(program)
            .map_err(|_| "program")
            .unwrap();
    }

    // A kernel of another registry is not one this device has.
    assert_eq!(
        backend
            .load_program(
                &context,
                artifact(kore_svc_accel_spmd::kernel::KERNEL_ADD_F32)
            )
            .err(),
        Some(BackendError::Incompatible)
    );
}
