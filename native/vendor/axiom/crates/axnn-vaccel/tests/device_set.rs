#![feature(allocator_api)]
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use axnn_vaccel::{AxCompileError, AxHetero, AxInvocationStatus, AxTosaAccelerator, Session};
use std::alloc::Global;
fn session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap()
}
#[test]
fn third_target_executes_and_releases_its_own_programs() {
    let mut runtime = AxHetero::new(session(), Some(session()));
    assert_eq!(runtime.add_accelerator(session()).ok(), Some(2));
    let mut b = GraphBuilder::new_in(Global);
    let x = b.tensor(DType::F32, TensorFlags::INPUT, &[4], &[]).unwrap();
    let y = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    b.node(OpCode::ABS, NodeFlags::default(), &[x], &[y], &[])
        .unwrap();
    let bytes = b.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let mut model = runtime.automatic_device_set();
    assert_eq!(model.accelerators.len(), 2);
    assert!(
        model
            .accelerators
            .iter()
            .all(|m| m.accelerator_tosa_capability.is_some())
    );
    model.cpu.cpu = false;
    model.accelerators[0].accelerator = false;
    let executable = runtime.compile_set_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.single_backend(), Some(2));
    let input: Vec<u8> = [-1f32, 2., -3., 4.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut invocation = runtime.begin_in(&executable, &[&input], Global).unwrap();
    let mut other = AxHetero::new(session(), Some(session()));
    assert!(matches!(
        other.advance_in(&mut invocation),
        Err(AxCompileError::WrongExecutor)
    ));
    assert!(
        matches!(runtime.advance_in(&mut invocation).unwrap(),AxInvocationStatus::Progress(p) if p.backend==2)
    );
    let output = loop {
        match runtime.advance_in(&mut invocation).unwrap() {
            AxInvocationStatus::Complete(output) => break output,
            AxInvocationStatus::Progress(p) => assert_eq!(p.backend, 2),
            _ => panic!("unexpected cancellation"),
        }
    };
    assert_eq!(
        output[0],
        [1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    drop(invocation);
    runtime.release_executable(executable).unwrap();
    // The same graph can be freshly admitted after retirement.
    let executable = runtime.compile_set_in(&graph, &model, Global).unwrap();
    runtime.run_in(&executable, &[&input], Global).unwrap();
    runtime.release_executable(executable).unwrap();
    model.routes.retain(|r| r.destination != 2);
    assert!(runtime.compile_set_in(&graph, &model, Global).is_err());
}

#[test]
fn bounded_stream_overlaps_targets_and_returns_rejected_input_ownership() {
    use axnn_vaccel::{AxStream, AxStreamProgress, BackendError};
    let mut runtime = AxHetero::new(session(), Some(session()));
    runtime.add_accelerator(session()).ok().unwrap();
    let mut b = GraphBuilder::new_in(Global);
    let x = b.tensor(DType::F32, TensorFlags::INPUT, &[4], &[]).unwrap();
    let y = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    b.node(OpCode::ABS, NodeFlags::default(), &[x], &[y], &[])
        .unwrap();
    let bytes = b.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let mut model = runtime.automatic_device_set();
    model.cpu.cpu = false;
    model.accelerators[1].accelerator = false;
    let first = runtime.compile_set_in(&graph, &model, Global).unwrap();
    model.accelerators[0].accelerator = false;
    model.accelerators[1].accelerator = true;
    let second = runtime.compile_set_in(&graph, &model, Global).unwrap();
    assert!(AxStream::new([&first, &first], 32, Global).is_err());
    let mut stream = AxStream::new([&first, &second], 32, Global).unwrap();
    let data = || vec![vec![0u8; 16]];
    stream
        .try_submit(&runtime, 0, data())
        .unwrap_or_else(|_| panic!("first admission"));
    let retry = data();
    let pointer = retry[0].as_ptr();
    let rejected = stream.try_submit(&runtime, 0, retry).err().unwrap();
    assert!(matches!(
        rejected.error,
        AxCompileError::Backend(BackendError::Busy)
    ));
    assert_eq!(rejected.inputs[0].as_ptr(), pointer);
    stream
        .try_submit(&runtime, 1, rejected.inputs)
        .unwrap_or_else(|_| panic!("second admission"));
    assert_eq!(stream.input_bytes(), 32);
    assert!(
        matches!(stream.poll(&mut runtime),AxStreamProgress::Progress {slot:0,progress} if progress.backend==1)
    );
    assert!(
        matches!(stream.poll(&mut runtime),AxStreamProgress::Progress {slot:1,progress} if progress.backend==2)
    );
    assert_eq!(stream.active_count(), 2);
    let mut completed = 0;
    for _ in 0..10 {
        match stream.poll(&mut runtime) {
            AxStreamProgress::Complete { outputs, .. } => {
                assert_eq!(outputs[0].as_bytes(), [0u8; 16]);
                completed += 1;
            }
            AxStreamProgress::Progress { .. } | AxStreamProgress::Idle => {}
            _ => panic!("unexpected terminal state"),
        }
    }
    assert_eq!(completed, 2);
    assert_eq!(stream.input_bytes(), 0);
    drop(stream);
    let mut stream = AxStream::new([&first, &second], 15, Global).unwrap();
    assert!(matches!(
        stream.try_submit(&runtime, 0, data()).err().unwrap().error,
        AxCompileError::Backend(BackendError::ResourceLimit)
    ));
    assert_eq!(stream.active_count(), 0);
    drop(stream);
    runtime.release_executable(first).unwrap();
    runtime.release_executable(second).unwrap();
}
