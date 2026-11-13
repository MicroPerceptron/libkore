use super::*;
use crate::fp8::Op;
use kore_svc_accel_spmd::{
    FORMAT_REGISTERED, IDENTITY_MAGIC, REGISTERED_MANIFEST, REGISTERED_RESIDENT_BYTES,
};
use std::vec::Vec;
use virtio_accel_device::{SegmentedSink, SegmentedSource};
use virtio_accel_proto::{
    AllocateBufferRequest, CreateContextRequest, CreateQueueRequest, KnownEventState, KnownOpcode,
    Le64, LoadProgramRequest, ObjectPayload, RequestFlags, RequestHeader, ResponseHeader,
    StatusCode, SubmitRequest, SubmitResponse, TransferBufferRequest, WireBinding, WireDeviceInfo,
    WireEventState, read_exact,
};
use zerocopy::IntoBytes;

const HEADER: usize = core::mem::size_of::<ResponseHeader>();
/// The registered-kernel manifest's length. Spelled through `size_of_val`:
/// in this module an array's `len` is also the byte ports'.
const MANIFEST_BYTES: u64 = core::mem::size_of_val(&REGISTERED_MANIFEST) as u64;

/// The driver's side of one command: a request frame in one readable region
/// and room for the response in one writable region, as a two-descriptor
/// chain carries them.
struct Driver {
    device: AccelDevice,
    next_request: u64,
}

impl Driver {
    fn new() -> Self {
        Self {
            device: AccelDevice::new().expect("device"),
            next_request: 1,
        }
    }

    /// Send `opcode` with `payload`; answer the status and response payload.
    fn call(&mut self, opcode: KnownOpcode, payload: &[u8], room: usize) -> (StatusCode, Vec<u8>) {
        let id = self.next_request;
        self.next_request += 1;
        let header = RequestHeader::new(opcode, RequestFlags::empty(), payload.len() as u32, id);
        let mut request = header.as_bytes().to_vec();
        request.extend_from_slice(payload);
        let mut response = std::vec![0u8; HEADER + room];
        let regions = [
            ChainRegion::readable(request.len() as u64),
            ChainRegion::writable(response.len() as u64),
        ];
        let outcome = {
            let segments = [request.as_slice()];
            let source = SegmentedSource::new(&segments).unwrap();
            let mut parts = [response.as_mut_slice()];
            let mut sink = SegmentedSink::new(&mut parts).unwrap();
            self.device
                .command(&regions, &source, &mut sink)
                .expect("the chain's memory holds")
        };
        let CommandOutcome::Response {
            request_id,
            status,
            used,
        } = outcome
        else {
            panic!("the chain could not carry a response: {outcome:?}");
        };
        assert_eq!(request_id, id, "a response answers its request");
        let header: ResponseHeader = read_exact(&response[..HEADER]).unwrap();
        assert_eq!(header.request_id.get(), id);
        assert_eq!(header.status.get(), status.0);
        let payload_bytes = header.payload_bytes.get() as usize;
        assert_eq!(used as usize, HEADER + payload_bytes);
        (status, response[HEADER..HEADER + payload_bytes].to_vec())
    }

    fn ok(&mut self, opcode: KnownOpcode, payload: &[u8], room: usize) -> Vec<u8> {
        let (status, payload) = self.call(opcode, payload, room);
        assert_eq!(status, StatusCode::OK, "{opcode:?}");
        payload
    }

    fn object(&mut self, opcode: KnownOpcode, payload: &[u8]) -> u64 {
        let reply = self.ok(opcode, payload, 8);
        read_exact::<ObjectPayload>(&reply).unwrap().object_id.get()
    }

    fn buffer(&mut self, context: u64, bytes: u64, usage: u32) -> u64 {
        self.object(
            KnownOpcode::AllocateBuffer,
            AllocateBufferRequest {
                context_id: Le64::new(context),
                bytes: Le64::new(bytes),
                alignment: Le64::new(1),
                memory_domain: MEMORY_DOMAIN_HOST,
                reserved0: [0; 7],
                usage: Le32::new(usage),
                reserved1: Le32::new(0),
            }
            .as_bytes(),
        )
    }

    fn program(&mut self, context: u64, kernel: u32) -> u64 {
        let mut target = [Le32::new(0); 12];
        target[0] = Le32::new(IDENTITY_MAGIC);
        target[1] = Le32::new(kernel);
        let mut payload = LoadProgramRequest {
            context_id: Le64::new(context),
            format: Le32::new(FORMAT_REGISTERED),
            flags: Le32::new(0),
            target,
            payload_bytes: Le64::new(MANIFEST_BYTES),
            resident_bytes: Le64::new(REGISTERED_RESIDENT_BYTES),
        }
        .as_bytes()
        .to_vec();
        payload.extend_from_slice(&REGISTERED_MANIFEST);
        self.object(KnownOpcode::LoadProgram, &payload)
    }
}

/// Provider memory a driver reaches by transfer: `MemoryDomain::Host`.
const MEMORY_DOMAIN_HOST: u8 = 1;
const TRANSFER_SOURCE: u32 = 1 << 0;
const TRANSFER_DESTINATION: u32 = 1 << 1;
const PROGRAM_INPUT: u32 = 1 << 2;
const PROGRAM_OUTPUT: u32 = 1 << 3;
const ACCESS_READ: u8 = 1;
const ACCESS_WRITE: u8 = 2;

fn transfer(buffer: u64, bytes: u64) -> TransferBufferRequest {
    TransferBufferRequest {
        buffer_id: Le64::new(buffer),
        offset: Le64::new(0),
        bytes: Le64::new(bytes),
    }
}

fn binding(buffer: u64, slot: u32, access: u8) -> WireBinding {
    WireBinding {
        buffer_id: Le64::new(buffer),
        offset: Le64::new(0),
        bytes: Le64::new(256),
        slot: Le32::new(slot),
        access,
        reserved: [0; 3],
    }
}

/// The configuration is one the protocol accepts, and is what the engine
/// was built with.
#[test]
fn the_configuration_is_the_baseline_protocol() {
    assert_eq!(AccelDevice::CONFIG.validate(), Ok(()));
    assert_eq!(AccelDevice::CONFIG.protocol_major.get(), 1);
    assert_eq!(AccelDevice::CONFIG.command_queue_count.get(), 1);
}

/// A whole driver session in the protocol's own frames: discover the device,
/// make a context, two buffers, a program and a queue, write every E4M3
/// value, submit, poll, and read the results back. Then again for each
/// function, on the same buffers.
#[test]
fn a_driver_session_computes_each_function_through_protocol_frames() {
    let mut driver = Driver::new();

    let info: WireDeviceInfo = read_exact(&driver.ok(
        KnownOpcode::GetDeviceInfo,
        &[],
        core::mem::size_of::<WireDeviceInfo>(),
    ))
    .unwrap();
    assert_eq!(&info.uuid[..13], b"kore-spmd-cpu");
    assert_eq!(info.max_contexts.get(), 1);
    assert!(info.max_bindings_per_submission.get() >= 2);

    let context = driver.object(
        KnownOpcode::CreateContext,
        CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    let operands = driver.buffer(context, 256, PROGRAM_INPUT | TRANSFER_DESTINATION);
    let results = driver.buffer(context, 256, PROGRAM_OUTPUT | TRANSFER_SOURCE);
    let queue = driver.object(
        KnownOpcode::CreateQueue,
        CreateQueueRequest {
            context_id: Le64::new(context),
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );

    let every_byte: [u8; 256] = core::array::from_fn(|i| i as u8);
    let mut write = transfer(operands, 256).as_bytes().to_vec();
    write.extend_from_slice(&every_byte);
    driver.ok(KnownOpcode::WriteBuffer, &write, 0);

    for op in Op::ALL {
        let program = driver.program(context, op.kernel_id());
        let mut submit = SubmitRequest {
            queue_id: Le64::new(queue),
            program_id: Le64::new(program),
            binding_count: Le32::new(2),
            flags: Le32::new(0),
            // Zero: no deadline.
            timeout_ns: Le64::new(0),
        }
        .as_bytes()
        .to_vec();
        submit.extend_from_slice(binding(operands, 0, ACCESS_READ).as_bytes());
        submit.extend_from_slice(binding(results, 1, ACCESS_WRITE).as_bytes());
        let event = read_exact::<SubmitResponse>(&driver.ok(KnownOpcode::Submit, &submit, 8))
            .unwrap()
            .event_id
            .get();

        // Admitted and answered, not yet run: the work is the device's turn.
        let poll = |driver: &mut Driver| {
            let reply = driver.ok(
                KnownOpcode::PollEvent,
                ObjectPayload {
                    object_id: Le64::new(event),
                }
                .as_bytes(),
                8,
            );
            read_exact::<WireEventState>(&reply)
                .unwrap()
                .known_state()
                .unwrap()
        };
        assert_eq!(poll(&mut driver), KnownEventState::Pending);
        assert!(driver.device.drive(), "the admitted submission runs");
        assert!(!driver.device.drive(), "and nothing is left to run");
        assert_eq!(poll(&mut driver), KnownEventState::Complete);

        // A buffer a submission bound stays the event's until the driver
        // destroys the event, terminal or not: the protocol's rule, and the
        // order a driver must follow.
        let read = transfer(results, 256);
        assert_eq!(
            driver.call(KnownOpcode::ReadBuffer, read.as_bytes(), 256).0,
            StatusCode::BUSY
        );
        let object = |id: u64| ObjectPayload {
            object_id: Le64::new(id),
        };
        driver.ok(KnownOpcode::DestroyEvent, object(event).as_bytes(), 0);
        let read = driver.ok(KnownOpcode::ReadBuffer, read.as_bytes(), 256);
        assert_eq!(read.as_slice(), op.table(), "{}", op.name());
        driver.ok(KnownOpcode::UnloadProgram, object(program).as_bytes(), 0);
    }
    assert_eq!(driver.device.health(), DeviceHealth::Running);
}

/// What a driver gets wrong is answered in a response, and the device goes
/// on serving.
#[test]
fn a_drivers_mistakes_are_answered_and_the_device_keeps_serving() {
    let mut driver = Driver::new();
    let context = driver.object(
        KnownOpcode::CreateContext,
        CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );

    // A program this device does not have: another registry's kernel.
    let mut target = [Le32::new(0); 12];
    target[0] = Le32::new(IDENTITY_MAGIC);
    target[1] = Le32::new(1);
    let mut load = LoadProgramRequest {
        context_id: Le64::new(context),
        format: Le32::new(FORMAT_REGISTERED),
        flags: Le32::new(0),
        target,
        payload_bytes: Le64::new(MANIFEST_BYTES),
        resident_bytes: Le64::new(REGISTERED_RESIDENT_BYTES),
    }
    .as_bytes()
    .to_vec();
    load.extend_from_slice(&REGISTERED_MANIFEST);
    assert_eq!(
        driver.call(KnownOpcode::LoadProgram, &load, 8).0,
        StatusCode::INCOMPATIBLE
    );

    // An id that names nothing.
    let stranger = ObjectPayload {
        object_id: Le64::new(0xdead_beef),
    };
    let (status, _) = driver.call(KnownOpcode::PollEvent, stranger.as_bytes(), 8);
    assert_ne!(status, StatusCode::OK);

    // A second context, where the device has one.
    assert_eq!(
        driver
            .call(
                KnownOpcode::CreateContext,
                CreateContextRequest {
                    flags: Le32::new(0),
                    reserved: Le32::new(0),
                }
                .as_bytes(),
                8
            )
            .0,
        StatusCode::RESOURCE_LIMIT
    );

    // A frame past the configured limit is refused before it is read.
    let oversized = std::vec![0u8; MAX_FRAME_BYTES as usize];
    let (status, _) = driver.call(KnownOpcode::WriteBuffer, &oversized, 0);
    assert_ne!(status, StatusCode::OK);

    assert_eq!(driver.device.health(), DeviceHealth::Running);
    driver.ok(KnownOpcode::GetDeviceInfo, &[], 128);
}

/// A reset releases what the driver made and renames every object: an id
/// from before it names nothing after.
#[test]
fn a_reset_releases_everything_and_stales_every_id() {
    let mut driver = Driver::new();
    let create = CreateContextRequest {
        flags: Le32::new(0),
        reserved: Le32::new(0),
    };
    let context = driver.object(KnownOpcode::CreateContext, create.as_bytes());
    let buffer = driver.buffer(context, 4096, TRANSFER_SOURCE | TRANSFER_DESTINATION);

    VirtualDevice::reset(&mut driver.device).expect("reset");
    assert_eq!(driver.device.health(), DeviceHealth::Running);

    let (status, _) = driver.call(KnownOpcode::ReadBuffer, transfer(buffer, 16).as_bytes(), 16);
    assert_eq!(status, StatusCode::STALE_OBJECT);
    // The one context the device has is free again.
    let again = driver.object(KnownOpcode::CreateContext, create.as_bytes());
    assert_ne!(again, context);

    // And again: each reset is a new namespace, so the id from between the
    // two is as stale as the one from before the first.
    VirtualDevice::reset(&mut driver.device).expect("a second reset");
    assert_eq!(driver.device.health(), DeviceHealth::Running);
    let object = ObjectPayload {
        object_id: Le64::new(again),
    };
    let (status, _) = driver.call(KnownOpcode::DestroyContext, object.as_bytes(), 0);
    assert_eq!(status, StatusCode::STALE_OBJECT);
    let third = driver.object(KnownOpcode::CreateContext, create.as_bytes());
    assert!(third != again && third != context);
}

/// A namespace is never reused, and there are sixteen bits of them. A
/// device reset past the last is not brought back: its transport then
/// reports it dead, and the driver is given a new device.
#[test]
fn a_device_out_of_namespaces_is_not_brought_back() {
    let mut device = AccelDevice::new().expect("device");
    // The first life is namespace one; each reset takes the next.
    for reset in 1..u32::from(u16::MAX) {
        assert_eq!(VirtualDevice::reset(&mut device), Ok(()), "reset {reset}");
    }
    assert_eq!(device.health(), DeviceHealth::Running);
    assert_eq!(VirtualDevice::reset(&mut device), Err(Unrecoverable));
    assert_eq!(VirtualDevice::reset(&mut device), Err(Unrecoverable));
}
