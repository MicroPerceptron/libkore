//! A driver's whole session with the virtual accelerator, end to end and
//! with nothing of the driver written here: the driver is the protocol's
//! reference guest client (`virtio-accel-guest`), over kore's register
//! block and shared-memory queue (`kore-vdev`'s driver half), against this
//! crate's device behind `kore-vdev`'s transport.
//!
//! What it shows is that the pieces are one device. The reference client
//! knows only the protocol's queue contract; the device model knows only
//! requests and responses; the registers and the ring between them are what
//! a driver in C will map.

use core::num::{NonZeroU32, NonZeroU64};
use core::ptr::NonNull;

use kore_svc_accel_spmd::{
    FORMAT_REGISTERED, IDENTITY_MAGIC, REGISTERED_MANIFEST, REGISTERED_RESIDENT_BYTES,
};
use kore_vdev::device::VirtualDevice;
use kore_vdev::driver::{self, DriverChain, DriverRing, Layout, Slots};
use kore_vdev::mem::Window;
use kore_vdev::regs;
use kore_vdev::transport::Transport;
use kore_vdev_accel::device::{AccelDevice, DEVICE_ID};
use kore_vdev_accel::fp8::Op;
use virtio_accel_guest::{
    AccessMode, Binding, Buffer, BufferDesc, BufferRange, BufferUsage, Completion, Context,
    EventState, ExecutionQueue, GuestClient, GuestConfig, MemoryDomain, Operation, ProgramDesc,
    RequestPoll, StartResult, SubmissionOutcome,
};
use virtio_accel_proto::{
    AllocateBufferRequest, CreateContextRequest, CreateQueueRequest, LoadProgramRequest,
    ObjectPayload, RequestHeader, ResponseHeader, StatusCode, SubmitRequest, SubmitResponse,
    TransferBufferRequest, WireBinding, WireConfig, WireDeviceInfo, WireEventState, read_exact,
};
use virtio_accel_transport::{DriverChainBuffer, QueueSize};

type Ring = DriverRing<Window>;
type Chain = DriverChain<Window>;
type Client = GuestClient<Ring>;

const BUDGET: NonZeroU32 = NonZeroU32::new(8).unwrap();
const REQUEST: u32 = core::mem::size_of::<RequestHeader>() as u32;
const RESPONSE: u32 = core::mem::size_of::<ResponseHeader>() as u32;
/// Every E4M3 value, once.
const VALUES: u32 = 256;

const fn bytes_of<T>() -> u32 {
    core::mem::size_of::<T>() as u32
}

/// Host memory standing for a lease both sides map.
fn window(bytes: u64) -> Window {
    let words: &'static mut [u64] =
        Box::leak(vec![0u64; (bytes as usize).div_ceil(8)].into_boxed_slice());
    // SAFETY: a leaked, eight-byte aligned allocation of at least `bytes`
    // bytes that nothing else holds a reference into.
    unsafe {
        Window::new(
            NonNull::new(words.as_mut_ptr().cast()).unwrap(),
            bytes as usize,
        )
    }
}

/// A driver process and the device's host, in one test.
struct Session {
    registers: Window,
    memory: Window,
    layout: Layout,
    transport: Transport<AccelDevice, Window>,
    /// How often the device raised the driver's interrupt.
    interrupts: u32,
}

/// What a bring-up gives the driver.
struct Driver {
    client: Client,
    slots: Slots<Window>,
}

impl Session {
    fn new() -> Self {
        // Sixteen descriptors, the device's longest chain, so eight commands
        // in flight; a slot holds a request or a response of a page.
        let layout = Layout::new(QueueSize::new(16).unwrap(), 4096, 4096).unwrap();
        let registers = window(regs::BYTES);
        let memory = window(layout.bytes());
        let device = AccelDevice::new().expect("the device");
        let mut transport = Transport::new(registers, device).expect("a transport");
        // The driver lends its memory before it brings the device up.
        transport.lend(memory).expect("a device in its first state");
        Self {
            registers,
            memory,
            layout,
            transport,
            interrupts: 0,
        }
    }

    /// The doorbell: the host services the transport.
    fn ring(&mut self) {
        loop {
            let serviced = self.transport.service(BUDGET);
            self.interrupts += u32::from(serviced.interrupt);
            if !serviced.pending {
                return;
            }
        }
    }

    /// What every driver does first: find the device, bring it up and read
    /// its configuration.
    fn bring_up(&mut self) -> Driver {
        let identity = driver::identify(&self.registers).expect("a kore virtual device");
        assert_eq!(identity.device_id, DEVICE_ID);
        assert_eq!(identity.vendor_id, regs::VENDOR_KORE);
        let mut config = [0u8; core::mem::size_of::<WireConfig>()];
        assert_eq!(identity.config_bytes as usize, config.len());
        driver::read_config(&self.registers, &mut config).unwrap();
        let wire: WireConfig = read_exact(&config).unwrap();
        assert_eq!(wire, AccelDevice::CONFIG);

        let (registers, memory, layout) = (self.registers, self.memory, self.layout);
        let (ring, slots) = driver::bring_up(&registers, memory, layout, || self.ring())
            .expect("the device comes up");
        // The protocol has no feature of its own: what is left of the offer
        // once the transport's bit is taken away must be nothing.
        let config = GuestConfig::new(wire, identity.features & !regs::F_VERSION_1, layout.slots())
            .expect("a configuration the reference driver accepts");
        Driver {
            client: GuestClient::new(ring, config).expect("the reference driver"),
            slots,
        }
    }

    /// One command: take a slot for a request of `request` payload bytes
    /// and a response of `response`, start the operation, ring, and take
    /// the answer.
    fn call<O: Operation>(
        &mut self,
        driver: &mut Driver,
        request: u32,
        response: u32,
        start: impl FnOnce(&mut Client, Chain) -> StartResult<Ring, O>,
    ) -> Completion<O, Chain, <Chain as DriverChainBuffer>::Error> {
        let chain = driver
            .slots
            .chain(REQUEST + request, RESPONSE + response)
            .expect("a free slot");
        let pending = match start(&mut driver.client, chain) {
            Ok(pending) => pending,
            Err(error) => panic!("the request was not started: {:?}", error.into_parts().2),
        };
        let before = self.interrupts;
        self.ring();
        assert_eq!(self.interrupts, before + 1, "one command, one interrupt");
        match driver.client.poll(pending) {
            RequestPoll::Ready(completion) => completion,
            _ => panic!("a serviced command has its answer"),
        }
    }

    /// A command that must succeed: its output, with the slot returned.
    fn ok<O: Operation>(
        &mut self,
        driver: &mut Driver,
        request: u32,
        response: u32,
        start: impl FnOnce(&mut Client, Chain) -> StartResult<Ring, O>,
    ) -> O::Output {
        let (output, chain) = self.ok_with_chain(driver, request, response, start);
        driver.slots.recycle(chain);
        output
    }

    fn ok_with_chain<O: Operation>(
        &mut self,
        driver: &mut Driver,
        request: u32,
        response: u32,
        start: impl FnOnce(&mut Client, Chain) -> StartResult<Ring, O>,
    ) -> (O::Output, Chain) {
        match self.call(driver, request, response, start) {
            Completion::Success { output, chain } => (output, chain),
            Completion::DeviceError { status, .. } => panic!("the device answered {status:?}"),
            Completion::InvalidResponse { error, .. } => panic!("a malformed answer: {error:?}"),
        }
    }

    /// A command the device must refuse: the status it refused with.
    fn refused<O: Operation>(
        &mut self,
        driver: &mut Driver,
        request: u32,
        response: u32,
        start: impl FnOnce(&mut Client, Chain) -> StartResult<Ring, O>,
    ) -> StatusCode {
        match self.call(driver, request, response, start) {
            Completion::DeviceError { status, chain, .. } => {
                driver.slots.recycle(chain);
                status
            }
            _ => panic!("the device was expected to refuse"),
        }
    }
}

/// The objects a session computes with.
struct Objects {
    context: Context,
    operands: Buffer,
    results: Buffer,
    queue: ExecutionQueue,
}

fn objects(session: &mut Session, driver: &mut Driver) -> Objects {
    let info = session.ok(driver, 0, bytes_of::<WireDeviceInfo>(), |client, chain| {
        client.get_device_info(chain)
    });
    assert_eq!(&info.uuid[..13], b"kore-spmd-cpu");

    let context = session.ok(
        driver,
        bytes_of::<CreateContextRequest>(),
        bytes_of::<ObjectPayload>(),
        |client, chain| client.create_context(chain),
    );
    let mut buffer = |usage| {
        let desc = BufferDesc::new(u64::from(VALUES), 1, MemoryDomain::Host, usage).unwrap();
        session.ok(
            driver,
            bytes_of::<AllocateBufferRequest>(),
            bytes_of::<ObjectPayload>(),
            |client, chain| client.allocate_buffer(chain, &context, desc),
        )
    };
    let operands = buffer(BufferUsage::PROGRAM_INPUT | BufferUsage::TRANSFER_DESTINATION);
    let results = buffer(BufferUsage::PROGRAM_OUTPUT | BufferUsage::TRANSFER_SOURCE);
    let queue = session.ok(
        driver,
        bytes_of::<CreateQueueRequest>(),
        bytes_of::<ObjectPayload>(),
        |client, chain| client.create_execution_queue(chain, &context),
    );
    Objects {
        context,
        operands,
        results,
        queue,
    }
}

/// Run `op` over the operands buffer and answer what the results buffer
/// then holds.
fn compute(session: &mut Session, driver: &mut Driver, objects: &Objects, op: Op) -> Vec<u8> {
    let mut target = [0u32; 12];
    target[0] = IDENTITY_MAGIC;
    target[1] = op.kernel_id();
    let desc = ProgramDesc::new(
        NonZeroU32::new(FORMAT_REGISTERED).unwrap(),
        target,
        NonZeroU64::new(REGISTERED_RESIDENT_BYTES).unwrap(),
    );
    let program = session.ok(
        driver,
        bytes_of::<LoadProgramRequest>() + REGISTERED_MANIFEST.len() as u32,
        bytes_of::<ObjectPayload>(),
        |client, chain| {
            client.load_program(
                chain,
                &objects.context,
                desc,
                REGISTERED_MANIFEST.as_slice(),
            )
        },
    );

    let whole = BufferRange::new(0, u64::from(VALUES)).unwrap();
    let bindings = [
        Binding {
            buffer: &objects.operands,
            range: whole,
            slot: 0,
            access: AccessMode::Read,
        },
        Binding {
            buffer: &objects.results,
            range: whole,
            slot: 1,
            access: AccessMode::Write,
        },
    ];
    let outcome = session.ok(
        driver,
        bytes_of::<SubmitRequest>() + 2 * bytes_of::<WireBinding>(),
        bytes_of::<SubmitResponse>(),
        // Zero: no deadline.
        |client, chain| client.submit(chain, &objects.queue, &program, &bindings, 0),
    );
    let SubmissionOutcome::Accepted(event) = outcome else {
        panic!("the submission was not accepted: {outcome:?}");
    };

    // The submission was answered and then run in the same service: the
    // first poll finds it complete.
    let state = session.ok(
        driver,
        bytes_of::<ObjectPayload>(),
        bytes_of::<WireEventState>(),
        |client, chain| client.poll_event(chain, &event),
    );
    assert_eq!(state, EventState::Complete, "{}", op.name());

    // The results are the event's until the event is destroyed.
    let read = |client: &mut Client, chain| client.read_buffer(chain, &objects.results, whole);
    assert_eq!(
        session.refused(driver, bytes_of::<TransferBufferRequest>(), VALUES, read),
        StatusCode::BUSY
    );
    session.ok(driver, bytes_of::<ObjectPayload>(), 0, |client, chain| {
        client.destroy_event(chain, event)
    });
    let (output, chain) =
        session.ok_with_chain(driver, bytes_of::<TransferBufferRequest>(), VALUES, read);
    assert_eq!(output.bytes, u64::from(VALUES));
    let mut results = vec![0u8; VALUES as usize];
    chain
        .read_device_writable(u64::from(RESPONSE), &mut results)
        .unwrap();
    driver.slots.recycle(chain);

    session.ok(driver, bytes_of::<ObjectPayload>(), 0, |client, chain| {
        client.unload_program(chain, program)
    });
    results
}

/// The session a driver is written to hold: bring the device up, make its
/// objects, write every E4M3 value once, and for each of the six functions
/// submit, poll, and read back exactly the function's table.
#[test]
fn the_reference_driver_computes_every_function() {
    let mut session = Session::new();
    let mut driver = session.bring_up();
    let objects = objects(&mut session, &mut driver);

    let every_value: [u8; VALUES as usize] = core::array::from_fn(|value| value as u8);
    session.ok(
        &mut driver,
        bytes_of::<TransferBufferRequest>() + VALUES,
        0,
        |client, chain| client.write_buffer(chain, &objects.operands, 0, every_value.as_slice()),
    );

    for op in Op::ALL {
        let results = compute(&mut session, &mut driver, &objects, op);
        assert_eq!(results.as_slice(), op.table(), "{}", op.name());
        for (value, result) in results.iter().enumerate() {
            assert_eq!(*result, op.apply(value as u8));
        }
    }
    assert_eq!(driver.slots.free(), u32::from(session.layout.slots()));
    assert!(!session.transport.device().needs_reset());
    assert!(!session.transport.is_dead());
}

/// A second bring-up is a reset. The device forgets every object, an id
/// kept from before names nothing, and the device computes again from its
/// first state.
#[test]
fn a_reset_gives_the_driver_a_device_in_its_first_state() {
    let mut session = Session::new();
    let mut driver = session.bring_up();
    let before = objects(&mut session, &mut driver);

    let mut driver = session.bring_up();
    // Discovery again: the client is new, as the device is.
    session.ok(
        &mut driver,
        0,
        bytes_of::<WireDeviceInfo>(),
        |client, chain| client.get_device_info(chain),
    );
    let stale = session.refused(
        &mut driver,
        bytes_of::<TransferBufferRequest>(),
        VALUES,
        |client, chain| {
            client.read_buffer(chain, &before.results, BufferRange::new(0, 16).unwrap())
        },
    );
    assert_eq!(stale, StatusCode::STALE_OBJECT);

    // The one context the device has is free again, and everything works.
    let objects = objects(&mut session, &mut driver);
    assert_ne!(objects.context.raw(), before.context.raw());
    let sevens = [0x77u8; VALUES as usize];
    session.ok(
        &mut driver,
        bytes_of::<TransferBufferRequest>() + VALUES,
        0,
        |client, chain| client.write_buffer(chain, &objects.operands, 0, sevens.as_slice()),
    );
    let results = compute(&mut session, &mut driver, &objects, Op::Sqrt);
    assert_eq!(results, [Op::Sqrt.apply(0x77); VALUES as usize]);
}

/// The reference driver with every slot in flight at once: the commands are
/// served in the order they were published, and each answer reaches the
/// request it belongs to.
#[test]
fn every_slot_in_flight_is_answered() {
    let mut session = Session::new();
    let mut driver = session.bring_up();
    let slots = session.layout.slots();

    let pending: Vec<_> = (0..slots)
        .map(|_| {
            let chain = driver
                .slots
                .chain(REQUEST, RESPONSE + bytes_of::<WireDeviceInfo>())
                .unwrap();
            driver.client.get_device_info(chain).ok().expect("started")
        })
        .collect();
    assert_eq!(driver.slots.free(), 0);
    session.ring();
    for pending in pending {
        let RequestPoll::Ready(Completion::Success { output, chain }) = driver.client.poll(pending)
        else {
            panic!("every command was answered");
        };
        assert_eq!(&output.uuid[..13], b"kore-spmd-cpu");
        driver.slots.recycle(chain);
    }
    assert_eq!(driver.slots.free(), u32::from(slots));
}
