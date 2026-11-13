use super::*;
use crate::device::{ChainRegion, ReadableBytes, Unrecoverable, WritableBytes};
use crate::mem::{Window, tests::window};
use crate::ring::tests::{Guest, Part, WINDOW};
use crate::ring::{VIRTQ_AVAIL_F_NO_INTERRUPT, VIRTQ_USED_F_NO_NOTIFY};
use std::vec;
use std::vec::Vec;

/// A model that answers a request with its bytes reversed, and misbehaves
/// on request.
#[derive(Debug, Default)]
pub(crate) struct Echo {
    pub(crate) commands: u32,
    pub(crate) drives: u32,
    pub(crate) resets: u32,
    /// Set by the request `sick`: the model can no longer answer.
    sick: bool,
    /// A reset does not bring this model back.
    pub(crate) unrecoverable: bool,
}

pub(crate) const ECHO_CONFIG: &[u8] = b"echo config";

impl VirtualDevice for Echo {
    const DEVICE_ID: u32 = u32::from_le_bytes(*b"echo");
    const MAX_QUEUE_SIZE: u16 = 8;
    const MAX_CHAIN_DESCRIPTORS: u16 = 4;

    fn config(&self) -> &[u8] {
        ECHO_CONFIG
    }

    fn command(
        &mut self,
        _regions: &[ChainRegion],
        request: &dyn ReadableBytes,
        response: &mut dyn WritableBytes,
    ) -> u32 {
        self.commands += 1;
        let mut bytes = vec![0u8; request.len() as usize];
        request.read_at(0, &mut bytes).unwrap();
        match bytes.as_slice() {
            b"sick" => self.sick = true,
            // Claims more than the chain gave it.
            b"over" => return response.len() as u32 + 1,
            _ => {}
        }
        bytes.reverse();
        bytes.truncate(response.len() as usize);
        response.write_at(0, &bytes).unwrap();
        bytes.len() as u32
    }

    fn drive(&mut self) -> bool {
        self.drives += 1;
        false
    }

    fn needs_reset(&self) -> bool {
        self.sick
    }

    fn reset(&mut self) -> Result<(), Unrecoverable> {
        self.resets += 1;
        if self.unrecoverable {
            return Err(Unrecoverable);
        }
        self.sick = false;
        Ok(())
    }
}

const BUDGET: NonZeroU32 = NonZeroU32::new(16).unwrap();
const NAMED: u32 = STATUS_ACKNOWLEDGE | STATUS_DRIVER;
const AGREED: u32 = NAMED | STATUS_FEATURES_OK;
const LIVE: u32 = AGREED | STATUS_DRIVER_OK;

/// A driver written the way a C driver is: the register block and its own
/// memory, with every step done by hand.
struct Bench {
    registers: Window,
    guest: Guest,
    transport: Transport<Echo, Window>,
}

impl Bench {
    fn new() -> Self {
        Self::with(Echo::default())
    }

    fn with(model: Echo) -> Self {
        let registers = window(regs::BYTES as usize);
        let guest = Guest::over(window(WINDOW), Echo::MAX_QUEUE_SIZE);
        let mut transport = Transport::new(registers, model).expect("a transport");
        assert!(transport.lend(guest.memory).unwrap().is_none());
        Self {
            registers,
            guest,
            transport,
        }
    }

    /// Ring the doorbell.
    fn ring(&mut self) -> Serviced {
        self.transport.service(BUDGET)
    }

    fn write_status(&self, status: u32) {
        self.registers
            .store_u32(regs::DRIVER_STATUS, status)
            .unwrap();
    }

    /// The status the device publishes, which is also what it holds.
    fn status(&self) -> u32 {
        let published = self.registers.load_u32(regs::DEVICE_STATUS).unwrap();
        assert_eq!(published, self.transport.status());
        published
    }

    fn describe_queue(&self, size: u32, at: RingAddresses) {
        let registers = &self.registers;
        registers.store_u32(regs::QUEUE_NUM, size).unwrap();
        registers
            .store_u64(regs::QUEUE_DESC, at.descriptors)
            .unwrap();
        registers
            .store_u64(regs::QUEUE_DRIVER, at.available)
            .unwrap();
        registers.store_u64(regs::QUEUE_DEVICE, at.used).unwrap();
        registers.store_u32(regs::QUEUE_READY, 1).unwrap();
    }

    /// Everything up to `FEATURES_OK`.
    fn agree(&mut self) {
        self.write_status(NAMED);
        self.registers
            .store_u64(regs::DRIVER_FEATURES, F_VERSION_1)
            .unwrap();
        self.write_status(AGREED);
        assert!(self.ring().interrupt);
        assert_eq!(self.status(), AGREED);
    }

    /// The whole bring-up, with the queue the guest laid out.
    fn bring_up(&mut self) {
        self.agree();
        self.describe_queue(u32::from(self.guest.size), self.guest.at);
        self.write_status(LIVE);
        assert!(self.ring().interrupt);
        assert_eq!(self.status(), LIVE);
    }

    /// Publish `request` with `room` bytes for the answer in descriptors
    /// `head` and `head + 1`, and say where the answer goes.
    fn send(&mut self, head: u16, request: &[u8], room: u32) -> u64 {
        let places = self
            .guest
            .chain(head, &[Part::Read(request), Part::Write(room)]);
        self.guest.publish(head);
        places[1]
    }

    /// A reset, and a queue with nothing in it for the bring-up after.
    fn reset(&mut self) {
        self.write_status(0);
        assert!(
            self.ring().interrupt,
            "a reset of a device that is not is answered"
        );
        self.guest = Guest::over(self.guest.memory, self.guest.size);
        for ring in [self.guest.at.available, self.guest.at.used] {
            self.guest.memory.write(ring, &[0; 4]).unwrap();
        }
    }
}

#[test]
fn the_block_says_what_the_device_is() {
    let bench = Bench::new();
    let registers = &bench.registers;
    assert_eq!(registers.load_u32(regs::MAGIC), Ok(regs::MAGIC_VALUE));
    assert_eq!(registers.load_u32(regs::VERSION), Ok(regs::VERSION_VALUE));
    assert_eq!(registers.load_u32(regs::DEVICE_ID), Ok(Echo::DEVICE_ID));
    assert_eq!(registers.load_u32(regs::VENDOR_ID), Ok(regs::VENDOR_KORE));
    assert_eq!(registers.load_u64(regs::DEVICE_FEATURES), Ok(F_VERSION_1));
    assert_eq!(registers.load_u32(regs::DEVICE_STATUS), Ok(0));
    assert_eq!(registers.load_u32(regs::QUEUE_NUM_MAX), Ok(8));
    assert_eq!(registers.load_u32(regs::CONFIG_GENERATION), Ok(0));
    assert_eq!(registers.load_u32(regs::INTERRUPTS), Ok(0));
    assert_eq!(
        registers.load_u32(regs::CONFIG_BYTES),
        Ok(ECHO_CONFIG.len() as u32)
    );
    let mut config = [0u8; ECHO_CONFIG.len()];
    registers.read(regs::CONFIG, &mut config).unwrap();
    assert_eq!(config, ECHO_CONFIG);

    // The words are where a C header says they are.
    let mut head = [0u8; 16];
    registers.read(0, &mut head).unwrap();
    assert_eq!(&head[..4], b"kvdv");
    assert_eq!(head[4..8], 1u32.to_le_bytes());
    assert_eq!(&head[8..12], b"echo");
    assert_eq!(&head[12..], b"kore");
}

#[test]
fn a_transport_is_refused_what_it_cannot_carry() {
    struct Wide;
    impl VirtualDevice for Wide {
        const DEVICE_ID: u32 = 0;
        const MAX_QUEUE_SIZE: u16 = 8;
        const MAX_CHAIN_DESCRIPTORS: u16 = 4;
        fn config(&self) -> &[u8] {
            &[0; (regs::BYTES - regs::CONFIG) as usize + 1]
        }
        fn command(
            &mut self,
            _: &[ChainRegion],
            _: &dyn ReadableBytes,
            _: &mut dyn WritableBytes,
        ) -> u32 {
            0
        }
        fn drive(&mut self) -> bool {
            false
        }
        fn needs_reset(&self) -> bool {
            false
        }
        fn reset(&mut self) -> Result<(), Unrecoverable> {
            Ok(())
        }
    }
    struct Odd<const SIZE: u16, const CHAIN: u16>;
    impl<const SIZE: u16, const CHAIN: u16> VirtualDevice for Odd<SIZE, CHAIN> {
        const DEVICE_ID: u32 = 0;
        const MAX_QUEUE_SIZE: u16 = SIZE;
        const MAX_CHAIN_DESCRIPTORS: u16 = CHAIN;
        fn config(&self) -> &[u8] {
            &[]
        }
        fn command(
            &mut self,
            _: &[ChainRegion],
            _: &dyn ReadableBytes,
            _: &mut dyn WritableBytes,
        ) -> u32 {
            0
        }
        fn drive(&mut self) -> bool {
            false
        }
        fn needs_reset(&self) -> bool {
            false
        }
        fn reset(&mut self) -> Result<(), Unrecoverable> {
            Ok(())
        }
    }

    let page = regs::BYTES as usize;
    let attach = |registers: usize| Transport::new(window(registers), Echo::default()).map(|_| ());
    assert_eq!(attach(page - 8), Err(AttachError::Registers));
    assert_eq!(attach(page), Ok(()));
    assert_eq!(
        Transport::new(window(page), Wide).map(|_| ()),
        Err(AttachError::Config)
    );
    let odd = |made: Result<(), AttachError>| assert_eq!(made, Err(AttachError::Queue));
    odd(Transport::new(window(page), Odd::<6, 2>).map(|_| ()));
    odd(Transport::new(window(page), Odd::<128, 2>).map(|_| ()));
    odd(Transport::new(window(page), Odd::<8, 1>).map(|_| ()));
    odd(Transport::new(window(page), Odd::<8, 9>).map(|_| ()));
}

#[test]
fn a_driver_steps_through_bring_up() {
    let mut bench = Bench::new();
    // A look at a device in its reset state announces nothing, however
    // often it comes: whoever hosts the device looks on every wake, and a
    // driver that asked for nothing is not interrupted. Nothing is undone.
    for _ in 0..3 {
        assert_eq!(
            bench.ring(),
            Serviced {
                interrupt: false,
                pending: false
            }
        );
    }
    assert_eq!(bench.registers.load_u32(regs::INTERRUPTS), Ok(0));
    assert_eq!(bench.transport.device().resets, 0);
    // Zero written over zero is the same look.
    bench.write_status(0);
    assert!(!bench.ring().interrupt);

    bench.write_status(STATUS_ACKNOWLEDGE);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), STATUS_ACKNOWLEDGE);
    // The same status again announces nothing.
    assert!(!bench.ring().interrupt);

    bench.write_status(NAMED);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), NAMED);
    assert_eq!(bench.transport.features(), 0);

    bench
        .registers
        .store_u64(regs::DRIVER_FEATURES, F_VERSION_1)
        .unwrap();
    bench.write_status(AGREED);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), AGREED);
    assert_eq!(bench.transport.features(), F_VERSION_1);

    bench.describe_queue(8, bench.guest.at);
    bench.write_status(LIVE);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), LIVE);
    assert_eq!(bench.transport.device().commands, 0);
}

/// A driver may write every step before it rings once: the device takes
/// them in order all the same.
#[test]
fn steps_written_together_are_taken_in_order() {
    let mut bench = Bench::new();
    bench
        .registers
        .store_u64(regs::DRIVER_FEATURES, F_VERSION_1)
        .unwrap();
    bench.describe_queue(8, bench.guest.at);
    bench.write_status(LIVE);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), LIVE);
}

#[test]
fn a_step_whose_predecessor_is_missing_is_not_accepted() {
    for skipped in [
        STATUS_DRIVER,
        STATUS_FEATURES_OK,
        STATUS_DRIVER_OK,
        STATUS_ACKNOWLEDGE | STATUS_FEATURES_OK,
        STATUS_ACKNOWLEDGE | STATUS_DRIVER_OK,
        NAMED | STATUS_DRIVER_OK,
    ] {
        let mut bench = Bench::new();
        bench
            .registers
            .store_u64(regs::DRIVER_FEATURES, F_VERSION_1)
            .unwrap();
        bench.describe_queue(8, bench.guest.at);
        bench.write_status(skipped);
        assert!(bench.ring().interrupt, "refused, and told so");
        let accepted = bench.status();
        assert_eq!(
            accepted & !skipped,
            0,
            "nothing the driver did not write: {skipped:#x}"
        );
        assert_eq!(
            accepted & (STATUS_DRIVER_OK | STATUS_DEVICE_NEEDS_RESET),
            0,
            "not live from {skipped:#x}"
        );
        // And nothing is served.
        bench.send(0, b"hello", 8);
        bench.ring();
        assert!(bench.guest.used().is_none());
    }
}

#[test]
fn features_the_device_did_not_offer_are_refused() {
    for refused in [0, 1, F_VERSION_1 | 1, F_VERSION_1 | 1 << 33, u64::MAX] {
        let mut bench = Bench::new();
        bench.write_status(NAMED);
        bench
            .registers
            .store_u64(regs::DRIVER_FEATURES, refused)
            .unwrap();
        bench.write_status(AGREED);
        assert!(bench.ring().interrupt, "the refusal is an answer");
        assert_eq!(bench.status(), NAMED, "{refused:#x}");
        assert_eq!(bench.transport.features(), 0);

        // The queue cannot go live behind a refusal.
        bench.describe_queue(8, bench.guest.at);
        bench.write_status(LIVE);
        bench.ring();
        assert_eq!(bench.status(), NAMED);
    }
}

#[test]
fn a_queue_the_device_cannot_serve_asks_for_a_reset() {
    let good = Guest::over(window(64), 8).at;
    let outside = |part: fn(&mut RingAddresses) -> &mut u64| {
        let mut at = good;
        *part(&mut at) = WINDOW as u64;
        at
    };
    let cases: [(&str, u32, RingAddresses, bool); 9] = [
        ("not ready", 8, good, false),
        ("no size", 0, good, true),
        ("not a power of two", 6, good, true),
        ("larger than the device takes", 16, good, true),
        ("larger than a queue is", 0x1_0008, good, true),
        ("shorter than a chain", 2, good, true),
        (
            "descriptors outside",
            8,
            outside(|at| &mut at.descriptors),
            true,
        ),
        (
            "available outside",
            8,
            outside(|at| &mut at.available),
            true,
        ),
        ("used outside", 8, outside(|at| &mut at.used), true),
    ];
    for (what, size, at, ready) in cases {
        let mut bench = Bench::new();
        bench.agree();
        bench.describe_queue(size, at);
        if !ready {
            bench.registers.store_u32(regs::QUEUE_READY, 0).unwrap();
        }
        bench.write_status(LIVE);
        assert!(bench.ring().interrupt);
        assert_eq!(bench.status(), AGREED | STATUS_DEVICE_NEEDS_RESET, "{what}");

        // Nothing is served, and putting the queue right is too late.
        bench.describe_queue(8, good);
        bench.send(0, b"hello", 8);
        assert!(!bench.ring().interrupt);
        assert!(bench.guest.used().is_none());
        assert_eq!(bench.status(), AGREED | STATUS_DEVICE_NEEDS_RESET);

        // A reset is the way back.
        bench.reset();
        assert_eq!(bench.status(), 0);
        bench.bring_up();
    }
}

#[test]
fn commands_are_served_and_answered() {
    let mut bench = Bench::new();
    bench.bring_up();

    let first = bench.send(0, b"hello", 16);
    let second = bench.send(2, b"device", 3);
    assert_eq!(
        bench.ring(),
        Serviced {
            interrupt: true,
            pending: false
        }
    );
    assert_eq!(bench.guest.used(), Some((0, 5)));
    assert_eq!(bench.guest.used(), Some((2, 3)));
    assert_eq!(bench.guest.used(), None);
    assert_eq!(bench.guest.read(first, 5), b"olleh");
    assert_eq!(bench.guest.read(second, 3), b"eci", "no more than the room");
    assert_eq!(bench.transport.device().commands, 2);
    assert_eq!(bench.transport.device().drives, 2, "a turn after each");

    // A doorbell with nothing new is not answered.
    assert_eq!(bench.ring(), Serviced::default());

    // VQ-017: a driver that asked not to be interrupted is not, and still
    // finds its answer.
    let flags = bench.guest.at.available;
    bench
        .guest
        .memory
        .store_u16(flags, VIRTQ_AVAIL_F_NO_INTERRUPT)
        .unwrap();
    let third = bench.send(4, b"quiet", 8);
    assert_eq!(bench.ring(), Serviced::default());
    assert_eq!(bench.guest.used(), Some((4, 5)));
    assert_eq!(bench.guest.read(third, 5), b"teiuq");
}

#[test]
fn a_service_takes_no_more_than_its_budget() {
    let mut bench = Bench::new();
    bench.bring_up();
    // Four chains of two descriptors fill the queue of eight.
    for head in [0u16, 2, 4, 6] {
        bench.send(head, &[b'a' + head as u8; 4], 4);
    }
    let two = NonZeroU32::new(2).unwrap();
    let notify = bench.guest.at.used;
    let suppressed =
        |bench: &Bench| bench.guest.memory.load_u16(notify).unwrap() == VIRTQ_USED_F_NO_NOTIFY;

    assert_eq!(
        bench.transport.service(two),
        Serviced {
            interrupt: true,
            pending: true
        }
    );
    assert_eq!(bench.transport.device().commands, 2);
    assert!(suppressed(&bench), "the host comes back by itself");

    // Exactly the budget again: the device cannot yet tell it is done.
    assert!(bench.transport.service(two).pending);
    assert_eq!(bench.transport.device().commands, 4);
    assert!(suppressed(&bench));

    assert_eq!(bench.transport.service(two), Serviced::default());
    assert!(!suppressed(&bench), "idle: the driver rings again");
    let heads: Vec<_> = core::iter::from_fn(|| bench.guest.used()).collect();
    assert_eq!(heads, [(0, 4), (2, 4), (4, 4), (6, 4)]);
}

/// VQ-004..VQ-006 through the transport: a chain that cannot carry a
/// command is returned used with nothing written, the model never sees it,
/// and the chains after it are served.
#[test]
fn a_malformed_chain_fails_alone() {
    let mut bench = Bench::new();
    bench.bring_up();
    bench
        .guest
        .chain(0, &[Part::Read(b"no room for an answer")]);
    bench.guest.publish(0);
    // A head that is no descriptor at all.
    bench.guest.publish(200);
    let answer = bench.send(2, b"fine", 8);

    assert!(bench.ring().interrupt);
    assert_eq!(bench.guest.used(), Some((0, 0)));
    assert_eq!(bench.guest.used(), Some((200, 0)));
    assert_eq!(bench.guest.used(), Some((2, 4)));
    assert_eq!(bench.guest.read(answer, 4), b"enif");
    assert_eq!(bench.transport.device().commands, 1);
    assert_eq!(
        bench.status(),
        LIVE,
        "one chain's failure is not the device's"
    );
}

#[test]
fn a_queue_broken_while_live_asks_for_a_reset() {
    let mut bench = Bench::new();
    bench.bring_up();
    bench.send(0, b"hello", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.guest.used(), Some((0, 5)));

    // The driver claims more chains than the queue is long.
    let idx = bench.guest.at.available + 2;
    bench.guest.memory.store_u16(idx, 1 + 9).unwrap();
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), LIVE | STATUS_DEVICE_NEEDS_RESET);

    // Nothing more is taken, however the driver rings.
    bench.guest.memory.store_u16(idx, 1).unwrap();
    bench.send(2, b"again", 8);
    assert_eq!(bench.ring(), Serviced::default());
    assert!(bench.guest.used().is_none());
    assert_eq!(bench.transport.device().commands, 1);

    bench.reset();
    assert_eq!(bench.status(), 0);
    bench.bring_up();
    let answer = bench.send(0, b"back", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.guest.used(), Some((0, 4)));
    assert_eq!(bench.guest.read(answer, 4), b"kcab");
}

#[test]
fn a_model_that_cannot_go_on_asks_for_a_reset() {
    // It says so itself: the command that made it sick is still answered.
    let mut bench = Bench::new();
    bench.bring_up();
    bench.send(0, b"sick", 8);
    bench.send(2, b"after", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), LIVE | STATUS_DEVICE_NEEDS_RESET);
    assert_eq!(bench.guest.used(), Some((0, 4)));
    assert_eq!(bench.guest.used(), None, "nothing after it is taken");
    assert_eq!(bench.transport.device().commands, 1);
    bench.reset();
    assert_eq!(bench.status(), 0);
    assert_eq!(bench.transport.device().resets, 1);
    bench.bring_up();
    bench.send(0, b"well", 8);
    bench.ring();
    assert_eq!(bench.guest.used(), Some((0, 4)));

    // Or it claims to have written more than the chain gave it: nothing is
    // returned used that would tell the driver to read bytes it never lent.
    let mut bench = Bench::new();
    bench.bring_up();
    bench.send(0, b"over", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), LIVE | STATUS_DEVICE_NEEDS_RESET);
    assert_eq!(bench.guest.used(), None);
}

/// VQ-018, VQ-019 at the transport: a reset returns the device to its first
/// state, and nothing of the queue from before is used afterwards.
#[test]
fn a_reset_returns_the_device_to_its_first_state() {
    let mut bench = Bench::new();
    bench.bring_up();
    bench.send(0, b"hello", 8);
    bench.ring();
    // Published, and never served: the reset comes first.
    let unanswered = bench.send(2, b"never", 8);
    let used_before = bench
        .guest
        .memory
        .load_u16(bench.guest.at.used + 2)
        .unwrap();

    bench.write_status(0);
    assert_eq!(
        bench.ring(),
        Serviced {
            interrupt: true,
            pending: false
        }
    );
    assert_eq!(bench.status(), 0);
    assert_eq!(bench.transport.features(), 0);
    assert_eq!(bench.transport.device().resets, 1);
    assert_eq!(bench.transport.device().commands, 1);
    assert_eq!(
        bench
            .guest
            .memory
            .load_u16(bench.guest.at.used + 2)
            .unwrap(),
        used_before,
        "no used entry after the reset"
    );
    assert_eq!(bench.guest.read(unanswered, 5), [0; 5], "and no bytes");

    // Zero again announces nothing: nothing was accepted since, so nothing
    // is undone, and a reset that asked nothing is not answered.
    assert!(!bench.ring().interrupt);
    assert_eq!(bench.transport.device().resets, 1);

    // The old queue is not live: ringing without a bring-up serves nothing.
    bench.write_status(NAMED);
    bench.ring();
    assert_eq!(
        bench
            .guest
            .memory
            .load_u16(bench.guest.at.used + 2)
            .unwrap(),
        used_before
    );

    bench.reset();
    assert_eq!(bench.transport.device().resets, 2);
    bench.bring_up();
    let answer = bench.send(0, b"anew", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.guest.used(), Some((0, 4)), "from the first entry");
    assert_eq!(bench.guest.read(answer, 4), b"wena");
}

#[test]
fn a_reset_that_fails_is_final() {
    let mut bench = Bench::with(Echo {
        unrecoverable: true,
        ..Echo::default()
    });
    bench.bring_up();
    assert!(!bench.transport.is_dead());

    bench.write_status(0);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), STATUS_DEVICE_NEEDS_RESET);
    assert!(bench.transport.is_dead());

    // No bring-up and no further reset changes it.
    bench.write_status(NAMED);
    bench.ring();
    assert_eq!(bench.status(), STATUS_DEVICE_NEEDS_RESET);
    bench.write_status(0);
    assert!(bench.ring().interrupt, "still answered");
    assert_eq!(bench.status(), STATUS_DEVICE_NEEDS_RESET);
    assert_eq!(bench.transport.device().resets, 1, "and not tried again");
}

#[test]
fn a_driver_that_gives_up_stops_the_device() {
    let mut bench = Bench::new();
    bench.bring_up();
    bench.write_status(LIVE | STATUS_FAILED);
    bench.send(0, b"hello", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.status(), LIVE | STATUS_FAILED);
    assert!(bench.guest.used().is_none());

    // Taking the bit back does not undo it.
    bench.write_status(LIVE);
    bench.ring();
    assert_eq!(bench.status(), LIVE | STATUS_FAILED);
    assert!(bench.guest.used().is_none());

    bench.reset();
    assert_eq!(bench.status(), 0);
}

/// The device acts on what it accepted, not on what the registers say now.
#[test]
fn what_was_accepted_stays_accepted() {
    let mut bench = Bench::new();
    bench.bring_up();
    let registers = bench.registers;

    // The driver scribbles over everything it can, the device's own
    // registers included, and takes back steps it reached.
    registers
        .store_u64(regs::DRIVER_FEATURES, u64::MAX)
        .unwrap();
    registers.store_u32(regs::QUEUE_NUM, 2).unwrap();
    registers.store_u32(regs::QUEUE_READY, 0).unwrap();
    for part in [regs::QUEUE_DESC, regs::QUEUE_DRIVER, regs::QUEUE_DEVICE] {
        registers.store_u64(part, WINDOW as u64).unwrap();
    }
    registers.store_u32(regs::DEVICE_STATUS, 0).unwrap();
    registers
        .store_u64(regs::DEVICE_FEATURES, u64::MAX)
        .unwrap();
    registers.store_u32(regs::QUEUE_NUM_MAX, 0x8000).unwrap();
    bench.write_status(STATUS_ACKNOWLEDGE);

    let answer = bench.send(0, b"still", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(bench.guest.used(), Some((0, 5)));
    assert_eq!(bench.guest.read(answer, 5), b"llits");
    assert_eq!(bench.transport.status(), LIVE);
    assert_eq!(bench.transport.features(), F_VERSION_1);
}

/// Driver memory in which a chain is published at the worst moment: after
/// the device found the queue empty and before it says it is listening
/// again. The driver saw the device not listening and rang no doorbell.
#[derive(Clone, Debug)]
struct Racing {
    window: Window,
    /// `(the device's listening flag, the available index, its new value)`,
    /// until the race has been run.
    race: std::rc::Rc<core::cell::Cell<Option<(u64, u64, u16)>>>,
}

impl Memory for Racing {
    fn len(&self) -> u64 {
        self.window.len()
    }
    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), crate::mem::MemoryFault> {
        self.window.read(at, target)
    }
    fn write(&self, at: u64, source: &[u8]) -> Result<(), crate::mem::MemoryFault> {
        self.window.write(at, source)
    }
    fn load_u16(&self, at: u64) -> Result<u16, crate::mem::MemoryFault> {
        self.window.load_u16(at)
    }
    fn store_u16(&self, at: u64, value: u16) -> Result<(), crate::mem::MemoryFault> {
        if let Some((listening, index, published)) = self.race.get()
            && at == listening
            && value == 0
        {
            self.race.set(None);
            self.window.store_u16(index, published)?;
        }
        self.window.store_u16(at, value)
    }
    fn load_u32(&self, at: u64) -> Result<u32, crate::mem::MemoryFault> {
        self.window.load_u32(at)
    }
    fn store_u32(&self, at: u64, value: u32) -> Result<(), crate::mem::MemoryFault> {
        self.window.store_u32(at, value)
    }
    fn load_u64(&self, at: u64) -> Result<u64, crate::mem::MemoryFault> {
        self.window.load_u64(at)
    }
    fn store_u64(&self, at: u64, value: u64) -> Result<(), crate::mem::MemoryFault> {
        self.window.store_u64(at, value)
    }
}

/// VQ-017, the half a single thread cannot reach by itself: the device
/// looks once more after it says it is listening, so the chain no doorbell
/// announced is served in the same service.
#[test]
fn a_chain_published_as_the_device_stops_looking_is_served() {
    let race = std::rc::Rc::new(core::cell::Cell::new(None));
    let racing = |window| Racing {
        window,
        race: race.clone(),
    };
    let registers = window(regs::BYTES as usize);
    let mut guest = Guest::over(window(WINDOW), Echo::MAX_QUEUE_SIZE);
    let mut transport = Transport::new(racing(registers), Echo::default()).unwrap();
    transport.lend(racing(guest.memory)).unwrap();

    registers
        .store_u64(regs::DRIVER_FEATURES, F_VERSION_1)
        .unwrap();
    registers.store_u32(regs::QUEUE_NUM, 8).unwrap();
    registers
        .store_u64(regs::QUEUE_DESC, guest.at.descriptors)
        .unwrap();
    registers
        .store_u64(regs::QUEUE_DRIVER, guest.at.available)
        .unwrap();
    registers
        .store_u64(regs::QUEUE_DEVICE, guest.at.used)
        .unwrap();
    registers.store_u32(regs::QUEUE_READY, 1).unwrap();
    registers.store_u32(regs::DRIVER_STATUS, LIVE).unwrap();
    transport.service(BUDGET);
    assert_eq!(transport.status(), LIVE);

    // One chain announced, and a second written but not yet published.
    guest.chain(0, &[Part::Read(b"first"), Part::Write(8)]);
    guest.publish(0);
    let late = guest.chain(2, &[Part::Read(b"late"), Part::Write(8)])[1];
    guest
        .memory
        .write(guest.at.available + 4 + 2, &2u16.to_le_bytes())
        .unwrap();
    race.set(Some((guest.at.used, guest.at.available + 2, 2)));

    assert_eq!(
        transport.service(BUDGET),
        Serviced {
            interrupt: true,
            pending: false
        }
    );
    assert!(race.get().is_none(), "the race was run");
    assert_eq!(guest.used(), Some((0, 5)));
    assert_eq!(guest.used(), Some((2, 4)), "served without a doorbell");
    assert_eq!(guest.read(late, 4), b"etal");
    assert_eq!(
        guest.memory.load_u16(guest.at.used).unwrap(),
        0,
        "and the device is listening when it stops"
    );
}

/// Driver memory whose available index says one chain is published on
/// every second look and none on the looks between: what a driver on
/// another processor can make the device see by writing the index and
/// taking it back.
#[derive(Clone, Debug)]
struct Flapping {
    window: Window,
    /// The available index, once the queue is live.
    index: std::rc::Rc<core::cell::Cell<Option<u64>>>,
    looks: std::rc::Rc<core::cell::Cell<u32>>,
}

impl Memory for Flapping {
    fn len(&self) -> u64 {
        self.window.len()
    }
    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), crate::mem::MemoryFault> {
        self.window.read(at, target)
    }
    fn write(&self, at: u64, source: &[u8]) -> Result<(), crate::mem::MemoryFault> {
        self.window.write(at, source)
    }
    fn load_u16(&self, at: u64) -> Result<u16, crate::mem::MemoryFault> {
        if self.index.get() == Some(at) {
            // The device looks to take a chain, then looks again as it
            // says it is listening: nothing, then one, and so on.
            self.looks.set(self.looks.get() + 1);
            return Ok(u16::from(self.looks.get() % 2 == 0));
        }
        self.window.load_u16(at)
    }
    fn store_u16(&self, at: u64, value: u16) -> Result<(), crate::mem::MemoryFault> {
        self.window.store_u16(at, value)
    }
    fn load_u32(&self, at: u64) -> Result<u32, crate::mem::MemoryFault> {
        self.window.load_u32(at)
    }
    fn store_u32(&self, at: u64, value: u32) -> Result<(), crate::mem::MemoryFault> {
        self.window.store_u32(at, value)
    }
    fn load_u64(&self, at: u64) -> Result<u64, crate::mem::MemoryFault> {
        self.window.load_u64(at)
    }
    fn store_u64(&self, at: u64, value: u64) -> Result<(), crate::mem::MemoryFault> {
        self.window.store_u64(at, value)
    }
}

/// A driver that shows the device a chain and takes it back, over and
/// over, gets the budget's worth of looks from one service and no more.
#[test]
fn a_driver_that_flaps_its_index_cannot_hold_the_device() {
    let index = std::rc::Rc::new(core::cell::Cell::new(None));
    let looks = std::rc::Rc::new(core::cell::Cell::new(0));
    let flapping = |window| Flapping {
        window,
        index: index.clone(),
        looks: looks.clone(),
    };
    let registers = window(regs::BYTES as usize);
    let guest = Guest::over(window(WINDOW), Echo::MAX_QUEUE_SIZE);
    let mut transport = Transport::new(flapping(registers), Echo::default()).unwrap();
    transport.lend(flapping(guest.memory)).unwrap();

    registers
        .store_u64(regs::DRIVER_FEATURES, F_VERSION_1)
        .unwrap();
    registers.store_u32(regs::QUEUE_NUM, 8).unwrap();
    registers
        .store_u64(regs::QUEUE_DESC, guest.at.descriptors)
        .unwrap();
    registers
        .store_u64(regs::QUEUE_DRIVER, guest.at.available)
        .unwrap();
    registers
        .store_u64(regs::QUEUE_DEVICE, guest.at.used)
        .unwrap();
    registers.store_u32(regs::QUEUE_READY, 1).unwrap();
    registers.store_u32(regs::DRIVER_STATUS, LIVE).unwrap();
    transport.service(BUDGET);
    assert_eq!(transport.status(), LIVE);

    index.set(Some(guest.at.available + 2));
    looks.set(0);
    let serviced = transport.service(BUDGET);
    assert!(serviced.pending, "the host is asked to come back");
    assert_eq!(
        looks.get(),
        2 * BUDGET.get(),
        "a look to take and a look to listen, the budget's worth"
    );
    assert_eq!(transport.device().commands, 0);
    assert_eq!(transport.status(), LIVE, "flapping is not a broken queue");
}

/// The block is there before any memory is: a driver identifies the device
/// first. A queue made ready with no memory lent is one the device cannot
/// serve.
#[test]
fn a_queue_with_no_memory_lent_asks_for_a_reset() {
    let registers = window(regs::BYTES as usize);
    let guest = Guest::over(window(WINDOW), Echo::MAX_QUEUE_SIZE);
    let mut transport = Transport::new(registers, Echo::default()).unwrap();
    assert!(!transport.has_memory());
    assert_eq!(registers.load_u32(regs::MAGIC), Ok(regs::MAGIC_VALUE));
    assert_eq!(registers.load_u32(regs::DEVICE_ID), Ok(Echo::DEVICE_ID));

    registers
        .store_u64(regs::DRIVER_FEATURES, F_VERSION_1)
        .unwrap();
    registers.store_u32(regs::DRIVER_STATUS, AGREED).unwrap();
    assert!(transport.service(BUDGET).interrupt);
    assert_eq!(transport.status(), AGREED, "no memory is needed to agree");

    registers.store_u32(regs::QUEUE_NUM, 8).unwrap();
    registers
        .store_u64(regs::QUEUE_DRIVER, guest.at.available)
        .unwrap();
    registers
        .store_u64(regs::QUEUE_DEVICE, guest.at.used)
        .unwrap();
    registers.store_u32(regs::QUEUE_READY, 1).unwrap();
    registers.store_u32(regs::DRIVER_STATUS, LIVE).unwrap();
    assert!(transport.service(BUDGET).interrupt);
    assert_eq!(transport.status(), AGREED | STATUS_DEVICE_NEEDS_RESET);

    // Lending now is too late; after the reset it is not.
    assert_eq!(transport.lend(guest.memory).map(|_| ()), Err(Busy));
    registers.store_u32(regs::DRIVER_STATUS, 0).unwrap();
    transport.service(BUDGET);
    assert_eq!(transport.status(), 0);
    assert!(transport.lend(guest.memory).unwrap().is_none());
    registers.store_u32(regs::DRIVER_STATUS, LIVE).unwrap();
    transport.service(BUDGET);
    assert_eq!(transport.status(), LIVE);
}

/// The driver's memory is the device's for as long as the device has
/// accepted anything: it is replaced or taken back only in the first state.
#[test]
fn memory_changes_only_in_the_first_state() {
    let mut bench = Bench::new();
    let other = window(WINDOW);
    // In the first state it is replaced, and the old is handed back.
    assert!(bench.transport.lend(other).unwrap().is_some());
    assert!(bench.transport.lend(bench.guest.memory).unwrap().is_some());

    bench.write_status(STATUS_ACKNOWLEDGE);
    bench.ring();
    assert_eq!(bench.transport.lend(other).map(|_| ()), Err(Busy));
    assert_eq!(bench.transport.withdraw().map(|_| ()), Err(Busy));

    bench.write_status(0);
    bench.ring();
    bench.bring_up();
    let answer = bench.send(0, b"hello", 8);
    bench.ring();
    assert_eq!(bench.guest.read(answer, 5), b"olleh");
    assert_eq!(bench.transport.withdraw().map(|_| ()), Err(Busy));

    // After a reset the device holds no queue, and gives the memory up.
    bench.reset();
    assert!(bench.transport.withdraw().unwrap().is_some());
    assert!(!bench.transport.has_memory());
    assert!(bench.transport.withdraw().unwrap().is_none());
}

/// The interrupt is a count: it moves exactly when a service says to
/// interrupt, after what the interrupt is about has been written.
#[test]
fn the_interrupt_count_moves_with_each_interrupt() {
    let mut bench = Bench::new();
    let count = |bench: &Bench| bench.registers.load_u32(regs::INTERRUPTS).unwrap();
    assert_eq!(count(&bench), 0);

    bench.bring_up();
    let raised = count(&bench);
    assert_eq!(raised, 2, "one for the features, one for the queue");

    // Nothing to answer: no interrupt, and the count stays.
    assert!(!bench.ring().interrupt);
    assert_eq!(count(&bench), raised);

    // Two commands in one service are one interrupt.
    bench.send(0, b"one", 8);
    bench.send(2, b"two", 8);
    assert!(bench.ring().interrupt);
    assert_eq!(count(&bench), raised + 1);

    // A driver that asked not to be interrupted is not counted either.
    let flags = bench.guest.at.available;
    bench
        .guest
        .memory
        .store_u16(flags, VIRTQ_AVAIL_F_NO_INTERRUPT)
        .unwrap();
    bench.send(4, b"quiet", 8);
    assert!(!bench.ring().interrupt);
    assert_eq!(count(&bench), raised + 1);
    assert_eq!(bench.guest.used(), Some((0, 3)));

    // The driver cannot wind the count: the device keeps its own.
    bench.registers.store_u32(regs::INTERRUPTS, 0).unwrap();
    bench.write_status(0);
    assert!(bench.ring().interrupt);
    assert_eq!(count(&bench), raised + 2);
}
