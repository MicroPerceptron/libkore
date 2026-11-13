use super::*;
use crate::mem::{Window, tests::window};
use std::vec;
use std::vec::Vec;
use virtio_accel_split_queue::{Descriptor, DriverChain, SplitQueue};
use virtio_accel_transport::RegionDirection;

/// One descriptor a test driver writes: readable bytes, or room to write.
#[derive(Clone, Copy)]
pub(crate) enum Part<'a> {
    Read(&'a [u8]),
    Write(u32),
}

/// A driver's half of the queue, written the way a C driver would write it:
/// the three parts laid out in its own memory, and descriptors, available
/// entries and used entries moved by hand.
pub(crate) struct Guest {
    pub(crate) memory: Window,
    pub(crate) size: u16,
    pub(crate) at: RingAddresses,
    /// Where the next buffer goes: buffers are never reused in a test.
    bump: u64,
    published: u16,
    seen: u16,
}

pub(crate) const WINDOW: usize = 64 * 1024;
const BUFFERS: u64 = 0x2000;

impl Guest {
    /// A queue of `size` whose chains are at most `max_chain` long, ready.
    fn ready(size: u16, max_chain: u16) -> (Self, DeviceRing<Window>) {
        let (guest, mut ring) = Self::unready(size, max_chain);
        ring.set_ready(true).expect("a layout inside the window");
        (guest, ring)
    }

    fn unready(size: u16, max_chain: u16) -> (Self, DeviceRing<Window>) {
        let guest = Self::over(window(WINDOW), size);
        let mut ring = DeviceRing::new(
            guest.memory,
            QueueSize::new(MAX_QUEUE_SIZE).unwrap(),
            max_chain,
        )
        .expect("limits fit");
        ring.configure(QueueSize::new(size).unwrap())
            .expect("configured");
        ring.set_addresses(guest.at).expect("addresses");
        (guest, ring)
    }

    /// A driver's queue of `size` laid out in `memory`, which no device has
    /// been told about yet.
    pub(crate) fn over(memory: Window, size: u16) -> Self {
        Self {
            memory,
            size,
            at: RingAddresses {
                descriptors: 0,
                available: 0x800,
                used: 0x1000,
            },
            bump: BUFFERS,
            published: 0,
            seen: 0,
        }
    }

    pub(crate) fn descriptor(&self, index: u16, addr: u64, len: u32, flags: u16, next: u16) {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&addr.to_le_bytes());
        bytes[8..12].copy_from_slice(&len.to_le_bytes());
        bytes[12..14].copy_from_slice(&flags.to_le_bytes());
        bytes[14..].copy_from_slice(&next.to_le_bytes());
        self.memory
            .write(self.at.descriptors + u64::from(index) * 16, &bytes)
            .unwrap();
    }

    fn space(&mut self, len: u32) -> u64 {
        let at = self.bump;
        self.bump += u64::from(len.max(1)).next_multiple_of(8);
        at
    }

    /// Write `parts` as a direct chain in descriptors `first..`, and answer
    /// where each part's bytes are.
    pub(crate) fn chain(&mut self, first: u16, parts: &[Part<'_>]) -> Vec<u64> {
        let mut places = Vec::new();
        for (offset, part) in parts.iter().enumerate() {
            let index = first + offset as u16;
            let last = offset + 1 == parts.len();
            let (len, mut flags) = match part {
                Part::Read(bytes) => (bytes.len() as u32, 0),
                Part::Write(len) => (*len, VIRTQ_DESC_F_WRITE),
            };
            let at = self.space(len);
            if let Part::Read(bytes) = part {
                self.memory.write(at, bytes).unwrap();
            }
            if !last {
                flags |= VIRTQ_DESC_F_NEXT;
            }
            self.descriptor(index, at, len, flags, if last { 0 } else { index + 1 });
            places.push(at);
        }
        places
    }

    /// Add `head` to the available ring and publish it.
    pub(crate) fn publish(&mut self, head: u16) {
        let slot = u64::from(self.published % self.size);
        self.memory
            .write(self.at.available + 4 + slot * 2, &head.to_le_bytes())
            .unwrap();
        self.published = self.published.wrapping_add(1);
        self.memory
            .store_u16(self.at.available + 2, self.published)
            .unwrap();
    }

    /// The next used entry the driver has not seen: `(head, bytes written)`.
    pub(crate) fn used(&mut self) -> Option<(u32, u32)> {
        if self.memory.load_u16(self.at.used + 2).unwrap() == self.seen {
            return None;
        }
        let slot = u64::from(self.seen % self.size);
        let entry = self.at.used + 4 + slot * 8;
        self.seen = self.seen.wrapping_add(1);
        Some((
            self.memory.load_u32(entry).unwrap(),
            self.memory.load_u32(entry + 4).unwrap(),
        ))
    }

    pub(crate) fn read(&self, at: u64, len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        self.memory.read(at, &mut bytes).unwrap();
        bytes
    }
}

fn request(chain: &mut RingChain<Window>) -> Vec<u8> {
    let (_, source, _) = chain.io().expect("a valid chain").into_parts();
    let mut bytes = vec![0u8; source.len() as usize];
    source.read_at(0, &mut bytes).unwrap();
    bytes
}

/// VQ-001, VQ-003, VQ-013: a command crosses in readable descriptors and its
/// response comes back in writable ones, each side one run of bytes whatever
/// the descriptors' split, with the used length exactly what was written.
#[test]
fn a_chain_carries_a_request_out_and_a_response_back() {
    let (mut guest, mut ring) = Guest::ready(8, 8);
    assert!(ring.pop_available().unwrap().is_none());

    let places = guest.chain(
        2,
        &[
            Part::Read(b"hello, "),
            Part::Read(b"device"),
            Part::Write(5),
            Part::Write(16),
        ],
    );
    guest.publish(2);

    let mut chain = ring.pop_available().unwrap().expect("one chain");
    assert_eq!(ring.in_flight(), 1);
    assert_eq!(request(&mut chain), b"hello, device");
    {
        let (regions, source, sink) = chain.io().unwrap().into_parts();
        let shape: Vec<_> = regions.iter().map(|r| (r.direction, r.bytes)).collect();
        assert_eq!(
            shape,
            [
                (RegionDirection::DeviceReadable, 7),
                (RegionDirection::DeviceReadable, 6),
                (RegionDirection::DeviceWritable, 5),
                (RegionDirection::DeviceWritable, 16),
            ]
        );
        // A read may start and end anywhere, across the split.
        let mut middle = [0u8; 6];
        source.read_at(4, &mut middle).unwrap();
        assert_eq!(&middle, b"o, dev");
        assert_eq!(
            source.read_at(8, &mut middle),
            Err(ByteAccessError::OutOfBounds)
        );
        // The response crosses its split the same way.
        assert_eq!(sink.len(), 21);
        sink.write_at(0, b"hi, driver").unwrap();
        assert_eq!(sink.write_at(20, b"xy"), Err(ByteAccessError::OutOfBounds));
    }
    assert_eq!(
        ring.complete(chain, UsedLength::new(10)),
        Ok(NotificationHint::Notify)
    );
    assert_eq!(ring.in_flight(), 0);
    assert_eq!(guest.used(), Some((2, 10)));
    assert_eq!(guest.used(), None);
    assert_eq!(guest.read(places[2], 5), b"hi, d");
    assert_eq!(guest.read(places[3], 5), b"river");
    // Room past the response is untouched.
    assert_eq!(guest.read(places[3] + 5, 11), [0u8; 11]);
}

/// A chain is read once. A driver that rewrites a descriptor after it
/// published the chain changes nothing the device does with it.
#[test]
fn a_descriptor_rewritten_after_publication_changes_nothing() {
    let (mut guest, mut ring) = Guest::ready(4, 4);
    let places = guest.chain(0, &[Part::Read(b"the request"), Part::Write(8)]);
    let decoy = guest.space(64);
    guest.memory.write(decoy, &[0xaa; 64]).unwrap();
    guest.publish(0);
    let mut chain = ring.pop_available().unwrap().unwrap();

    // Point both descriptors at the decoy, longer, and swap their roles.
    guest.descriptor(0, decoy, 64, VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT, 1);
    guest.descriptor(1, decoy, 64, 0, 0);

    assert_eq!(request(&mut chain), b"the request");
    let (_, _, sink) = chain.io().unwrap().into_parts();
    assert_eq!(sink.len(), 8);
    sink.write_at(0, b"answered").unwrap();
    ring.complete(chain, UsedLength::new(8)).unwrap();
    assert_eq!(guest.read(places[1], 8), b"answered");
    assert_eq!(guest.read(decoy, 64), [0xaa; 64]);
}

/// A chain's shape as plain numbers: `(length, flags, next)` per descriptor.
type Shape = &'static [(u32, u16, u16)];

const N: u16 = VIRTQ_DESC_F_NEXT;
const W: u16 = VIRTQ_DESC_F_WRITE;

/// Shapes a driver may publish, good and bad, with what the protocol calls
/// each. The chain limit in these tests is four descriptors.
const SHAPES: &[(&str, Shape, u16, Option<MalformedChain>)] = &[
    (
        "one readable, one writable",
        &[(24, N, 1), (32, W, 0)],
        0,
        None,
    ),
    (
        "two and two",
        &[(8, N, 1), (8, N, 2), (8, N | W, 3), (8, W, 0)],
        0,
        None,
    ),
    (
        "a head that is not descriptor zero",
        &[(16, W, 0), (16, N, 0)],
        1,
        None,
    ),
    (
        "no writable descriptor",
        &[(8, N, 1), (8, 0, 0)],
        0,
        Some(MalformedChain::Direction),
    ),
    (
        "no readable descriptor",
        &[(8, N | W, 1), (8, W, 0)],
        0,
        Some(MalformedChain::Direction),
    ),
    (
        "readable after writable",
        &[(8, N, 1), (8, N | W, 2), (8, 0, 0)],
        0,
        Some(MalformedChain::Direction),
    ),
    (
        "a single descriptor",
        &[(8, W, 0)],
        0,
        Some(MalformedChain::DescriptorCount),
    ),
    (
        "longer than the chain limit",
        &[(8, N, 1), (8, N, 2), (8, N, 3), (8, N | W, 4), (8, W, 0)],
        0,
        Some(MalformedChain::DescriptorCount),
    ),
    (
        "a zero-length descriptor",
        &[(0, N, 1), (8, W, 0)],
        0,
        Some(MalformedChain::ZeroLength),
    ),
    (
        "a loop",
        &[(8, N, 1), (8, N | W, 0)],
        0,
        Some(MalformedChain::DescriptorLoop),
    ),
    (
        "a descriptor that follows itself",
        &[(8, N, 0)],
        0,
        Some(MalformedChain::DescriptorLoop),
    ),
    (
        "an indirect descriptor",
        &[(16, VIRTQ_DESC_F_INDIRECT, 0)],
        0,
        Some(MalformedChain::IndirectUnsupported),
    ),
    (
        "a flag the profile does not have",
        &[(8, N | 0x80, 1), (8, W, 0)],
        0,
        Some(MalformedChain::DescriptorFlags),
    ),
];

/// What a consumed chain says of itself: its regions, or why it is malformed.
type Verdict = Result<Vec<ChainRegion>, MalformedChain>;

fn verdict<C: DeviceChain>(chain: &mut C, name: &str) -> Verdict
where
    C::Error: core::fmt::Debug,
{
    match chain.io() {
        Ok(io) => Ok(io.into_parts().0.to_vec()),
        Err(ChainError::Malformed(malformed)) => Err(malformed),
        Err(other) => panic!("{name}: {other:?}"),
    }
}

/// VQ-004, VQ-005, VQ-006, and the oracle: every shape gets the same answer
/// from this ring as from the protocol's own in-memory split queue, and a
/// malformed one is handed up, says why, and is used with nothing written.
#[test]
fn every_chain_shape_is_judged_as_the_protocols_own_queue_judges_it() {
    for &(name, shape, head, expected) in SHAPES {
        // This ring, in a driver's memory.
        let (mut guest, mut ring) = Guest::ready(8, 4);
        for (index, &(len, flags, next)) in shape.iter().enumerate() {
            let at = guest.space(len);
            guest.descriptor(index as u16, at, len, flags, next);
        }
        guest.publish(head);
        let mut chain = ring.pop_available().unwrap().expect(name);
        let ours = verdict(&mut chain, name);
        assert_eq!(ours.as_ref().err().copied(), expected, "{name}");

        // The protocol's model, given the same descriptors.
        let mut model = SplitQueue::new(QueueSize::new(8).unwrap(), 4).unwrap();
        model.configure(QueueSize::new(8).unwrap()).unwrap();
        QueueControl::set_ready(&mut model, true).unwrap();
        let descriptors = shape
            .iter()
            .map(|&(len, flags, next)| Descriptor::raw(vec![0; len as usize], flags, next))
            .collect();
        let raw = DriverChain::raw(descriptors, head).expect(name);
        model.inject_available(raw).map_err(|_| name).unwrap();
        let mut modelled = DeviceQueue::pop_available(&mut model).unwrap().expect(name);
        assert_eq!(
            ours,
            verdict(&mut modelled, name),
            "{name}: this ring against the model"
        );

        // Either way the driver gets its chain back; a malformed one with
        // nothing written.
        let used = u32::from(ours.is_ok());
        ring.complete(chain, UsedLength::new(used)).expect(name);
        assert_eq!(guest.used(), Some((u32::from(head), used)), "{name}");
    }
}

/// What only a ring in real memory can get wrong: indices and addresses the
/// driver made up.
#[test]
fn indices_and_addresses_outside_what_was_lent_are_malformed_chains() {
    let malformed = |build: &dyn Fn(&mut Guest) -> u16| {
        let (mut guest, mut ring) = Guest::ready(8, 4);
        let head = build(&mut guest);
        guest.publish(head);
        let mut chain = ring.pop_available().unwrap().unwrap();
        let reason = verdict(&mut chain, "made up").expect_err("malformed");
        // It is still the driver's chain to get back, with nothing written.
        ring.complete(chain, UsedLength::new(0)).unwrap();
        assert_eq!(guest.used(), Some((u32::from(head), 0)));
        reason
    };

    // A head past the queue.
    assert_eq!(malformed(&|_| 8), MalformedChain::DescriptorIndex);
    assert_eq!(malformed(&|_| u16::MAX), MalformedChain::DescriptorIndex);
    // A `next` past the queue.
    assert_eq!(
        malformed(&|guest| {
            let at = guest.space(8);
            guest.descriptor(0, at, 8, N, 9);
            0
        }),
        MalformedChain::DescriptorIndex
    );
    // Bytes that begin or end outside the window, readable or writable.
    for (at, len) in [
        (WINDOW as u64, 8u32),
        (WINDOW as u64 - 4, 8),
        (u64::MAX - 3, 8),
        (0x1_0000_0000, 8),
    ] {
        assert_eq!(
            malformed(&|guest| {
                let good = guest.space(8);
                guest.descriptor(0, at, len, N, 1);
                guest.descriptor(1, good, 8, W, 0);
                0
            }),
            MalformedChain::Address,
            "readable at {at:#x}"
        );
        assert_eq!(
            malformed(&|guest| {
                let good = guest.space(8);
                guest.descriptor(0, good, 8, N, 1);
                guest.descriptor(1, at, len, W, 0);
                0
            }),
            MalformedChain::Address,
            "writable at {at:#x}"
        );
    }
}

/// A used length is the bytes the device wrote, and it cannot claim more
/// than the chain gave it room for.
#[test]
fn a_used_length_past_the_chains_room_is_refused() {
    let (mut guest, mut ring) = Guest::ready(4, 4);
    guest.chain(0, &[Part::Read(b"q"), Part::Write(4)]);
    guest.publish(0);
    let chain = ring.pop_available().unwrap().unwrap();
    assert_eq!(
        ring.complete(chain, UsedLength::new(5)).err(),
        Some(QueueError::UsedLengthExceeded {
            used: UsedLength::new(5),
            capacity: 4
        })
    );
    assert_eq!(guest.used(), None, "nothing was published");
}

/// VQ-014: the device uses chains in whatever order it finishes them.
#[test]
fn chains_are_used_in_the_order_the_device_finishes_them() {
    let (mut guest, mut ring) = Guest::ready(8, 4);
    guest.chain(0, &[Part::Read(b"first"), Part::Write(4)]);
    guest.chain(4, &[Part::Read(b"second"), Part::Write(4)]);
    guest.publish(0);
    guest.publish(4);
    let first = ring.pop_available().unwrap().unwrap();
    let second = ring.pop_available().unwrap().unwrap();
    assert!(ring.pop_available().unwrap().is_none());
    assert_eq!(ring.in_flight(), 2);
    ring.complete(second, UsedLength::new(3)).unwrap();
    ring.complete(first, UsedLength::new(1)).unwrap();
    assert_eq!(guest.used(), Some((4, 3)));
    assert_eq!(guest.used(), Some((0, 1)));
}

/// VQ-017: either side may ask not to be woken, and neither loses work for
/// it.
#[test]
fn suppression_is_honoured_and_loses_no_work() {
    let (mut guest, mut ring) = Guest::ready(4, 4);

    // The driver asks not to be interrupted.
    guest
        .memory
        .store_u16(guest.at.available, VIRTQ_AVAIL_F_NO_INTERRUPT)
        .unwrap();
    guest.chain(0, &[Part::Read(b"q"), Part::Write(1)]);
    guest.publish(0);
    let chain = ring.pop_available().unwrap().unwrap();
    assert_eq!(
        ring.complete(chain, UsedLength::new(1)),
        Ok(NotificationHint::Suppressed)
    );
    // The entry is there all the same.
    assert_eq!(guest.used(), Some((0, 1)));

    // The device asks not to be notified while it drains, and says so where
    // the driver reads it.
    ring.disable_available_notifications().unwrap();
    assert_eq!(
        guest.memory.load_u16(guest.at.used),
        Ok(VIRTQ_USED_F_NO_NOTIFY)
    );
    assert!(ring.pop_available().unwrap().is_none());
    // A chain published before it listens again is found by the recheck.
    guest.chain(2, &[Part::Read(b"r"), Part::Write(1)]);
    guest.publish(2);
    assert_eq!(
        ring.enable_available_notifications(),
        Ok(NotificationRecheck::WorkPending)
    );
    assert_eq!(guest.memory.load_u16(guest.at.used), Ok(0));
    let chain = ring.pop_available().unwrap().unwrap();
    ring.complete(chain, UsedLength::new(0)).unwrap();
    assert_eq!(
        ring.enable_available_notifications(),
        Ok(NotificationRecheck::Idle)
    );
}

/// VQ-018: after a reset nothing of the old epoch is written. A chain the
/// device still holds can neither read, write nor be used.
#[test]
fn a_reset_silences_every_chain_the_device_still_holds() {
    let (mut guest, mut ring) = Guest::ready(4, 4);
    let places = guest.chain(0, &[Part::Read(b"before"), Part::Write(8)]);
    guest.publish(0);
    let mut chain = ring.pop_available().unwrap().unwrap();
    let old = chain.id().epoch();
    let next = old.checked_next().unwrap();

    assert_eq!(
        ring.reset(old).err(),
        Some(QueueError::InvalidConfiguration(
            QueueConfigError::NonIncreasingEpoch
        ))
    );
    ring.reset(next).unwrap();
    assert!(!ring.state().ready());
    assert_eq!(ring.in_flight(), 0);

    assert!(matches!(
        chain.io().map(|_| ()),
        Err(ChainError::ResetRace { chain, current }) if chain == old && current == next
    ));
    assert_eq!(
        chain.response.write_at(0, b"too late"),
        Err(ByteAccessError::Reset)
    );
    let mut bytes = [0u8; 6];
    assert_eq!(
        chain.request.read_at(0, &mut bytes),
        Err(ByteAccessError::Reset)
    );
    assert_eq!(
        ring.complete(chain, UsedLength::new(0)).err(),
        Some(QueueError::ResetRace {
            operation: old,
            current: next
        })
    );
    assert_eq!(guest.used(), None, "no used entry from the old epoch");
    assert_eq!(guest.read(places[1], 8), [0u8; 8]);

    // The queue serves again once the driver sets it up again.
    assert_eq!(
        ring.pop_available().map(|chain| chain.is_some()),
        Err(QueueError::NotReady)
    );
    ring.configure(QueueSize::new(4).unwrap()).unwrap();
    ring.set_addresses(guest.at).unwrap();
    ring.set_ready(true).unwrap();
    // The driver starts its rings over too, as a reset requires of it.
    guest.memory.store_u16(guest.at.available + 2, 0).unwrap();
    guest.memory.store_u16(guest.at.used + 2, 0).unwrap();
    (guest.published, guest.seen) = (0, 0);
    guest.chain(0, &[Part::Read(b"after"), Part::Write(8)]);
    guest.publish(0);
    let mut chain = ring.pop_available().unwrap().unwrap();
    assert_eq!(chain.id().epoch(), next);
    assert_eq!(request(&mut chain), b"after");
    ring.complete(chain, UsedLength::new(0)).unwrap();
    assert_eq!(guest.used(), Some((0, 0)));
}

/// The queue's own shape is checked when it is made ready, and a driver that
/// breaks the ring's rules afterwards stops the queue, not the device.
#[test]
fn a_queue_the_driver_set_up_wrongly_is_refused() {
    // Limits that do not fit each other.
    let memory = window(4096);
    let size = |n| QueueSize::new(n).unwrap();
    assert!(DeviceRing::new(memory, size(128), 4).is_none());
    assert!(DeviceRing::new(memory, size(8), 1).is_none());
    assert!(DeviceRing::new(memory, size(8), 16).is_none());

    let (guest, mut ring) = Guest::unready(8, 4);
    // Smaller than the longest chain.
    assert_eq!(
        ring.configure(size(2)).err(),
        Some(QueueError::Transport(RingError::ChainLimitExceedsQueue))
    );
    for wrong in [
        RingAddresses {
            descriptors: 8,
            ..guest.at
        },
        RingAddresses {
            available: 0x801,
            ..guest.at
        },
        RingAddresses {
            used: 0x1002,
            ..guest.at
        },
        RingAddresses {
            descriptors: WINDOW as u64 - 16,
            ..guest.at
        },
        RingAddresses {
            used: WINDOW as u64 - 8,
            ..guest.at
        },
        RingAddresses {
            available: u64::MAX - 1,
            ..guest.at
        },
    ] {
        ring.set_addresses(wrong).unwrap();
        assert_eq!(
            ring.set_ready(true).err(),
            Some(QueueError::Transport(RingError::Layout)),
            "{wrong:?}"
        );
        assert!(!ring.state().ready());
    }
    ring.set_addresses(guest.at).unwrap();
    ring.set_ready(true).unwrap();
    // Ready: its shape is fixed until it is made unready or reset.
    assert_eq!(
        ring.set_addresses(guest.at).err(),
        Some(QueueError::Transport(RingError::Busy))
    );
    assert_eq!(
        ring.configure(size(8)).err(),
        Some(QueueError::Transport(RingError::Busy))
    );

    // An available index further ahead than the queue is long.
    guest.memory.store_u16(guest.at.available + 2, 9).unwrap();
    assert_eq!(
        ring.pop_available().map(|chain| chain.is_some()),
        Err(QueueError::Transport(RingError::AvailableIndex))
    );

    // A head published twice while the device holds it.
    let (mut guest, mut ring) = Guest::ready(8, 4);
    guest.chain(0, &[Part::Read(b"once"), Part::Write(4)]);
    guest.publish(0);
    guest.publish(0);
    let held = ring.pop_available().unwrap().unwrap();
    assert_eq!(
        ring.pop_available().map(|chain| chain.is_some()),
        Err(QueueError::Transport(RingError::HeadInFlight))
    );
    drop(held);
}

/// The ring's indices are sixteen bits and wrap; the queue goes on across
/// the wrap, slot by slot.
#[test]
fn the_indices_wrap_and_the_queue_goes_on() {
    let (mut guest, mut ring) = Guest::ready(2, 2);
    let places = guest.chain(0, &[Part::Read(b"ping"), Part::Write(4)]);
    for round in 0..70_000u32 {
        guest.publish(0);
        let mut chain = ring.pop_available().unwrap().expect("published");
        let (_, _, sink) = chain.io().unwrap().into_parts();
        sink.write_at(0, &round.to_le_bytes()).unwrap();
        ring.complete(chain, UsedLength::new(4)).unwrap();
        assert_eq!(guest.used(), Some((0, 4)), "round {round}");
        assert_eq!(guest.read(places[1], 4), round.to_le_bytes());
    }
    assert!(ring.pop_available().unwrap().is_none());
}
