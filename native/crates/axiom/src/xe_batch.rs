//! The admission window of a batched Xe executor: which prepared
//! instructions (and MoE wave members) are admitted, where each one's operands
//! and output live in resident scratch, and which of them one ordered
//! submission carries.
//!
//! This is bookkeeping only. It holds no graph borrow, performs no copy and
//! issues no call: the executor stages bytes where [`BatchArena`] places
//! them, builds each member's bindings from its [`Placement`]s, and submits
//! the staged members as one ordered sequence. A member reads a pending
//! producer's output where that producer writes it, so a dependent run of
//! instructions never round-trips through the host.
//!
//! Ordering, the property everything here serves:
//! - members are admitted, submitted and retired in one order, the order the
//!   prepared engine offered them;
//! - a member reading a producer that is staged in the same submission
//!   carries a barrier, so the device finishes the producer's writes first;
//!   a producer submitted earlier is ordered by the same queue's job-end
//!   visibility fence, even if the CPU has not yet observed its completion;
//! - scratch is only reclaimed once every member has retired, so no admitted
//!   member's region is reused while the device or the host may still read it.

use crate::xe_graph::{Kernel, MAX_PORTS, Port};

/// Members one ordered submission may carry, on every lane: the accelerator
/// protocol's sequence ceiling (`MAX_ACCEL_SEQUENCE_ENTRIES`). Each
/// submission costs its client a submit, a poll and a retirement round trip
/// to the service, so a window should end where its work does, not at a
/// count below the table. The provider's command window and the scratch
/// arena still bound it at admission ([`Window::fits`],
/// [`BatchArena::allocate`]): a window that runs out of either submits what
/// it holds.
pub const WINDOW: usize = kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES;

/// Scratch placement granularity: every region starts on a cache line and
/// ends before the next one begins, so CPU publication and device writes of
/// different members never share a line.
pub const ALIGN: u64 = 64;

/// A bump allocator over one resident scratch region, `[0, capacity)`.
/// Regions are never freed individually: the whole arena resets once the
/// window drains.
#[derive(Clone, Copy, Debug)]
pub struct BatchArena {
    capacity: u64,
    top: u64,
}

impl BatchArena {
    pub const fn new(capacity: u64) -> Self {
        Self { capacity, top: 0 }
    }

    /// The offset of `bytes` fresh bytes, or `None` when they do not fit.
    pub fn allocate(&mut self, bytes: u64) -> Option<u64> {
        let start = self.top.checked_add(ALIGN - 1)? / ALIGN * ALIGN;
        let end = start.checked_add(bytes.max(1))?;
        if end > self.capacity {
            return None;
        }
        self.top = end;
        Some(start)
    }

    /// Where the arena stands, to undo a partial admission.
    pub const fn mark(&self) -> u64 {
        self.top
    }

    /// Undo every allocation made since `mark`.
    pub fn rewind(&mut self, mark: u64) {
        debug_assert!(mark <= self.top);
        self.top = mark;
    }

    pub fn reset(&mut self) {
        self.top = 0;
    }

    pub const fn used(&self) -> u64 {
        self.top
    }
}

/// Which prepared-engine admission a member answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKey {
    /// A pure instruction offered through `advance_chained`.
    Instruction(usize),
    /// Member `part` of a MoE instruction's expert wave `wave`: its grouped
    /// up and down launches, then its combine, admitted together.
    Wave {
        instruction: usize,
        wave: usize,
        part: u8,
    },
}

/// Where the device finds one port's bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// An already-imported buffer (resident weights), at this byte offset.
    Resident { buffer: u64, offset: u64 },
    /// Resident scratch at this byte offset.
    Scratch { offset: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberState {
    /// Admitted and placed; not yet part of a submission.
    Staged,
    /// Carried by the submission in flight.
    Submitted,
    /// Its submission completed; its output awaits the host's copy.
    Done,
}

/// One admitted member: its package, program, where every port lives, and
/// the part of its output region the host copies back.
#[derive(Clone, Copy, Debug)]
pub struct Member {
    pub key: MemberKey,
    pub kernel: Kernel,
    pub program: u64,
    pub ports: [Placement; MAX_PORTS],
    /// Scratch offset of the output region, and the bytes the host reads
    /// back (a batch of fewer rows than its package reads back only those).
    pub output: u64,
    pub output_bytes: u64,
    /// Successor states, in scratch or an independently retained import.
    /// Only scratch placements need a host copy at retirement.
    pub recurrent: Option<(Placement, u64)>,
    pub barrier: bool,
    pub state: MemberState,
}

impl Member {
    /// The scratch region of this member's output, for a consumer reading
    /// `elements` F32 values starting `offset` values in.
    pub fn output_view(&self, offset: usize, elements: usize) -> Option<Placement> {
        let start = (offset as u64).checked_mul(4)?;
        let end = start.checked_add((elements as u64).checked_mul(4)?)?;
        (end <= self.output_bytes).then(|| Placement::Scratch {
            offset: self.output + start,
        })
    }
}

/// Why a member could not be admitted. Every refusal leaves the window and
/// the arena exactly as they were: the engine then drains and retries, or
/// runs the instruction natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The window already holds [`WINDOW`] members.
    Full,
    /// A producer the member reads is not admitted here.
    UnknownProducer,
    /// A pending operand's range falls outside its producer's output.
    ProducerRange,
}

/// The provider's command window for one ordered sequence: `COMMAND_PAGES`
/// pages in drivers/xe/src/accel.rs, which this must not exceed. The
/// encoder lays out the commands, binding tables and push blocks, then the
/// ISA of every distinct package the sequence names and a surface state per
/// binding; a sequence that does not fit is refused (`OutOfBounds`, naming
/// nothing). The lane bounds a window by the same accounting, from above.
pub const SEQUENCE_WINDOW_BYTES: u64 = 128 * 4096;
/// The window's fixed cost: the prologue and the page and line alignments
/// between its regions, rounded up.
const WINDOW_FIXED_BYTES: u64 = 2 * 4096;
/// A member's cost beyond its package: a barrier and a walker, its binding
/// table, and a surface state per binding and one more, rounded up.
const MEMBER_BYTES: u64 = 1024;

/// Admitted members, oldest first, and the arena their scratch comes from.
#[derive(Debug)]
pub struct Window {
    members: [Option<Member>; WINDOW],
    len: usize,
    pub arena: BatchArena,
    /// The distinct programs admitted, with their packages' bytes, and the
    /// window's estimated use of the provider's command window.
    packages: [(u64, u64); WINDOW],
    distinct: usize,
    bytes: u64,
}

impl Window {
    pub const fn new(capacity: u64) -> Self {
        Self {
            members: [None; WINDOW],
            len: 0,
            arena: BatchArena::new(capacity),
            packages: [(0, 0); WINDOW],
            distinct: 0,
            bytes: WINDOW_FIXED_BYTES,
        }
    }

    /// The window's estimated use of the provider's command window.
    pub const fn sequence_bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether members of `programs` (each with its package's bytes) would
    /// all keep the sequence within the provider's window. A program
    /// repeated among them, or already admitted, is charged once.
    pub fn fits_all(&self, programs: &[(u64, u64)]) -> bool {
        let mut bytes = self.bytes;
        let mut added = [0u64; WINDOW];
        let mut count = 0;
        for &(program, package_bytes) in programs {
            let known = self.packages[..self.distinct]
                .iter()
                .any(|(known, _)| *known == program)
                || added[..count].contains(&program);
            if !known {
                bytes = bytes.saturating_add(package_bytes.div_ceil(64) * 64);
                if count < WINDOW {
                    added[count] = program;
                    count += 1;
                }
            }
            bytes = bytes.saturating_add(MEMBER_BYTES);
        }
        bytes <= SEQUENCE_WINDOW_BYTES
    }

    /// [`Self::fits_all`] for one member.
    pub fn fits(&self, program: u64, package_bytes: u64) -> bool {
        self.fits_all(&[(program, package_bytes)])
    }

    pub(crate) fn charge(&mut self, program: u64, package_bytes: u64) {
        if !self.packages[..self.distinct]
            .iter()
            .any(|(known, _)| *known == program)
        {
            self.packages[self.distinct] = (program, package_bytes);
            self.distinct += 1;
            self.bytes = self.bytes.saturating_add(package_bytes.div_ceil(64) * 64);
        }
        self.bytes = self.bytes.saturating_add(MEMBER_BYTES);
    }

    fn reset_layout(&mut self) {
        self.distinct = 0;
        self.bytes = WINDOW_FIXED_BYTES;
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn is_full(&self) -> bool {
        self.len == WINDOW
    }

    pub fn members(&self) -> impl Iterator<Item = &Member> {
        self.members[..self.len].iter().flatten()
    }

    pub fn find(&self, key: MemberKey) -> Option<(usize, &Member)> {
        self.members[..self.len]
            .iter()
            .enumerate()
            .find_map(|(index, member)| {
                member.as_ref().filter(|m| m.key == key).map(|m| (index, m))
            })
    }

    /// The member producing instruction `instruction`'s output.
    pub fn producer(&self, instruction: usize) -> Option<&Member> {
        self.members().find(|member| match member.key {
            MemberKey::Instruction(index) => index == instruction,
            // Only the combine publishes a wave output. Axiom offers this
            // instruction as a dependency only for its final pending wave.
            MemberKey::Wave {
                instruction: index, ..
            } => index == instruction && member.output_bytes != 0,
        })
    }

    /// Admit `member` at the back. Its barrier is set here: it needs one
    /// exactly when one of its reads names a member still staged, which the
    /// caller reports through `reads_staged`.
    pub fn admit(
        &mut self,
        mut member: Member,
        reads_staged: bool,
        package_bytes: u64,
    ) -> Result<(), Refusal> {
        if self.is_full() || !self.fits(member.program, package_bytes) {
            return Err(Refusal::Full);
        }
        self.charge(member.program, package_bytes);
        member.barrier = reads_staged;
        member.state = MemberState::Staged;
        self.members[self.len] = Some(member);
        self.len += 1;
        Ok(())
    }

    /// The staged members, in order, that the next submission carries.
    pub fn staged(&self) -> impl Iterator<Item = &Member> {
        self.members().filter(|m| m.state == MemberState::Staged)
    }

    /// Whether a submission is in flight.
    pub fn in_flight(&self) -> bool {
        self.members().any(|m| m.state == MemberState::Submitted)
    }

    /// Mark every staged member submitted (one sequence now carries them).
    /// The first of them needs no barrier: nothing earlier shares its
    /// submission.
    pub fn submit_staged(&mut self) -> usize {
        self.submit_first(WINDOW)
    }

    /// Mark the first `limit` staged members submitted; later ones wait for
    /// a submission of their own. A member whose producer went in an earlier
    /// submission keeps its barrier, which is then merely redundant.
    pub fn submit_first(&mut self, limit: usize) -> usize {
        let mut first = true;
        let mut count = 0;
        for member in self.members[..self.len].iter_mut().flatten() {
            if count == limit {
                break;
            }
            if member.state == MemberState::Staged {
                if first {
                    member.barrier = false;
                    first = false;
                }
                member.state = MemberState::Submitted;
                count += 1;
            }
        }
        // Only staged members occupy the next command window. Commands of
        // earlier submissions live in separate provider slots.
        let packages = self.packages;
        let distinct = self.distinct;
        self.reset_layout();
        for index in 0..self.len {
            if let Some(member) = self.members[index]
                && member.state == MemberState::Staged
            {
                let bytes = packages[..distinct]
                    .iter()
                    .find(|(program, _)| *program == member.program)
                    .expect("every staged member retains its package charge")
                    .1;
                self.charge(member.program, bytes);
            }
        }
        count
    }

    /// The submission in flight completed.
    pub fn complete_submitted(&mut self) {
        self.complete_first(WINDOW);
    }

    /// Complete only the oldest submission's members. Later queued work
    /// retains its own completion and cannot be published by this event.
    pub fn complete_first(&mut self, mut count: usize) {
        for member in self.members[..self.len].iter_mut().flatten() {
            if member.state == MemberState::Submitted && count > 0 {
                member.state = MemberState::Done;
                count -= 1;
            }
        }
    }

    /// Remove the oldest member once it is Done; the arena resets when the
    /// window drains. Returns it for the host's copy.
    pub fn retire_front(&mut self, key: MemberKey) -> Option<Member> {
        let front = self.members[0]?;
        if front.key != key || front.state != MemberState::Done {
            return None;
        }
        self.members.copy_within(1..self.len, 0);
        self.len -= 1;
        self.members[self.len] = None;
        if self.len == 0 {
            self.arena.reset();
            self.reset_layout();
        }
        Some(front)
    }

    /// Forget every member that was never submitted (an abandoned
    /// invocation's staging). Refuses while a submission is in flight.
    pub fn discard_staged(&mut self) -> bool {
        if self.in_flight() {
            return false;
        }
        self.members = [None; WINDOW];
        self.len = 0;
        self.arena.reset();
        self.reset_layout();
        true
    }

    /// Where a pending operand of `producer` lives, and whether reading it
    /// needs a barrier (its producer is still staged, not yet submitted).
    pub fn chained(
        &self,
        producer: usize,
        offset: usize,
        elements: usize,
    ) -> Result<(Placement, bool), Refusal> {
        let member = self.producer(producer).ok_or(Refusal::UnknownProducer)?;
        let placement = member
            .output_view(offset, elements)
            .ok_or(Refusal::ProducerRange)?;
        Ok((placement, member.state == MemberState::Staged))
    }
}

/// Resolve a member's placements into `(buffer, offset)` pairs for its
/// package bindings: scratch placements name the scratch import, and the
/// ones run lives at `ones` within it.
pub fn place(member: &Member, scratch: u64, ones: u64) -> impl FnMut(Port) -> (u64, u64) + '_ {
    let mut slot = 0usize;
    move |port| {
        let placement = member.ports[slot];
        slot += 1;
        match (port, placement) {
            (Port::Ones(_), _) => (scratch, ones),
            (_, Placement::Resident { buffer, offset }) => (buffer, offset),
            (_, Placement::Scratch { offset }) => (scratch, offset),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use axnn::lowering::{KernelAccuracy, KernelRequest};
    use axnn::{BinaryOp, Shape, TensorDesc};

    fn add_kernel(elements: usize) -> Kernel {
        let desc = TensorDesc::try_new(Shape::from([elements]), axnn::DType::F32).unwrap();
        let request = KernelRequest::binary(BinaryOp::Add, &desc, &desc).unwrap();
        Kernel::select(&request, KernelAccuracy::Approximate).unwrap()
    }

    fn member(window: &mut Window, instruction: usize, elements: usize) -> Member {
        let kernel = add_kernel(elements);
        let output = window.arena.allocate(kernel.output_bytes()).unwrap();
        Member {
            key: MemberKey::Instruction(instruction),
            kernel,
            program: 1,
            ports: [Placement::Scratch { offset: 0 }; MAX_PORTS],
            output,
            output_bytes: kernel.output_bytes(),
            recurrent: None,
            barrier: false,
            state: MemberState::Staged,
        }
    }

    #[test]
    fn wave_consumers_bind_only_the_combined_output_and_keep_it_alive() {
        use kore_xe_kernels::MoeWave;
        let mut window = Window::new(1 << 20);
        let region_bytes = 64 << 10;
        let region = window.arena.allocate(region_bytes).unwrap();
        let output_bytes = (crate::xe_moe::MOE_HIDDEN * 4) as u64;
        let output = region + region_bytes - output_bytes;
        let key = MemberKey::Wave {
            instruction: 2,
            wave: 0,
            part: 2,
        };
        let combine = Member {
            key,
            kernel: Kernel::moe_wave(MoeWave::Combine, 1, region_bytes, [4096; 5]).unwrap(),
            program: 1,
            ports: [Placement::Scratch { offset: region }; MAX_PORTS],
            output,
            output_bytes,
            recurrent: None,
            barrier: false,
            state: MemberState::Staged,
        };
        window.admit(combine, false, 4096).unwrap();
        assert_eq!(
            window.chained(2, 0, crate::xe_moe::MOE_HIDDEN),
            Ok((Placement::Scratch { offset: output }, true))
        );
        assert_eq!(
            window.chained(2, crate::xe_moe::MOE_HIDDEN, 1),
            Err(Refusal::ProducerRange)
        );
        assert_eq!(
            window.chained(2, usize::MAX, 1),
            Err(Refusal::ProducerRange)
        );
        let consumer = member(&mut window, 3, crate::xe_moe::MOE_HIDDEN);
        window.admit(consumer, true, 4096).unwrap();
        // Separate submissions preserve the producer region until the
        // consumer retires; one completed event cannot free the later one.
        assert_eq!(window.submit_first(1), 1);
        assert_eq!(
            window.chained(2, 0, 1),
            Ok((Placement::Scratch { offset: output }, false))
        );
        assert_eq!(window.submit_first(1), 1);
        window.complete_first(1);
        assert!(window.retire_front(key).is_some());
        assert!(window.arena.used() >= region + region_bytes);
        assert!(window.retire_front(consumer.key).is_none());
        window.complete_first(1);
        assert!(window.retire_front(consumer.key).is_some());
        assert_eq!(window.arena.used(), 0);
    }

    #[test]
    fn the_arena_aligns_rewinds_and_refuses_what_does_not_fit() {
        let mut arena = BatchArena::new(256);
        assert_eq!(arena.allocate(4), Some(0));
        assert_eq!(arena.allocate(100), Some(64));
        let mark = arena.mark();
        assert_eq!(arena.allocate(64), Some(192));
        assert_eq!(arena.allocate(1), None, "past the capacity");
        arena.rewind(mark);
        assert_eq!(arena.used(), 164);
        arena.reset();
        assert_eq!(arena.allocate(256), Some(0));
    }

    /// A limited submission takes the oldest staged members; the rest wait
    /// for a submission of their own, first among them without a barrier.
    #[test]
    fn a_limited_submission_takes_the_oldest_staged_members() {
        let mut window = Window::new(1 << 20);
        for instruction in 0..3 {
            let m = member(&mut window, instruction, 16);
            window.admit(m, instruction > 0, 4096).unwrap();
        }
        assert_eq!(window.submit_first(1), 1);
        assert_eq!(window.staged().count(), 2);
        assert!(!window.find(MemberKey::Instruction(0)).unwrap().1.barrier);
        window.complete_submitted();
        assert_eq!(window.submit_first(1), 1);
        assert!(!window.find(MemberKey::Instruction(1)).unwrap().1.barrier);
        assert!(window.find(MemberKey::Instruction(2)).unwrap().1.barrier);
        assert_eq!(
            window.find(MemberKey::Instruction(2)).unwrap().1.state,
            MemberState::Staged
        );
    }

    #[test]
    fn queued_consumers_keep_scratch_and_only_their_own_event_publishes_them() {
        let mut window = Window::new(1 << 20);
        let a = member(&mut window, 0, 16);
        window.admit(a, false, 4096).unwrap();
        let command_bytes = window.sequence_bytes();
        assert_eq!(window.submit_first(1), 1);
        assert_eq!(window.sequence_bytes(), WINDOW_FIXED_BYTES);
        let b = member(&mut window, 1, 16);
        window.admit(b, false, 4096).unwrap();
        assert_eq!(window.sequence_bytes(), command_bytes);
        assert_eq!(window.submit_first(1), 1);
        window.complete_first(1);
        assert_eq!(window.find(b.key).unwrap().1.state, MemberState::Submitted);
        let scratch_used = window.arena.used();
        assert!(window.retire_front(a.key).is_some());
        assert_eq!(window.arena.used(), scratch_used);
        assert!(window.retire_front(b.key).is_none());
        window.complete_first(1);
        assert!(window.retire_front(b.key).is_some());
        assert_eq!(window.arena.used(), 0);
    }

    /// A member reading a staged producer carries a barrier; one reading a
    /// producer already submitted does not; the first member of a
    /// submission never does. Members retire only in order, and the arena
    /// resets when the window drains.
    #[test]
    fn barriers_follow_staged_producers_and_members_retire_in_order() {
        let mut window = Window::new(1 << 20);
        let a = member(&mut window, 3, 16);
        window.admit(a, false, 4096).unwrap();
        let (view, staged) = window.chained(3, 4, 8).unwrap();
        assert_eq!(
            view,
            Placement::Scratch {
                offset: a.output + 16
            }
        );
        assert!(staged);
        assert_eq!(window.chained(3, 12, 8), Err(Refusal::ProducerRange));
        assert_eq!(window.chained(9, 0, 1), Err(Refusal::UnknownProducer));
        let b = member(&mut window, 4, 16);
        window.admit(b, true, 4096).unwrap();
        assert!(window.find(MemberKey::Instruction(4)).unwrap().1.barrier);
        assert_eq!(window.submit_staged(), 2);
        assert!(window.in_flight());
        // Admitted while the first submission flies: its producer is
        // submitted, so the next sequence needs no barrier for it.
        let (_, staged) = window.chained(4, 0, 16).unwrap();
        assert!(!staged);
        let c = member(&mut window, 5, 16);
        window.admit(c, true, 4096).unwrap();
        assert_eq!(window.staged().count(), 1);
        assert!(
            window.retire_front(MemberKey::Instruction(3)).is_none(),
            "not done yet"
        );
        window.complete_submitted();
        assert!(
            window.retire_front(MemberKey::Instruction(4)).is_none(),
            "out of order"
        );
        assert_eq!(
            window.retire_front(MemberKey::Instruction(3)).unwrap().key,
            a.key
        );
        assert_eq!(
            window.retire_front(MemberKey::Instruction(4)).unwrap().key,
            b.key
        );
        // The last member alone in its submission: its barrier is dropped.
        assert_eq!(window.submit_staged(), 1);
        assert!(!window.find(MemberKey::Instruction(5)).unwrap().1.barrier);
        assert!(!window.discard_staged(), "a submission is in flight");
        window.complete_submitted();
        assert!(window.arena.used() > 0);
        window.retire_front(MemberKey::Instruction(5)).unwrap();
        assert!(window.is_empty());
        assert_eq!(window.arena.used(), 0, "a drained window resets its arena");
        assert_eq!(
            window.sequence_bytes(),
            WINDOW_FIXED_BYTES,
            "a drained window forgets its packages"
        );
    }

    /// A window drains before its sequence would outgrow the provider's
    /// command window: each distinct program is charged its package once, a
    /// repeated one only its member's share.
    #[test]
    fn a_window_charges_each_distinct_package_once_against_the_provider_window() {
        let mut window = Window::new(1 << 20);
        let package = 90 * 1024;
        for program in 1..=5u64 {
            let mut m = member(&mut window, program as usize, 16);
            m.program = program;
            window.admit(m, false, package).unwrap();
        }
        // Five distinct packages and their members, under 512 KiB with the
        // fixed cost; a sixth distinct one would not fit, while a repeat of
        // the first costs only its member.
        assert_eq!(
            window.sequence_bytes(),
            WINDOW_FIXED_BYTES + 5 * (package + MEMBER_BYTES)
        );
        assert!(window.fits(1, package));
        assert!(!window.fits(6, package));
        assert!(!window.fits_all(&[(1, package), (6, package)]));
        assert!(window.fits_all(&[(1, package), (2, package), (1, package)]));
        let mut repeat = member(&mut window, 6, 16);
        repeat.program = 1;
        window.admit(repeat, false, package).unwrap();
        let mut sixth = member(&mut window, 7, 16);
        sixth.program = 6;
        assert_eq!(window.admit(sixth, false, package), Err(Refusal::Full));
        assert!(window.discard_staged());
        assert!(
            window.fits(6, package),
            "a drained window forgets its packages"
        );
    }

    /// A group RMS member reading a staged producer binds, slot by slot, the
    /// producer's output region, the ones run, its resident weight, and its
    /// own output region.
    #[test]
    fn a_chained_member_binds_each_package_slot_where_it_lives() {
        let mut window = Window::new(1 << 20);
        let producer = member(&mut window, 7, 8 * 4096);
        window.admit(producer, false, 4096).unwrap();
        let tensor = |dims: &[usize]| {
            TensorDesc::try_new(Shape::try_from(dims).unwrap(), axnn::DType::F32).unwrap()
        };
        let request =
            KernelRequest::rms_norm_over(&tensor(&[8, 8, 512]), &tensor(&[8, 512]), 1e-5, 2)
                .unwrap();
        let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        let (input, staged) = window.chained(7, 0, 8 * 4096).unwrap();
        assert!(staged);
        let output = window.arena.allocate(kernel.output_bytes()).unwrap();
        let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
        ports[0] = input;
        ports[2] = Placement::Resident {
            buffer: 42,
            offset: 4096 + 64,
        };
        ports[3] = Placement::Scratch { offset: output };
        let rms = Member {
            key: MemberKey::Instruction(8),
            kernel,
            program: 2,
            ports,
            output,
            output_bytes: kernel.output_bytes(),
            recurrent: None,
            barrier: false,
            state: MemberState::Staged,
        };
        window.admit(rms, staged, 4096).unwrap();
        let ones = 1 << 21;
        let bindings = kernel.port_bindings(place(&rms, 9, ones)).unwrap();
        let slots: Vec<_> = bindings
            .iter()
            .map(|b| (b.buffer, b.offset, b.len, b.slot))
            .collect();
        assert_eq!(
            slots,
            [
                (9, producer.output, 8 * 4096 * 4, 0),
                (9, ones, 8 * 4096 * 4, 1),
                (42, 4096 + 64, 4096 * 4, 2),
                (9, output, 8 * 4096 * 4, 3),
            ]
        );
        assert!(window.find(MemberKey::Instruction(8)).unwrap().1.barrier);
    }

    #[test]
    fn a_full_window_refuses_without_changing_anything() {
        let mut window = Window::new(1 << 20);
        for instruction in 0..WINDOW {
            let m = member(&mut window, instruction, 4);
            window.admit(m, false, 4096).unwrap();
        }
        let used = window.arena.used();
        let extra = Member {
            key: MemberKey::Wave {
                instruction: 99,
                wave: 0,
                part: 0,
            },
            ..*window.members().next().unwrap()
        };
        assert_eq!(window.admit(extra, false, 4096), Err(Refusal::Full));
        assert_eq!(window.len(), WINDOW);
        assert_eq!(window.arena.used(), used);
    }
}
