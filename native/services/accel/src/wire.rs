//! Fail-closed codecs over [`kore_abi::svc`]'s frozen accel forms.
//!
//! Every decode is **byte-exact fail-closed**: after the structural
//! lift, the decoded value is re-encoded and compared to the incoming
//! record — any stray bit (a set reserve, a non-zero unused per-verb
//! word, a live byte in a dead binding slot, or target words outside
//! `LoadProgram`/`Submit`/`SubmitLaunch`) refuses. The peer wrote garbage or a future revision,
//! and a wrong guess would be a protocol break, not a convenience.
//!
//! Two types speak this wire: [`AccelWire`], the backend-free protocol
//! marker a client names (`SvcClient<…, AccelWire>` — a client face
//! must not import a hardware backend to talk to a row), and every
//! [`AccelService<A>`] host instantiation.  Both delegate every codec
//! and const to the same free functions, so byte agreement across all
//! backends holds **by construction**, not by test (the swap tier's
//! delegation discipline, one step stricter).

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::session::SlotRef;
use kore_abi::svc::{
    AccelBindingWire, AccelRepWire, AccelReqWire, AccelSequenceEntryWire, AccelSequenceSyncWire,
    MAX_ACCEL_BINDINGS,
};
use zerocopy::FromBytes;
#[allow(deprecated)]
use kore_driver::accel::Dispatch;
use kore_driver::accel::LaunchShape;
use kore_driver::accel::NativeAccelerator;
use kore_driver::accel::Workgroups;
use kore_rt::svc::SvcProtocol;
use virtio_accel_core::{AccessMode, DeviceLimits};
use zerocopy::little_endian::{U32, U64};

use crate::service::{
    AccelBinding, AccelBindings, AccelEventState, AccelInfo, AccelRefusal, AccelRequest,
    AccelResponse, AccelSequenceEntry, AccelService, AccelSync,
};

const EMPTY_BINDING_WIRE: AccelBindingWire = AccelBindingWire {
    buffer: U64::ZERO,
    offset: U64::ZERO,
    len: U64::ZERO,
    slot: U32::ZERO,
    access: 0,
    reserved: [0; 3],
};

/// Encode one staged ordered-submission member.
///
/// The artifact-defined launch is the all-zero geometry — never a magic
/// `1x1x1`, which stays an explicit one-work-item launch — and explicit
/// workgroups ride the same count/local order as `SubmitLaunch`'s `target`.
pub fn encode_sequence_entry(entry: AccelSequenceEntry) -> AccelSequenceEntryWire {
    let (groups, local) = match entry.launch {
        LaunchShape::ArtifactDefined => ([0; 3], [0; 3]),
        LaunchShape::Workgroups(workgroups) => (workgroups.group_count(), workgroups.local_size()),
    };
    let flag = |set: bool, bit: u32| if set { bit } else { 0 };
    let mut wire = AccelSequenceEntryWire {
        program: U64::new(entry.program),
        binding_count: U32::new(entry.bindings.len() as u32),
        flags: U32::new(
            flag(entry.barrier, AccelSequenceEntryWire::FLAG_BARRIER)
                | flag(entry.gate.is_some(), AccelSequenceEntryWire::FLAG_GATE)
                | flag(entry.signal.is_some(), AccelSequenceEntryWire::FLAG_SIGNAL),
        ),
        groups: groups.map(U32::new),
        local: local.map(U32::new),
        bindings: [EMPTY_BINDING_WIRE; MAX_ACCEL_BINDINGS],
    };
    for (dst, binding) in wire.bindings.iter_mut().zip(entry.bindings.iter()) {
        *dst = AccelBindingWire {
            buffer: U64::new(binding.buffer),
            offset: U64::new(binding.offset),
            len: U64::new(binding.len),
            slot: U32::new(binding.slot),
            access: binding.access as u8,
            reserved: [0; 3],
        };
    }
    wire
}

/// One member's sync records, in table order: its gate's, then its signal's.
pub fn encode_sequence_syncs(entry: &AccelSequenceEntry) -> impl Iterator<Item = AccelSequenceSyncWire> {
    [entry.gate, entry.signal].into_iter().flatten().map(|sync| AccelSequenceSyncWire {
        buffer: U64::new(sync.buffer),
        offset: U64::new(sync.offset),
        value: U64::new(sync.value),
        reserved: [0; 8],
    })
}

fn decode_sync(wire: &AccelSequenceSyncWire) -> Option<AccelSync> {
    (wire.reserved == [0; 8]).then_some(AccelSync {
        buffer: wire.buffer.get(),
        offset: wire.offset.get(),
        value: wire.value.get(),
    })
}

/// Decode one staged member with the same byte-exact reserved/dead-tail gate
/// as the request ring, taking its gate's and signal's records from `syncs`.
/// A reserved flag bit, a record with reserved bytes set or none left, or
/// geometry that is neither all zero nor a valid [`Workgroups`] (a zero axis
/// beside a nonzero one), refuses.
pub fn decode_sequence_entry<'a>(
    wire: &AccelSequenceEntryWire,
    syncs: &mut impl Iterator<Item = &'a AccelSequenceSyncWire>,
) -> Option<AccelSequenceEntry> {
    let count = usize::try_from(wire.binding_count.get()).ok()?;
    if count == 0 || count > MAX_ACCEL_BINDINGS {
        return None;
    }
    let flags = wire.flags.get();
    if flags & !AccelSequenceEntryWire::FLAGS != 0 {
        return None;
    }
    let mut take = |bit: u32| -> Option<Option<AccelSync>> {
        if flags & bit == 0 {
            return Some(None);
        }
        syncs.next().and_then(decode_sync).map(Some)
    };
    let gate = take(AccelSequenceEntryWire::FLAG_GATE)?;
    let signal = take(AccelSequenceEntryWire::FLAG_SIGNAL)?;
    let groups = wire.groups.map(|axis| axis.get());
    let local = wire.local.map(|axis| axis.get());
    let launch = if groups == [0; 3] && local == [0; 3] {
        LaunchShape::ArtifactDefined
    } else {
        LaunchShape::Workgroups(Workgroups::new(groups, local).ok()?)
    };
    let mut live = heapless::Vec::<AccelBinding, MAX_ACCEL_BINDINGS>::new();
    for source in wire.bindings.iter().take(count) {
        live.push(AccelBinding {
            buffer: source.buffer.get(),
            offset: source.offset.get(),
            len: source.len.get(),
            slot: source.slot.get(),
            access: decode_access(source.access)?,
        })
        .ok()?;
    }
    let entry = AccelSequenceEntry {
        program: wire.program.get(),
        bindings: AccelBindings::new(live.as_slice())?,
        launch,
        barrier: flags & AccelSequenceEntryWire::FLAG_BARRIER != 0,
        gate,
        signal,
    };
    (encode_sequence_entry(entry) == *wire).then_some(entry)
}

/// A staged table's `count` members: the entries, then exactly the sync
/// records their flags name, no more (`AccelReqWire::OP_SUBMIT_SEQUENCE`).
pub fn decode_sequence(bytes: &[u8], count: usize) -> Result<alloc::vec::Vec<AccelSequenceEntry>, AccelRefusal> {
    if count == 0 || count > kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES {
        return Err(AccelRefusal::InvalidArgument);
    }
    let entry_bytes = core::mem::size_of::<AccelSequenceEntryWire>();
    let sync_bytes = core::mem::size_of::<AccelSequenceSyncWire>();
    let table = count.checked_mul(entry_bytes).ok_or(AccelRefusal::InvalidArgument)?;
    let (entries, tail) = bytes.split_at_checked(table).ok_or(AccelRefusal::InvalidArgument)?;
    let entries: alloc::vec::Vec<AccelSequenceEntryWire> = entries
        .chunks_exact(entry_bytes)
        .map(|chunk| AccelSequenceEntryWire::read_from_bytes(chunk).map_err(|_| AccelRefusal::InvalidArgument))
        .collect::<Result<_, _>>()?;
    let records: usize = entries.iter().map(AccelSequenceEntryWire::sync_records).sum();
    if records > kore_abi::svc::MAX_ACCEL_SEQUENCE_SYNCS || tail.len() != records * sync_bytes {
        return Err(AccelRefusal::InvalidArgument);
    }
    let syncs: alloc::vec::Vec<AccelSequenceSyncWire> = tail
        .chunks_exact(sync_bytes)
        .map(|chunk| AccelSequenceSyncWire::read_from_bytes(chunk).map_err(|_| AccelRefusal::InvalidArgument))
        .collect::<Result<_, _>>()?;
    let mut syncs = syncs.iter();
    let mut decoded = alloc::vec::Vec::new();
    decoded.try_reserve_exact(count).map_err(|_| AccelRefusal::OutOfMemory)?;
    for wire in &entries {
        decoded.push(decode_sequence_entry(wire, &mut syncs).ok_or(AccelRefusal::InvalidArgument)?);
    }
    Ok(decoded)
}

/// Grant channels an accel client pre-pins — exactly one, carrying
/// [`AccelResponse::Signaled`]'s retained completion-channel recv.
pub const ACCEL_REPLY_GRANTS: usize = 1;
/// Init's fixed validation/diagnostic face.
pub const ACCEL_BOOT_PAIR: usize = 0;
/// Exec's fixed application-face administration pair.
pub const ACCEL_EXEC_PAIR: usize = 1;
/// Fixed founding pairs. Dynamic arrivals begin after boot and exec.
pub const ACCEL_SERVE_PAIRS: usize = ACCEL_EXEC_PAIR + 1;

/// Whether one runtime-authenticated fixed pair may administer recyclable
/// peers. Pair zero preserves the boot validation witness; pair one belongs
/// to exec. Generation-stamped dynamic arrivals can never satisfy this test.
pub const fn is_admin_pair(via: u16) -> bool {
    (via as usize) < ACCEL_SERVE_PAIRS
}
/// The backend-free wire marker — what a *client* names to speak the
/// protocol.  It can never be a host (no `SystemService` impl): a
/// consumer of the wire must not have to prove it could be one, and
/// must not import a hardware backend type to place a call.
pub struct AccelWire;

/// Lower a request to its wire form (total).
pub fn encode_req(req: &AccelRequest) -> AccelReqWire {
    let mut w = AccelReqWire {
        op: 0,
        reserved: [0; 3],
        flags: U32::ZERO,
        subject: U64::ZERO,
        a: U64::ZERO,
        b: U64::ZERO,
        c: U64::ZERO,
        d: U64::ZERO,
        target: [0; 48],
        bindings: [EMPTY_BINDING_WIRE; MAX_ACCEL_BINDINGS],
    };
    match *req {
        AccelRequest::AdmitPeer { part, announced } => {
            w.op = AccelReqWire::OP_ADMIT_PEER;
            w.a = U64::new(u64::from(announced.index()));
            w.b = U64::new(match part {
                kore_rt::svc::PeerBundlePart::Ring => 0,
                kore_rt::svc::PeerBundlePart::Kick => 1,
                kore_rt::svc::PeerBundlePart::ReturnRecv => 2,
                kore_rt::svc::PeerBundlePart::Grant(i) => 128 + u64::from(i),
            });
        }
        AccelRequest::TakePeerAdmission => w.op = AccelReqWire::OP_TAKE_PEER_ADMISSION,
        AccelRequest::ClosePeer => w.op = AccelReqWire::OP_CLOSE_PEER,
        AccelRequest::RetirePeer { peer } => {
            w.op = AccelReqWire::OP_RETIRE_PEER;
            w.subject = U64::new(u64::from(peer.arrival()));
        }
        AccelRequest::ConfirmRetired { peer } => {
            w.op = AccelReqWire::OP_CONFIRM_RETIRED;
            w.subject = U64::new(u64::from(peer.arrival()));
        }
        AccelRequest::AttachOwned { announced } => {
            w.op = AccelReqWire::OP_ATTACH_OWNED;
            w.a = U64::new(u64::from(announced.index()));
        }
        AccelRequest::DetachStaging => w.op = AccelReqWire::OP_DETACH_STAGING,
        AccelRequest::Attach { announced } => {
            w.op = AccelReqWire::OP_ATTACH;
            w.a = U64::new(u64::from(announced.index()));
        }
        AccelRequest::Query => {
            w.op = AccelReqWire::OP_QUERY;
        }
        AccelRequest::ProvisionMemory {
            announced,
            buffer_bytes,
            program_bytes,
        } => {
            w.op = AccelReqWire::OP_PROVISION_MEMORY;
            w.a = U64::new(u64::from(announced.index()));
            w.b = U64::new(buffer_bytes);
            w.c = U64::new(program_bytes);
        }
        AccelRequest::ReserveMemory {
            buffer_bytes,
            program_bytes,
        } => {
            w.op = AccelReqWire::OP_RESERVE_MEMORY;
            w.b = U64::new(buffer_bytes);
            w.c = U64::new(program_bytes);
        }
        AccelRequest::CreateContext { flags } => {
            w.op = AccelReqWire::OP_CREATE_CONTEXT;
            w.a = U64::new(u64::from(flags));
        }
        AccelRequest::DestroyContext { ctx } => {
            w.op = AccelReqWire::OP_DESTROY_CONTEXT;
            w.subject = U64::new(ctx);
        }
        AccelRequest::CreateBuffer {
            ctx,
            bytes,
            align,
            domain,
            usage,
        } => {
            w.op = AccelReqWire::OP_CREATE_BUFFER;
            w.subject = U64::new(ctx);
            w.a = U64::new(bytes);
            w.b = U64::new(align);
            w.c = U64::new(u64::from(domain) | (u64::from(usage) << 8));
        }
        AccelRequest::ImportBuffer {
            ctx,
            bytes,
            align,
            usage,
            announced,
        } => {
            w.op = AccelReqWire::OP_IMPORT_BUFFER;
            w.subject = U64::new(ctx);
            w.a = U64::new(bytes);
            w.b = U64::new(align);
            w.c = U64::new(u64::from(usage));
            w.d = U64::new(u64::from(announced.index()));
        }
        AccelRequest::ExportBuffer { buffer } => {
            w.op = AccelReqWire::OP_EXPORT_BUFFER;
            w.subject = U64::new(buffer);
        }
        AccelRequest::FreeBuffer { buffer } => {
            w.op = AccelReqWire::OP_FREE_BUFFER;
            w.subject = U64::new(buffer);
        }
        AccelRequest::WriteBuffer {
            buffer,
            offset,
            stage,
            len,
        } => {
            w.op = AccelReqWire::OP_WRITE_BUFFER;
            w.subject = U64::new(buffer);
            w.a = U64::new(offset);
            w.b = U64::new(stage);
            w.c = U64::new(len);
        }
        AccelRequest::ReadBuffer {
            buffer,
            offset,
            stage,
            len,
        } => {
            w.op = AccelReqWire::OP_READ_BUFFER;
            w.subject = U64::new(buffer);
            w.a = U64::new(offset);
            w.b = U64::new(stage);
            w.c = U64::new(len);
        }
        AccelRequest::ReadFailure { offset } => {
            w.op = AccelReqWire::OP_READ_FAILURE;
            w.a = U64::new(offset);
        }
        AccelRequest::ConfigureProfiling { ctx, enabled } => {
            w.op = AccelReqWire::OP_CONFIGURE_PROFILING;
            w.subject = U64::new(ctx);
            w.a = U64::new(u64::from(enabled));
        }
        AccelRequest::ReadTelemetry { offset } => {
            w.op = AccelReqWire::OP_READ_TELEMETRY;
            w.a = U64::new(offset);
        }
        AccelRequest::ReadLog { stage, len } => {
            w.op = AccelReqWire::OP_READ_LOG;
            w.b = U64::new(stage);
            w.c = U64::new(len);
        }
        AccelRequest::LoadProgram {
            ctx,
            stage,
            len,
            resident_bytes,
            format,
            target,
        } => {
            w.op = AccelReqWire::OP_LOAD_PROGRAM;
            w.subject = U64::new(ctx);
            w.a = U64::new(stage);
            w.b = U64::new(len);
            w.c = U64::new(resident_bytes);
            w.d = U64::new(u64::from(format));
            for (chunk, word) in w.target.chunks_exact_mut(4).zip(target) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
        }
        AccelRequest::UnloadProgram { program } => {
            w.op = AccelReqWire::OP_UNLOAD_PROGRAM;
            w.subject = U64::new(program);
        }
        AccelRequest::CreateQueue { ctx, flags } => {
            w.op = AccelReqWire::OP_CREATE_QUEUE;
            w.subject = U64::new(ctx);
            w.a = U64::new(u64::from(flags));
        }
        AccelRequest::DestroyQueue { queue } => {
            w.op = AccelReqWire::OP_DESTROY_QUEUE;
            w.subject = U64::new(queue);
        }
        #[allow(deprecated)]
        AccelRequest::Submit {
            queue,
            program,
            timeout_ns,
            bindings,
            dispatch,
        } => {
            w.op = AccelReqWire::OP_SUBMIT;
            w.subject = U64::new(queue);
            w.a = U64::new(program);
            w.b = U64::new(timeout_ns);
            w.c = U64::new(bindings.len() as u64);
            encode_dispatch(&mut w.target, dispatch);
            for (dst, binding) in w.bindings.iter_mut().zip(bindings.iter()) {
                *dst = AccelBindingWire {
                    buffer: U64::new(binding.buffer),
                    offset: U64::new(binding.offset),
                    len: U64::new(binding.len),
                    slot: U32::new(binding.slot),
                    access: binding.access as u8,
                    reserved: [0; 3],
                };
            }
        }
        AccelRequest::SubmitLaunch {
            queue,
            program,
            timeout_ns,
            bindings,
            launch,
        } => {
            match launch {
                LaunchShape::ArtifactDefined => {
                    w.op = AccelReqWire::OP_SUBMIT;
                    #[allow(deprecated)]
                    encode_dispatch(&mut w.target, Dispatch::scalar());
                }
                LaunchShape::Workgroups(workgroups) => {
                    w.op = AccelReqWire::OP_SUBMIT_LAUNCH;
                    encode_workgroups(&mut w.target, workgroups);
                }
            }
            w.subject = U64::new(queue);
            w.a = U64::new(program);
            w.b = U64::new(timeout_ns);
            w.c = U64::new(bindings.len() as u64);
            for (dst, binding) in w.bindings.iter_mut().zip(bindings.iter()) {
                *dst = AccelBindingWire {
                    buffer: U64::new(binding.buffer),
                    offset: U64::new(binding.offset),
                    len: U64::new(binding.len),
                    slot: U32::new(binding.slot),
                    access: binding.access as u8,
                    reserved: [0; 3],
                };
            }
        }
        AccelRequest::SubmitSequence {
            queue,
            stage,
            len,
            count,
            timeout_ns,
        } => {
            w.op = AccelReqWire::OP_SUBMIT_SEQUENCE;
            w.subject = U64::new(queue);
            w.a = U64::new(stage);
            w.b = U64::new(len);
            w.c = U64::new(u64::from(count));
            w.d = U64::new(timeout_ns);
        }
        AccelRequest::PollEvent { event } => {
            w.op = AccelReqWire::OP_POLL_EVENT;
            w.subject = U64::new(event);
        }
        AccelRequest::CancelEvent { event } => {
            w.op = AccelReqWire::OP_CANCEL_EVENT;
            w.subject = U64::new(event);
        }
        AccelRequest::DestroyEvent { event } => {
            w.op = AccelReqWire::OP_DESTROY_EVENT;
            w.subject = U64::new(event);
        }
        AccelRequest::Signal => {
            w.op = AccelReqWire::OP_SIGNAL;
        }
    }
    w
}

/// Lift a wire request — byte-exact fail-closed (see the module docs).
pub fn decode_req(w: &AccelReqWire) -> Option<AccelRequest> {
    let req = raw_decode_req(w)?;
    (encode_req(&req) == *w).then_some(req)
}

/// The structural lift: known op, in-vocabulary field values.  Stray
/// bits are the re-encode comparison's to catch.
fn raw_decode_req(w: &AccelReqWire) -> Option<AccelRequest> {
    Some(match w.op {
        AccelReqWire::OP_ADMIT_PEER => AccelRequest::AdmitPeer {
            announced: SlotRef::from_wire(u32::try_from(w.a.get()).ok()?),
            part: match w.b.get() {
                0 => kore_rt::svc::PeerBundlePart::Ring,
                1 => kore_rt::svc::PeerBundlePart::Kick,
                2 => kore_rt::svc::PeerBundlePart::ReturnRecv,
                128..=383 => kore_rt::svc::PeerBundlePart::Grant((w.b.get() - 128) as u8),
                _ => return None,
            },
        },
        AccelReqWire::OP_TAKE_PEER_ADMISSION => AccelRequest::TakePeerAdmission,
        AccelReqWire::OP_CLOSE_PEER => AccelRequest::ClosePeer,
        AccelReqWire::OP_RETIRE_PEER => AccelRequest::RetirePeer {
            peer: kore_rt::svc::PeerId::from_arrival(u16::try_from(w.subject.get()).ok()?)?,
        },
        AccelReqWire::OP_CONFIRM_RETIRED => AccelRequest::ConfirmRetired {
            peer: kore_rt::svc::PeerId::from_arrival(u16::try_from(w.subject.get()).ok()?)?,
        },
        AccelReqWire::OP_ATTACH_OWNED => AccelRequest::AttachOwned {
            announced: SlotRef::from_wire(u32::try_from(w.a.get()).ok()?),
        },
        AccelReqWire::OP_DETACH_STAGING => AccelRequest::DetachStaging,
        AccelReqWire::OP_PROVISION_MEMORY => AccelRequest::ProvisionMemory {
            announced: SlotRef::from_wire(u32::try_from(w.a.get()).ok()?),
            buffer_bytes: w.b.get(),
            program_bytes: w.c.get(),
        },
        AccelReqWire::OP_RESERVE_MEMORY => AccelRequest::ReserveMemory {
            buffer_bytes: w.b.get(),
            program_bytes: w.c.get(),
        },
        AccelReqWire::OP_ATTACH => AccelRequest::Attach {
            announced: SlotRef::from_wire(u32::try_from(w.a.get()).ok()?),
        },
        AccelReqWire::OP_QUERY => AccelRequest::Query,
        AccelReqWire::OP_CREATE_CONTEXT => AccelRequest::CreateContext {
            flags: u32::try_from(w.a.get()).ok()?,
        },
        AccelReqWire::OP_DESTROY_CONTEXT => AccelRequest::DestroyContext {
            ctx: w.subject.get(),
        },
        AccelReqWire::OP_CREATE_BUFFER => AccelRequest::CreateBuffer {
            ctx: w.subject.get(),
            bytes: w.a.get(),
            align: w.b.get(),
            domain: (w.c.get() & 0xff) as u8,
            usage: u32::try_from(w.c.get() >> 8).ok()?,
        },
        AccelReqWire::OP_IMPORT_BUFFER => AccelRequest::ImportBuffer {
            ctx: w.subject.get(),
            bytes: w.a.get(),
            align: w.b.get(),
            usage: u32::try_from(w.c.get()).ok()?,
            announced: SlotRef::from_wire(u32::try_from(w.d.get()).ok()?),
        },
        AccelReqWire::OP_EXPORT_BUFFER => AccelRequest::ExportBuffer {
            buffer: w.subject.get(),
        },
        AccelReqWire::OP_FREE_BUFFER => AccelRequest::FreeBuffer {
            buffer: w.subject.get(),
        },
        AccelReqWire::OP_WRITE_BUFFER => AccelRequest::WriteBuffer {
            buffer: w.subject.get(),
            offset: w.a.get(),
            stage: w.b.get(),
            len: w.c.get(),
        },
        AccelReqWire::OP_READ_BUFFER => AccelRequest::ReadBuffer {
            buffer: w.subject.get(),
            offset: w.a.get(),
            stage: w.b.get(),
            len: w.c.get(),
        },
        AccelReqWire::OP_READ_FAILURE => AccelRequest::ReadFailure { offset: w.a.get() },
        AccelReqWire::OP_READ_TELEMETRY => AccelRequest::ReadTelemetry { offset: w.a.get() },
        AccelReqWire::OP_CONFIGURE_PROFILING => AccelRequest::ConfigureProfiling {
            ctx: w.subject.get(),
            enabled: match w.a.get() {
                0 => false,
                1 => true,
                _ => return None,
            },
        },
        AccelReqWire::OP_READ_LOG => AccelRequest::ReadLog {
            stage: w.b.get(),
            len: w.c.get(),
        },
        AccelReqWire::OP_LOAD_PROGRAM => {
            let mut target = [0u32; 12];
            for (word, chunk) in target.iter_mut().zip(w.target.chunks_exact(4)) {
                *word = u32::from_le_bytes(chunk.try_into().ok()?);
            }
            AccelRequest::LoadProgram {
                ctx: w.subject.get(),
                stage: w.a.get(),
                len: w.b.get(),
                resident_bytes: w.c.get(),
                format: u32::try_from(w.d.get()).ok()?,
                target,
            }
        }
        AccelReqWire::OP_UNLOAD_PROGRAM => AccelRequest::UnloadProgram {
            program: w.subject.get(),
        },
        AccelReqWire::OP_CREATE_QUEUE => AccelRequest::CreateQueue {
            ctx: w.subject.get(),
            flags: u32::try_from(w.a.get()).ok()?,
        },
        AccelReqWire::OP_DESTROY_QUEUE => AccelRequest::DestroyQueue {
            queue: w.subject.get(),
        },
        AccelReqWire::OP_SUBMIT | AccelReqWire::OP_SUBMIT_LAUNCH => {
            let count = usize::try_from(w.c.get()).ok()?;
            if count == 0 || count > MAX_ACCEL_BINDINGS {
                return None;
            }
            let mut live = heapless::Vec::<AccelBinding, MAX_ACCEL_BINDINGS>::new();
            for src in w.bindings.iter().take(count) {
                live.push(AccelBinding {
                    buffer: src.buffer.get(),
                    offset: src.offset.get(),
                    len: src.len.get(),
                    slot: src.slot.get(),
                    access: decode_access(src.access)?,
                })
                .ok()?;
            }
            let bindings = AccelBindings::new(live.as_slice())?;
            if w.op == AccelReqWire::OP_SUBMIT {
                #[allow(deprecated)]
                let dispatch = decode_dispatch(&w.target)?;
                #[allow(deprecated)]
                AccelRequest::Submit {
                    queue: w.subject.get(),
                    program: w.a.get(),
                    timeout_ns: w.b.get(),
                    bindings,
                    dispatch,
                }
            } else {
                AccelRequest::SubmitLaunch {
                    queue: w.subject.get(),
                    program: w.a.get(),
                    timeout_ns: w.b.get(),
                    bindings,
                    // Semantic geometry validation lives here: re-encoding
                    // catches reserved tail bytes, while `Workgroups::new`
                    // rejects zero axes.
                    launch: LaunchShape::Workgroups(decode_workgroups(&w.target)?),
                }
            }
        }
        AccelReqWire::OP_SUBMIT_SEQUENCE => AccelRequest::SubmitSequence {
            queue: w.subject.get(),
            stage: w.a.get(),
            len: w.b.get(),
            count: u32::try_from(w.c.get()).ok()?,
            timeout_ns: w.d.get(),
        },
        AccelReqWire::OP_POLL_EVENT => AccelRequest::PollEvent {
            event: w.subject.get(),
        },
        AccelReqWire::OP_CANCEL_EVENT => AccelRequest::CancelEvent {
            event: w.subject.get(),
        },
        AccelReqWire::OP_DESTROY_EVENT => AccelRequest::DestroyEvent {
            event: w.subject.get(),
        },
        AccelReqWire::OP_SIGNAL => AccelRequest::Signal,
        _ => return None,
    })
}

fn decode_access(value: u8) -> Option<AccessMode> {
    Some(match value {
        1 => AccessMode::Read,
        2 => AccessMode::Write,
        3 => AccessMode::ReadWrite,
        _ => return None,
    })
}

/// Preserve the deployed OP13 geometry byte-for-byte: grid `[0..12]`, group
/// `[12..24]`, SIMD width `[24..28]`, all LE; `[28..48]` stays zero.
#[allow(deprecated)]
fn encode_dispatch(target: &mut [u8; 48], dispatch: Dispatch) {
    for (i, word) in dispatch.grid().iter().chain(&dispatch.group()).enumerate() {
        target[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    target[24..28].copy_from_slice(&dispatch.simd_width().to_le_bytes());
}

/// Decode the deployed OP13 geometry without mapping it to portable
/// [`Workgroups`]. Reserved tail bytes are rejected by byte-exact re-encoding.
#[allow(deprecated)]
fn decode_dispatch(target: &[u8; 48]) -> Option<Dispatch> {
    let word = |at: usize| u32::from_le_bytes(target[at..at + 4].try_into().unwrap());
    Dispatch::new(
        [word(0), word(4), word(8)],
        [word(12), word(16), word(20)],
        word(24),
    )
    .ok()
}

/// Pack explicit workgroup geometry into a `SubmitLaunch` record's `target`:
/// group count `[0..12]`, local size `[12..24]`, all LE; `[24..48]` stays zero.
fn encode_workgroups(target: &mut [u8; 48], workgroups: Workgroups) {
    let group_count = workgroups.group_count();
    let local_size = workgroups.local_size();
    for (i, word) in group_count.iter().chain(&local_size).enumerate() {
        target[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
}

/// Lift explicit geometry back out, refusing values `Workgroups::new` rejects.
fn decode_workgroups(target: &[u8; 48]) -> Option<Workgroups> {
    let word = |at: usize| u32::from_le_bytes(target[at..at + 4].try_into().unwrap());
    let group_count = [word(0), word(4), word(8)];
    let local_size = [word(12), word(16), word(20)];
    Workgroups::new(group_count, local_size).ok()
}

/// Lower a response + the effected grant count (total).
pub fn encode_rep(rep: &AccelResponse, grants: u8) -> AccelRepWire {
    debug_assert_eq!(
        grants,
        u8::from(matches!(
            rep,
            AccelResponse::Signaled
                | AccelResponse::ImportReturned { .. }
                | AccelResponse::BufferExported { .. }
        )),
        "completion and import hand-back replies carry one retained grant",
    );
    let mut w = AccelRepWire {
        tag: 0,
        grants,
        refusal: 0,
        state: 0,
        reserved: [0; 4],
        id: U64::ZERO,
        info: [0; 80],
    };
    match *rep {
        AccelResponse::PeerAccepted => w.tag = AccelRepWire::TAG_PEER_ACCEPTED,
        AccelResponse::PeerAdmitted { peer } => {
            w.tag = AccelRepWire::TAG_PEER_ADMITTED;
            w.id = U64::new(u64::from(peer.arrival()));
        }
        AccelResponse::PeerClosing => w.tag = AccelRepWire::TAG_PEER_CLOSING,
        AccelResponse::PeerRetired => w.tag = AccelRepWire::TAG_PEER_RETIRED,
        AccelResponse::Detached => w.tag = AccelRepWire::TAG_DETACHED,
        AccelResponse::Provisioned { bytes } => {
            w.tag = AccelRepWire::TAG_PROVISIONED;
            w.id = U64::new(bytes);
        }
        AccelResponse::Attached { bytes } => {
            w.tag = AccelRepWire::TAG_ATTACHED;
            w.id = U64::new(bytes);
        }
        AccelResponse::Info(info) => {
            w.tag = AccelRepWire::TAG_INFO;
            w.info = encode_info(&info);
        }
        AccelResponse::ContextMade { ctx } => {
            w.tag = AccelRepWire::TAG_CONTEXT_MADE;
            w.id = U64::new(ctx);
        }
        AccelResponse::ImportReturned { refusal } => {
            w.tag = AccelRepWire::TAG_IMPORT_RETURNED;
            w.refusal = refusal_code(refusal);
        }
        AccelResponse::BufferExported { buffer } => {
            w.tag = AccelRepWire::TAG_BUFFER_EXPORTED;
            w.id = U64::new(buffer);
        }
        AccelResponse::BufferMade { buffer } => {
            w.tag = AccelRepWire::TAG_BUFFER_MADE;
            w.id = U64::new(buffer);
        }
        AccelResponse::ProgramMade { program } => {
            w.tag = AccelRepWire::TAG_PROGRAM_MADE;
            w.id = U64::new(program);
        }
        AccelResponse::ProfilingConfigured { ctx, enabled } => {
            w.tag = AccelRepWire::TAG_PROFILING_CONFIGURED;
            w.id = U64::new(ctx);
            w.info[0] = u8::from(enabled);
        }
        AccelResponse::QueueMade { queue } => {
            w.tag = AccelRepWire::TAG_QUEUE_MADE;
            w.id = U64::new(queue);
        }
        AccelResponse::Released { subject } => {
            w.tag = AccelRepWire::TAG_RELEASED;
            w.id = U64::new(subject);
        }
        AccelResponse::Wrote { buffer } => {
            w.tag = AccelRepWire::TAG_WROTE;
            w.id = U64::new(buffer);
        }
        AccelResponse::Read { buffer } => {
            w.tag = AccelRepWire::TAG_READ;
            w.id = U64::new(buffer);
        }
        AccelResponse::Log { bytes } => {
            w.tag = AccelRepWire::TAG_LOG;
            w.id = U64::new(bytes);
        }
        AccelResponse::Submitted { event } => {
            w.tag = AccelRepWire::TAG_SUBMITTED;
            w.id = U64::new(event);
        }
        AccelResponse::SubmitIndeterminate { event, refusal } => {
            w.tag = AccelRepWire::TAG_SUBMIT_INDETERMINATE;
            w.id = U64::new(event);
            w.refusal = refusal_code(refusal);
        }
        AccelResponse::Event { event, state } => {
            w.tag = AccelRepWire::TAG_EVENT;
            w.id = U64::new(event);
            let (state, refusal) = match state {
                AccelEventState::Pending => (1, 0),
                AccelEventState::Complete => (2, 0),
                AccelEventState::Failed(refusal) => (3, refusal_code(refusal)),
                AccelEventState::Cancelled => (4, 0),
            };
            w.state = state;
            w.refusal = refusal;
        }
        AccelResponse::Cancelled { event } => {
            w.tag = AccelRepWire::TAG_CANCELLED;
            w.id = U64::new(event);
        }
        AccelResponse::Failure { offset, len, bytes } => {
            w.tag = AccelRepWire::TAG_FAILURE;
            w.id = U64::new(offset);
            w.state = len;
            w.info = bytes;
        }
        AccelResponse::Signaled => {
            w.tag = AccelRepWire::TAG_SIGNALED;
        }
        AccelResponse::Refused(refusal) => {
            w.tag = AccelRepWire::TAG_REFUSED;
            w.refusal = refusal_code(refusal);
        }
    }
    let refusal = match rep {
        AccelResponse::Refused(refusal)
        | AccelResponse::ImportReturned { refusal }
        | AccelResponse::SubmitIndeterminate { refusal, .. }
        | AccelResponse::Event {
            state: AccelEventState::Failed(refusal),
            ..
        } => Some(refusal),
        _ => None,
    };
    if let Some(AccelRefusal::External { domain, code }) = refusal {
        w.info[0..4].copy_from_slice(&domain.to_le_bytes());
        w.info[4..12].copy_from_slice(&code.to_le_bytes());
    }
    w
}

/// Lift a wire reply + its grant count — byte-exact fail-closed,
/// including the per-tag grant law for completion channels and lease hand-back.
pub fn decode_rep(w: &AccelRepWire) -> Option<(AccelResponse, u8)> {
    let rep = raw_decode_rep(w)?;
    if w.grants
        != u8::from(matches!(
            rep,
            AccelResponse::Signaled
                | AccelResponse::ImportReturned { .. }
                | AccelResponse::BufferExported { .. }
        ))
    {
        return None;
    }
    (encode_rep(&rep, w.grants) == *w).then_some((rep, w.grants))
}

fn raw_decode_rep(w: &AccelRepWire) -> Option<AccelResponse> {
    Some(match w.tag {
        AccelRepWire::TAG_FAILURE => {
            let len = usize::from(w.state);
            if len > w.info.len() || w.info[len..].iter().any(|byte| *byte != 0) {
                return None;
            }
            AccelResponse::Failure { offset: w.id.get(), len: w.state, bytes: w.info }
        }
        AccelRepWire::TAG_PEER_ACCEPTED => AccelResponse::PeerAccepted,
        AccelRepWire::TAG_PEER_ADMITTED => AccelResponse::PeerAdmitted {
            peer: kore_rt::svc::PeerId::from_arrival(u16::try_from(w.id.get()).ok()?)?,
        },
        AccelRepWire::TAG_PEER_CLOSING => AccelResponse::PeerClosing,
        AccelRepWire::TAG_PEER_RETIRED => AccelResponse::PeerRetired,
        AccelRepWire::TAG_DETACHED => AccelResponse::Detached,
        AccelRepWire::TAG_ATTACHED => AccelResponse::Attached { bytes: w.id.get() },
        AccelRepWire::TAG_PROVISIONED => AccelResponse::Provisioned { bytes: w.id.get() },
        AccelRepWire::TAG_INFO => AccelResponse::Info(decode_info(&w.info)?),
        AccelRepWire::TAG_CONTEXT_MADE => AccelResponse::ContextMade { ctx: w.id.get() },
        AccelRepWire::TAG_IMPORT_RETURNED => AccelResponse::ImportReturned {
            refusal: decode_refusal(w.refusal, &w.info)?,
        },
        AccelRepWire::TAG_BUFFER_EXPORTED => AccelResponse::BufferExported { buffer: w.id.get() },
        AccelRepWire::TAG_BUFFER_MADE => AccelResponse::BufferMade { buffer: w.id.get() },
        AccelRepWire::TAG_PROGRAM_MADE => AccelResponse::ProgramMade {
            program: w.id.get(),
        },
        AccelRepWire::TAG_QUEUE_MADE => AccelResponse::QueueMade { queue: w.id.get() },
        AccelRepWire::TAG_PROFILING_CONFIGURED => AccelResponse::ProfilingConfigured {
            ctx: w.id.get(),
            enabled: match w.info[0] {
                0 => false,
                1 => true,
                _ => return None,
            },
        },
        AccelRepWire::TAG_RELEASED => AccelResponse::Released {
            subject: w.id.get(),
        },
        AccelRepWire::TAG_WROTE => AccelResponse::Wrote { buffer: w.id.get() },
        AccelRepWire::TAG_READ => AccelResponse::Read { buffer: w.id.get() },
        AccelRepWire::TAG_LOG => AccelResponse::Log { bytes: w.id.get() },
        AccelRepWire::TAG_SUBMITTED => AccelResponse::Submitted { event: w.id.get() },
        AccelRepWire::TAG_SUBMIT_INDETERMINATE => AccelResponse::SubmitIndeterminate {
            event: w.id.get(),
            refusal: decode_refusal(w.refusal, &w.info)?,
        },
        AccelRepWire::TAG_EVENT => AccelResponse::Event {
            event: w.id.get(),
            state: match (w.state, w.refusal) {
                (1, 0) => AccelEventState::Pending,
                (2, 0) => AccelEventState::Complete,
                (3, code) => AccelEventState::Failed(decode_refusal(code, &w.info)?),
                (4, 0) => AccelEventState::Cancelled,
                _ => return None,
            },
        },
        AccelRepWire::TAG_CANCELLED => AccelResponse::Cancelled { event: w.id.get() },
        AccelRepWire::TAG_SIGNALED => AccelResponse::Signaled,
        AccelRepWire::TAG_REFUSED => AccelResponse::Refused(decode_refusal(w.refusal, &w.info)?),
        _ => return None,
    })
}

fn refusal_code(refusal: AccelRefusal) -> u8 {
    match refusal {
        AccelRefusal::Unsupported => AccelRepWire::REFUSE_UNSUPPORTED,
        AccelRefusal::Incompatible => AccelRepWire::REFUSE_INCOMPATIBLE,
        AccelRefusal::InvalidArgument => AccelRepWire::REFUSE_INVALID_ARGUMENT,
        AccelRefusal::OutOfBounds => AccelRepWire::REFUSE_OUT_OF_BOUNDS,
        AccelRefusal::Busy => AccelRepWire::REFUSE_BUSY,
        AccelRefusal::OutOfMemory => AccelRepWire::REFUSE_OUT_OF_MEMORY,
        AccelRefusal::ResourceLimit => AccelRepWire::REFUSE_RESOURCE_LIMIT,
        AccelRefusal::DeadlineExpired => AccelRepWire::REFUSE_DEADLINE_EXPIRED,
        AccelRefusal::DeviceLost => AccelRepWire::REFUSE_DEVICE_LOST,
        AccelRefusal::PermissionDenied => AccelRepWire::REFUSE_PERMISSION_DENIED,
        AccelRefusal::Internal => AccelRepWire::REFUSE_INTERNAL,
        AccelRefusal::BadObject => AccelRepWire::REFUSE_BAD_OBJECT,
        AccelRefusal::NoStaging => AccelRepWire::REFUSE_NO_STAGING,
        AccelRefusal::Denied => AccelRepWire::REFUSE_DENIED,
        AccelRefusal::NoChannel => AccelRepWire::REFUSE_NO_CHANNEL,
        AccelRefusal::NotReady => AccelRepWire::REFUSE_NOT_READY,
        AccelRefusal::External { .. } => AccelRepWire::REFUSE_EXTERNAL,
    }
}

fn decode_refusal(code: u8, detail: &[u8; 80]) -> Option<AccelRefusal> {
    Some(match code {
        AccelRepWire::REFUSE_UNSUPPORTED => AccelRefusal::Unsupported,
        AccelRepWire::REFUSE_INCOMPATIBLE => AccelRefusal::Incompatible,
        AccelRepWire::REFUSE_INVALID_ARGUMENT => AccelRefusal::InvalidArgument,
        AccelRepWire::REFUSE_OUT_OF_BOUNDS => AccelRefusal::OutOfBounds,
        AccelRepWire::REFUSE_BUSY => AccelRefusal::Busy,
        AccelRepWire::REFUSE_OUT_OF_MEMORY => AccelRefusal::OutOfMemory,
        AccelRepWire::REFUSE_RESOURCE_LIMIT => AccelRefusal::ResourceLimit,
        AccelRepWire::REFUSE_DEADLINE_EXPIRED => AccelRefusal::DeadlineExpired,
        AccelRepWire::REFUSE_DEVICE_LOST => AccelRefusal::DeviceLost,
        AccelRepWire::REFUSE_PERMISSION_DENIED => AccelRefusal::PermissionDenied,
        AccelRepWire::REFUSE_INTERNAL => AccelRefusal::Internal,
        AccelRepWire::REFUSE_BAD_OBJECT => AccelRefusal::BadObject,
        AccelRepWire::REFUSE_NO_STAGING => AccelRefusal::NoStaging,
        AccelRepWire::REFUSE_DENIED => AccelRefusal::Denied,
        AccelRepWire::REFUSE_NO_CHANNEL => AccelRefusal::NoChannel,
        AccelRepWire::REFUSE_NOT_READY => AccelRefusal::NotReady,
        AccelRepWire::REFUSE_EXTERNAL => AccelRefusal::External {
            domain: u32::from_le_bytes(detail[0..4].try_into().ok()?),
            code: i64::from_le_bytes(detail[4..12].try_into().ok()?),
        },
        _ => return None,
    })
}

/// The 80-byte `TAG_INFO` payload: uuid ‖ class ‖ reserved ‖ vendor ‖
/// device ‖ capabilities ‖ the eight limit words ‖ zero tail.
fn encode_info(info: &AccelInfo) -> [u8; 80] {
    let mut out = [0u8; 80];
    out[0..16].copy_from_slice(&info.uuid);
    out[16..18].copy_from_slice(&info.class.to_le_bytes());
    out[20..24].copy_from_slice(&info.vendor.to_le_bytes());
    out[24..28].copy_from_slice(&info.device.to_le_bytes());
    out[28..36].copy_from_slice(&info.capabilities.to_le_bytes());
    let l = &info.limits;
    out[36..40].copy_from_slice(&l.max_contexts.to_le_bytes());
    out[40..44].copy_from_slice(&l.max_buffers_per_context.to_le_bytes());
    out[44..48].copy_from_slice(&l.max_programs_per_context.to_le_bytes());
    out[48..52].copy_from_slice(&l.max_queues_per_context.to_le_bytes());
    out[52..56].copy_from_slice(&l.max_events_per_context.to_le_bytes());
    out[56..60].copy_from_slice(&l.max_bindings_per_submission.to_le_bytes());
    out[60..68].copy_from_slice(&l.max_buffer_bytes.to_le_bytes());
    out[68..76].copy_from_slice(&l.max_artifact_bytes.to_le_bytes());
    out
}

fn decode_info(bytes: &[u8; 80]) -> Option<AccelInfo> {
    let word32 =
        |at: usize| -> Option<u32> { Some(u32::from_le_bytes(bytes[at..at + 4].try_into().ok()?)) };
    let word64 =
        |at: usize| -> Option<u64> { Some(u64::from_le_bytes(bytes[at..at + 8].try_into().ok()?)) };
    Some(AccelInfo {
        uuid: bytes[0..16].try_into().ok()?,
        class: u16::from_le_bytes(bytes[16..18].try_into().ok()?),
        vendor: word32(20)?,
        device: word32(24)?,
        capabilities: word64(28)?,
        limits: DeviceLimits {
            max_contexts: word32(36)?,
            max_buffers_per_context: word32(40)?,
            max_programs_per_context: word32(44)?,
            max_queues_per_context: word32(48)?,
            max_events_per_context: word32(52)?,
            max_bindings_per_submission: word32(56)?,
            max_buffer_bytes: word64(60)?,
            max_artifact_bytes: word64(68)?,
        },
    })
}

/// Capability return slots needed before this request becomes visible.
/// Data-only execution traffic must not reserve or post a retained grant lane.
/// Import can return its original lease on refusal, and export/signal deliver
/// authority on success; all three must keep their receive posted first.
pub fn request_grants(request: &AccelReqWire) -> usize {
    usize::from(matches!(
        request.op,
        AccelReqWire::OP_SIGNAL | AccelReqWire::OP_IMPORT_BUFFER | AccelReqWire::OP_EXPORT_BUFFER
    ))
}

/// The uncorrelated-pair reply gate: each op's legal tags, plus the
/// subject echo where the reply names one — a stale reply naming a
/// different object than the outstanding request never correlates.
pub fn reply_matches(request: &AccelReqWire, response: &AccelResponse, grants: u8) -> bool {
    use AccelResponse as R;
    if grants
        != u8::from(matches!(
            response,
            R::Signaled | R::ImportReturned { .. } | R::BufferExported { .. }
        ))
    {
        return false;
    }
    let subject = request.subject.get();
    match (request.op, response) {
        (AccelReqWire::OP_EXPORT_BUFFER, R::BufferExported { buffer }) => *buffer == subject,
        (AccelReqWire::OP_IMPORT_BUFFER, R::ImportReturned { .. }) => true,
        (AccelReqWire::OP_ADMIT_PEER, R::PeerAccepted) => true,
        (AccelReqWire::OP_TAKE_PEER_ADMISSION, R::PeerAdmitted { .. }) => true,
        (AccelReqWire::OP_CLOSE_PEER, R::PeerClosing) => true,
        (AccelReqWire::OP_RETIRE_PEER, R::PeerClosing) => true,
        (AccelReqWire::OP_CONFIRM_RETIRED, R::PeerRetired) => true,
        (_, R::Refused(_)) => {
            request.op >= AccelReqWire::OP_ATTACH
                && request.op <= AccelReqWire::OP_CONFIGURE_PROFILING
        }
        (AccelReqWire::OP_ATTACH, R::Attached { .. }) => true,
        (AccelReqWire::OP_ATTACH_OWNED, R::Attached { bytes }) => *bytes != 0 && *bytes & 4095 == 0,
        (AccelReqWire::OP_DETACH_STAGING, R::Detached) => true,
        (AccelReqWire::OP_PROVISION_MEMORY, R::Provisioned { bytes }) => request
            .b
            .get()
            .checked_add(request.c.get())
            .is_some_and(|requested| requested != 0 && requested <= *bytes),
        (AccelReqWire::OP_RESERVE_MEMORY, R::Provisioned { bytes }) => request
            .b
            .get()
            .checked_add(request.c.get())
            .is_some_and(|requested| requested != 0 && requested == *bytes),
        (AccelReqWire::OP_QUERY, R::Info(_)) => true,
        (AccelReqWire::OP_CONFIGURE_PROFILING, R::ProfilingConfigured { ctx, enabled }) => {
            *ctx == subject && u64::from(*enabled) == request.a.get()
        }
        (AccelReqWire::OP_CREATE_CONTEXT, R::ContextMade { .. }) => true,
        (AccelReqWire::OP_CREATE_BUFFER | AccelReqWire::OP_IMPORT_BUFFER, R::BufferMade { .. }) => {
            true
        }
        (AccelReqWire::OP_LOAD_PROGRAM, R::ProgramMade { .. }) => true,
        (AccelReqWire::OP_CREATE_QUEUE, R::QueueMade { .. }) => true,
        (
            AccelReqWire::OP_DESTROY_CONTEXT
            | AccelReqWire::OP_FREE_BUFFER
            | AccelReqWire::OP_UNLOAD_PROGRAM
            | AccelReqWire::OP_DESTROY_QUEUE
            | AccelReqWire::OP_DESTROY_EVENT,
            R::Released { subject: echoed },
        ) => *echoed == subject,
        (AccelReqWire::OP_WRITE_BUFFER, R::Wrote { buffer }) => *buffer == subject,
        (AccelReqWire::OP_READ_BUFFER, R::Read { buffer }) => *buffer == subject,
        (AccelReqWire::OP_READ_LOG, R::Log { .. }) => true,
        (
            AccelReqWire::OP_READ_FAILURE | AccelReqWire::OP_READ_TELEMETRY,
            R::Failure { offset, len, bytes },
        ) => {
            *offset == request.a.get() && usize::from(*len) <= bytes.len()
                && bytes[usize::from(*len)..].iter().all(|byte| *byte == 0)
        }
        (
            AccelReqWire::OP_SUBMIT
            | AccelReqWire::OP_SUBMIT_LAUNCH
            | AccelReqWire::OP_SUBMIT_SEQUENCE,
            R::Submitted { .. } | R::SubmitIndeterminate { .. },
        ) => true,
        (AccelReqWire::OP_POLL_EVENT, R::Event { event, .. }) => *event == subject,
        (AccelReqWire::OP_CANCEL_EVENT, R::Cancelled { event }) => *event == subject,
        (AccelReqWire::OP_SIGNAL, R::Signaled) => true,
        _ => false,
    }
}

/// The retained-grant policy both speakers cite: channel 0 is heterogeneous,
/// like the exec protocol, with reply-specific authenticated possession.
fn grant_policy(_index: usize) -> Option<(ResourceKind, Rights)> {
    // This retained bank carries completion channels, allocation leases and scoped routes.
    // The canonical reply identifies the kind; each client possession checks
    // the authenticated record before exposing a typed affine owner.
    None
}

impl SvcProtocol for AccelWire {
    type Request = AccelRequest;
    type Response = AccelResponse;
    type ReqWire = AccelReqWire;
    type RepWire = AccelRepWire;

    const REPLY_GRANTS: usize = ACCEL_REPLY_GRANTS;
    const SERVE_PAIRS: usize = ACCEL_SERVE_PAIRS;

    fn request_grants(request: &AccelReqWire) -> usize {
        request_grants(request)
    }
    fn grant_policy(index: usize) -> Option<(ResourceKind, Rights)> {
        grant_policy(index)
    }
    fn retain_grant(index: usize) -> bool {
        index == 0
    }
    fn reply_matches(request: &AccelReqWire, response: &AccelResponse, grants: u8) -> bool {
        reply_matches(request, response, grants)
    }
    fn encode_req(req: &AccelRequest) -> AccelReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &AccelReqWire) -> Option<AccelRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &AccelResponse, grants: u8) -> AccelRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &AccelRepWire) -> Option<(AccelResponse, u8)> {
        decode_rep(wire)
    }
}

impl<A: NativeAccelerator> SvcProtocol for AccelService<A> {
    type Request = AccelRequest;
    type Response = AccelResponse;
    type ReqWire = AccelReqWire;
    type RepWire = AccelRepWire;

    const REPLY_GRANTS: usize = ACCEL_REPLY_GRANTS;
    const SERVE_PAIRS: usize = ACCEL_SERVE_PAIRS;

    fn request_grants(request: &AccelReqWire) -> usize {
        request_grants(request)
    }
    fn grant_policy(index: usize) -> Option<(ResourceKind, Rights)> {
        grant_policy(index)
    }
    fn retain_grant(index: usize) -> bool {
        index == 0
    }
    fn reply_matches(request: &AccelReqWire, response: &AccelResponse, grants: u8) -> bool {
        reply_matches(request, response, grants)
    }
    fn encode_req(req: &AccelRequest) -> AccelReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &AccelReqWire) -> Option<AccelRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &AccelResponse, grants: u8) -> AccelRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &AccelRepWire) -> Option<(AccelResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]

    use virtio_accel_core::AccessMode;
    use zerocopy::FromBytes;

    use super::*;
    use crate::service::AccelBinding;

    #[test]
    fn profiling_configuration_is_canonical_and_echoes_context_and_setting() {
        for enabled in [false, true] {
            let request = AccelRequest::ConfigureProfiling { ctx: 19, enabled };
            let wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            assert_eq!(request_grants(&wire), 0);
            let reply = AccelResponse::ProfilingConfigured { ctx: 19, enabled };
            let encoded = encode_rep(&reply, 0);
            assert_eq!(decode_rep(&encoded), Some((reply, 0)));
            assert!(reply_matches(&wire, &reply, 0));
            assert!(!reply_matches(&wire, &reply, 1));
            assert!(!reply_matches(
                &wire,
                &AccelResponse::ProfilingConfigured { ctx: 20, enabled },
                0
            ));
            assert!(!reply_matches(
                &wire,
                &AccelResponse::ProfilingConfigured {
                    ctx: 19,
                    enabled: !enabled
                },
                0
            ));
            assert!(reply_matches(
                &wire,
                &AccelResponse::Refused(AccelRefusal::Unsupported),
                0
            ));
            let mut invalid = wire;
            invalid.a = U64::new(2);
            assert_eq!(decode_req(&invalid), None);
            let mut invalid = wire;
            invalid.flags = U32::new(1);
            assert_eq!(decode_req(&invalid), None);
            let mut invalid = wire;
            invalid.b = U64::new(1);
            assert_eq!(decode_req(&invalid), None);
            let mut invalid = encoded;
            invalid.info[0] = 2;
            assert_eq!(decode_rep(&invalid), None);
            let mut invalid = encoded;
            invalid.info[1] = 1;
            assert_eq!(decode_rep(&invalid), None);
        }
    }

    #[test]
    fn failure_pages_echo_offsets_and_reject_noncanonical_payloads() {
        let req = encode_req(&AccelRequest::ReadFailure { offset: 80 });
        assert_eq!(decode_req(&req), Some(AccelRequest::ReadFailure { offset: 80 }));
        let req = encode_req(&AccelRequest::ReadTelemetry { offset: 80 });
        assert_eq!(
            decode_req(&req),
            Some(AccelRequest::ReadTelemetry { offset: 80 })
        );
        let mut bytes = [0; AccelRepWire::FAILURE_CHUNK_BYTES];
        bytes[..3].copy_from_slice(b"abc");
        let rep = AccelResponse::Failure { offset: 80, len: 3, bytes };
        let wire = encode_rep(&rep, 0);
        assert_eq!(decode_rep(&wire), Some((rep, 0)));
        assert!(reply_matches(&req, &rep, 0));
        assert!(!reply_matches(&encode_req(&AccelRequest::ReadFailure { offset: 0 }), &rep, 0));
        assert!(!reply_matches(&req, &rep, 1));
        let mut bad = wire;
        bad.info[3] = 1;
        assert_eq!(decode_rep(&bad), None);
        bad = wire;
        bad.state = 81;
        assert_eq!(decode_rep(&bad), None);
        bad = wire;
        bad.grants = 1;
        assert_eq!(decode_rep(&bad), None);
    }

    #[test]
    fn request_grant_budget_covers_every_correlated_authority_reply() {
        for request in every_request() {
            let wire = encode_req(&request);
            let expected = usize::from(matches!(
                request,
                AccelRequest::Signal
                    | AccelRequest::ImportBuffer { .. }
                    | AccelRequest::ExportBuffer { .. }
            ));
            assert_eq!(AccelWire::request_grants(&wire), expected);
            assert_eq!(
                <AccelService<
                    kore_driver::accel::ArtifactBackend<virtio_accel_mock::MockAccelerator>,
                > as SvcProtocol>::request_grants(&wire),
                expected
            );
            for (response, grants) in every_response() {
                if reply_matches(&wire, &response, grants) {
                    assert!(usize::from(grants) <= expected);
                }
            }
        }
    }

    fn every_request() -> alloc::vec::Vec<AccelRequest> {
        let bindings = AccelBindings::new(&[
            AccelBinding {
                buffer: 0x10,
                offset: 0,
                len: 64,
                slot: 7,
                access: AccessMode::ReadWrite,
            },
            AccelBinding {
                buffer: 0x11,
                offset: 128,
                len: 32,
                slot: 2,
                access: AccessMode::Read,
            },
        ])
        .unwrap();
        alloc::vec![
            AccelRequest::Attach {
                announced: SlotRef::from_wire(9)
            },
            AccelRequest::AttachOwned {
                announced: SlotRef::from_wire(9)
            },
            AccelRequest::DetachStaging,
            AccelRequest::ExportBuffer { buffer: 0x1234 },
            AccelRequest::Query,
            AccelRequest::ProvisionMemory {
                announced: SlotRef::from_wire(9),
                buffer_bytes: 2 << 30,
                program_bytes: 1 << 30,
            },
            AccelRequest::ReserveMemory {
                buffer_bytes: 4096,
                program_bytes: 8192,
            },
            AccelRequest::CreateContext { flags: 0 },
            AccelRequest::DestroyContext { ctx: 0x1234 },
            AccelRequest::ConfigureProfiling { ctx: 0x1234, enabled: true },
            AccelRequest::ConfigureProfiling { ctx: 0x1234, enabled: false },
            AccelRequest::CreateBuffer {
                ctx: 0x1234,
                bytes: 4096,
                align: 64,
                domain: 3,
                usage: 0b11111,
            },
            AccelRequest::FreeBuffer { buffer: 0x55 },
            AccelRequest::WriteBuffer {
                buffer: 0x55,
                offset: 8,
                stage: 512,
                len: 128
            },
            AccelRequest::ReadBuffer {
                buffer: 0x55,
                offset: 0,
                stage: 1024,
                len: 256
            },
            AccelRequest::ReadFailure { offset: 80 },
            AccelRequest::ReadLog {
                stage: 2048,
                len: 4096
            },
            AccelRequest::LoadProgram {
                ctx: 0x1234,
                stage: 0,
                len: 900,
                resident_bytes: 4096,
                format: 1,
                target: [7; 12],
            },
            AccelRequest::UnloadProgram { program: 0x77 },
            AccelRequest::CreateQueue {
                ctx: 0x1234,
                flags: 0
            },
            AccelRequest::DestroyQueue { queue: 0x88 },
            AccelRequest::Submit {
                queue: 0x88,
                program: 0x77,
                timeout_ns: 1_000_000,
                bindings,
                dispatch: Dispatch::scalar(),
            },
            AccelRequest::SubmitLaunch {
                queue: 0x88,
                program: 0x77,
                timeout_ns: 1_000_000,
                bindings,
                launch: LaunchShape::Workgroups(Workgroups::new([16, 8, 2], [64, 1, 1]).unwrap(),),
            },
            AccelRequest::SubmitSequence {
                queue: 0x88,
                stage: 4096,
                len: 2 * core::mem::size_of::<AccelSequenceEntryWire>() as u64,
                count: 2,
                timeout_ns: 1_000_000,
            },
            AccelRequest::Submit {
                queue: 0x88,
                program: 0x77,
                timeout_ns: 1_000_000,
                bindings,
                dispatch: Dispatch::new([16, 8, 2], [64, 1, 1], 16).unwrap(),
            },
            AccelRequest::PollEvent { event: 0x99 },
            AccelRequest::CancelEvent { event: 0x99 },
            AccelRequest::DestroyEvent { event: 0x99 },
            AccelRequest::Signal,
        ]
    }

    fn every_response() -> alloc::vec::Vec<(AccelResponse, u8)> {
        let info = AccelInfo {
            uuid: *b"virtio-accelmock",
            class: 1,
            vendor: 0x8086,
            device: 0x643e,
            capabilities: 0b100111,
            limits: DeviceLimits {
                max_contexts: 64,
                max_buffers_per_context: 1024,
                max_programs_per_context: 256,
                max_queues_per_context: 16,
                max_events_per_context: 4096,
                max_bindings_per_submission: 256,
                max_buffer_bytes: 1 << 30,
                max_artifact_bytes: 1 << 30,
            },
        };
        alloc::vec![
            (AccelResponse::Attached { bytes: 65536 }, 0),
            (AccelResponse::Detached, 0),
            (AccelResponse::Provisioned { bytes: 3 << 30 }, 0),
            (AccelResponse::Info(info), 0),
            (AccelResponse::ContextMade { ctx: 1 }, 0),
            (AccelResponse::ProfilingConfigured { ctx: 1, enabled: true }, 0),
            (AccelResponse::ProfilingConfigured { ctx: 1, enabled: false }, 0),
            (AccelResponse::BufferMade { buffer: 2 }, 0),
            (AccelResponse::ProgramMade { program: 3 }, 0),
            (AccelResponse::QueueMade { queue: 4 }, 0),
            (AccelResponse::Released { subject: 5 }, 0),
            (AccelResponse::Wrote { buffer: 2 }, 0),
            (AccelResponse::Read { buffer: 2 }, 0),
            (AccelResponse::Log { bytes: 1200 }, 0),
            (AccelResponse::Submitted { event: 6 }, 0),
            (
                AccelResponse::SubmitIndeterminate {
                    event: 6,
                    refusal: AccelRefusal::DeviceLost
                },
                0
            ),
            (
                AccelResponse::Event {
                    event: 6,
                    state: AccelEventState::Pending
                },
                0
            ),
            (
                AccelResponse::Event {
                    event: 6,
                    state: AccelEventState::Failed(AccelRefusal::DeadlineExpired)
                },
                0
            ),
            (AccelResponse::Cancelled { event: 6 }, 0),
            (AccelResponse::Signaled, 1),
            (AccelResponse::Refused(AccelRefusal::NotReady), 0),
        ]
    }

    /// Every vocabulary member survives a round trip in both directions.
    #[test]
    fn export_reply_requires_exact_buffer_and_one_lease_grant() {
        let request = encode_req(&AccelRequest::ExportBuffer { buffer: 42 });
        let response = AccelResponse::BufferExported { buffer: 42 };
        let wire = encode_rep(&response, 1);
        assert_eq!(decode_rep(&wire), Some((response, 1)));
        assert!(reply_matches(&request, &response, 1));
        assert!(!reply_matches(&request, &response, 0));
        assert!(!reply_matches(
            &request,
            &AccelResponse::BufferExported { buffer: 43 },
            1
        ));
        assert!(!reply_matches(
            &encode_req(&AccelRequest::FreeBuffer { buffer: 42 }),
            &response,
            1
        ));
        let mut missing = wire;
        missing.grants = 0;
        assert!(decode_rep(&missing).is_none());
        assert!(reply_matches(
            &request,
            &AccelResponse::Refused(AccelRefusal::Busy),
            0
        ));
    }

    #[test]
    fn import_handback_requires_one_grant_and_the_import_request() {
        let request = encode_req(&AccelRequest::ImportBuffer {
            ctx: 42,
            bytes: 4096,
            align: 4096,
            usage: 1,
            announced: SlotRef::from_wire(7),
        });
        for refusal in [
            AccelRefusal::NotReady,
            AccelRefusal::External {
                domain: u32::MAX,
                code: i64::MIN,
            },
        ] {
            let reply = AccelResponse::ImportReturned { refusal };
            let wire = encode_rep(&reply, 1);
            assert_eq!(decode_rep(&wire), Some((reply, 1)));
            assert!(reply_matches(&request, &reply, 1));
            assert!(!reply_matches(&encode_req(&AccelRequest::Query), &reply, 1));
            let mut malformed = wire;
            malformed.grants = 0;
            assert!(decode_rep(&malformed).is_none());
            malformed.grants = 2;
            assert!(decode_rep(&malformed).is_none());
        }
    }

    #[test]
    fn import_buffer_codec_is_canonical_and_correlated() {
        let request = AccelRequest::ImportBuffer {
            ctx: 42,
            bytes: 24,
            align: 4096,
            usage: 1,
            announced: SlotRef::from_wire(7),
        };
        let wire = encode_req(&request);
        assert_eq!(decode_req(&wire), Some(request));
        assert!(reply_matches(
            &wire,
            &AccelResponse::BufferMade { buffer: 9 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 4096 },
            0
        ));
        let mut malformed = wire;
        malformed.c = U64::new(1 << 32);
        assert_eq!(decode_req(&malformed), None);
        malformed = wire;
        malformed.target[0] = 1;
        assert_eq!(decode_req(&malformed), None);
    }

    #[test]
    fn peer_control_round_trips_and_correlates_only_its_outcome() {
        let peer = kore_rt::svc::PeerId::from_arrival(crate::service::arrival_of(1, 1)).unwrap();
        let cases = [
            (
                AccelRequest::AdmitPeer {
                    part: kore_rt::svc::PeerBundlePart::Grant(0),
                    announced: SlotRef::from_wire(7),
                },
                AccelResponse::PeerAccepted,
            ),
            (
                AccelRequest::TakePeerAdmission,
                AccelResponse::PeerAdmitted { peer },
            ),
            (AccelRequest::ClosePeer, AccelResponse::PeerClosing),
        ];
        for (request, response) in cases {
            let req = encode_req(&request);
            assert_eq!(decode_req(&req), Some(request));
            assert_eq!(decode_rep(&encode_rep(&response, 0)), Some((response, 0)));
            assert!(reply_matches(&req, &response, 0));
            assert!(!reply_matches(&req, &AccelResponse::Signaled, 1));
            let mut bad = req;
            bad.subject = U64::new(1);
            assert_eq!(decode_req(&bad), None);
        }
        let mut bad = encode_rep(&AccelResponse::PeerAdmitted { peer }, 0);
        bad.id = U64::new(1); // static identity is not an admission ticket
        assert_eq!(decode_rep(&bad), None);
        bad.id = U64::new(1 << 16);
        assert_eq!(decode_rep(&bad), None);
        assert!(!reply_matches(
            &encode_req(&AccelRequest::TakePeerAdmission),
            &AccelResponse::PeerAccepted,
            0
        ));
    }

    #[test]
    fn round_trips() {
        for req in every_request() {
            assert_eq!(decode_req(&encode_req(&req)), Some(req), "{req:?}");
        }
        for (rep, grants) in every_response() {
            assert_eq!(
                decode_rep(&encode_rep(&rep, grants)),
                Some((rep, grants)),
                "{rep:?}"
            );
        }
    }

    fn sequence_entry(launch: LaunchShape, barrier: bool) -> AccelSequenceEntry {
        AccelSequenceEntry {
            program: 0x77,
            bindings: AccelBindings::new(&[AccelBinding {
                buffer: 0x10,
                offset: 64,
                len: 128,
                slot: 2,
                access: AccessMode::ReadWrite,
            }])
            .unwrap(),
            launch,
            barrier,
            gate: None,
            signal: None,
        }
    }

    /// A table's gates and signals ride after its entries, one record a
    /// flag in member order, and decode back to the members that named them;
    /// a record too few, one too many, or one with reserved bytes refuses.
    #[test]
    fn a_staged_table_carries_its_gates_and_signals_after_its_entries() {
        use zerocopy::IntoBytes;
        let word = |offset, value| AccelSync { buffer: 0x30, offset, value };
        let signalled = AccelSequenceEntry {
            signal: Some(word(8, 5)),
            ..sequence_entry(LaunchShape::ArtifactDefined, true)
        };
        let gated = AccelSequenceEntry {
            gate: Some(word(16, 5)),
            signal: Some(word(24, 9)),
            ..sequence_entry(LaunchShape::ArtifactDefined, true)
        };
        let plain = sequence_entry(LaunchShape::ArtifactDefined, false);
        let members = [signalled, plain, gated];
        let mut bytes = alloc::vec::Vec::new();
        for member in &members {
            bytes.extend_from_slice(encode_sequence_entry(*member).as_bytes());
        }
        for member in &members {
            for sync in encode_sequence_syncs(member) {
                bytes.extend_from_slice(sync.as_bytes());
            }
        }
        let sync = core::mem::size_of::<AccelSequenceSyncWire>();
        assert_eq!(bytes.len(), 3 * core::mem::size_of::<AccelSequenceEntryWire>() + 3 * sync);
        assert_eq!(decode_sequence(&bytes, 3).unwrap(), members);
        // The gate's record comes before its member's signal's.
        let tail = &bytes[bytes.len() - 2 * sync..];
        assert_eq!(AccelSequenceSyncWire::read_from_bytes(&tail[..sync]).unwrap().offset.get(), 16);
        // A record missing, a record extra, a reserved byte set.
        assert_eq!(decode_sequence(&bytes[..bytes.len() - sync], 3), Err(AccelRefusal::InvalidArgument));
        let mut extra = bytes.clone();
        extra.extend_from_slice(&bytes[bytes.len() - sync..]);
        assert_eq!(decode_sequence(&extra, 3), Err(AccelRefusal::InvalidArgument));
        let mut reserved = bytes.clone();
        let last = reserved.len() - 1;
        reserved[last] = 1;
        assert_eq!(decode_sequence(&reserved, 3), Err(AccelRefusal::InvalidArgument));
        // A table of no members, or past the entry ceiling, refuses.
        assert_eq!(decode_sequence(&[], 0), Err(AccelRefusal::InvalidArgument));
    }

    #[test]
    fn staged_sequence_entry_is_canonical_and_rejects_dead_tail_bytes() {
        let entry = sequence_entry(LaunchShape::ArtifactDefined, true);
        let wire = encode_sequence_entry(entry);
        // Artifact-defined is the all-zero geometry, never a magic 1x1x1.
        assert_eq!(wire.groups, [U32::ZERO; 3]);
        assert_eq!(wire.local, [U32::ZERO; 3]);
        assert_eq!(wire.flags.get(), AccelSequenceEntryWire::FLAG_BARRIER);
        assert_eq!(decode_sequence_entry(&wire, &mut core::iter::empty()), Some(entry));
        let mut bad = wire;
        bad.bindings[1].len = U64::new(1);
        assert_eq!(decode_sequence_entry(&bad, &mut core::iter::empty()), None);
    }

    #[test]
    fn staged_sequence_entry_carries_launch_geometry_and_barrier_flag() {
        use zerocopy::IntoBytes;
        let explicit = LaunchShape::Workgroups(Workgroups::new([257, 3, 2], [32, 1, 1]).unwrap());
        let entry = sequence_entry(explicit, false);
        let wire = encode_sequence_entry(entry);
        assert_eq!(wire.flags.get(), 0);
        // The frozen byte layout: count/flags at 8..16, group count at
        // 16..28, local size at 28..40, bindings from 40, all LE.
        let bytes = wire.as_bytes();
        assert_eq!(bytes.len(), 232);
        assert_eq!(&bytes[8..12], &1u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &257u32.to_le_bytes());
        assert_eq!(&bytes[24..28], &2u32.to_le_bytes());
        assert_eq!(&bytes[28..32], &32u32.to_le_bytes());
        assert_eq!(&bytes[40..48], &0x10u64.to_le_bytes());
        let lifted = AccelSequenceEntryWire::read_from_bytes(bytes).unwrap();
        assert_eq!(decode_sequence_entry(&lifted, &mut core::iter::empty()), Some(entry));

        // An explicit one-work-item launch stays distinct from the artifact's.
        let one = sequence_entry(
            LaunchShape::Workgroups(Workgroups::new([1; 3], [1; 3]).unwrap()),
            true,
        );
        let one_wire = encode_sequence_entry(one);
        assert_ne!(
            one_wire,
            encode_sequence_entry(sequence_entry(LaunchShape::ArtifactDefined, true))
        );
        assert_eq!(decode_sequence_entry(&one_wire, &mut core::iter::empty()), Some(one));

        // Every reserved flag bit refuses, with or without the barrier bit.
        for bit in 1..32 {
            for base in [0, AccelSequenceEntryWire::FLAG_BARRIER] {
                let mut bad = wire;
                bad.flags = U32::new(base | (1 << bit));
                assert_eq!(decode_sequence_entry(&bad, &mut core::iter::empty()), None, "flag bit {bit}");
            }
        }
        // Geometry is all zero or a valid Workgroups: a zero axis beside a
        // nonzero one refuses in either vector, as SubmitLaunch's would.
        for axis in 0..3 {
            let mut bad = wire;
            bad.groups[axis] = U32::ZERO;
            assert_eq!(decode_sequence_entry(&bad, &mut core::iter::empty()), None);
            let mut bad = wire;
            bad.local[axis] = U32::ZERO;
            assert_eq!(decode_sequence_entry(&bad, &mut core::iter::empty()), None);
            let mut bad = encode_sequence_entry(sequence_entry(LaunchShape::ArtifactDefined, true));
            bad.local[axis] = U32::new(1);
            assert_eq!(decode_sequence_entry(&bad, &mut core::iter::empty()), None);
        }
    }

    #[test]
    fn provider_errors_preserve_domain_and_signed_code_in_every_outcome() {
        for (domain, code) in [
            (0, 0),
            (u32::MAX, i64::MIN),
            (0x5550_5649, 0xfedc_ba98),
            (17, i64::MAX),
        ] {
            let refusal = AccelRefusal::External { domain, code };
            let state = AccelEventState::from_contract(virtio_accel_core::EventState::Failed(
                virtio_accel_core::BackendError::External { domain, code },
            ));
            assert_eq!(state, AccelEventState::Failed(refusal));
            for response in [
                AccelResponse::Refused(refusal),
                AccelResponse::SubmitIndeterminate { event: 42, refusal },
                AccelResponse::Event { event: 42, state },
            ] {
                let wire = encode_rep(&response, 0);
                assert_eq!(wire.refusal, AccelRepWire::REFUSE_EXTERNAL);
                assert_eq!(&wire.info[..4], &domain.to_le_bytes());
                assert_eq!(&wire.info[4..12], &code.to_le_bytes());
                assert_eq!(decode_rep(&wire), Some((response, 0)));
                for byte in 12..80 {
                    let mut bad = wire;
                    bad.info[byte] = 1;
                    assert_eq!(decode_rep(&bad), None, "reserved byte {byte}");
                }
                let mut bad = wire;
                bad.grants = 1;
                assert_eq!(decode_rep(&bad), None);
            }
        }
    }

    #[test]
    fn provider_detail_cannot_hide_in_other_reply_shapes() {
        let external = encode_rep(
            &AccelResponse::Event {
                event: 42,
                state: AccelEventState::Failed(AccelRefusal::External {
                    domain: 7,
                    code: -9,
                }),
            },
            0,
        );
        for state in [1, 2, 4] {
            let mut bad = external;
            bad.state = state;
            bad.refusal = 0;
            assert_eq!(decode_rep(&bad), None);
        }
        let mut bad = external;
        bad.refusal = AccelRepWire::REFUSE_INTERNAL;
        assert_eq!(decode_rep(&bad), None);
        for tag in [
            AccelRepWire::TAG_WROTE,
            AccelRepWire::TAG_SUBMITTED,
            AccelRepWire::TAG_RELEASED,
        ] {
            let mut bad = external;
            bad.tag = tag;
            bad.state = 0;
            bad.refusal = 0;
            assert_eq!(decode_rep(&bad), None);
        }
    }

    #[test]
    fn provisioning_correlates_capacity_and_refuses_reserved_words() {
        let request = AccelRequest::ProvisionMemory {
            announced: SlotRef::from_wire(17),
            buffer_bytes: 8192,
            program_bytes: 4096,
        };
        let wire = encode_req(&request);
        assert!(reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 12288 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 8192 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 12288 },
            1
        ));
        assert!(!reply_matches(
            &wire,
            &AccelResponse::Attached { bytes: 12288 },
            0
        ));
        assert!(reply_matches(
            &wire,
            &AccelResponse::Refused(AccelRefusal::Unsupported),
            0
        ));
        let mut malformed = wire;
        malformed.d = U64::new(1);
        assert!(decode_req(&malformed).is_none());
        malformed = wire;
        malformed.target[0] = 1;
        assert!(decode_req(&malformed).is_none());
        malformed = wire;
        malformed.a = U64::new(u64::from(u32::MAX) + 1);
        assert!(decode_req(&malformed).is_none());
        let overflow = encode_req(&AccelRequest::ProvisionMemory {
            announced: SlotRef::from_wire(17),
            buffer_bytes: u64::MAX,
            program_bytes: 1,
        });
        assert!(!reply_matches(
            &overflow,
            &AccelResponse::Provisioned { bytes: u64::MAX },
            0
        ));
    }

    #[test]
    fn existing_pool_reservation_conveys_no_capability_and_requires_exact_budget() {
        let request = AccelRequest::ReserveMemory {
            buffer_bytes: 8192,
            program_bytes: 4096,
        };
        let wire = encode_req(&request);
        assert_eq!(decode_req(&wire), Some(request));
        assert!(reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 12288 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 8192 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 16384 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &AccelResponse::Provisioned { bytes: 12288 },
            1
        ));
        let mut malformed = wire;
        malformed.a = U64::new(1);
        assert_eq!(decode_req(&malformed), None);
        malformed = wire;
        malformed.d = U64::new(1);
        assert_eq!(decode_req(&malformed), None);
    }

    /// Cross-version fixtures copied from origin/main's deployed OP13 layout:
    /// scalar remains artifact-defined, non-scalar remains a legacy dispatch,
    /// and OP18 is the disjoint portable workgroup dialect.
    #[test]
    fn origin_main_submit_bytes_remain_compatible_and_op18_is_distinct() {
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer: 1,
            offset: 0,
            len: 8,
            slot: 0,
            access: AccessMode::Read,
        }])
        .unwrap();
        let artifact = AccelRequest::Submit {
            queue: 2,
            program: 3,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::scalar(),
        };
        let artifact_wire = encode_req(&artifact);
        assert_eq!(artifact_wire.op, 13);
        let mut origin_scalar = [0u8; 48];
        for at in [0usize, 4, 8, 12, 16, 20, 24] {
            origin_scalar[at] = 1;
        }
        assert_eq!(artifact_wire.target, origin_scalar);
        assert_eq!(decode_req(&artifact_wire), Some(artifact));
        let additive_artifact = AccelRequest::SubmitLaunch {
            queue: 2,
            program: 3,
            timeout_ns: 0,
            bindings,
            launch: LaunchShape::ArtifactDefined,
        };
        assert_eq!(
            encode_req(&additive_artifact),
            artifact_wire,
            "the additive source API still speaks the deployed scalar OP13 bytes"
        );
        assert_eq!(
            decode_req(&encode_req(&additive_artifact)),
            Some(artifact),
            "scalar OP13 canonicalizes to the source-compatible legacy carrier"
        );

        let mut origin_non_scalar = artifact_wire;
        origin_non_scalar.target = [0; 48];
        for (at, value) in [
            (0usize, 16u8),
            (4, 8),
            (8, 2),
            (12, 64),
            (16, 1),
            (20, 1),
            (24, 16),
        ] {
            origin_non_scalar.target[at] = value;
        }
        let legacy = AccelRequest::Submit {
            queue: 2,
            program: 3,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::new([16, 8, 2], [64, 1, 1], 16).unwrap(),
        };
        assert_eq!(decode_req(&origin_non_scalar), Some(legacy));
        assert_eq!(encode_req(&legacy), origin_non_scalar);

        let workgroups = Workgroups::new([16, 8, 2], [64, 1, 1]).unwrap();
        let explicit = AccelRequest::SubmitLaunch {
            queue: 2,
            program: 3,
            timeout_ns: 0,
            bindings,
            launch: LaunchShape::Workgroups(workgroups),
        };
        let explicit_wire = encode_req(&explicit);
        assert_eq!(explicit_wire.op, 18);
        assert_eq!(&explicit_wire.target[24..], &[0; 24]);
        assert_eq!(decode_req(&explicit_wire), Some(explicit));
        assert!(reply_matches(
            &explicit_wire,
            &AccelResponse::Submitted { event: 4 },
            0
        ));
    }

    /// The fail-closed obligations — each a distinct way a peer can lie.
    #[test]
    fn decode_fails_closed() {
        // An all-zero record is not a verb (1-based discriminants).
        let zero = AccelReqWire::read_from_bytes(&[0u8; 288]).unwrap();
        assert_eq!(decode_req(&zero), None);
        // Unknown op.
        let mut w = encode_req(&AccelRequest::Query);
        w.op = 200;
        assert_eq!(decode_req(&w), None);
        // A set reserve refuses.
        let mut w = encode_req(&AccelRequest::Query);
        w.reserved[1] = 1;
        assert_eq!(decode_req(&w), None);
        // A set protocol flag refuses (reserved-zero in v1).
        let mut w = encode_req(&AccelRequest::Query);
        w.flags = U32::new(1);
        assert_eq!(decode_req(&w), None);
        // A live unused per-verb word refuses: Query carries nothing.
        let mut w = encode_req(&AccelRequest::Query);
        w.subject = U64::new(7);
        assert_eq!(decode_req(&w), None);
        // Target words outside LoadProgram refuse.
        let mut w = encode_req(&AccelRequest::PollEvent { event: 9 });
        w.target[0] = 1;
        assert_eq!(decode_req(&w), None);
        // A dead binding slot with live bytes refuses.
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer: 1,
            offset: 0,
            len: 8,
            slot: 0,
            access: AccessMode::Read,
        }])
        .unwrap();
        let mut w = encode_req(&AccelRequest::Submit {
            queue: 1,
            program: 2,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::scalar(),
        });
        w.bindings[3].len = U64::new(4);
        assert_eq!(decode_req(&w), None);
        // A binding access outside the vocabulary refuses.
        let mut w = encode_req(&AccelRequest::Submit {
            queue: 1,
            program: 2,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::scalar(),
        });
        w.bindings[0].access = 4;
        assert_eq!(decode_req(&w), None);
        // Binding counts outside 1..=MAX refuse.
        let mut w = encode_req(&AccelRequest::Submit {
            queue: 1,
            program: 2,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::scalar(),
        });
        w.c = U64::new(0);
        assert_eq!(decode_req(&w), None);
        w.c = U64::new(MAX_ACCEL_BINDINGS as u64 + 1);
        assert_eq!(decode_req(&w), None);
        // Legacy Submit rejects invalid axes/SIMD and reserved tail bytes.
        let mut w = encode_req(&AccelRequest::Submit {
            queue: 1,
            program: 2,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::scalar(),
        });
        w.target[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(decode_req(&w), None);
        let mut w = encode_req(&AccelRequest::Submit {
            queue: 1,
            program: 2,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::scalar(),
        });
        w.target[24..28].copy_from_slice(&24u32.to_le_bytes());
        assert_eq!(decode_req(&w), None);
        let mut w = encode_req(&AccelRequest::Submit {
            queue: 1,
            program: 2,
            timeout_ns: 0,
            bindings,
            dispatch: Dispatch::scalar(),
        });
        w.target[28] = 1;
        assert_eq!(decode_req(&w), None);
        // Explicit geometry refuses zero axes and reserved tail bytes.
        let explicit = AccelRequest::SubmitLaunch {
            queue: 1,
            program: 2,
            timeout_ns: 0,
            bindings,
            launch: LaunchShape::Workgroups(Workgroups::new([1; 3], [1; 3]).unwrap()),
        };
        let mut w = encode_req(&explicit);
        w.target[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(decode_req(&w), None);
        let mut w = encode_req(&explicit);
        w.target[24] = 1;
        assert_eq!(decode_req(&w), None);

        // Reply side: zero/unknown tags refuse.
        let zero = AccelRepWire::read_from_bytes(&[0u8; 96]).unwrap();
        assert_eq!(decode_rep(&zero), None);
        let mut w = encode_rep(&AccelResponse::Signaled, 1);
        w.tag = 99;
        assert_eq!(decode_rep(&w), None);
        // The per-tag grant law: Signaled must carry exactly one, and a
        // grant-bearing data reply is a protocol break.
        let mut w = encode_rep(&AccelResponse::Signaled, 1);
        w.grants = 0;
        assert_eq!(decode_rep(&w), None);
        let mut w = encode_rep(&AccelResponse::Wrote { buffer: 1 }, 0);
        w.grants = 1;
        assert_eq!(decode_rep(&w), None);
        // A set reserve refuses.
        let mut w = encode_rep(&AccelResponse::Wrote { buffer: 1 }, 0);
        w.reserved[0] = 1;
        assert_eq!(decode_rep(&w), None);
        // Event-state / refusal inconsistencies refuse: a failed state
        // needs a refusal code, a healthy one forbids it.
        let mut w = encode_rep(
            &AccelResponse::Event {
                event: 1,
                state: AccelEventState::Complete,
            },
            0,
        );
        w.refusal = AccelRepWire::REFUSE_BUSY;
        assert_eq!(decode_rep(&w), None);
        let mut w = encode_rep(
            &AccelResponse::Event {
                event: 1,
                state: AccelEventState::Failed(AccelRefusal::Busy),
            },
            0,
        );
        w.refusal = 0;
        assert_eq!(decode_rep(&w), None);
        // An unknown refusal code refuses.
        let mut w = encode_rep(&AccelResponse::Refused(AccelRefusal::Busy), 0);
        w.refusal = 200;
        assert_eq!(decode_rep(&w), None);
        // Info payload tail bytes are reserved-zero.
        let info = every_response()[1].0;
        let mut w = encode_rep(&info, 0);
        w.info[78] = 1;
        assert_eq!(decode_rep(&w), None);
    }

    /// The uncorrelated-pair gate: legal pairings pass, echo mismatches
    /// and cross-op strays refuse.
    #[test]
    fn reply_matching_pins_the_op_and_the_echo() {
        let free = encode_req(&AccelRequest::FreeBuffer { buffer: 7 });
        assert!(reply_matches(
            &free,
            &AccelResponse::Released { subject: 7 },
            0
        ));
        assert!(!reply_matches(
            &free,
            &AccelResponse::Released { subject: 8 },
            0
        ));
        assert!(!reply_matches(
            &free,
            &AccelResponse::Wrote { buffer: 7 },
            0
        ));
        assert!(reply_matches(
            &free,
            &AccelResponse::Refused(AccelRefusal::BadObject),
            0
        ));
        // A grant beside anything but Signaled refuses.
        assert!(!reply_matches(
            &free,
            &AccelResponse::Released { subject: 7 },
            1
        ));
        let signal = encode_req(&AccelRequest::Signal);
        assert!(reply_matches(&signal, &AccelResponse::Signaled, 1));
        assert!(!reply_matches(&signal, &AccelResponse::Signaled, 0));
        // A create's reply must be its own kind — a stale BufferMade
        // cannot answer CreateContext.
        let create = encode_req(&AccelRequest::CreateContext { flags: 0 });
        assert!(!reply_matches(
            &create,
            &AccelResponse::BufferMade { buffer: 1 },
            0
        ));
    }

    /// The pair window fits the founder's one-page ring lease at the
    /// standard capacities.
    #[test]
    fn the_pair_window_fits_one_page() {
        let bytes =
            kore_rt::svc::window_bytes::<AccelWire>(kore_rt::svc::REQ_CAP, kore_rt::svc::REP_CAP);
        assert!(bytes <= 4096, "window is {bytes} bytes");
    }
    #[test]
    fn owned_staging_replies_are_operation_specific_and_grant_free() {
        let attach = encode_req(&AccelRequest::AttachOwned {
            announced: SlotRef::from_wire(9),
        });
        let detach = encode_req(&AccelRequest::DetachStaging);
        assert!(reply_matches(
            &attach,
            &AccelResponse::Attached { bytes: 4096 },
            0
        ));
        assert!(!reply_matches(&attach, &AccelResponse::Detached, 0));
        assert!(!reply_matches(
            &attach,
            &AccelResponse::Attached { bytes: 0 },
            0
        ));
        assert!(!reply_matches(
            &attach,
            &AccelResponse::Attached { bytes: 4097 },
            0
        ));
        assert!(reply_matches(&detach, &AccelResponse::Detached, 0));
        assert!(!reply_matches(
            &detach,
            &AccelResponse::Attached { bytes: 4096 },
            0
        ));
        assert!(!reply_matches(&detach, &AccelResponse::Detached, 1));
        let mut invalid = detach;
        invalid.a = U64::new(1);
        assert!(decode_req(&invalid).is_none());
        let mut invalid = attach;
        invalid.a = U64::new(1u64 << 32);
        assert!(decode_req(&invalid).is_none());
        let mut invalid = encode_rep(&AccelResponse::Detached, 0);
        invalid.id = U64::new(1);
        assert!(decode_rep(&invalid).is_none());
    }
}
