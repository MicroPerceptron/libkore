//! `wire` — the exec enum ⇄ wire codecs (`res/USERLAND.md` §5.3), v3:
//! the request-side authority-transfer protocol (2026-07-28 ruling) in
//! its *convey roots, carve leaves* shape.
//!
//! The zerocopy forms live in [`kore_abi::svc`]; decode is fail-closed
//! (unknown ops and the retired legacy spawn, bad counts, out-of-range
//! convey indices,
//! `W && X` or empty segment access, out-of-vocabulary rights/kind
//! envelopes, a window `write` byte outside `{0,1}`).  The nested
//! [`SpawnSegment`]/[`SpawnWire`]/[`SpawnWindow`]/[`SpawnData`] tables
//! ride the request whole — they are themselves frozen wire types, and
//! the *service* still validates them against the LANDED convey count
//! (`validate_spawn`) exactly as `Bootstrap::of` validated the legacy
//! block: the codec refuses shapes, the serve step refuses semantics.

use kore_abi::cap::Rights;
use kore_abi::image::ImageName;
use kore_abi::session::{SHARE_ACCESS_R, SHARE_ACCESS_W, SHARE_ACCESS_X};
use kore_abi::svc::{
    MAX_CONVEY, MAX_SPAWN_DATA, MAX_SPAWN_SEGMENTS, MAX_SPAWN_WINDOWS, MAX_SPAWN_WIRING,
    SPAWN_WIRE_MOVE, SpawnData, SpawnSegment, SpawnWindow, SpawnWire, TaskRepWire, TaskReqWire,
};
use kore_svc_dagfs::Rights as FsRights;

use kore_rt::svc::{MAX_INVOCATION_AUXILIARIES, SvcProtocol};

fn encode_spawn(w: &mut TaskReqWire, s: &SpawnSpec, op: u8) {
    w.op = op;
    w.reserved = if s.return_handle {
        TaskReqWire::SPAWN_RETURN_HANDLE
    } else {
        0
    } | if s.linked {
        TaskReqWire::SPAWN_LINKED
    } else {
        0
    } | if s.delegable_handle {
        TaskReqWire::SPAWN_DELEGABLE_HANDLE
    } else {
        0
    } | if s.return_shared_wake {
        TaskReqWire::SPAWN_RETURN_SHARED_WAKE
    } else {
        0
    };
    w.seg_count = s.seg_count;
    w.wiring_count = s.wiring_count;
    w.window_count = s.window_count;
    w.data_count = s.data_count;
    w.boot_write_idx = s.boot_write_idx;
    w.boot_share_idx = s.boot_share_idx;
    w.entry = s.entry.into();
    w.stack = s.stack.into();
    w.arena_slots = s.arena_slots.into();
    w.boot_va = s.boot_va.into();
    w.segments = s.segments;
    w.wiring = s.wiring;
    w.windows = s.windows;
    w.data = s.data;
    w.target = kore_abi::image::ImageName::empty();
}

/// Lower a request to its wire form (total).
pub fn encode_req(req: &TaskRequest) -> TaskReqWire {
    let mut w = TaskReqWire {
        op: 0,
        seg_count: 0,
        wiring_count: 0,
        boot_write_idx: 0,
        boot_share_idx: 0,
        window_count: 0,
        data_count: 0,
        reserved: 0,
        slot: 0.into(),
        status: 0.into(),
        entry: 0.into(),
        stack: 0.into(),
        arena_slots: 0.into(),
        reserved2: 0.into(),
        boot_va: 0.into(),
        segments: [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS],
        wiring: [SpawnWire::EMPTY; MAX_SPAWN_WIRING],
        windows: [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS],
        data: [SpawnData::EMPTY; MAX_SPAWN_DATA],
        target: kore_abi::image::ImageName::empty(),
    };
    match *req {
        TaskRequest::ProcessorStatus { first } => {
            w.op = TaskReqWire::OP_PROCESSOR;
            w.entry = u64::from(first).into();
        }
        TaskRequest::Install {
            target,
            profile,
            include_context,
            from_tree,
        } => {
            w.op = TaskReqWire::OP_INSTALL;
            w.target = target;
            w.reserved2 = profile.into();
            if include_context {
                w.reserved |= TaskReqWire::INSTALL_CONTEXT;
            }
            if from_tree {
                w.reserved |= TaskReqWire::INSTALL_FROM_TREE;
            }
        }
        TaskRequest::RetireGeneration { target } => {
            w.op = TaskReqWire::OP_RETIRE_GENERATION;
            w.target = target;
        }
        TaskRequest::SeedApp { target } => {
            w.op = TaskReqWire::OP_SEED_APP;
            w.target = target;
        }
        TaskRequest::SeedLibrary { target } => {
            w.op = TaskReqWire::OP_SEED_LIBRARY;
            w.target = target;
        }
        TaskRequest::TreeLoadProbe { target } => {
            w.op = TaskReqWire::OP_TREE_LOAD_PROBE;
            w.target = target;
        }
        TaskRequest::TreeAcquire { target } => {
            w.op = TaskReqWire::OP_TREE_ACQUIRE;
            w.target = target;
        }
        // No live caller constructs this; it exists so the tombstone family
        // round-trips through one vocabulary. Any frozen value serves.
        TaskRequest::Retired => w.op = TaskReqWire::OP_LAUNCH,
        TaskRequest::MintClient { fs_rights } => {
            w.op = TaskReqWire::OP_MINT_CLIENT;
            w.reserved2 = u32::from(fs_rights.bits()).into();
        }
        TaskRequest::ClientBank { face } => {
            w.op = TaskReqWire::OP_CLIENT_BANK;
            w.slot = u32::from(face).into();
        }
        TaskRequest::ClientContext { face } => {
            w.op = TaskReqWire::OP_CLIENT_CONTEXT;
            w.slot = u32::from(face).into();
        }
        TaskRequest::RetireClient { face } => {
            w.op = TaskReqWire::OP_RETIRE_CLIENT;
            w.slot = u32::from(face).into();
        }
        TaskRequest::ConfirmClientRetired { face, retirement } => {
            w.op = TaskReqWire::OP_CONFIRM_CLIENT_RETIRED;
            w.slot = u32::from(face).into();
            w.entry = retirement.into();
        }
        TaskRequest::DeviceCensus { page } => {
            w.op = TaskReqWire::OP_DEVICE_CENSUS;
            w.slot = u32::from(page).into();
        }
        TaskRequest::SealInstaller => w.op = TaskReqWire::OP_SEAL_INSTALLER,
        TaskRequest::Invoke(ref s) => encode_spawn(&mut w, s, TaskReqWire::OP_INVOKE),
    }
    w
}

/// One wire segment's decode-level shape check ([`MAX_CONVEY`] bound,
/// non-empty access within the R/W/X vocabulary, W^X) — representable
/// refusals live here; the landed-count and geometry semantics are the
/// serve step's.
fn seg_shape_ok(s: &SpawnSegment) -> bool {
    (s.convey_idx as usize) < MAX_CONVEY
        && s.access != 0
        && s.access & !(SHARE_ACCESS_R | SHARE_ACCESS_W | SHARE_ACCESS_X) == 0
        && !(s.access & SHARE_ACCESS_W != 0 && s.access & SHARE_ACCESS_X != 0)
}

/// One wire wiring entry's decode-level shape check: convey bound, and
/// the `kind`/`rights` byte envelopes (a derive carries a `u8` kind and
/// `u8` rights — wider values are never expressible downstream).
fn wire_shape_ok(w: &SpawnWire) -> bool {
    let moving = w.reserved & SPAWN_WIRE_MOVE != 0;
    (w.convey_idx as usize) < MAX_CONVEY
        && w.reserved & !SPAWN_WIRE_MOVE == 0
        && w.kind.get() <= u8::MAX as u16
        && w.rights.get() <= u8::MAX as u32
        && kore_abi::cap::Rights::from_bits(w.rights.get() as u8).is_some()
        && (!moving || w.rights.get() == 0)
}

/// One wire window entry's decode-level shape check: convey bound, a
/// boolean `write` byte, and flags inside the known envelope (an unknown
/// bit is a future revision or garbage — refuse, never guess).  Geometry
/// (alignment, non-zero length, the landed-count reference) is the serve
/// step's, exactly as for segments.
fn window_shape_ok(w: &SpawnWindow) -> bool {
    (w.convey_idx as usize) < MAX_CONVEY
        && w.write <= 1
        && w.flags & !(kore_abi::svc::WINDOW_FORWARD | kore_abi::svc::WINDOW_SUBLEASE) == 0
        // A sublease delivery is a full-grant lessor move: `write` is
        // mandatorily 1 and FORWARD has no coherent composition with it.
        && (w.flags & kore_abi::svc::WINDOW_SUBLEASE == 0
            || (w.write == 1 && w.flags & kore_abi::svc::WINDOW_FORWARD == 0))
        && w.reserved == [0]
}

fn client_control_shape_ok(w: &TaskReqWire, carries_face: bool) -> bool {
    w.target.is_empty()
        && w.seg_count == 0
        && w.wiring_count == 0
        && w.boot_write_idx == 0
        && w.boot_share_idx == 0
        && w.window_count == 0
        && w.data_count == 0
        && (!carries_face || w.slot.get() <= u8::MAX as u32)
        && (carries_face || w.slot.get() == 0)
        && w.status.get() == 0
        && w.entry.get() == 0
        && w.stack.get() == 0
        && w.arena_slots.get() == 0
        && w.reserved2.get() == 0
        && w.boot_va.get() == 0
        && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
        && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
        && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
        && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA]
}

fn mint_client_shape_ok(w: &TaskReqWire) -> bool {
    let rights = kore_svc_dagfs::Rights::from_bits_exact(w.reserved2.get() as u8);
    client_control_shape_ok_except_rights(w)
        && w.reserved2.get() <= u32::from(u8::MAX)
        && rights.is_some()
}

fn client_control_shape_ok_except_rights(w: &TaskReqWire) -> bool {
    w.target.is_empty()
        && w.seg_count == 0
        && w.wiring_count == 0
        && w.boot_write_idx == 0
        && w.boot_share_idx == 0
        && w.window_count == 0
        && w.data_count == 0
        && w.slot.get() == 0
        && w.status.get() == 0
        && w.entry.get() == 0
        && w.stack.get() == 0
        && w.arena_slots.get() == 0
        && w.boot_va.get() == 0
        && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
        && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
        && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
        && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA]
}

/// Lift a wire request — fail-closed.
pub fn decode_req(w: &TaskReqWire) -> Option<TaskRequest> {
    if w.op == TaskReqWire::OP_PROCESSOR {
        let req = TaskRequest::ProcessorStatus {
            first: u16::try_from(w.entry.get()).ok()?,
        };
        use zerocopy::IntoBytes;
        return (w.as_bytes() == encode_req(&req).as_bytes()).then_some(req);
    }
    // Tombstones outrank every shape rule: a frozen discriminant answers a
    // typed refusal no matter what payload rides with it, so a stale caller
    // is denied rather than dropped, and no payload interpretation of a
    // retired protocol survives anywhere.
    if TaskReqWire::op_retired(w.op) {
        return Some(TaskRequest::Retired);
    }
    let reserved_valid = match w.op {
        TaskReqWire::OP_INVOKE => true,
        TaskReqWire::OP_INSTALL => {
            w.reserved & !(TaskReqWire::INSTALL_CONTEXT | TaskReqWire::INSTALL_FROM_TREE) == 0
        }
        _ => w.reserved == 0,
    };
    if !reserved_valid {
        return None;
    }
    Some(match w.op {
        TaskReqWire::OP_INVOKE => {
            let ns = w.seg_count as usize;
            let nw = w.wiring_count as usize;
            let nwin = w.window_count as usize;
            let nd = w.data_count as usize;
            if ns > MAX_SPAWN_SEGMENTS
                || nw > MAX_SPAWN_WIRING
                || nwin > MAX_SPAWN_WINDOWS
                || nd > MAX_SPAWN_DATA
                || w.reserved
                    & !(TaskReqWire::SPAWN_RETURN_HANDLE
                        | TaskReqWire::SPAWN_LINKED
                        | TaskReqWire::SPAWN_DELEGABLE_HANDLE
                        | TaskReqWire::SPAWN_RETURN_SHARED_WAKE)
                    != 0
                // Supply-mode boot: the low bits name the supply's convey
                // entry; `boot_share_idx` is a page offset, not an index,
                // so neither the convey bound nor the distinctness rule
                // applies to it (a `u8` of pages cannot overflow — the
                // kernel's over-grant guard owns the real span check).
                || if w.boot_write_idx & kore_abi::svc::BOOT_FROM_SUPPLY != 0 {
                    ((w.boot_write_idx & !kore_abi::svc::BOOT_FROM_SUPPLY) as usize) >= MAX_CONVEY
                } else {
                    (w.boot_write_idx as usize) >= MAX_CONVEY
                        || (w.boot_share_idx as usize) >= MAX_CONVEY
                        || w.boot_write_idx == w.boot_share_idx
                }
                || !w.segments[..ns].iter().all(seg_shape_ok)
                || !w.wiring[..nw].iter().all(wire_shape_ok)
                || !w.windows[..nwin].iter().all(window_shape_ok)
                || w.reserved
                    & (TaskReqWire::SPAWN_RETURN_HANDLE | TaskReqWire::SPAWN_LINKED)
                    != (TaskReqWire::SPAWN_RETURN_HANDLE | TaskReqWire::SPAWN_LINKED)
                || !w.target.is_empty()
            {
                return None;
            }
            let spec = SpawnSpec {
                return_handle: w.reserved & TaskReqWire::SPAWN_RETURN_HANDLE != 0,
                delegable_handle: w.reserved & TaskReqWire::SPAWN_DELEGABLE_HANDLE != 0,
                return_shared_wake: w.reserved & TaskReqWire::SPAWN_RETURN_SHARED_WAKE != 0,
                linked: w.reserved & TaskReqWire::SPAWN_LINKED != 0,
                entry: w.entry.get(),
                stack: w.stack.get(),
                arena_slots: w.arena_slots.get(),
                boot_va: w.boot_va.get(),
                boot_write_idx: w.boot_write_idx,
                boot_share_idx: w.boot_share_idx,
                seg_count: w.seg_count,
                wiring_count: w.wiring_count,
                window_count: w.window_count,
                data_count: w.data_count,
                segments: w.segments,
                wiring: w.wiring,
                windows: w.windows,
                data: w.data,
            };
            TaskRequest::Invoke(spec)
        }
        TaskReqWire::OP_INSTALL
            if w.target.as_str().is_some()
                && w.seg_count == 0
                && w.wiring_count == 0
                && w.boot_write_idx == 0
                && w.boot_share_idx == 0
                && w.window_count == 0
                && w.data_count == 0
                && w.slot.get() == 0
                && w.status.get() == 0
                && w.entry.get() == 0
                && w.stack.get() == 0
                && w.arena_slots.get() == 0
                && kore_abi::image::profile_valid(w.reserved2.get())
                && w.boot_va.get() == 0
                && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
                && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
                && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
                && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA]
                && (w.reserved & TaskReqWire::INSTALL_FROM_TREE == 0 || w.reserved2.get() == 0) =>
        {
            TaskRequest::Install {
                target: w.target,
                profile: w.reserved2.get(),
                include_context: w.reserved & TaskReqWire::INSTALL_CONTEXT != 0,
                from_tree: w.reserved & TaskReqWire::INSTALL_FROM_TREE != 0,
            }
        }
        TaskReqWire::OP_RETIRE_GENERATION
            if w.target.as_str().is_some()
                && w.seg_count == 0
                && w.wiring_count == 0
                && w.boot_write_idx == 0
                && w.boot_share_idx == 0
                && w.window_count == 0
                && w.data_count == 0
                && w.slot.get() == 0
                && w.status.get() == 0
                && w.entry.get() == 0
                && w.stack.get() == 0
                && w.arena_slots.get() == 0
                && w.reserved2.get() == 0
                && w.boot_va.get() == 0
                && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
                && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
                && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
                && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA] =>
        {
            TaskRequest::RetireGeneration { target: w.target }
        }
        TaskReqWire::OP_SEED_APP
            if w.target.as_str().is_some()
                && w.seg_count == 0
                && w.wiring_count == 0
                && w.boot_write_idx == 0
                && w.boot_share_idx == 0
                && w.window_count == 0
                && w.data_count == 0
                && w.slot.get() == 0
                && w.status.get() == 0
                && w.entry.get() == 0
                && w.stack.get() == 0
                && w.arena_slots.get() == 0
                && w.reserved2.get() == 0
                && w.boot_va.get() == 0
                && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
                && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
                && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
                && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA] =>
        {
            TaskRequest::SeedApp { target: w.target }
        }
        TaskReqWire::OP_SEED_LIBRARY
            if w.target.as_str().is_some()
                && w.seg_count == 0
                && w.wiring_count == 0
                && w.boot_write_idx == 0
                && w.boot_share_idx == 0
                && w.window_count == 0
                && w.data_count == 0
                && w.slot.get() == 0
                && w.status.get() == 0
                && w.entry.get() == 0
                && w.stack.get() == 0
                && w.arena_slots.get() == 0
                && w.reserved2.get() == 0
                && w.boot_va.get() == 0
                && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
                && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
                && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
                && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA] =>
        {
            TaskRequest::SeedLibrary { target: w.target }
        }
        TaskReqWire::OP_TREE_LOAD_PROBE
            if w.target.as_str().is_some()
                && w.seg_count == 0
                && w.wiring_count == 0
                && w.boot_write_idx == 0
                && w.boot_share_idx == 0
                && w.window_count == 0
                && w.data_count == 0
                && w.slot.get() == 0
                && w.status.get() == 0
                && w.entry.get() == 0
                && w.stack.get() == 0
                && w.arena_slots.get() == 0
                && w.reserved2.get() == 0
                && w.boot_va.get() == 0
                && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
                && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
                && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
                && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA] =>
        {
            TaskRequest::TreeLoadProbe { target: w.target }
        }
        TaskReqWire::OP_TREE_ACQUIRE
            if w.target.as_str().is_some()
                && w.seg_count == 0
                && w.wiring_count == 0
                && w.boot_write_idx == 0
                && w.boot_share_idx == 0
                && w.window_count == 0
                && w.data_count == 0
                && w.slot.get() == 0
                && w.status.get() == 0
                && w.entry.get() == 0
                && w.stack.get() == 0
                && w.arena_slots.get() == 0
                && w.reserved2.get() == 0
                && w.boot_va.get() == 0
                && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
                && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
                && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
                && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA] =>
        {
            TaskRequest::TreeAcquire { target: w.target }
        }
        TaskReqWire::OP_SEAL_INSTALLER
            if w.target.is_empty()
                && w.seg_count == 0
                && w.wiring_count == 0
                && w.boot_write_idx == 0
                && w.boot_share_idx == 0
                && w.window_count == 0
                && w.data_count == 0
                && w.slot.get() == 0
                && w.status.get() == 0
                && w.entry.get() == 0
                && w.stack.get() == 0
                && w.arena_slots.get() == 0
                && w.reserved2.get() == 0
                && w.boot_va.get() == 0
                && w.segments == [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS]
                && w.wiring == [SpawnWire::EMPTY; MAX_SPAWN_WIRING]
                && w.windows == [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS]
                && w.data == [SpawnData::EMPTY; MAX_SPAWN_DATA] =>
        {
            TaskRequest::SealInstaller
        }
        TaskReqWire::OP_MINT_CLIENT if mint_client_shape_ok(w) => TaskRequest::MintClient {
            fs_rights: kore_svc_dagfs::Rights::from_bits_exact(w.reserved2.get() as u8)?,
        },
        TaskReqWire::OP_CLIENT_BANK if client_control_shape_ok(w, true) => {
            TaskRequest::ClientBank {
                face: w.slot.get() as u8,
            }
        }
        TaskReqWire::OP_CLIENT_CONTEXT if client_control_shape_ok(w, true) => {
            TaskRequest::ClientContext {
                face: w.slot.get() as u8,
            }
        }
        TaskReqWire::OP_RETIRE_CLIENT if client_control_shape_ok(w, true) => {
            TaskRequest::RetireClient {
                face: w.slot.get() as u8,
            }
        }
        TaskReqWire::OP_CONFIRM_CLIENT_RETIRED
            if w.entry.get() != 0
                && client_control_shape_ok(
                    &TaskReqWire {
                        entry: 0.into(),
                        ..*w
                    },
                    true,
                ) =>
        {
            TaskRequest::ConfirmClientRetired {
                face: w.slot.get() as u8,
                retirement: w.entry.get(),
            }
        }
        TaskReqWire::OP_DEVICE_CENSUS
            if client_control_shape_ok(w, true)
                && (w.slot.get() as usize) < kore_abi::device::DEVICE_CENSUS_PAGES =>
        {
            TaskRequest::DeviceCensus {
                page: w.slot.get() as u8,
            }
        }
        // The retired opcodes (1..=9 and 16) were answered above; everything
        // unknown drops here — garbage earns no reply.
        _ => return None,
    })
}

/// Lower a response + the effected grant count (total).
pub fn encode_rep(rep: &TaskResponse, grants: u8) -> TaskRepWire {
    let mut w = TaskRepWire {
        tag: 0,
        refusal: 0,
        grants,
        index: 0,
        status: 0.into(),
        id: 0.into(),
        snapshot: kore_abi::monitor::TelemetrySnapshot::EMPTY,
        filesystem: kore_abi::monitor::FilesystemSnapshot::EMPTY,
        net: kore_abi::monitor::NetSnapshot::EMPTY,
        processor: Default::default(),
        census: kore_abi::device::DeviceCensusPage::EMPTY,
    };
    match *rep {
        TaskResponse::ProcessorStatus(snapshot) => {
            w.tag = TaskRepWire::TAG_PROCESSOR;
            w.processor = snapshot;
        }
        TaskResponse::DeviceCensus(page) => {
            w.tag = TaskRepWire::TAG_DEVICE_CENSUS;
            w.census = page;
        }
        TaskResponse::Installed { segments } => {
            w.tag = TaskRepWire::TAG_INSTALLED;
            w.index = segments;
        }
        TaskResponse::InstallerSealed => w.tag = TaskRepWire::TAG_INSTALLER_SEALED,
        TaskResponse::AppSeeded { seeded, failure } => {
            w.tag = TaskRepWire::TAG_APP_SEEDED;
            w.index = (u8::from(seeded) * TaskRepWire::APP_SEEDED)
                | ((failure << TaskRepWire::APP_FAILURE_SHIFT) & TaskRepWire::APP_FAILURE_MASK);
        }
        TaskResponse::LibrarySeeded {
            seeded,
            manifest,
            failure,
        } => {
            w.tag = TaskRepWire::TAG_LIBRARY_SEEDED;
            w.index = (u8::from(seeded) * TaskRepWire::LIBRARY_SEEDED)
                | (u8::from(manifest) * TaskRepWire::LIBRARY_MANIFEST)
                | ((failure << TaskRepWire::LIBRARY_FAILURE_SHIFT)
                    & TaskRepWire::LIBRARY_FAILURE_MASK);
        }
        TaskResponse::TreeProbed {
            linked,
            already_resident,
        } => {
            w.tag = TaskRepWire::TAG_TREE_PROBED;
            w.index = u8::from(linked) * TaskRepWire::TREE_PROBE_LINKED
                | u8::from(already_resident) * TaskRepWire::TREE_PROBE_ALREADY_RESIDENT;
        }
        TaskResponse::TreeAcquired { segments, profile } => {
            w.tag = TaskRepWire::TAG_TREE_ACQUIRED;
            w.index = segments;
            // `status` is tag-contextual like `index`: under this tag it is
            // the granted generation's launch-profile word (a small bitflag
            // set — `profile_valid` bounds it well inside i32 on decode).
            w.status = (profile as i32).into();
        }
        TaskResponse::GenerationRetired => w.tag = TaskRepWire::TAG_GENERATION_RETIRED,
        TaskResponse::ClientMinted { face } => {
            w.tag = TaskRepWire::TAG_CLIENT_MINTED;
            w.index = face;
        }
        TaskResponse::ClientBanked { face } => {
            w.tag = TaskRepWire::TAG_CLIENT_BANKED;
            w.index = face;
        }
        TaskResponse::ClientContext { face } => {
            w.tag = TaskRepWire::TAG_CLIENT_CONTEXT;
            w.index = face;
        }
        TaskResponse::ClientRetired => w.tag = TaskRepWire::TAG_CLIENT_RETIRED,
        TaskResponse::ClientRetiring { face, retirement } => {
            w.tag = TaskRepWire::TAG_CLIENT_RETIRING;
            w.index = face;
            w.id = retirement.into();
        }
        TaskResponse::ClientRetirementConfirmed { face, retirement } => {
            w.tag = TaskRepWire::TAG_CLIENT_RETIREMENT_CONFIRMED;
            w.index = face;
            w.id = retirement.into();
        }
        TaskResponse::Spawned {
            id,
            segments,
            snapshot,
            filesystem,
            net,
        } => {
            w.tag = TaskRepWire::TAG_SPAWNED;
            w.id = id.into();
            w.index = segments;
            w.snapshot = snapshot;
            w.filesystem = filesystem;
            w.net = net;
        }
        TaskResponse::Refused(r) => {
            w.tag = TaskRepWire::TAG_REFUSED;
            w.refusal = match r {
                TaskRefusal::BadBlock => TaskRepWire::REFUSE_BAD_BLOCK,
                TaskRefusal::PlanTooLarge => TaskRepWire::REFUSE_PLAN_TOO_LARGE,
                TaskRefusal::BadHandle => TaskRepWire::REFUSE_BAD_HANDLE,
                TaskRefusal::Unfounded => TaskRepWire::REFUSE_UNFOUNDED,
                TaskRefusal::LedgerFull => TaskRepWire::REFUSE_LEDGER_FULL,
                TaskRefusal::Unwired => TaskRepWire::REFUSE_UNWIRED,
                TaskRefusal::TableFull => TaskRepWire::REFUSE_TABLE_FULL,
                TaskRefusal::NothingConveyed => TaskRepWire::REFUSE_NOTHING_CONVEYED,
                TaskRefusal::Denied => TaskRepWire::REFUSE_DENIED,
                TaskRefusal::ImageUnavailable => TaskRepWire::REFUSE_IMAGE_UNAVAILABLE,
                TaskRefusal::Retired => TaskRepWire::REFUSE_RETIRED,
                TaskRefusal::Contended => TaskRepWire::REFUSE_CONTENDED,
                TaskRefusal::PeerAdmissionPending => TaskRepWire::REFUSE_PEER_ADMISSION_PENDING,
                TaskRefusal::PeerAdmissionWedged => TaskRepWire::REFUSE_PEER_ADMISSION_WEDGED,
                TaskRefusal::PeerRefused => TaskRepWire::REFUSE_PEER_REFUSED,
                TaskRefusal::PeerAdmissionLost => TaskRepWire::REFUSE_PEER_ADMISSION_LOST,
                TaskRefusal::PeerServiceUnwired => TaskRepWire::REFUSE_PEER_SERVICE_UNWIRED,
                TaskRefusal::PeerRingTransfer => TaskRepWire::REFUSE_PEER_RING_TRANSFER,
                TaskRefusal::PeerKickTransfer => TaskRepWire::REFUSE_PEER_KICK_TRANSFER,
                TaskRefusal::PeerGrantTransfer => TaskRepWire::REFUSE_PEER_GRANT_TRANSFER,
                TaskRefusal::PeerReturnTransfer => TaskRepWire::REFUSE_PEER_RETURN_TRANSFER,
                TaskRefusal::PeerTransientUnwired => TaskRepWire::REFUSE_PEER_TRANSIENT_UNWIRED,
                TaskRefusal::PeerMemoryUnwired => TaskRepWire::REFUSE_PEER_MEMORY_UNWIRED,
            };
        }
    }
    w
}

/// Lift a wire reply + its grant count — fail-closed.
pub fn decode_rep(w: &TaskRepWire) -> Option<(TaskResponse, u8)> {
    if matches!(
        w.tag,
        TaskRepWire::TAG_CLIENT_RETIRING | TaskRepWire::TAG_CLIENT_RETIREMENT_CONFIRMED
    ) {
        let rep = if w.tag == TaskRepWire::TAG_CLIENT_RETIRING {
            TaskResponse::ClientRetiring {
                face: w.index,
                retirement: w.id.get(),
            }
        } else {
            TaskResponse::ClientRetirementConfirmed {
                face: w.index,
                retirement: w.id.get(),
            }
        };
        return (w.id.get() != 0 && *w == encode_rep(&rep, 0)).then_some((rep, 0));
    }
    if w.tag == TaskRepWire::TAG_PROCESSOR {
        let rep = TaskResponse::ProcessorStatus(w.processor);
        return (w.processor.available.get() == 1
            && w.processor.validate()
            && *w == encode_rep(&rep, 0))
        .then_some((rep, 0));
    }
    if w.processor != Default::default()
        || (w.tag != TaskRepWire::TAG_DEVICE_CENSUS
            && w.census != kore_abi::device::DeviceCensusPage::EMPTY)
    {
        return None;
    }
    let rep = match (w.tag, w.refusal) {
        (TaskRepWire::TAG_INSTALLED, 0) if (w.index as usize) <= kore_abi::image::MAX_SEGMENTS => {
            TaskResponse::Installed { segments: w.index }
        }
        (TaskRepWire::TAG_INSTALLER_SEALED, 0) if w.index == 0 => TaskResponse::InstallerSealed,
        (TaskRepWire::TAG_APP_SEEDED, 0) if w.index & !TaskRepWire::APP_SEED_MASK == 0 => {
            TaskResponse::AppSeeded {
                seeded: w.index & TaskRepWire::APP_SEEDED != 0,
                failure: (w.index & TaskRepWire::APP_FAILURE_MASK)
                    >> TaskRepWire::APP_FAILURE_SHIFT,
            }
        }
        (TaskRepWire::TAG_LIBRARY_SEEDED, 0) if w.index & !TaskRepWire::LIBRARY_SEED_MASK == 0 => {
            TaskResponse::LibrarySeeded {
                seeded: w.index & TaskRepWire::LIBRARY_SEEDED != 0,
                manifest: w.index & TaskRepWire::LIBRARY_MANIFEST != 0,
                failure: (w.index & TaskRepWire::LIBRARY_FAILURE_MASK)
                    >> TaskRepWire::LIBRARY_FAILURE_SHIFT,
            }
        }
        (TaskRepWire::TAG_TREE_PROBED, 0)
            if w.index & !TaskRepWire::TREE_PROBE_MASK == 0
                && (w.index & TaskRepWire::TREE_PROBE_ALREADY_RESIDENT == 0
                    || w.index & TaskRepWire::TREE_PROBE_LINKED != 0) =>
        {
            TaskResponse::TreeProbed {
                linked: w.index & TaskRepWire::TREE_PROBE_LINKED != 0,
                already_resident: w.index & TaskRepWire::TREE_PROBE_ALREADY_RESIDENT != 0,
            }
        }
        (TaskRepWire::TAG_GENERATION_RETIRED, 0) if w.index == 0 => TaskResponse::GenerationRetired,
        (TaskRepWire::TAG_TREE_ACQUIRED, 0)
            if (w.index as usize) <= kore_abi::image::MAX_SEGMENTS
                && kore_abi::image::profile_valid(w.status.get() as u32) =>
        {
            TaskResponse::TreeAcquired {
                segments: w.index,
                profile: w.status.get() as u32,
            }
        }
        (TaskRepWire::TAG_CLIENT_MINTED, 0) => TaskResponse::ClientMinted { face: w.index },
        (TaskRepWire::TAG_CLIENT_BANKED, 0) => TaskResponse::ClientBanked { face: w.index },
        (TaskRepWire::TAG_CLIENT_CONTEXT, 0) => TaskResponse::ClientContext { face: w.index },
        (TaskRepWire::TAG_CLIENT_RETIRED, 0) if w.index == 0 => TaskResponse::ClientRetired,
        (TaskRepWire::TAG_DEVICE_CENSUS, 0) if w.index == 0 && w.census.validate() => {
            TaskResponse::DeviceCensus(w.census)
        }
        (TaskRepWire::TAG_SPAWNED, 0) if (w.index as usize) <= kore_abi::image::MAX_SEGMENTS => {
            if !w.snapshot.exec.available()
                || !w.snapshot.validate()
                || !w.filesystem.validate()
                || !w.net.validate()
            {
                return None;
            }
            TaskResponse::Spawned {
                id: w.id.get(),
                segments: w.index,
                snapshot: w.snapshot,
                filesystem: w.filesystem,
                net: w.net,
            }
        }
        (TaskRepWire::TAG_REFUSED, r) => TaskResponse::Refused(match r {
            TaskRepWire::REFUSE_BAD_BLOCK => TaskRefusal::BadBlock,
            TaskRepWire::REFUSE_PLAN_TOO_LARGE => TaskRefusal::PlanTooLarge,
            TaskRepWire::REFUSE_BAD_HANDLE => TaskRefusal::BadHandle,
            TaskRepWire::REFUSE_UNFOUNDED => TaskRefusal::Unfounded,
            TaskRepWire::REFUSE_LEDGER_FULL => TaskRefusal::LedgerFull,
            TaskRepWire::REFUSE_UNWIRED => TaskRefusal::Unwired,
            TaskRepWire::REFUSE_TABLE_FULL => TaskRefusal::TableFull,
            TaskRepWire::REFUSE_NOTHING_CONVEYED => TaskRefusal::NothingConveyed,
            TaskRepWire::REFUSE_DENIED => TaskRefusal::Denied,
            TaskRepWire::REFUSE_IMAGE_UNAVAILABLE => TaskRefusal::ImageUnavailable,
            TaskRepWire::REFUSE_RETIRED => TaskRefusal::Retired,
            TaskRepWire::REFUSE_CONTENDED => TaskRefusal::Contended,
            TaskRepWire::REFUSE_PEER_ADMISSION_PENDING => TaskRefusal::PeerAdmissionPending,
            TaskRepWire::REFUSE_PEER_ADMISSION_WEDGED => TaskRefusal::PeerAdmissionWedged,
            TaskRepWire::REFUSE_PEER_REFUSED => TaskRefusal::PeerRefused,
            TaskRepWire::REFUSE_PEER_ADMISSION_LOST => TaskRefusal::PeerAdmissionLost,
            TaskRepWire::REFUSE_PEER_SERVICE_UNWIRED => TaskRefusal::PeerServiceUnwired,
            TaskRepWire::REFUSE_PEER_RING_TRANSFER => TaskRefusal::PeerRingTransfer,
            TaskRepWire::REFUSE_PEER_KICK_TRANSFER => TaskRefusal::PeerKickTransfer,
            TaskRepWire::REFUSE_PEER_GRANT_TRANSFER => TaskRefusal::PeerGrantTransfer,
            TaskRepWire::REFUSE_PEER_RETURN_TRANSFER => TaskRefusal::PeerReturnTransfer,
            TaskRepWire::REFUSE_PEER_TRANSIENT_UNWIRED => TaskRefusal::PeerTransientUnwired,
            TaskRepWire::REFUSE_PEER_MEMORY_UNWIRED => TaskRefusal::PeerMemoryUnwired,
            _ => return None,
        }),
        _ => return None,
    };
    if !matches!(rep, TaskResponse::Spawned { .. })
        && w.snapshot != kore_abi::monitor::TelemetrySnapshot::EMPTY
    {
        return None;
    }
    Some((rep, w.grants))
}

/// Refusals the atomic spawn path can actually emit.
pub(crate) const fn is_spawn_refusal(refusal: TaskRefusal) -> bool {
    matches!(
        refusal,
        TaskRefusal::BadBlock
            | TaskRefusal::Contended
            | TaskRefusal::PlanTooLarge
            | TaskRefusal::BadHandle
            | TaskRefusal::LedgerFull
            | TaskRefusal::Unwired
            | TaskRefusal::TableFull
            | TaskRefusal::NothingConveyed
            | TaskRefusal::Denied
            | TaskRefusal::ImageUnavailable
            // A face's peer assembly is part of the atomic spawn, so its
            // admission refusals — this launch's own verdict still pending,
            // and an earlier launch's face left wedged — are spawn refusals.
            | TaskRefusal::PeerAdmissionPending
            | TaskRefusal::PeerAdmissionWedged
            | TaskRefusal::PeerRefused
            | TaskRefusal::PeerAdmissionLost
            | TaskRefusal::PeerServiceUnwired
            | TaskRefusal::PeerRingTransfer
            | TaskRefusal::PeerKickTransfer
            | TaskRefusal::PeerGrantTransfer
            | TaskRefusal::PeerReturnTransfer
            | TaskRefusal::PeerTransientUnwired
            | TaskRefusal::PeerMemoryUnwired
    )
}

// ── The exec vocabulary ────────────────────────────────────────────────────
// A client speaks these; the server answers them. They were declared in
// `service/mod.rs` only because that is where they were written, which meant a
// client had to link the 13k-line spawn engine to name a request. They live
// beside `ExecProtocol` now, and `service` re-exports them under their existing
// names so no caller had to be renamed.

/// Second-result authority for a composite window launch contract.
/// `ACCESS` can ring the shared child doorbell; `REVOKE` lets the receiver keep
/// retryable teardown custody. Delegation remains with exec's retained parent.
pub const SHARED_WAKE_RIGHTS: Rights = Rights::ACCESS.union(Rights::REVOKE);

/// The validated capability-selected founding record: every capability it
/// names is an index in the bounded atomic-invocation landing bank, never a
/// caller-arena slot. The nested tables are the frozen
/// [`kore_abi::svc`] wire forms carried whole (the `BootBlock`-in-enum
/// precedent).
#[derive(Clone, Copy, Debug)]
pub struct SpawnSpec {
    /// Request an affine parent handle after the optional wake grant. Always
    /// required: the data-only reply shape died with the kernel's ring-0 shell founding
    /// (a task nobody holds a handle to is unreapable), so a cleared bit
    /// refuses [`Denied`](TaskRefusal::Denied) before validation.
    pub return_handle: bool,
    /// Preserve `DERIVE` on the returned parent handle. This is an explicit
    /// controller contract, not the default exec-client authority, and the
    /// service accepts it only for the authenticated window-runtime profile.
    pub delegable_handle: bool,
    /// Return the window compositor's response-send sibling as grant 0.
    /// This bit is meaningful only for the exact [`kore_app::FilesWindow`]
    /// profile: exec derives the sibling from the same filesystem-response
    /// lineage whose receive end becomes the child's sole executor doorbell.
    pub return_shared_wake: bool,
    /// Add the **standing stub** to this founding
    /// ([`SPAWN_LINKED`](kore_abi::svc::TaskReqWire::SPAWN_LINKED)): exec
    /// shares the provided spans into the child beside the record's own
    /// segments and the task enters at the stub's validated entry —
    /// [`entry`](Self::entry) must then be zero (the client never chooses
    /// where the linker starts).  Refused while no stub stands.
    pub linked: bool,
    /// Entry-point VA in the new task.
    pub entry: u64,
    /// Initial stack-pointer VA.
    pub stack: u64,
    /// Cap-arena slots the founded task gets (#207; bounded by
    /// [`kore_rt::entry::ARENA_SLOT_CEILING`]).
    pub arena_slots: u32,
    /// The boot page's VA in the child (page-aligned, non-zero).
    pub boot_va: u64,
    /// `BOOT_FROM_SUPPLY` plus the invocation supply's semantic index.
    /// The historical detached-view pair decodes but Invoke refuses it before
    /// provisioning: its RO sibling cannot authorize receiver-map cleanup.
    pub boot_write_idx: u8,
    /// Boot page offset within that supply, in 4 KiB pages. Exec carves its
    /// temporary writable view and the child's RO share from the same page.
    pub boot_share_idx: u8,
    /// Live prefix of [`segments`](Self::segments).
    pub seg_count: u8,
    /// Live prefix of [`wiring`](Self::wiring).
    pub wiring_count: u8,
    /// Live prefix of [`windows`](Self::windows).
    pub window_count: u8,
    /// Live prefix of [`data`](Self::data).
    pub data_count: u8,
    /// Shares of conveyed leases (each at its own offset) into the fresh
    /// space (W^X refused).
    pub segments: [SpawnSegment; MAX_SPAWN_SEGMENTS],
    /// Attenuation derives delivered into the child's arena — child
    /// slots sequential from 1, wiring **before** windows.
    pub wiring: [SpawnWire; MAX_SPAWN_WIRING],
    /// Detached views carved off conveyed leases and delivered into the
    /// child's arena, which installs them itself — child slots continue
    /// the same sequence, after the wiring block.
    pub windows: [SpawnWindow; MAX_SPAWN_WINDOWS],
    /// Literal boot-block entries: stamped verbatim, no capability, **no
    /// child slot consumed**.
    pub data: [SpawnData; MAX_SPAWN_DATA],
}

/// The exec vocabulary: privileged generation administration and atomic task
/// founding.
///
/// Task activation, observation, exit, and reap are performed directly with
/// typed `TaskToken` verbs. Historical lifecycle opcodes are absent and fail
/// closed in the wire decoder.
#[derive(Clone, Copy, Debug)]
pub enum TaskRequest {
    /// Bounded read-only processor report, relayed from monitor. No grants.
    ProcessorStatus { first: u16 },
    /// Privileged name-resolution boundary. A successful reply carries a
    /// generation-bound executable capability package; ordinary invocation
    /// never carries this name.
    Install {
        target: ImageName,
        profile: u32,
        /// Return the task-wide generic invocation context before this
        /// generation endpoint. Accepted exactly once for the privileged
        /// installer pair.
        include_context: bool,
        /// Resolve the generation from the durable executable store (`app/`)
        /// rather than the boot archive — the first-class "install from the
        /// store" path. Forces a fresh `register_fs_image` load (shadowing any
        /// archive entry) and always registers launch profile zero, so
        /// `profile` must be 0 when this is set.
        from_tree: bool,
    },
    /// Irreversibly close the bootstrap-only global-name resolver before the
    /// pair is handed to an ordinary task.
    SealInstaller,
    /// Privileged: copy one already-registered generation into the durable
    /// store (`app` beneath the persisted root), so the filesystem rather
    /// than the boot archive is where that executable lives. Idempotent —
    /// a name already present in the tree is left alone — and answered
    /// rather than refused when no durable store exists, because a machine
    /// with no disk must still boot on the archive it already has.
    SeedApp { target: ImageName },
    /// Privileged: copy one archive library into the durable library
    /// directory (`lib/`, every task's READ-only `libs/` view).  The
    /// `SeedApp` contract minus the executable half — no peer, no
    /// projection, no launch authority — and the last one to land also
    /// writes the system package manifest (`pkgs/kore/manifest`).
    SeedLibrary { target: ImageName },
    /// Load-and-discard probe of the durable store: resolve `target` from
    /// `app/` and answer whether it links, without minting a peer or any
    /// launch authority. Seal-exempt (via==0 only), so the store read-back can
    /// be witnessed after installation is sealed.
    TreeLoadProbe { target: ImageName },
    /// Acquire one durable-store revision as an execute capability — the
    /// `app/<name>` command spelling.  The first CLIENT-callable store verb:
    /// seal-exempt and via-open, because it mints only what the store
    /// already authenticates — no name is bound, no entry shadowed, no
    /// installer touched.  Digest dedup resolves an unchanged revision to
    /// the resident generation, whose archive-authenticated profile the
    /// reply carries (authenticated bytes keep their authenticated
    /// metadata); a genuinely fresh tree revision registers at profile 0.
    /// Bootstrap-prefixed names refuse.
    TreeAcquire { target: ImageName },
    /// Privileged generation retirement. The name is installation metadata
    /// used only while the resolver is still open; it never authorizes an
    /// ordinary launch.
    RetireGeneration { target: ImageName },
    /// Privileged construction of one post-less client face. The reply grants
    /// its ring sublease and complete task-result receive bank; request wake
    /// authority remains a separately delegated descendant of exec's common
    /// doorbell lineage.
    /// `NONE` leaves the default filesystem context unbound: invocation must
    /// then supply a separately held view. Nonempty rights register the
    /// existing private home used by shells.
    MintClient { fs_rights: FsRights },
    /// Collect the reusable invocation-context sends for one exact client
    /// face. The face id is lifecycle correlation only and is accepted solely
    /// on the still-open administration pair.
    ClientBank { face: u8 },
    /// Collect the reusable invocation-context sends for one exact client
    /// face.
    ClientContext { face: u8 },
    /// Begin retirement of one exact task-owned client face.
    RetireClient { face: u8 },
    /// Observe physical closure of the exact service-issued retirement.
    ConfirmClientRetired { face: u8, retirement: u64 },
    /// One page of the PCI function census (`kore_abi::device::census`),
    /// relayed from the monitor for a launcher stamping an executable's
    /// context prefix.  Descriptions only; no grants.
    DeviceCensus { page: u8 },
    /// Invoke the immutable generation selected by this serving pair. The
    /// affine invocation inputs arrive with the admission proof in one
    /// atomic session.
    Invoke(SpawnSpec),
    /// A frozen tombstone discriminant arrived. Decoded explicitly so the
    /// caller receives a typed refusal instead of a dropped request; it
    /// carries no payload and can have no effect.
    Retired,
}

/// The exec replies. Authority remains out-of-band: a `Spawned` reply
/// correlates with exactly one task-handle grant on grant channel 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskResponse {
    ProcessorStatus(kore_abi::processor_status::ProcessorPage),
    /// A generation-bound executable endpoint package was returned. The
    /// segment count is inert display metadata, never an authority selector.
    Installed {
        segments: u8,
    },
    /// Init closed the bootstrap-only installer; future Install requests on
    /// this service instance refuse regardless of pair handoff.
    InstallerSealed,
    /// The generation is in the durable store: either this call copied it, or
    /// an earlier boot already did. `seeded` is false when no store could be
    /// reached, which is a posture (no disk), not a refusal.
    AppSeeded {
        seeded: bool,
        /// Why not, when `seeded` is false (`kore_abi::svc::LibraryFailure`).
        failure: u8,
    },
    /// The library is in the durable store, and whether the system package
    /// manifest stands (written once every executable and library has
    /// seeded).  Both false is the no-store posture, not a refusal.
    LibrarySeeded {
        seeded: bool,
        manifest: bool,
        failure: u8,
    },
    /// Reply to [`TaskRequest::TreeLoadProbe`]: whether the store revision of
    /// the named executable loaded and validated, and whether digest identity
    /// resolved it to an existing executable peer without registering a new
    /// image generation. `already_resident` implies `linked` on the wire.
    TreeProbed {
        linked: bool,
        already_resident: bool,
    },
    /// Reply to [`TaskRequest::TreeAcquire`]: a store revision's executable
    /// endpoint package was granted.  `segments` is inert display metadata
    /// (the [`Installed`](Self::Installed) discipline); `profile` is the
    /// granted generation's launch profile — authenticated metadata on a
    /// digest dedup, `0` for a fresh tree registration — which the caller
    /// needs to wire the launch (an FS-profiled tool takes its view
    /// auxiliaries through `app/` exactly as through its bare name).
    TreeAcquired {
        segments: u8,
        profile: u32,
    },
    /// The newest immutable image generation and its generation-bound
    /// executable service root entered retirement.
    GenerationRetired,
    /// A post-less exec client face was attached. `face` is bounded inert
    /// lifecycle correlation, never launch authority.
    ClientMinted {
        face: u8,
    },
    /// The client received all reply-grant receive endpoints.
    ClientBanked {
        face: u8,
    },
    /// The client received all task-wide invocation-context endpoints.
    ClientContext {
        face: u8,
    },
    /// Legacy queued acknowledgement, rejected by live retirement callers.
    ClientRetired,
    /// Accepted/pending retirement. The slot remains reserved.
    ClientRetiring {
        face: u8,
        retirement: u64,
    },
    /// The serving pair and all service-owned native resources are closed.
    ClientRetirementConfirmed {
        face: u8,
        retirement: u64,
    },
    /// One census page.
    DeviceCensus(kore_abi::device::DeviceCensusPage),
    /// Founded. `id` is diagnostic (hash-derived, unordered; no PID trap);
    /// ownership is the affine task handle carried out-of-band when the
    /// request selected it.
    Spawned {
        /// The founded task's id — hash-derived and unordered.
        id: u64,
        /// The linked executable's validated segment count; zero for an
        /// unlinked spawn.  This lets the caller testify to exec's metadata
        /// gate without owning a duplicate image-service face.
        segments: u8,
        /// Process-table snapshot captured after this child entered the
        /// ledger. The launcher publishes it before START.
        snapshot: kore_abi::monitor::TelemetrySnapshot,
        /// Filesystem-service posture captured in the same breath. Exec is
        /// the only party holding a filesystem face, so it federates this
        /// half the way the monitor federates the rest.
        filesystem: kore_abi::monitor::FilesystemSnapshot,
        /// Frame-service facts captured for a `PROFILE_NETWORK` launch —
        /// the link's and the bound radio's. Exec holds the frame service's
        /// one pair for exactly this, so it federates this half too.
        net: kore_abi::monitor::NetSnapshot,
    },
    /// Refused — no grant, no ledger change; a refused `Spawn` leaves the
    /// convey table intact (the client re-issues with corrected indices).
    Refused(TaskRefusal),
}

/// Why the exec manager refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskRefusal {
    /// Terminal Found committed, but no child handle was published. The
    /// client must retire any preceding shared-wake grant before reusing it.
    Unfounded,
    /// The spawn record failed validation: counts over their bounds,
    /// indices naming a convey slot that is not landed (spawning before
    /// conveying lands here), `W && X` or empty segment access, geometry
    /// misalignment/overflow, a wiring entry claiming the synthesized
    /// `DOORBELL@1` landing role, or a child plan that overflows the boot
    /// block / the child arena.
    BadBlock,
    /// The linked target could not be resolved as a validated dynamic
    /// executable. Unknown, non-grantable, static, library, and malformed
    /// descriptors all fail closed before a child is founded.
    ImageUnavailable,
    /// The individually bounded spawn tables compose into more operations
    /// than one atomic session can hold, including revocation of conveyed but
    /// unreferenced entries. Reduce unused conveys or split the launch plan;
    /// no mapping or boot-page write has occurred when this is returned.
    PlanTooLarge,
    /// A presented capability did not bind: the conveyed boot view was
    /// absent, not mappable, or lacked `ACCESS|MODIFY`; the founding
    /// session refused at commit (an over-asked derive or stale convey), or a
    /// returned child handle could not be possessed after commit.
    BadHandle,
    /// Cleanup storage or resource provisioning could not grow. Settle its
    /// retryable remote cleanup or obtain more backing. The cleanup ledger
    /// grows from mm and has no fixed task-count ceiling.
    LedgerFull,
    /// The request needs authority transfer this pair does not carry: the
    /// serving return leg is absent (the boot-chain constructor
    /// posture).  Arena-local slot numbers are never interpreted across
    /// arena namespaces — every verb that names a capability names a
    /// convey index instead.
    Unwired,
    /// A bounded task-construction table has no usable slot: either the
    /// bootstrap convey bank is occupied or ImageService cannot mint another
    /// loader face within the generation-plus-child capacity law.
    TableFull,
    /// No capability record has landed in the current convey-table slot —
    /// `Convey` before the transfer, or the transfer refused.
    NothingConveyed,
    /// The request exceeded the authority available to it: a data-only
    /// spawn (`return_handle = false` — an unowned task is unreapable), or
    /// a frozen service-lifecycle opcode was presented. Typed task handles,
    /// not this ring protocol, authorize start/wait/exit/reap.
    Denied,
    /// The request named a frozen tombstone discriminant
    /// ([`TaskReqWire::op_retired`]); nothing was attempted.
    Retired,
    /// Concurrent invocations landed admissions for more than one
    /// generation, so no landing could be attributed to this request.
    /// Every in-flight admission was retired and its posts re-armed; the
    /// caller re-derives its invocation inputs and retries.
    /// Also returned when another client face owns the detach/cleanup turn;
    /// that RetireClient had no effect and may be retried.
    Contended,
    /// The filesystem accepted every component of the child's peer bundle
    /// but had not published the admission verdict before the bounded
    /// re-ask was spent.  `admit_peer` only ENQUEUES a host action; the
    /// verdict lands after the host drains it through its own awaited
    /// mapping, so "not yet" is an ordinary schedule, not a fault — which
    /// is exactly why it must not share a rung with a full table.
    PeerAdmissionPending,
    /// An earlier launch owns an unresolved admission verdict. This request
    /// has published no peer components and can retry after reconciliation.
    PeerAdmissionWedged,
    /// The filesystem refused the peer admission for a reason that is
    /// neither capacity nor a pending verdict (wrong authority, an
    /// unsupported map plan, a malformed ring).  Terminal: the serving host
    /// closed the accepted owners before surfacing the refusal.
    PeerRefused,
    /// The admission verdict could not be observed — a reply outside the
    /// response vocabulary, or a transport boundary.  Whether the
    /// filesystem admitted the peer is UNKNOWN, so the peer is left for
    /// lazy remote reconciliation instead of being assumed closed.
    PeerAdmissionLost,
    /// A filesystem-profile launch reached peer construction without an
    /// attached filesystem service. This is narrower than [`Unwired`]: the
    /// exec invocation pair and its return leg are live, but the service
    /// needed to realize the child's private filesystem face is absent.
    PeerServiceUnwired,
    /// The private filesystem peer's shared ring did not cross its exact
    /// transfer. No later peer component was attempted.
    PeerRingTransfer,
    /// The private filesystem peer's response-send (kick) endpoint did not
    /// cross its exact transfer. The ring component had already landed.
    PeerKickTransfer,
    /// An accelerator or memory peer's reply-grant sender was not transferred.
    PeerGrantTransfer,
    /// The private filesystem peer's return-receive endpoint did not cross
    /// its exact transfer. The ring and kick components had already landed.
    PeerReturnTransfer,
    /// Exec had no transient mapping aperture with which to format the
    /// child's private filesystem ring.
    PeerTransientUnwired,
    /// Exec had no memory-service face from which to allocate the child's
    /// private filesystem ring.
    PeerMemoryUnwired,
}

/// Rights the parent's handle grant carries: wait (`ACCESS`) + explicit
/// activation (`MODIFY`) + kill (`REVOKE`) — and nothing deeper; the
/// delegating root stays with the server.
pub const HANDLE_RIGHTS: Rights = Rights::ACCESS.union(Rights::MODIFY).union(Rights::REVOKE);
/// Opt-in founder authority for controllers that must attenuate a lifecycle
/// observer into a sibling thread. Ordinary exec clients receive
/// [`HANDLE_RIGHTS`] and cannot delegate the task.
pub const DELEGABLE_HANDLE_RIGHTS: Rights = HANDLE_RIGHTS.union(Rights::DERIVE);
/// Grant channels a client of this protocol pre-pins.
pub const EXEC_REPLY_GRANTS: usize = 4 + kore_rt::svc::MAX_INVOCATION_AUXILIARIES;

/// The exec **protocol** as a type of its own.
///
/// A client names a protocol to size its ring window and to encode its
/// requests; it has no business naming the server that answers them. While
/// the only `SvcProtocol` witness for exec was `TaskManager` — the spawn engine
/// itself — every client —
/// `RingExec`, the agent runner, init — had to link the whole spawn engine to
/// say "exec". `client` referenced `service` 14 times and `service` referenced
/// `client` zero times, so the coupling was one-way and structural rather
/// than behavioural: the protocol identity was simply bolted onto the server
/// struct. This carries the identity instead.
///
/// `kore-svc-exec`'s `TaskManager` keeps its own `SvcProtocol` impl — `SystemService` requires
/// it of every service — but that impl now delegates here, so there is exactly
/// one encoding, one grant policy and one reply-matching rule.
pub struct ExecProtocol;

impl SvcProtocol for ExecProtocol {
    type Request = TaskRequest;
    type Response = TaskResponse;
    type ReqWire = TaskReqWire;
    type RepWire = TaskRepWire;

    /// Privileged install returns the task-wide supply, ProcessIo, and
    /// filesystem-view sends plus the sole generation-bound admission
    /// endpoint. Invoke publishes an optional shared wake first, then its
    /// exact TaskToken; Unfounded declares only the wake prefix.
    const REPLY_GRANTS: usize = EXEC_REPLY_GRANTS;

    fn request_grants(request: &TaskReqWire) -> usize {
        match request.op {
            TaskReqWire::OP_INSTALL if request.reserved & TaskReqWire::INSTALL_CONTEXT != 0 => {
                Self::REPLY_GRANTS
            }
            TaskReqWire::OP_INSTALL => 1,
            TaskReqWire::OP_TREE_ACQUIRE => 1,
            TaskReqWire::OP_MINT_CLIENT => 1,
            TaskReqWire::OP_CLIENT_BANK => EXEC_REPLY_GRANTS,
            TaskReqWire::OP_CLIENT_CONTEXT => 3 + MAX_INVOCATION_AUXILIARIES,
            TaskReqWire::OP_INVOKE if request.reserved & TaskReqWire::SPAWN_RETURN_HANDLE != 0 => {
                1 + usize::from(request.reserved & TaskReqWire::SPAWN_RETURN_SHARED_WAKE != 0)
            }
            _ => 0,
        }
    }

    fn grant_policy(_index: usize) -> Option<(kore_abi::cap::ResourceKind, Rights)> {
        None
    }

    fn retain_grant(index: usize) -> bool {
        index < Self::REPLY_GRANTS
    }

    fn reply_matches(request: &TaskReqWire, response: &TaskResponse, grants: u8) -> bool {
        let Some(request) = decode_req(request) else {
            return false;
        };
        match (request, response) {
            (TaskRequest::ProcessorStatus { first }, TaskResponse::ProcessorStatus(page)) => {
                grants == 0
                    && page.available.get() == 1
                    && page.first.get() == first
                    && page.validate()
            }
            (TaskRequest::ProcessorStatus { .. }, TaskResponse::Refused(_)) => grants == 0,
            (
                TaskRequest::Install {
                    include_context, ..
                },
                TaskResponse::Installed { segments },
            ) => {
                grants
                    == if include_context {
                        Self::REPLY_GRANTS as u8
                    } else {
                        1
                    }
                    && *segments != 0
            }
            (TaskRequest::Install { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::RetireGeneration { .. }, TaskResponse::GenerationRetired)
            | (TaskRequest::RetireGeneration { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::SealInstaller, TaskResponse::InstallerSealed)
            | (TaskRequest::SealInstaller, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::SeedApp { .. }, TaskResponse::AppSeeded { .. })
            | (TaskRequest::SeedApp { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::SeedLibrary { .. }, TaskResponse::LibrarySeeded { .. })
            | (TaskRequest::SeedLibrary { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::TreeLoadProbe { .. }, TaskResponse::TreeProbed { .. })
            | (TaskRequest::TreeLoadProbe { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::TreeAcquire { .. }, TaskResponse::TreeAcquired { segments, .. }) => {
                grants == 1 && *segments != 0
            }
            (TaskRequest::TreeAcquire { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::MintClient { .. }, TaskResponse::ClientMinted { .. }) => grants == 1,
            (TaskRequest::MintClient { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::ClientBank { face }, TaskResponse::ClientBanked { face: got }) => {
                face == *got && grants == EXEC_REPLY_GRANTS as u8
            }
            (TaskRequest::ClientBank { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::ClientContext { face }, TaskResponse::ClientContext { face: got }) => {
                face == *got && grants == (3 + MAX_INVOCATION_AUXILIARIES) as u8
            }
            (TaskRequest::ClientContext { .. }, TaskResponse::Refused(_)) => grants == 0,
            (
                TaskRequest::RetireClient { face },
                TaskResponse::ClientRetiring {
                    face: got,
                    retirement,
                },
            ) => face == *got && *retirement != 0 && grants == 0,
            (
                TaskRequest::ConfirmClientRetired { face, retirement },
                TaskResponse::ClientRetiring {
                    face: got,
                    retirement: token,
                }
                | TaskResponse::ClientRetirementConfirmed {
                    face: got,
                    retirement: token,
                },
            ) => face == *got && retirement != 0 && retirement == *token && grants == 0,
            (TaskRequest::RetireClient { .. }, TaskResponse::Refused(_))
            | (TaskRequest::ConfirmClientRetired { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::DeviceCensus { page }, TaskResponse::DeviceCensus(got)) => {
                page == got.page && grants == 0
            }
            (TaskRequest::DeviceCensus { .. }, TaskResponse::Refused(_)) => grants == 0,
            (TaskRequest::Invoke(spec), TaskResponse::Spawned { segments, .. }) => {
                grants == 1 + u8::from(spec.return_shared_wake)
                    && spec.return_handle
                    && spec.linked
                    && *segments != 0
            }
            (TaskRequest::Invoke(spec), TaskResponse::Refused(TaskRefusal::Unfounded)) => {
                spec.return_handle && spec.linked && grants == u8::from(spec.return_shared_wake)
            }
            (TaskRequest::Invoke(_), TaskResponse::Refused(refusal)) => {
                grants == 0 && is_spawn_refusal(*refusal)
            }
            _ => false,
        }
    }

    fn encode_req(req: &TaskRequest) -> TaskReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &TaskReqWire) -> Option<TaskRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &TaskResponse, grants: u8) -> TaskRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &TaskRepWire) -> Option<(TaskResponse, u8)> {
        decode_rep(wire)
    }
}

// The ring-window fit (the founder's `[layout] ring_bytes` contract):
// the fat v3 request record must keep the whole pair — request ring +
// reply ring at the REQ_CAP/REP_CAP geometry — inside one founded ring
// window.  Asserted const-side so a record growth that bursts the
// window is a build error, never a mis-attached ring.
//
// **This is the binding budget on the v3 table capacities.**  At
// `REQ_CAP = REP_CAP = 8` and `ring_bytes = 0x4000`:
//
//   reply  region = align64(192 + 8·848) = 6976 B
//   request region = align64(192 + 8·864) = 7104 B
//   ─────────────────────────────────────────────
//   pair                                 = 14080 B of 16384
//
// The 848-byte reply record is the 752-byte one plus the launcher's
// 76-byte `NetSnapshot` (the frame service's facts for a `PROFILE_NETWORK`
// launch — the link's, the device's counters, the radio's) and the
// filesystem snapshot's 20-byte completion-latency moments, riding
// `Spawned` together.
//
// The 864-byte record combines the append-only privileged-install target
// metadata with the bounded shell plan: fifteen protocol wiring rows,
// seventeen task-local executable endpoints, seven task-wide context lanes,
// and nine windows. Ordinary invoke leaves the target empty.
// The four-page window carries the record; the assert keeps carrying
// the day a table grows past it — a burst window is a build error, never a
// mis-attached ring.  Future growth is checked against
// `[layout].ring_bytes`, never paid for by reducing ring depth.
const _: () = {
    use kore_rt::svc::{REP_CAP, REQ_CAP, window_bytes};
    assert!(
        window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP) <= kore_abi::task::FACE_RING_STRIDE as usize,
        "exec pair (8-deep request + reply rings) must fit LAYOUT.ring_bytes"
    );
    // The spelled-out arithmetic above, pinned: a silent drift in
    // REQ_CAP/REP_CAP or either indexed telemetry page moves this number.
    // 14_080 → 14_208 when `SystemSnapshot` gained the scheduler-timer arm
    // census (16 bytes, carried by each of the eight reply slots), and
    // 14_208 → 14_336 when it gained the frame-class census (another 16),
    // 14_336 → 14_528 for the DRAM stream table (24), and 14_528 → 14_592
    // when each processor row gained its own DRAM stream pair (4 per row,
    // two rows per page).
    assert!(window_bytes::<ExecProtocol>(8, 8) == 14_592);
};

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::cap::Rights;
    use kore_abi::task::role;
    use zerocopy::IntoBytes;

    #[test]
    fn unfounded_reply_declares_only_the_requested_wake_prefix() {
        let reply = TaskResponse::Refused(TaskRefusal::Unfounded);
        for shared_wake in [false, true] {
            let mut spec = spawn_spec();
            spec.linked = true;
            spec.entry = 0;
            spec.return_shared_wake = shared_wake;
            let request = encode_req(&TaskRequest::Invoke(spec));
            for grants in 0..=2 {
                let wire = encode_rep(&reply, grants);
                assert_eq!(decode_rep(&wire), Some((reply, grants)));
                assert_eq!(
                    ExecProtocol::reply_matches(&request, &reply, grants),
                    grants == u8::from(shared_wake),
                );
                assert_eq!(
                    ExecProtocol::reply_matches(
                        &request,
                        &TaskResponse::Refused(TaskRefusal::BadHandle),
                        grants
                    ),
                    grants == 0,
                );
            }
            for clear_flag in [TaskReqWire::SPAWN_RETURN_HANDLE, TaskReqWire::SPAWN_LINKED] {
                let mut malformed = request;
                malformed.reserved &= !clear_flag;
                assert!(!ExecProtocol::reply_matches(
                    &malformed,
                    &reply,
                    u8::from(shared_wake)
                ));
            }
        }
    }

    #[test]
    fn wake_first_invoke_cannot_reinterpret_the_legacy_task_first_opcode() {
        let mut spec = spawn_spec();
        spec.linked = true;
        spec.entry = 0;
        spec.return_shared_wake = true;
        let mut request = encode_req(&TaskRequest::Invoke(spec));
        assert_eq!(request.op, 25);
        assert!(matches!(decode_req(&request), Some(TaskRequest::Invoke(_))));
        assert_eq!(TaskReqWire::INVOKE_WAKE_GRANT, 0);
        assert_eq!(TaskReqWire::invoke_task_grant(true), 1);
        assert_eq!(TaskReqWire::invoke_task_grant(false), 0);
        request.op = 16;
        assert_eq!(request.op, TaskReqWire::OP_INVOKE_LEGACY);
        assert!(matches!(decode_req(&request), Some(TaskRequest::Retired)));
        assert!(!ExecProtocol::reply_matches(
            &request,
            &TaskResponse::Refused(TaskRefusal::Unfounded),
            1
        ));
    }

    #[test]
    fn processor_reply_is_grant_free_canonical_and_correlated() {
        let req = TaskRequest::ProcessorStatus { first: 8 };
        let mut wire = encode_req(&req);
        assert!(matches!(
            decode_req(&wire),
            Some(TaskRequest::ProcessorStatus { first: 8 })
        ));
        wire.entry.set(65536);
        assert!(decode_req(&wire).is_none());
        let mut page = kore_abi::processor_status::ProcessorPage::default();
        page.available.set(1);
        page.first.set(8);
        page.total.set(8);
        let rep = TaskResponse::ProcessorStatus(page);
        let mut wire = encode_rep(&rep, 0);
        assert_eq!(decode_rep(&wire), Some((rep, 0)));
        assert!(ExecProtocol::reply_matches(&encode_req(&req), &rep, 0));
        assert!(!ExecProtocol::reply_matches(&encode_req(&req), &rep, 1));
        assert!(!ExecProtocol::reply_matches(
            &encode_req(&TaskRequest::ProcessorStatus { first: 7 }),
            &rep,
            0
        ));
        wire.id.set(1);
        assert!(decode_rep(&wire).is_none());
    }

    /// A representative fat spawn record: two segments (one RW at offset
    /// zero, one R|X at a non-zero offset — the v3 carve shape), three
    /// wiring entries (a doorbell recv among them), two windows (a
    /// sibling pair off ONE lease), two data literals sharing a value,
    /// and distinct boot indices.
    pub(crate) fn spawn_spec() -> SpawnSpec {
        let mut segments = [SpawnSegment::EMPTY; MAX_SPAWN_SEGMENTS];
        segments[0] = SpawnSegment {
            convey_idx: 2,
            access: SHARE_ACCESS_R | SHARE_ACCESS_W,
            reserved: [0; 2],
            offset: 0u32.into(),
            va: 0x7fff_0000_0000u64.into(),
            len: 0x1_0000u64.into(),
        };
        segments[1] = SpawnSegment {
            convey_idx: 3,
            access: SHARE_ACCESS_R | SHARE_ACCESS_X,
            reserved: [0; 2],
            offset: 0x3000u32.into(),
            va: 0x40_0000u64.into(),
            len: 0x2000u64.into(),
        };
        let mut wiring = [SpawnWire::EMPTY; MAX_SPAWN_WIRING];
        wiring[0] = SpawnWire {
            convey_idx: 4,
            reserved: 0,
            kind: (kore_abi::cap::ResourceKind::ChanRecv as u16).into(),
            role: role::DOORBELL.into(),
            index: 0.into(),
            rights: u32::from((Rights::ACCESS | Rights::REVOKE).bits()).into(),
        };
        wiring[1] = SpawnWire {
            convey_idx: 5,
            reserved: 0,
            kind: (kore_abi::cap::ResourceKind::ChanSend as u16).into(),
            role: role::SVC_SEND.into(),
            index: 0.into(),
            rights: u32::from((Rights::ACCESS | Rights::DERIVE).bits()).into(),
        };
        wiring[2] = SpawnWire {
            convey_idx: 6,
            reserved: 0,
            kind: (kore_abi::cap::ResourceKind::ChanSend as u16).into(),
            role: role::PARENT_SEND.into(),
            index: 0.into(),
            rights: u32::from(Rights::ACCESS.bits()).into(),
        };
        // Two windows off the SAME conveyed lease at the SAME geometry —
        // the sibling-view case (one half mapped here, the other handed
        // to the peer); they must stay two independent entries.
        let mut windows = [SpawnWindow::EMPTY; MAX_SPAWN_WINDOWS];
        windows[0] = SpawnWindow {
            convey_idx: 7,
            write: 1,
            // The half the child forwards to its peer — it needs the
            // transfer allowance; its sibling below does not.
            flags: kore_abi::svc::WINDOW_FORWARD,
            reserved: [0; 1],
            role: role::SVC_RING.into(),
            index: 0.into(),
            offset: 0u32.into(),
            len: 0x1000u32.into(),
        };
        windows[1] = SpawnWindow {
            convey_idx: 7,
            write: 1,
            flags: 0,
            reserved: [0; 1],
            role: role::SVC_RING.into(),
            index: 1.into(),
            offset: 0u32.into(),
            len: 0x1000u32.into(),
        };
        // Two data literals carrying the SAME value — two peers' landing
        // slots in their own arenas legitimately coincide.
        let mut data = [SpawnData::EMPTY; MAX_SPAWN_DATA];
        data[0] = SpawnData {
            role: role::SVC_RETURN.into(),
            index: 1.into(),
            value: 6u32.into(),
        };
        data[1] = SpawnData {
            role: role::SVC_RETURN.into(),
            index: 3.into(),
            value: 6u32.into(),
        };
        SpawnSpec {
            return_handle: true,
            delegable_handle: false,
            return_shared_wake: false,
            linked: false,
            entry: 0x40_0120,
            stack: 0x7fff_0001_0000,
            arena_slots: 24,
            boot_va: 0x7fff_fe00_0000,
            boot_write_idx: 0,
            boot_share_idx: 1,
            seg_count: 2,
            wiring_count: 3,
            window_count: 2,
            data_count: 2,
            segments,
            wiring,
            windows,
            data,
        }
    }

    /// Every current form round-trips byte-exactly.
    #[test]
    fn round_trips() {
        let reqs = [
            TaskRequest::Install {
                target: kore_abi::image::ImageName::new("hello").unwrap(),
                profile: 0,
                include_context: true,
                from_tree: false,
            },
            TaskRequest::Install {
                target: kore_abi::image::ImageName::new("hello").unwrap(),
                profile: 0,
                include_context: false,
                from_tree: true,
            },
            TaskRequest::RetireGeneration {
                target: kore_abi::image::ImageName::new("hello").unwrap(),
            },
            TaskRequest::TreeAcquire {
                target: kore_abi::image::ImageName::new("hello").unwrap(),
            },
            TaskRequest::SeedApp {
                target: kore_abi::image::ImageName::new("hello").unwrap(),
            },
            TaskRequest::SeedLibrary {
                target: kore_abi::image::ImageName::new("libdemo.so").unwrap(),
            },
            TaskRequest::MintClient {
                fs_rights: kore_svc_dagfs::Rights::READ_ONLY,
            },
            TaskRequest::MintClient {
                fs_rights: kore_svc_dagfs::Rights::NONE,
            },
            TaskRequest::ClientBank { face: 2 },
            TaskRequest::ClientContext { face: 2 },
            TaskRequest::RetireClient { face: 2 },
            TaskRequest::ConfirmClientRetired {
                face: 2,
                retirement: 123,
            },
            TaskRequest::SealInstaller,
            TaskRequest::Invoke({
                let mut invoke = spawn_spec();
                invoke.linked = true;
                invoke.entry = 0;
                invoke
            }),
        ];
        for req in reqs {
            let back = decode_req(&encode_req(&req)).expect("decodes");
            assert_eq!(
                encode_req(&back).as_bytes(),
                encode_req(&req).as_bytes(),
                "byte-exact round trip"
            );
        }
        for (rep, grants) in [
            (
                TaskResponse::Spawned {
                    id: 0xDEAD_BEEF,
                    segments: 4,
                    snapshot: kore_abi::monitor::TelemetrySnapshot::exec_only(
                        kore_abi::process::ProcessSnapshot::new(8, 2, 1, 0, 4, 0, 1),
                    ),
                    filesystem: kore_abi::monitor::FilesystemSnapshot::EMPTY,
                    net: kore_abi::monitor::NetSnapshot::EMPTY,
                },
                1,
            ),
            (TaskResponse::Installed { segments: 4 }, 5),
            (
                TaskResponse::TreeProbed {
                    linked: true,
                    already_resident: false,
                },
                0,
            ),
            (
                TaskResponse::TreeProbed {
                    linked: true,
                    already_resident: true,
                },
                0,
            ),
            (
                TaskResponse::TreeAcquired {
                    segments: 4,
                    profile: kore_abi::image::PROFILE_FILESYSTEM,
                },
                1,
            ),
            (
                TaskResponse::TreeAcquired {
                    segments: 2,
                    profile: 0,
                },
                1,
            ),
            (TaskResponse::GenerationRetired, 0),
            (
                TaskResponse::AppSeeded {
                    seeded: true,
                    failure: 0,
                },
                0,
            ),
            (
                TaskResponse::AppSeeded {
                    seeded: false,
                    failure: 0,
                },
                0,
            ),
            (
                TaskResponse::LibrarySeeded {
                    seeded: true,
                    manifest: false,
                    failure: 0,
                },
                0,
            ),
            (
                TaskResponse::LibrarySeeded {
                    seeded: true,
                    manifest: true,
                    failure: 0,
                },
                0,
            ),
            (
                TaskResponse::LibrarySeeded {
                    seeded: false,
                    manifest: false,
                    failure: kore_abi::svc::LibraryFailure::EXPORT,
                },
                0,
            ),
            (TaskResponse::ClientMinted { face: 2 }, 1),
            (
                TaskResponse::ClientBanked { face: 2 },
                EXEC_REPLY_GRANTS as u8,
            ),
            (
                TaskResponse::ClientContext { face: 2 },
                (3 + MAX_INVOCATION_AUXILIARIES) as u8,
            ),
            (TaskResponse::ClientRetired, 0),
            (
                TaskResponse::ClientRetiring {
                    face: 2,
                    retirement: 123,
                },
                0,
            ),
            (
                TaskResponse::ClientRetirementConfirmed {
                    face: 2,
                    retirement: 123,
                },
                0,
            ),
            (TaskResponse::InstallerSealed, 0),
            (TaskResponse::Refused(TaskRefusal::BadBlock), 0),
            (TaskResponse::Refused(TaskRefusal::Contended), 0),
            (TaskResponse::Refused(TaskRefusal::PlanTooLarge), 0),
            (TaskResponse::Refused(TaskRefusal::BadHandle), 0),
            (TaskResponse::Refused(TaskRefusal::LedgerFull), 0),
            (TaskResponse::Refused(TaskRefusal::Unwired), 0),
            (TaskResponse::Refused(TaskRefusal::TableFull), 0),
            (TaskResponse::Refused(TaskRefusal::NothingConveyed), 0),
            (TaskResponse::Refused(TaskRefusal::Denied), 0),
            (TaskResponse::Refused(TaskRefusal::ImageUnavailable), 0),
            (TaskResponse::Refused(TaskRefusal::PeerAdmissionPending), 0),
            (TaskResponse::Refused(TaskRefusal::PeerAdmissionWedged), 0),
            (TaskResponse::Refused(TaskRefusal::PeerRefused), 0),
            (TaskResponse::Refused(TaskRefusal::PeerAdmissionLost), 0),
            (TaskResponse::Refused(TaskRefusal::PeerServiceUnwired), 0),
            (TaskResponse::Refused(TaskRefusal::PeerRingTransfer), 0),
            (TaskResponse::Refused(TaskRefusal::PeerKickTransfer), 0),
            (TaskResponse::Refused(TaskRefusal::PeerGrantTransfer), 0),
            (TaskResponse::Refused(TaskRefusal::PeerReturnTransfer), 0),
            (TaskResponse::Refused(TaskRefusal::PeerTransientUnwired), 0),
            (TaskResponse::Refused(TaskRefusal::PeerMemoryUnwired), 0),
        ] {
            let (back, g) = decode_rep(&encode_rep(&rep, grants)).expect("decodes");
            assert_eq!(back, rep);
            assert_eq!(g, grants);
        }
    }

    #[test]
    fn tree_probe_disposition_bits_are_fail_closed() {
        for response in [
            TaskResponse::TreeProbed {
                linked: false,
                already_resident: false,
            },
            TaskResponse::TreeProbed {
                linked: true,
                already_resident: false,
            },
            TaskResponse::TreeProbed {
                linked: true,
                already_resident: true,
            },
        ] {
            let wire = encode_rep(&response, 0);
            assert_eq!(decode_rep(&wire), Some((response, 0)));
        }

        let resident_without_link = encode_rep(
            &TaskResponse::TreeProbed {
                linked: false,
                already_resident: true,
            },
            0,
        );
        assert!(decode_rep(&resident_without_link).is_none());

        let mut unknown_bit = encode_rep(
            &TaskResponse::TreeProbed {
                linked: true,
                already_resident: true,
            },
            0,
        );
        unknown_bit.index |= 1 << 7;
        assert!(decode_rep(&unknown_bit).is_none());
    }

    #[test]
    fn client_retirement_wire_requires_exact_nonzero_token_and_canonical_shape() {
        let begin = encode_req(&TaskRequest::RetireClient { face: 4 });
        let confirm = encode_req(&TaskRequest::ConfirmClientRetired {
            face: 4,
            retirement: 17,
        });
        assert!(decode_req(&confirm).is_some());
        assert!(ExecProtocol::reply_matches(
            &begin,
            &TaskResponse::ClientRetiring {
                face: 4,
                retirement: 17
            },
            0
        ));
        assert!(!ExecProtocol::reply_matches(
            &begin,
            &TaskResponse::ClientRetired,
            0
        ));
        assert!(!ExecProtocol::reply_matches(
            &confirm,
            &TaskResponse::ClientRetired,
            0
        ));
        for response in [
            TaskResponse::ClientRetiring {
                face: 4,
                retirement: 17,
            },
            TaskResponse::ClientRetirementConfirmed {
                face: 4,
                retirement: 17,
            },
        ] {
            assert!(ExecProtocol::reply_matches(&confirm, &response, 0));
            assert!(!ExecProtocol::reply_matches(&confirm, &response, 1));
            let wire = encode_rep(&response, 0);
            let mut wrong = wire;
            wrong.id = 0.into();
            assert!(decode_rep(&wrong).is_none());
            wrong = wire;
            wrong.grants = 1;
            assert!(decode_rep(&wrong).is_none());
            wrong = wire;
            wrong.status = 1.into();
            assert!(decode_rep(&wrong).is_none());
            wrong = wire;
            wrong.refusal = 1;
            assert!(decode_rep(&wrong).is_none());
        }
        for response in [
            TaskResponse::ClientRetiring {
                face: 5,
                retirement: 17,
            },
            TaskResponse::ClientRetiring {
                face: 4,
                retirement: 18,
            },
            TaskResponse::ClientRetirementConfirmed {
                face: 5,
                retirement: 17,
            },
            TaskResponse::ClientRetirementConfirmed {
                face: 4,
                retirement: 18,
            },
        ] {
            assert!(!ExecProtocol::reply_matches(&confirm, &response, 0));
        }
        let mut wrong = confirm;
        wrong.entry = 0.into();
        assert!(decode_req(&wrong).is_none());
        wrong = confirm;
        wrong.slot = 256.into();
        assert!(decode_req(&wrong).is_none());
        wrong = confirm;
        wrong.stack = 1.into();
        assert!(decode_req(&wrong).is_none());
        wrong = confirm;
        wrong.reserved2 = 1.into();
        assert!(decode_req(&wrong).is_none());
        wrong = confirm;
        wrong.reserved = 1;
        assert!(decode_req(&wrong).is_none());
    }

    #[test]
    fn reply_correlation_is_closed_per_exec_operation() {
        let install = encode_req(&TaskRequest::Install {
            target: kore_abi::image::ImageName::new("hello").unwrap(),
            profile: 0,
            include_context: true,
            from_tree: false,
        });
        let retire = encode_req(&TaskRequest::RetireGeneration {
            target: kore_abi::image::ImageName::new("hello").unwrap(),
        });
        let mut invoke_spec = spawn_spec();
        invoke_spec.linked = true;
        invoke_spec.entry = 0;
        let invoke = encode_req(&TaskRequest::Invoke(invoke_spec));
        assert!(ExecProtocol::reply_matches(
            &install,
            &TaskResponse::Installed { segments: 3 },
            ExecProtocol::REPLY_GRANTS as u8
        ));
        assert!(!ExecProtocol::reply_matches(
            &install,
            &TaskResponse::Installed { segments: 3 },
            3
        ));
        assert!(ExecProtocol::reply_matches(
            &retire,
            &TaskResponse::GenerationRetired,
            0
        ));
        assert!(!ExecProtocol::reply_matches(
            &retire,
            &TaskResponse::GenerationRetired,
            1
        ));
        let tree_acquire = encode_req(&TaskRequest::TreeAcquire {
            target: kore_abi::image::ImageName::new("hello").unwrap(),
        });
        assert!(ExecProtocol::reply_matches(
            &tree_acquire,
            &TaskResponse::TreeAcquired {
                segments: 3,
                profile: 0
            },
            1
        ));
        assert!(!ExecProtocol::reply_matches(
            &tree_acquire,
            &TaskResponse::TreeAcquired {
                segments: 3,
                profile: 0
            },
            0
        ));
        assert!(!ExecProtocol::reply_matches(
            &tree_acquire,
            &TaskResponse::TreeAcquired {
                segments: 0,
                profile: 0
            },
            1
        ));
        assert!(ExecProtocol::reply_matches(
            &tree_acquire,
            &TaskResponse::Refused(TaskRefusal::ImageUnavailable),
            0
        ));
        assert!(ExecProtocol::reply_matches(
            &invoke,
            &TaskResponse::Spawned {
                id: 0,
                segments: 3,
                snapshot: kore_abi::monitor::TelemetrySnapshot::exec_only(
                    kore_abi::process::ProcessSnapshot::new(8, 1, 0, 0, 1, 0, 0),
                ),
                filesystem: kore_abi::monitor::FilesystemSnapshot::EMPTY,
                net: kore_abi::monitor::NetSnapshot::EMPTY,
            },
            1
        ));
        let shared_invoke = encode_req(&TaskRequest::Invoke(SpawnSpec {
            return_shared_wake: true,
            ..invoke_spec
        }));
        assert_eq!(ExecProtocol::request_grants(&shared_invoke), 2);
        assert!(ExecProtocol::reply_matches(
            &shared_invoke,
            &TaskResponse::Spawned {
                id: 0,
                segments: 3,
                snapshot: kore_abi::monitor::TelemetrySnapshot::exec_only(
                    kore_abi::process::ProcessSnapshot::new(8, 1, 0, 0, 1, 0, 0),
                ),
                filesystem: kore_abi::monitor::FilesystemSnapshot::EMPTY,
                net: kore_abi::monitor::NetSnapshot::EMPTY,
            },
            2
        ));
        assert!(!ExecProtocol::reply_matches(
            &shared_invoke,
            &TaskResponse::Spawned {
                id: 0,
                segments: 3,
                snapshot: kore_abi::monitor::TelemetrySnapshot::exec_only(
                    kore_abi::process::ProcessSnapshot::new(8, 1, 0, 0, 1, 0, 0),
                ),
                filesystem: kore_abi::monitor::FilesystemSnapshot::EMPTY,
                net: kore_abi::monitor::NetSnapshot::EMPTY,
            },
            1
        ));
        // A filesystem-profile launch may discover bounded peer-table or
        // transferred-custody exhaustion while constructing its atomic
        // invocation context. Those are valid invoke/spawn refusals, not a
        // malformed reply.
        let mut unknown = invoke;
        unknown.op = 0xff;
        assert!(!ExecProtocol::reply_matches(
            &unknown,
            &TaskResponse::Refused(TaskRefusal::Denied),
            0
        ));
    }

    #[test]
    fn minted_client_filesystem_ceiling_is_explicit_and_fail_closed() {
        let request = TaskRequest::MintClient {
            fs_rights: kore_svc_dagfs::Rights::READ_ONLY,
        };
        let wire = encode_req(&request);
        assert!(matches!(
            decode_req(&wire),
            Some(TaskRequest::MintClient { fs_rights })
                if fs_rights == kore_svc_dagfs::Rights::READ_ONLY
        ));

        let mut empty = wire;
        empty.reserved2 = 0u32.into();
        assert!(
            matches!(
                decode_req(&empty),
                Some(TaskRequest::MintClient { fs_rights }) if fs_rights.is_empty()
            ),
            "an agent's explicit NONE leaves its default filesystem view unbound"
        );
        let mut oversized = wire;
        oversized.reserved2 = 256u32.into();
        assert!(decode_req(&oversized).is_none());

        let mut unknown = wire;
        unknown.reserved2 = u32::from(1u8 << 7).into();
        assert!(decode_req(&unknown).is_none());
    }

    /// Linked ordinary launch exists only on capability-selected Invoke.
    #[test]
    fn raw_name_linked_spawn_is_retired() {
        let mut linked = spawn_spec();
        linked.linked = true;
        linked.entry = 0;
        let mut w = encode_req(&TaskRequest::Invoke(linked));
        w.op = TaskReqWire::OP_SPAWN;
        assert_eq!(
            w.reserved,
            TaskReqWire::SPAWN_RETURN_HANDLE | TaskReqWire::SPAWN_LINKED
        );
        assert!(matches!(decode_req(&w), Some(TaskRequest::Retired)));
        let mut invoke = w;
        invoke.op = TaskReqWire::OP_INVOKE;
        assert!(matches!(decode_req(&invoke), Some(TaskRequest::Invoke(_))));
        let mut bad_flags = invoke;
        bad_flags.reserved |= 1 << 7;
        assert!(
            decode_req(&bad_flags).is_none(),
            "unknown flag bits refuse on the live invoke path"
        );

        let mut delegable = invoke;
        delegable.reserved |= TaskReqWire::SPAWN_DELEGABLE_HANDLE;
        assert!(matches!(
            decode_req(&delegable),
            Some(TaskRequest::Invoke(spec)) if spec.delegable_handle
        ));
        delegable.reserved &= !TaskReqWire::SPAWN_RETURN_HANDLE;
        assert!(
            decode_req(&delegable).is_none(),
            "delegability cannot survive without an affine returned owner"
        );

        let mut shared = invoke;
        shared.reserved |= TaskReqWire::SPAWN_RETURN_SHARED_WAKE;
        assert!(matches!(
            decode_req(&shared),
            Some(TaskRequest::Invoke(spec)) if spec.return_shared_wake
        ));

        let mut missing = encode_req(&TaskRequest::Invoke({
            let mut request = spawn_spec();
            request.linked = true;
            request.entry = 0;
            request
        }));
        missing.op = TaskReqWire::OP_SPAWN;
        missing.reserved |= TaskReqWire::SPAWN_LINKED;
        missing.entry = 0u64.into();
        assert!(
            matches!(decode_req(&missing), Some(TaskRequest::Retired)),
            "a raw spawn is a tombstone refusal regardless of target bytes"
        );

        let mut stray = missing;
        stray.target = kore_abi::image::ImageName::new("hello").unwrap();
        assert!(
            matches!(decode_req(&stray), Some(TaskRequest::Retired)),
            "target metadata cannot revive a tombstoned spawn"
        );
    }

    #[test]
    fn filesystem_auxiliary_profiles_require_a_filesystem_face_at_decode() {
        let request = TaskRequest::Install {
            target: kore_abi::image::ImageName::new("cd").unwrap(),
            profile: kore_abi::image::PROFILE_FS_VIEW_BINDER,
            include_context: true,
            from_tree: false,
        };
        assert!(decode_req(&encode_req(&request)).is_none());

        let request = TaskRequest::Install {
            target: kore_abi::image::ImageName::new("cd").unwrap(),
            profile: kore_abi::image::PROFILE_FILESYSTEM | kore_abi::image::PROFILE_FS_VIEW_BINDER,
            include_context: false,
            from_tree: false,
        };
        assert!(matches!(
            decode_req(&encode_req(&request)),
            Some(TaskRequest::Install { .. })
        ));

        let request = TaskRequest::Install {
            target: kore_abi::image::ImageName::new("grant").unwrap(),
            profile: kore_abi::image::PROFILE_FS_GRANT_TARGET,
            include_context: true,
            from_tree: false,
        };
        assert!(decode_req(&encode_req(&request)).is_none());

        let request = TaskRequest::Install {
            target: kore_abi::image::ImageName::new("grant").unwrap(),
            profile: kore_abi::image::PROFILE_FILESYSTEM | kore_abi::image::PROFILE_FS_GRANT_TARGET,
            include_context: false,
            from_tree: false,
        };
        assert!(matches!(
            decode_req(&encode_req(&request)),
            Some(TaskRequest::Install { .. })
        ));
    }

    /// Decode is fail-closed: unknown and every retired opcode/tag,
    /// over-bound counts, out-of-range convey indices, equal boot indices,
    /// empty / W|X / out-of-vocabulary segment access, and rights/kind
    /// envelopes all refuse.
    #[test]
    fn decode_fails_closed() {
        assert_eq!(
            [
                TaskReqWire::OP_WAIT,
                TaskReqWire::OP_KILL,
                TaskReqWire::OP_EXIT,
                TaskReqWire::OP_LAUNCH,
                TaskReqWire::OP_PREPARE_IMAGES,
                TaskReqWire::OP_SET_WORKSPACE,
            ],
            [2, 3, 4, 7, 8, 9],
            "retired request values stay frozen"
        );
        assert_eq!(
            [
                TaskRepWire::TAG_RUNNING,
                TaskRepWire::TAG_EXITED,
                TaskRepWire::TAG_KILLED,
                TaskRepWire::TAG_DEPARTING,
                TaskRepWire::TAG_IMAGES_PREPARING,
                TaskRepWire::TAG_IMAGES_READY,
                TaskRepWire::TAG_WORKSPACE_SET,
            ],
            [2, 3, 4, 5, 8, 9, 10],
            "retired reply values stay frozen"
        );
        let base_wire = encode_req(&TaskRequest::SealInstaller);
        // The whole frozen band answers a typed refusal — payload ignored,
        // effect impossible, caller never left hanging.
        for op in (1..=TaskReqWire::OP_SET_WORKSPACE).chain([TaskReqWire::OP_INVOKE_LEGACY]) {
            let mut w = base_wire;
            w.op = op;
            assert!(TaskReqWire::op_retired(op));
            assert!(
                matches!(decode_req(&w), Some(TaskRequest::Retired)),
                "tombstone {op} answers, never revives or drops"
            );
        }
        // Garbage outside every vocabulary still earns no reply.
        for op in [0, 18, u8::MAX] {
            let mut w = base_wire;
            w.op = op;
            assert!(decode_req(&w).is_none(), "opcode {op} is not current");
        }
        let retired_reply = encode_rep(&TaskResponse::Refused(TaskRefusal::Retired), 0);
        assert_eq!(retired_reply.refusal, TaskRepWire::REFUSE_RETIRED);
        assert_eq!(
            decode_rep(&retired_reply),
            Some((TaskResponse::Refused(TaskRefusal::Retired), 0))
        );
        for tag in [
            TaskRepWire::TAG_RUNNING,
            TaskRepWire::TAG_EXITED,
            TaskRepWire::TAG_KILLED,
            TaskRepWire::TAG_DEPARTING,
        ] {
            let mut retired = encode_rep(&TaskResponse::Refused(TaskRefusal::Denied), 0);
            retired.tag = tag;
            retired.index = 0;
            assert!(
                decode_rep(&retired).is_none(),
                "retired reply tag {tag} is absent from current vocabulary"
            );
        }

        let mut base = spawn_spec();
        base.linked = true;
        base.entry = 0;
        let ok = encode_req(&TaskRequest::Invoke(base));
        assert!(decode_req(&ok).is_some(), "the well-formed record decodes");

        let mut w = ok;
        w.seg_count = (MAX_SPAWN_SEGMENTS + 1) as u8;
        assert!(decode_req(&w).is_none(), "segment over-count refuses");
        let mut w = ok;
        w.wiring_count = (MAX_SPAWN_WIRING + 1) as u8;
        assert!(decode_req(&w).is_none(), "wiring over-count refuses");
        let mut w = ok;
        w.boot_write_idx = MAX_CONVEY as u8;
        assert!(decode_req(&w).is_none(), "boot index past the table");
        let mut w = ok;
        w.boot_share_idx = w.boot_write_idx;
        assert!(decode_req(&w).is_none(), "equal boot indices refuse");
        let mut w = ok;
        w.segments[0].access = SHARE_ACCESS_W | SHARE_ACCESS_X;
        assert!(decode_req(&w).is_none(), "W|X segment access refuses");
        let mut w = ok;
        w.segments[0].access = 0;
        assert!(decode_req(&w).is_none(), "empty segment access refuses");
        let mut w = ok;
        w.segments[0].access = 0xF0 | SHARE_ACCESS_R;
        assert!(decode_req(&w).is_none(), "unknown access bits refuse");
        let mut w = ok;
        w.segments[1].convey_idx = MAX_CONVEY as u8;
        assert!(decode_req(&w).is_none(), "segment convey index past table");
        let mut w = ok;
        w.wiring[2].convey_idx = 0xFF;
        assert!(decode_req(&w).is_none(), "wiring convey index past table");
        // ── The v3 tables ────────────────────────────────────────────────
        let mut w = ok;
        w.window_count = (MAX_SPAWN_WINDOWS + 1) as u8;
        assert!(decode_req(&w).is_none(), "window over-count refuses");
        let mut w = ok;
        w.data_count = (MAX_SPAWN_DATA + 1) as u8;
        assert!(decode_req(&w).is_none(), "data over-count refuses");
        let mut w = ok;
        w.windows[1].convey_idx = MAX_CONVEY as u8;
        assert!(decode_req(&w).is_none(), "window convey index past table");
        let mut w = ok;
        w.windows[0].write = 2;
        assert!(
            decode_req(&w).is_none(),
            "a window `write` byte outside {{0,1}} refuses"
        );
        // A window's offset/len geometry is the SERVE step's refusal, not
        // the codec's (the segment precedent) — the shape still decodes.
        let mut w = ok;
        w.windows[0].len = 0u32.into();
        assert!(
            decode_req(&w).is_some(),
            "window geometry is validate_spawn's, not the codec's"
        );
        // A data entry carries no capability and no bounded field: every
        // bit pattern is a lawful literal, so nothing here can refuse.
        let mut w = ok;
        w.data[0].value = u32::MAX.into();
        w.data[1] = w.data[0];
        assert!(
            decode_req(&w).is_some(),
            "duplicate data literals are lawful"
        );
        let mut w = ok;
        w.wiring[0].rights = 0x1_0000u32.into();
        assert!(decode_req(&w).is_none(), "rights past the u8 envelope");
        let mut w = ok;
        w.wiring[0].rights = 0xF0u32.into();
        assert!(decode_req(&w).is_none(), "unknown rights bits refuse");
        let mut w = ok;
        w.wiring[0].kind = 0x100u16.into();
        assert!(decode_req(&w).is_none(), "kind past the u8 envelope");
        let mut w = ok;
        w.wiring[0].reserved = SPAWN_WIRE_MOVE | (1 << 1);
        w.wiring[0].rights = 0u32.into();
        assert!(decode_req(&w).is_none(), "unknown exact-move flags refuse");
        let mut w = ok;
        w.wiring[0].reserved = SPAWN_WIRE_MOVE;
        assert!(
            decode_req(&w).is_none(),
            "an exact move cannot also request attenuation rights"
        );
        let mut w = ok;
        w.wiring[0].reserved = SPAWN_WIRE_MOVE;
        w.wiring[0].rights = 0u32.into();
        assert!(
            decode_req(&w).is_some(),
            "the known exact-move shape decodes"
        );
        // ── The program-loading flavors ──────────────────────────────────
        let mut w = ok;
        w.boot_write_idx = kore_abi::svc::BOOT_FROM_SUPPLY | 3;
        w.boot_share_idx = 0xFF;
        assert!(
            decode_req(&w).is_some(),
            "supply-mode boot decodes; the share byte is a page offset"
        );
        let mut w = ok;
        w.boot_write_idx = kore_abi::svc::BOOT_FROM_SUPPLY | MAX_CONVEY as u8;
        assert!(decode_req(&w).is_none(), "supply index past the table");
        let mut w = ok;
        w.boot_write_idx = kore_abi::svc::BOOT_FROM_SUPPLY | 5;
        w.boot_share_idx = 5;
        assert!(
            decode_req(&w).is_some(),
            "supply mode has no distinctness rule — the share byte is data"
        );
        let mut w = ok;
        w.windows[0].flags = kore_abi::svc::WINDOW_SUBLEASE;
        w.windows[0].write = 1;
        assert!(decode_req(&w).is_some(), "a sublease window decodes");
        let mut w = ok;
        w.windows[0].flags = kore_abi::svc::WINDOW_SUBLEASE;
        w.windows[0].write = 0;
        assert!(decode_req(&w).is_none(), "a read-only sublease refuses");
        let mut w = ok;
        w.windows[0].flags = kore_abi::svc::WINDOW_SUBLEASE | kore_abi::svc::WINDOW_FORWARD;
        w.windows[0].write = 1;
        assert!(decode_req(&w).is_none(), "sublease+forward has no reading");
        let mut w = ok;
        w.windows[0].flags = 1 << 2;
        assert!(decode_req(&w).is_none(), "unknown window flag bits refuse");
        // Entries past the live prefixes are NOT validated (they ride
        // zero on encode and are ignored on decode) — in every table.
        let mut w = ok;
        w.wiring[MAX_SPAWN_WIRING - 1].rights = 0xFFFFu32.into();
        w.windows[MAX_SPAWN_WINDOWS - 1].convey_idx = 0xFF;
        w.windows[MAX_SPAWN_WINDOWS - 1].write = 0xFF;
        assert!(decode_req(&w).is_some(), "dead tail entries are ignored");

        let base_reply = encode_rep(&TaskResponse::InstallerSealed, 0);
        for tag in [99] {
            let mut r = base_reply;
            r.tag = tag;
            assert!(decode_rep(&r).is_none(), "reply tag {tag} is not current");
        }
        let mut r = encode_rep(&TaskResponse::Refused(TaskRefusal::BadBlock), 0);
        // One past the committed-discard refusal; never reuse a live code.
        r.refusal = TaskRepWire::REFUSE_UNFOUNDED + 1;
        assert!(decode_rep(&r).is_none());
        let mut r = base_reply;
        r.refusal = 1;
        assert!(
            decode_rep(&r).is_none(),
            "a refusal byte on success refuses"
        );
        let mut r = encode_rep(&TaskResponse::InstallerSealed, 0);
        r.index = 1;
        assert!(decode_rep(&r).is_none(), "sealed reply carries no index");
    }
}
