//! Fail-closed codecs over `kore_abi::svc`'s frozen filesystem forms.
//!
//! Every decode is **total and suspicious**. The bytes come off a ring shared
//! with another process, so a decode's job is not to interpret them
//! charitably — it is to refuse anything that is not exactly a request this
//! service implements. An unknown verb, a right this build does not define, a
//! name length past the field, a non-zero reserved byte, or junk in a name's
//! unused tail all decode to `None`, and the serving loop turns that into a
//! refusal rather than a guess.
//!
//! Being stricter than necessary is the point. This service hands out
//! authority; a field it reads charitably today is a field an attacker shapes
//! tomorrow.
//!
//! # Per-verb field meanings
//!
//! One fixed record serves every verb, so a field's meaning is per-verb — the
//! same convention `OutReqWire`'s `slot`/`through` use. The mapping:
//!
//! | Verb | `handle` | `handle2` | `offset` | `length` | names |
//! |---|---|---|---|---|---|
//! | `OpenRoot` | — | — | — | — | — |
//! | `OpenAt` | parent | — | — | — | `name` |
//! | `OpenGrant` | — | grant id | — | — | — |
//! | `Close`/`Stat`/`Sync` | object | — | — | — | — |
//! | `List` | directory | — | — | cursor | — |
//! | `Create` | parent | — | — | — | `name` |
//! | `Read` | file | — | file offset | length | — |
//! | `WriteAt` | file | — | file offset | length | — |
//! | `Append` | file | — | — | length | — |
//! | `Resize` | file | — | new length | — | — |
//! | `Rename` | parent | — | — | — | `name`, `name2` |
//! | `Move` | src parent | dst parent | — | — | `name`, `name2` |
//! | `Link` | parent | file | — | — | `name` |
//! | `Unlink` | parent | — | — | — | `name` |
//! | `Grant` | object | — | — | — | optional `name` |
//! | `GrantScoped` | object | — | max bytes | max objects | optional `name` |
//! | `Revoke` | — | grant id | — | — | — |
//! | `LendVersion`/`LendExecutable` | file | — | — | — | — |
//! | `ReadVersion` | — | version lease | source offset | length | — |
//! | `ReleaseVersion` | — | version lease | — | — | — |
//!
//! Replies reuse `handle` for a minted handle, a minted grant id, or a stat's
//! object id; `count` for bytes transferred or entries listed; and `cursor`
//! (with `FLAG_MORE`) for a listing's resume point. A stat additionally rides
//! its link count in `count` and its grant count in `cursor`.

use kore_abi::session::SlotRef;
use kore_abi::svc::{FS_NAME_BYTES, FsRepWire, FsReqWire};
use kore_rt::svc::PeerBundlePart;
use zerocopy::byteorder::little_endian::{U16, U32, U64};

use crate::{
    AuthorityQuota, FsError, FsRequest, FsResponse, GrantId, Handle, Name, ObjectId, ObjectKind,
    ProviderError, Rights, SearchCursor, SearchQuery, SessionId, bounds::MAX_NAME_LEN,
};

/// The engine's inline name bound and the wire's name field must agree, or a
/// name that fits one would be truncated or refused by the other.
const _: () = assert!(
    MAX_NAME_LEN == FS_NAME_BYTES,
    "the engine's MAX_NAME_LEN must equal the wire's FS_NAME_BYTES"
);

/// The unconditional-write sentinel must be the same value on both sides, since
/// it shares its field with real revisions.
const _: () = assert!(crate::ANY_REVISION == FsReqWire::FS_ANY_REVISION);

// ─── requests ───────────────────────────────────────────────────────────────

/// A zeroed request record — every field must be *set* to be meaningful, so
/// encoders start from nothing rather than from stale bytes.
fn blank_req() -> FsReqWire {
    FsReqWire {
        op: 0,
        kind: 0,
        rights: 0,
        name_len: 0,
        name2_len: 0,
        reserved0: 0,
        session: U16::new(0),
        handle: U64::new(0),
        handle2: U64::new(0),
        offset: U64::new(0),
        expected_revision: U64::new(0),
        window_offset: U32::new(0),
        length: U32::new(0),
        reserved1: U32::new(0),
        reserved2: U32::new(0),
        name: [0; FS_NAME_BYTES],
        name2: [0; FS_NAME_BYTES],
    }
}

/// Place `name` into a wire name field, returning its live length.
fn put_name(dst: &mut [u8; FS_NAME_BYTES], name: &Name) -> u8 {
    let bytes = name.as_bytes();
    dst[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as u8
}

/// Lift a wire name field — fail-closed on a length past the field, and on
/// junk in the tail beyond the live bytes.
///
/// The tail check is not pedantry: the tail is padding this service must not
/// interpret, and a peer writing there is a peer whose framing does not match.
fn take_name(src: &[u8; FS_NAME_BYTES], len: u8) -> Option<Name> {
    let len = len as usize;
    if len > FS_NAME_BYTES {
        return None;
    }
    if src[len..].iter().any(|byte| *byte != 0) {
        return None;
    }
    Name::from_bytes(&src[..len]).ok()
}

fn put_search_term(dst: &mut [u8; FS_NAME_BYTES], term: &crate::SearchTerm) -> u8 {
    let bytes = term.as_bytes();
    dst[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as u8
}

fn take_search_term(src: &[u8; FS_NAME_BYTES], len: u8) -> Option<crate::SearchTerm> {
    let len = len as usize;
    if len > FS_NAME_BYTES || src[len..].iter().any(|byte| *byte != 0) {
        return None;
    }
    crate::SearchTerm::from_bytes(&src[..len]).ok()
}

fn peer_part_code(part: PeerBundlePart) -> u8 {
    match part {
        PeerBundlePart::Ring => FsReqWire::KIND_PEER_RING,
        PeerBundlePart::Kick => FsReqWire::KIND_PEER_KICK,
        PeerBundlePart::ReturnRecv => FsReqWire::KIND_PEER_RETURN_RECV,
        PeerBundlePart::Grant(index) => FsReqWire::KIND_PEER_GRANT_BASE | (index & 0x7f),
    }
}

fn decode_peer_part(code: u8) -> Option<PeerBundlePart> {
    Some(match code {
        FsReqWire::KIND_PEER_RING => PeerBundlePart::Ring,
        FsReqWire::KIND_PEER_KICK => PeerBundlePart::Kick,
        FsReqWire::KIND_PEER_RETURN_RECV => PeerBundlePart::ReturnRecv,
        value if value & FsReqWire::KIND_PEER_GRANT_BASE != 0 => {
            PeerBundlePart::Grant(value & 0x7f)
        }
        _ => return None,
    })
}

fn names_are_empty(w: &FsReqWire) -> bool {
    w.name_len == 0
        && w.name2_len == 0
        && w.name.iter().all(|byte| *byte == 0)
        && w.name2.iter().all(|byte| *byte == 0)
}

fn control_payload_is_empty(w: &FsReqWire) -> bool {
    w.handle.get() == 0
        && w.handle2.get() == 0
        && w.offset.get() == 0
        && w.expected_revision.get() == 0
        && w.length.get() == 0
        && w.session.get() == 0
        && names_are_empty(w)
}

fn control_payload_except_handle_is_empty(w: &FsReqWire) -> bool {
    w.handle2.get() == 0
        && w.offset.get() == 0
        && w.expected_revision.get() == 0
        && w.length.get() == 0
        && w.session.get() == 0
        && names_are_empty(w)
}

fn control_payload_except_name_is_empty(w: &FsReqWire) -> bool {
    w.handle.get() == 0
        && w.handle2.get() == 0
        && w.offset.get() == 0
        && w.expected_revision.get() == 0
        && w.length.get() == 0
        && w.session.get() == 0
        && w.name2_len == 0
        && w.name2.iter().all(|byte| *byte == 0)
}

/// Lower a request to its wire form (total).
pub fn encode_req(req: &FsRequest) -> FsReqWire {
    let mut w = blank_req();
    match *req {
        FsRequest::OpenRoot { rights } => {
            w.op = FsReqWire::OP_OPEN_ROOT;
            w.rights = rights.bits();
        }
        FsRequest::OpenAt {
            parent,
            name,
            rights,
        } => {
            w.op = FsReqWire::OP_OPEN_AT;
            w.handle = U64::new(parent.to_bits());
            w.rights = rights.bits();
            w.name_len = put_name(&mut w.name, &name);
        }
        FsRequest::OpenGrant { grant, rights } => {
            w.op = FsReqWire::OP_OPEN_GRANT;
            w.handle2 = U64::new(grant.to_bits());
            w.rights = rights.bits();
        }
        FsRequest::Close { handle } => {
            w.op = FsReqWire::OP_CLOSE;
            w.handle = U64::new(handle.to_bits());
        }
        FsRequest::Stat { handle } => {
            w.op = FsReqWire::OP_STAT;
            w.handle = U64::new(handle.to_bits());
        }
        FsRequest::List {
            handle,
            cursor,
            window_offset,
        } => {
            w.op = FsReqWire::OP_LIST;
            w.handle = U64::new(handle.to_bits());
            w.length = U32::new(cursor);
            w.window_offset = U32::new(window_offset);
        }
        FsRequest::ListGrants { window_offset } => {
            w.op = FsReqWire::OP_LIST_GRANTS;
            w.window_offset = U32::new(window_offset);
        }
        FsRequest::AttachWindow { slot, bytes } => {
            w.op = FsReqWire::OP_ATTACH_WINDOW;
            // Both fields describe the window itself: where the client's view
            // landed, and how long it is.
            w.window_offset = U32::new(slot);
            w.length = U32::new(bytes);
        }
        FsRequest::Create {
            parent,
            name,
            kind,
            subtype,
        } => {
            w.op = FsReqWire::OP_CREATE;
            w.handle = U64::new(parent.to_bits());
            w.kind = match kind {
                ObjectKind::File => FsReqWire::KIND_FILE,
                ObjectKind::Directory => FsReqWire::KIND_DIRECTORY,
            };
            // `Create` is the one verb whose reserved byte carries data: the
            // subtype the object is stamped with (data, never authority).
            w.reserved0 = subtype.to_bits();
            w.name_len = put_name(&mut w.name, &name);
        }
        FsRequest::Read {
            handle,
            offset,
            window_offset,
            length,
        } => {
            w.op = FsReqWire::OP_READ;
            w.handle = U64::new(handle.to_bits());
            w.offset = U64::new(offset);
            w.window_offset = U32::new(window_offset);
            w.length = U32::new(length);
        }
        FsRequest::WriteAt {
            handle,
            offset,
            window_offset,
            length,
            expected,
        } => {
            w.op = FsReqWire::OP_WRITE_AT;
            w.handle = U64::new(handle.to_bits());
            w.offset = U64::new(offset);
            w.window_offset = U32::new(window_offset);
            w.length = U32::new(length);
            w.expected_revision = U64::new(expected);
        }
        FsRequest::Append {
            handle,
            window_offset,
            length,
            expected,
        } => {
            w.op = FsReqWire::OP_APPEND;
            w.handle = U64::new(handle.to_bits());
            w.window_offset = U32::new(window_offset);
            w.length = U32::new(length);
            w.expected_revision = U64::new(expected);
        }
        FsRequest::Resize {
            handle,
            length,
            expected,
        } => {
            w.op = FsReqWire::OP_RESIZE;
            w.handle = U64::new(handle.to_bits());
            w.offset = U64::new(length);
            w.expected_revision = U64::new(expected);
        }
        FsRequest::Rename { parent, old, new } => {
            w.op = FsReqWire::OP_RENAME;
            w.handle = U64::new(parent.to_bits());
            w.name_len = put_name(&mut w.name, &old);
            w.name2_len = put_name(&mut w.name2, &new);
        }
        FsRequest::Relocate {
            src_parent,
            name,
            dst_parent,
            new_name,
        } => {
            w.op = FsReqWire::OP_MOVE;
            w.handle = U64::new(src_parent.to_bits());
            w.handle2 = U64::new(dst_parent.to_bits());
            w.name_len = put_name(&mut w.name, &name);
            w.name2_len = put_name(&mut w.name2, &new_name);
        }
        FsRequest::Link {
            parent,
            name,
            child,
        } => {
            w.op = FsReqWire::OP_LINK;
            w.handle = U64::new(parent.to_bits());
            w.handle2 = U64::new(child.to_bits());
            w.name_len = put_name(&mut w.name, &name);
        }
        FsRequest::Unlink { parent, name } => {
            w.op = FsReqWire::OP_UNLINK;
            w.handle = U64::new(parent.to_bits());
            w.name_len = put_name(&mut w.name, &name);
        }
        FsRequest::DetachEdge { parent, name } => {
            w.op = FsReqWire::OP_DETACH_EDGE;
            w.handle = U64::new(parent.to_bits());
            w.name_len = put_name(&mut w.name, &name);
        }
        FsRequest::Grant {
            handle,
            destination,
            rights,
            display_name,
        } => {
            w.op = FsReqWire::OP_GRANT;
            w.handle = U64::new(handle.to_bits());
            w.session = U16::new(destination.index());
            w.rights = rights.bits();
            if let Some(name) = display_name.as_ref() {
                w.name_len = put_name(&mut w.name, name);
            }
        }
        FsRequest::GrantScoped {
            handle,
            destination,
            rights,
            quota,
            display_name,
        } => {
            w.op = FsReqWire::OP_GRANT_SCOPED;
            w.handle = U64::new(handle.to_bits());
            w.session = U16::new(destination.index());
            w.rights = rights.bits();
            w.offset = U64::new(quota.max_bytes());
            w.window_offset = U32::new(quota.max_edges());
            w.length = U32::new(quota.max_objects());
            w.reserved1 = U32::new(quota.max_handles());
            if let Some(name) = display_name.as_ref() {
                w.name_len = put_name(&mut w.name, name);
            }
        }
        FsRequest::GrantView {
            handle,
            rights,
            display_name,
            announced,
        } => {
            w.op = FsReqWire::OP_GRANT_VIEW;
            w.handle = U64::new(handle.to_bits());
            w.rights = rights.bits();
            w.name_len = put_name(&mut w.name, &display_name);
            w.window_offset = U32::new(announced.index());
        }
        FsRequest::Revoke { grant } => {
            w.op = FsReqWire::OP_REVOKE;
            w.handle2 = U64::new(grant.to_bits());
        }
        FsRequest::Sync { handle } => {
            w.op = FsReqWire::OP_SYNC;
            w.handle = U64::new(handle.to_bits());
        }
        FsRequest::DeferCommits { hold } => {
            w.op = FsReqWire::OP_DEFER_COMMITS;
            w.kind = u8::from(hold);
        }
        FsRequest::SetSystemSeed { active } => {
            w.op = FsReqWire::OP_SET_SYSTEM_SEED;
            w.kind = u8::from(active);
        }
        FsRequest::LendVersion {
            handle,
            window_offset,
        } => {
            w.op = FsReqWire::OP_LEND_VERSION;
            w.handle = U64::new(handle.to_bits());
            w.window_offset = U32::new(window_offset);
        }
        FsRequest::LendExecutable {
            handle,
            window_offset,
        } => {
            w.op = FsReqWire::OP_LEND_EXECUTABLE;
            w.handle = U64::new(handle.to_bits());
            w.window_offset = U32::new(window_offset);
        }
        FsRequest::ReadVersion {
            lease,
            offset,
            window_offset,
            length,
        } => {
            w.op = FsReqWire::OP_READ_VERSION;
            w.handle2 = U64::new(lease.to_bits());
            w.offset = U64::new(offset);
            w.window_offset = U32::new(window_offset);
            w.length = U32::new(length);
        }
        FsRequest::CommitDerived {
            lease,
            window_offset,
            length,
            output_window_bytes,
        } => {
            w.op = FsReqWire::OP_COMMIT_DERIVED;
            w.handle2 = U64::new(lease.to_bits());
            w.window_offset = U32::new(window_offset);
            w.length = U32::new(length);
            w.expected_revision = U64::new(u64::from(output_window_bytes));
        }
        FsRequest::ReleaseVersion { lease } => {
            w.op = FsReqWire::OP_RELEASE_VERSION;
            w.handle2 = U64::new(lease.to_bits());
        }
        FsRequest::DerivedInfo { handle, parser } => {
            w.op = FsReqWire::OP_DERIVED_INFO;
            w.handle = U64::new(handle.to_bits());
            w.offset = U64::new(u64::from(parser.0));
        }
        FsRequest::Search {
            handle,
            query,
            cursor,
            window_offset,
        } => {
            w.op = FsReqWire::OP_SEARCH;
            w.handle = U64::new(handle.to_bits());
            w.handle2 = U64::new(query.min_len);
            w.expected_revision = U64::new(query.max_len);
            w.window_offset = U32::new(window_offset);
            w.kind = match query.kind {
                None => 0,
                Some(ObjectKind::File) => FsReqWire::KIND_FILE,
                Some(ObjectKind::Directory) => FsReqWire::KIND_DIRECTORY,
            };
            if let Some(name) = query.name_contains.as_ref() {
                w.name_len = put_name(&mut w.name, name);
            }
            if let Some(content) = query.content.as_ref() {
                w.rights = content.plane as u8;
                w.name2_len = put_search_term(&mut w.name2, &content.term);
            }
            w.reserved0 = query.freshness as u8;
            w.reserved1 = U32::new(query.max_work);
            w.reserved2 = U32::new(query.max_index_lag);
            if let Some(cursor) = cursor {
                w.offset = U64::new(cursor.observation);
                w.length = U32::new(cursor.position);
            }
        }
        FsRequest::RegisterSystemView { rights, announced } => {
            w.op = FsReqWire::OP_REGISTER_SYSTEM_VIEW;
            w.rights = rights.bits();
            w.window_offset = U32::new(announced.index());
        }
        FsRequest::RegisterPrivateView {
            rights,
            name,
            announced,
        } => {
            w.op = FsReqWire::OP_REGISTER_PRIVATE_VIEW;
            w.rights = rights.bits();
            w.name_len = put_name(&mut w.name, &name);
            w.window_offset = U32::new(announced.index());
        }
        FsRequest::SystemStatus { window_offset } => {
            w.op = FsReqWire::OP_SYSTEM_STATUS;
            w.window_offset = U32::new(window_offset);
        }
        FsRequest::ApplyView {
            peer,
            rights,
            announced,
        } => {
            w.op = FsReqWire::OP_APPLY_VIEW;
            w.handle = U64::new(u64::from(peer.arrival()));
            w.rights = rights.bits();
            w.window_offset = U32::new(announced.index());
        }
        FsRequest::BindView {
            handle,
            rights,
            announced,
        } => {
            w.op = FsReqWire::OP_BIND_VIEW;
            w.handle = U64::new(handle.to_bits());
            w.rights = rights.bits();
            w.window_offset = U32::new(announced.index());
        }
        FsRequest::AdmitPeer { part, announced } => {
            w.op = FsReqWire::OP_ADMIT_PEER;
            w.kind = peer_part_code(part);
            w.window_offset = U32::new(announced.index());
        }
        FsRequest::TakeAdmission => {
            w.op = FsReqWire::OP_TAKE_ADMISSION;
        }
        FsRequest::RetirePeer { peer } => {
            w.op = FsReqWire::OP_RETIRE_PEER;
            w.handle = U64::new(u64::from(peer.arrival()));
        }
        FsRequest::ConfirmRetired { peer } => {
            w.op = FsReqWire::OP_CONFIRM_RETIRED;
            w.handle = U64::new(u64::from(peer.arrival()));
        }
        FsRequest::ClosePeer => {
            w.op = FsReqWire::OP_CLOSE_PEER;
        }
    }
    w
}

/// Lift a wire request — fail-closed.
pub fn decode_req(w: &FsReqWire) -> Option<FsRequest> {
    // Reserves exist to be zero; a peer setting one is a peer this service
    // does not understand.
    if (!matches!(w.op, FsReqWire::OP_SEARCH | FsReqWire::OP_CREATE) && w.reserved0 != 0)
        || (!matches!(w.op, FsReqWire::OP_GRANT_SCOPED | FsReqWire::OP_SEARCH)
            && w.reserved1.get() != 0)
        || (w.op != FsReqWire::OP_SEARCH && w.reserved2.get() != 0)
    {
        return None;
    }
    if w.op == 0 || w.op > FsReqWire::OP_MAX {
        return None;
    }
    // Name lengths are checked here as well as in `take_name`, so a verb that
    // takes no name still refuses a record claiming one.
    if w.name_len as usize > FS_NAME_BYTES || w.name2_len as usize > FS_NAME_BYTES {
        return None;
    }

    let handle = Handle::from_bits(w.handle.get());
    let handle2 = Handle::from_bits(w.handle2.get());
    let grant = GrantId::from_bits(w.handle2.get());

    // `rights` is meaningful for this closed verb set and must be zero for the
    // rest — an ignored non-zero mask would be a field the peer thinks it set.
    let rights_verb = matches!(
        w.op,
        FsReqWire::OP_OPEN_ROOT
            | FsReqWire::OP_OPEN_AT
            | FsReqWire::OP_OPEN_GRANT
            | FsReqWire::OP_GRANT
            | FsReqWire::OP_GRANT_SCOPED
            | FsReqWire::OP_GRANT_VIEW
            | FsReqWire::OP_REGISTER_SYSTEM_VIEW
            | FsReqWire::OP_REGISTER_PRIVATE_VIEW
            | FsReqWire::OP_APPLY_VIEW
            | FsReqWire::OP_BIND_VIEW
    );
    let rights = if rights_verb {
        Rights::from_bits_exact(w.rights)?
    } else if w.op == FsReqWire::OP_SEARCH {
        Rights::NONE
    } else {
        if w.rights != 0 {
            return None;
        }
        Rights::NONE
    };

    // Likewise `kind` (Create/AdmitPeer/Search and the two boolean controls,
    // whose whole payload rides there) and `session` (Grant only). A verb missing
    // from this list cannot carry `kind` at all: its own arm never runs, the
    // record decodes to nothing, and the serve loop drops it without a reply
    // — which the caller cannot distinguish from a service that hangs.
    if !matches!(
        w.op,
        FsReqWire::OP_CREATE
            | FsReqWire::OP_ADMIT_PEER
            | FsReqWire::OP_SEARCH
            | FsReqWire::OP_DEFER_COMMITS
            | FsReqWire::OP_SET_SYSTEM_SEED
    ) && w.kind != 0
    {
        return None;
    }
    if !matches!(w.op, FsReqWire::OP_GRANT | FsReqWire::OP_GRANT_SCOPED) && w.session.get() != 0 {
        return None;
    }

    Some(match w.op {
        FsReqWire::OP_OPEN_ROOT => FsRequest::OpenRoot { rights },
        FsReqWire::OP_OPEN_AT => FsRequest::OpenAt {
            parent: handle,
            name: take_name(&w.name, w.name_len)?,
            rights,
        },
        FsReqWire::OP_OPEN_GRANT => FsRequest::OpenGrant { grant, rights },
        FsReqWire::OP_CLOSE => FsRequest::Close { handle },
        FsReqWire::OP_STAT => FsRequest::Stat { handle },
        FsReqWire::OP_LIST => FsRequest::List {
            handle,
            cursor: w.length.get(),
            window_offset: w.window_offset.get(),
        },
        FsReqWire::OP_LIST_GRANTS => FsRequest::ListGrants {
            window_offset: w.window_offset.get(),
        },
        FsReqWire::OP_SYSTEM_STATUS => FsRequest::SystemStatus {
            window_offset: w.window_offset.get(),
        },
        FsReqWire::OP_ATTACH_WINDOW => FsRequest::AttachWindow {
            slot: w.window_offset.get(),
            bytes: w.length.get(),
        },
        FsReqWire::OP_CREATE => FsRequest::Create {
            parent: handle,
            name: take_name(&w.name, w.name_len)?,
            kind: match w.kind {
                FsReqWire::KIND_FILE => ObjectKind::File,
                FsReqWire::KIND_DIRECTORY => ObjectKind::Directory,
                _ => return None,
            },
            subtype: crate::ObjectSubtype::from_bits(w.reserved0),
        },
        FsReqWire::OP_READ => FsRequest::Read {
            handle,
            offset: w.offset.get(),
            window_offset: w.window_offset.get(),
            length: w.length.get(),
        },
        FsReqWire::OP_WRITE_AT => FsRequest::WriteAt {
            handle,
            offset: w.offset.get(),
            window_offset: w.window_offset.get(),
            length: w.length.get(),
            expected: w.expected_revision.get(),
        },
        FsReqWire::OP_APPEND => FsRequest::Append {
            handle,
            window_offset: w.window_offset.get(),
            length: w.length.get(),
            expected: w.expected_revision.get(),
        },
        FsReqWire::OP_RESIZE => FsRequest::Resize {
            handle,
            length: w.offset.get(),
            expected: w.expected_revision.get(),
        },
        FsReqWire::OP_RENAME => FsRequest::Rename {
            parent: handle,
            old: take_name(&w.name, w.name_len)?,
            new: take_name(&w.name2, w.name2_len)?,
        },
        FsReqWire::OP_MOVE => FsRequest::Relocate {
            src_parent: handle,
            name: take_name(&w.name, w.name_len)?,
            dst_parent: handle2,
            new_name: take_name(&w.name2, w.name2_len)?,
        },
        FsReqWire::OP_LINK => FsRequest::Link {
            parent: handle,
            name: take_name(&w.name, w.name_len)?,
            child: handle2,
        },
        FsReqWire::OP_UNLINK => FsRequest::Unlink {
            parent: handle,
            name: take_name(&w.name, w.name_len)?,
        },
        FsReqWire::OP_DETACH_EDGE => FsRequest::DetachEdge {
            parent: handle,
            name: take_name(&w.name, w.name_len)?,
        },
        FsReqWire::OP_GRANT => FsRequest::Grant {
            handle,
            destination: SessionId::new(w.session.get()),
            rights,
            // A zero-length name means "no displayed attachment name", which
            // is legal here — unlike every other name-taking verb, where a
            // zero length is an invalid name.
            display_name: if w.name_len == 0 {
                if w.name.iter().any(|byte| *byte != 0) {
                    return None;
                }
                None
            } else {
                Some(take_name(&w.name, w.name_len)?)
            },
        },
        FsReqWire::OP_GRANT_SCOPED => {
            if w.handle2.get() != 0
                || w.expected_revision.get() != 0
                || w.name2_len != 0
                || w.name2.iter().any(|byte| *byte != 0)
            {
                return None;
            }
            FsRequest::GrantScoped {
                handle,
                destination: SessionId::new(w.session.get()),
                rights,
                quota: AuthorityQuota::new(
                    w.length.get(),
                    w.window_offset.get(),
                    w.offset.get(),
                    w.reserved1.get(),
                ),
                display_name: if w.name_len == 0 {
                    if w.name.iter().any(|byte| *byte != 0) {
                        return None;
                    }
                    None
                } else {
                    Some(take_name(&w.name, w.name_len)?)
                },
            }
        }
        FsReqWire::OP_GRANT_VIEW => {
            if w.handle2.get() != 0
                || w.expected_revision.get() != 0
                || w.offset.get() != 0
                || w.length.get() != 0
                || w.name2_len != 0
                || w.name2.iter().any(|byte| *byte != 0)
            {
                return None;
            }
            FsRequest::GrantView {
                handle,
                rights,
                display_name: take_name(&w.name, w.name_len)?,
                announced: SlotRef::from_wire(w.window_offset.get()),
            }
        }
        FsReqWire::OP_REVOKE => FsRequest::Revoke { grant },
        FsReqWire::OP_SYNC => FsRequest::Sync { handle },
        FsReqWire::OP_DEFER_COMMITS => {
            // `kind` is the whole payload: a deferral names no object, and a
            // value outside the two-state vocabulary is a framing mismatch.
            if w.kind > 1 || !control_payload_is_empty(w) || w.rights != 0 {
                return None;
            }
            FsRequest::DeferCommits { hold: w.kind == 1 }
        }
        FsReqWire::OP_SET_SYSTEM_SEED => {
            if w.kind > 1 || !control_payload_is_empty(w) || w.rights != 0 {
                return None;
            }
            FsRequest::SetSystemSeed {
                active: w.kind == 1,
            }
        }
        FsReqWire::OP_LEND_VERSION => FsRequest::LendVersion {
            handle,
            window_offset: w.window_offset.get(),
        },
        FsReqWire::OP_LEND_EXECUTABLE => FsRequest::LendExecutable {
            handle,
            window_offset: w.window_offset.get(),
        },
        FsReqWire::OP_READ_VERSION => FsRequest::ReadVersion {
            lease: crate::VersionLease::from_bits(w.handle2.get()),
            offset: w.offset.get(),
            window_offset: w.window_offset.get(),
            length: w.length.get(),
        },
        FsReqWire::OP_COMMIT_DERIVED => FsRequest::CommitDerived {
            lease: crate::VersionLease::from_bits(w.handle2.get()),
            window_offset: w.window_offset.get(),
            length: w.length.get(),
            output_window_bytes: u32::try_from(w.expected_revision.get()).ok()?,
        },
        FsReqWire::OP_RELEASE_VERSION => FsRequest::ReleaseVersion {
            lease: crate::VersionLease::from_bits(w.handle2.get()),
        },
        FsReqWire::OP_DERIVED_INFO => FsRequest::DerivedInfo {
            handle,
            parser: kore_svc_extract::ParserId(u16::try_from(w.offset.get()).ok()?),
        },
        FsReqWire::OP_SEARCH => {
            if (w.offset.get() == 0) != (w.length.get() == 0) {
                return None;
            }
            FsRequest::Search {
                handle,
                query: SearchQuery {
                    name_contains: if w.name_len == 0 {
                        if w.name.iter().any(|byte| *byte != 0) {
                            return None;
                        }
                        None
                    } else {
                        Some(take_name(&w.name, w.name_len)?)
                    },
                    kind: match w.kind {
                        0 => None,
                        FsReqWire::KIND_FILE => Some(ObjectKind::File),
                        FsReqWire::KIND_DIRECTORY => Some(ObjectKind::Directory),
                        _ => return None,
                    },
                    min_len: w.handle2.get(),
                    max_len: w.expected_revision.get(),
                    content: if w.name2_len == 0 {
                        if w.name2.iter().any(|byte| *byte != 0) {
                            return None;
                        }
                        if w.rights != 0 {
                            return None;
                        }
                        None
                    } else {
                        Some(crate::SearchContent {
                            plane: crate::SearchPlane::from_wire(w.rights)?,
                            term: take_search_term(&w.name2, w.name2_len)?,
                        })
                    },
                    freshness: crate::FreshnessPolicy::from_wire(w.reserved0)?,
                    max_index_lag: w.reserved2.get(),
                    max_work: w.reserved1.get(),
                },
                cursor: if w.offset.get() == 0 {
                    None
                } else {
                    Some(SearchCursor {
                        observation: w.offset.get(),
                        position: w.length.get(),
                    })
                },
                window_offset: w.window_offset.get(),
            }
        }
        FsReqWire::OP_REGISTER_SYSTEM_VIEW => {
            if w.kind != 0 || !control_payload_is_empty(w) || rights.is_empty() {
                return None;
            }
            FsRequest::RegisterSystemView {
                rights,
                announced: SlotRef::from_wire(w.window_offset.get()),
            }
        }
        FsReqWire::OP_REGISTER_PRIVATE_VIEW => {
            if w.kind != 0 || !control_payload_except_name_is_empty(w) || rights.is_empty() {
                return None;
            }
            FsRequest::RegisterPrivateView {
                rights,
                name: take_name(&w.name, w.name_len)?,
                announced: SlotRef::from_wire(w.window_offset.get()),
            }
        }
        FsReqWire::OP_APPLY_VIEW => {
            if w.kind != 0 || !control_payload_except_handle_is_empty(w) || rights.is_empty() {
                return None;
            }
            FsRequest::ApplyView {
                peer: kore_rt::svc::PeerId::from_arrival(u16::try_from(w.handle.get()).ok()?)?,
                rights,
                announced: SlotRef::from_wire(w.window_offset.get()),
            }
        }
        FsReqWire::OP_BIND_VIEW => {
            if w.kind != 0 || !control_payload_except_handle_is_empty(w) || rights.is_empty() {
                return None;
            }
            FsRequest::BindView {
                handle,
                rights,
                announced: SlotRef::from_wire(w.window_offset.get()),
            }
        }
        FsReqWire::OP_ADMIT_PEER => {
            if !control_payload_is_empty(w) {
                return None;
            }
            FsRequest::AdmitPeer {
                part: decode_peer_part(w.kind)?,
                announced: SlotRef::from_wire(w.window_offset.get()),
            }
        }
        FsReqWire::OP_CLOSE_PEER => {
            if w.kind != 0 || w.window_offset.get() != 0 || !control_payload_is_empty(w) {
                return None;
            }
            FsRequest::ClosePeer
        }
        FsReqWire::OP_TAKE_ADMISSION => {
            if w.kind != 0 || w.window_offset.get() != 0 || !control_payload_is_empty(w) {
                return None;
            }
            FsRequest::TakeAdmission
        }
        FsReqWire::OP_RETIRE_PEER | FsReqWire::OP_CONFIRM_RETIRED => {
            if w.kind != 0
                || w.window_offset.get() != 0
                || !control_payload_except_handle_is_empty(w)
            {
                return None;
            }
            let arrival = u16::try_from(w.handle.get()).ok()?;
            let peer = kore_rt::svc::PeerId::from_arrival(arrival)?;
            if w.op == FsReqWire::OP_RETIRE_PEER {
                FsRequest::RetirePeer { peer }
            } else {
                FsRequest::ConfirmRetired { peer }
            }
        }
        _ => return None,
    })
}

// ─── replies ────────────────────────────────────────────────────────────────

fn blank_rep() -> FsRepWire {
    FsRepWire {
        status: 0,
        kind: 0,
        flags: 0,
        grants: 0,
        count: U32::new(0),
        handle: U64::new(0),
        len: U64::new(0),
        revision: U64::new(0),
        cursor: U32::new(0),
        reserved0: U32::new(0),
        reserved1: U64::new(0),
    }
}

/// The wire status for a refusal.
fn status_of(err: FsError) -> u8 {
    match err {
        FsError::NotFound => FsRepWire::STATUS_NOT_FOUND,
        FsError::AlreadyExists => FsRepWire::STATUS_ALREADY_EXISTS,
        FsError::NotDirectory => FsRepWire::STATUS_NOT_DIRECTORY,
        FsError::IsDirectory => FsRepWire::STATUS_IS_DIRECTORY,
        FsError::InvalidName => FsRepWire::STATUS_INVALID_NAME,
        FsError::InvalidHandle => FsRepWire::STATUS_INVALID_HANDLE,
        FsError::AccessDenied => FsRepWire::STATUS_ACCESS_DENIED,
        FsError::AccessLost => FsRepWire::STATUS_ACCESS_LOST,
        FsError::Conflict => FsRepWire::STATUS_CONFLICT,
        FsError::Cycle => FsRepWire::STATUS_CYCLE,
        FsError::NotEmpty => FsRepWire::STATUS_NOT_EMPTY,
        FsError::OutOfObjects => FsRepWire::STATUS_OUT_OF_OBJECTS,
        FsError::OutOfEdges => FsRepWire::STATUS_OUT_OF_EDGES,
        FsError::OutOfHandles => FsRepWire::STATUS_OUT_OF_HANDLES,
        FsError::OutOfStorage => FsRepWire::STATUS_OUT_OF_STORAGE,
        FsError::OutOfPlacements => FsRepWire::STATUS_OUT_OF_PLACEMENTS,
        FsError::OutOfVersionLeases => FsRepWire::STATUS_OUT_OF_VERSION_LEASES,
        FsError::OutOfIndexSlots => FsRepWire::STATUS_OUT_OF_INDEX_SLOTS,
        FsError::Range => FsRepWire::STATUS_RANGE,
        FsError::BudgetExceeded => FsRepWire::STATUS_BUDGET_EXCEEDED,
        FsError::IndexIncomplete => FsRepWire::STATUS_INDEX_INCOMPLETE,
        FsError::Unsupported => FsRepWire::STATUS_UNSUPPORTED,
        FsError::NoWindow => FsRepWire::STATUS_NO_WINDOW,
        FsError::Busy => FsRepWire::STATUS_BUSY,
        FsError::ProviderFailure(_) => FsRepWire::STATUS_PROVIDER_FAILURE,
    }
}

/// The refusal for a wire status.
///
/// `PROVIDER_FAILURE` lands on [`ProviderError::Media`]: the provider's own
/// distinction between medium, range, and unknown-key does not survive the
/// boundary, and inventing one back would be a lie. What the client needs to
/// know — the provider failed and the request had no visible effect — does
/// survive.
fn error_of(status: u8) -> Option<FsError> {
    Some(match status {
        FsRepWire::STATUS_NOT_FOUND => FsError::NotFound,
        FsRepWire::STATUS_ALREADY_EXISTS => FsError::AlreadyExists,
        FsRepWire::STATUS_NOT_DIRECTORY => FsError::NotDirectory,
        FsRepWire::STATUS_IS_DIRECTORY => FsError::IsDirectory,
        FsRepWire::STATUS_INVALID_NAME => FsError::InvalidName,
        FsRepWire::STATUS_INVALID_HANDLE => FsError::InvalidHandle,
        FsRepWire::STATUS_ACCESS_DENIED => FsError::AccessDenied,
        FsRepWire::STATUS_ACCESS_LOST => FsError::AccessLost,
        FsRepWire::STATUS_CONFLICT => FsError::Conflict,
        FsRepWire::STATUS_CYCLE => FsError::Cycle,
        FsRepWire::STATUS_NOT_EMPTY => FsError::NotEmpty,
        FsRepWire::STATUS_OUT_OF_OBJECTS => FsError::OutOfObjects,
        FsRepWire::STATUS_OUT_OF_EDGES => FsError::OutOfEdges,
        FsRepWire::STATUS_OUT_OF_HANDLES => FsError::OutOfHandles,
        FsRepWire::STATUS_OUT_OF_STORAGE => FsError::OutOfStorage,
        FsRepWire::STATUS_OUT_OF_PLACEMENTS => FsError::OutOfPlacements,
        FsRepWire::STATUS_OUT_OF_VERSION_LEASES => FsError::OutOfVersionLeases,
        FsRepWire::STATUS_OUT_OF_INDEX_SLOTS => FsError::OutOfIndexSlots,
        FsRepWire::STATUS_RANGE => FsError::Range,
        FsRepWire::STATUS_BUDGET_EXCEEDED => FsError::BudgetExceeded,
        FsRepWire::STATUS_INDEX_INCOMPLETE => FsError::IndexIncomplete,
        FsRepWire::STATUS_UNSUPPORTED => FsError::Unsupported,
        FsRepWire::STATUS_NO_WINDOW => FsError::NoWindow,
        FsRepWire::STATUS_BUSY => FsError::Busy,
        FsRepWire::STATUS_PROVIDER_FAILURE => FsError::ProviderFailure(ProviderError::Media),
        _ => return None,
    })
}

fn kind_byte(kind: ObjectKind) -> u8 {
    match kind {
        ObjectKind::File => FsReqWire::KIND_FILE,
        ObjectKind::Directory => FsReqWire::KIND_DIRECTORY,
    }
}

fn kind_of(byte: u8) -> Option<ObjectKind> {
    match byte {
        FsReqWire::KIND_FILE => Some(ObjectKind::File),
        FsReqWire::KIND_DIRECTORY => Some(ObjectKind::Directory),
        _ => None,
    }
}

/// Lower a response (total).
///
/// `grants` is pinned zero and asserted, not honoured: the filesystem mints no
/// kernel authority, so a grant-bearing filesystem reply is a contradiction.
pub fn encode_rep(rep: &FsResponse, grants: u8) -> FsRepWire {
    debug_assert_eq!(grants, 0, "the filesystem grants no kernel authority");
    let mut w = blank_rep();
    match *rep {
        FsResponse::Opened {
            handle,
            stable_object,
            kind,
            subtype,
            len,
            revision,
        } => {
            w.status = FsRepWire::STATUS_OK;
            w.kind = kind_byte(kind);
            w.handle = U64::new(handle.to_bits());
            w.reserved0 = U32::new(u32::from(subtype.to_bits()));
            w.reserved1 = U64::new(stable_object.get());
            w.len = U64::new(len);
            w.revision = U64::new(revision);
        }
        FsResponse::Stat {
            object,
            stable_object,
            kind,
            subtype,
            len,
            revision,
            links,
            grants: grant_count,
        } => {
            w.status = FsRepWire::STATUS_OK;
            w.kind = kind_byte(kind);
            w.handle = U64::new(object.to_bits());
            w.reserved0 = U32::new(u32::from(subtype.to_bits()));
            w.reserved1 = U64::new(stable_object.get());
            w.len = U64::new(len);
            w.revision = U64::new(revision);
            w.count = U32::new(links);
            w.cursor = U32::new(grant_count);
        }
        FsResponse::Listed { entries, next } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(entries);
            if let Some(cursor) = next {
                w.flags = FsRepWire::FLAG_MORE;
                w.cursor = U32::new(cursor);
            }
        }
        FsResponse::GrantsListed { entries } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(entries);
        }
        FsResponse::SystemStatusReported { len } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(len);
        }
        FsResponse::Transferred { count } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(count);
        }
        FsResponse::Committed { revision } => {
            w.status = FsRepWire::STATUS_OK;
            w.revision = U64::new(revision);
        }
        FsResponse::Appended { offset, revision } => {
            w.status = FsRepWire::STATUS_OK;
            w.len = U64::new(offset);
            w.revision = U64::new(revision);
        }
        FsResponse::Granted { grant } => {
            w.status = FsRepWire::STATUS_OK;
            w.handle = U64::new(grant.to_bits());
        }
        FsResponse::WindowAttached { max_transfer } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(max_transfer);
        }
        FsResponse::Done => {
            w.status = FsRepWire::STATUS_OK;
        }
        FsResponse::VersionLent { lease } => {
            w.status = FsRepWire::STATUS_OK;
            w.handle = U64::new(lease.to_bits());
        }
        FsResponse::DerivedCommitted {
            parser,
            parser_version,
        } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(parser_version);
            w.cursor = U32::new(u32::from(parser.0));
        }
        FsResponse::DerivedInfo {
            input_version,
            parser_version,
        } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(parser_version);
            w.revision = U64::new(input_version);
        }
        FsResponse::Searched {
            hits,
            observation,
            snapshot,
            index_generation,
            freshness,
            next,
            scanned,
            work,
        } => {
            w.status = FsRepWire::STATUS_OK;
            w.count = U32::new(hits);
            w.revision = U64::new(observation);
            w.handle = U64::new(snapshot);
            w.reserved1 = U64::new(index_generation);
            w.kind = freshness as u8;
            w.len = U64::new(u64::from(scanned) | (u64::from(work) << 32));
            if let Some(cursor) = next {
                debug_assert_eq!(cursor.observation, observation);
                w.flags = FsRepWire::FLAG_MORE;
                w.cursor = U32::new(cursor.position);
            }
        }
        FsResponse::PeerAdmitted { peer } => {
            w.status = FsRepWire::STATUS_OK;
            w.handle = U64::new(u64::from(peer.arrival()));
        }
        FsResponse::Refused(err) => {
            w.status = status_of(err);
        }
    }
    w
}

/// Lift a wire reply — fail-closed.
///
/// A reply is decoded against the **request it answers**, because one fixed
/// record serves every verb: `count` is a byte count for a `Read` and a link
/// count for a `Stat`, and without the request there is no way to know which.
/// This is why [`decode_rep_for`] takes the request; a decode that guessed
/// would hand a client the wrong number with a successful status.
pub fn decode_rep_for(request: &FsRequest, w: &FsRepWire) -> Option<(FsResponse, u8)> {
    if w.grants != 0 {
        // Nothing here can grant kernel authority.
        return None;
    }
    // The reserved word carries one byte of data — the object subtype — and
    // only on the replies that describe an object; everywhere else it is
    // reserved and must be zero.
    let describes_object = matches!(
        request,
        FsRequest::OpenRoot { .. }
            | FsRequest::OpenAt { .. }
            | FsRequest::OpenGrant { .. }
            | FsRequest::Create { .. }
            | FsRequest::Stat { .. }
    );
    let subtype = u8::try_from(w.reserved0.get()).ok()?;
    if w.status != FsRepWire::STATUS_OK && w.reserved0.get() != 0 {
        return None;
    }
    if !describes_object && w.reserved0.get() != 0 {
        return None;
    }
    if w.status == 0 || w.status > FsRepWire::STATUS_MAX {
        return None;
    }
    if w.status != FsRepWire::STATUS_OK {
        // A refusal carries a status and nothing else; a refusal with payload
        // is a reply this service did not write.
        if w.kind != 0
            || w.flags != 0
            || w.count.get() != 0
            || w.handle.get() != 0
            || w.len.get() != 0
            || w.revision.get() != 0
            || w.cursor.get() != 0
            || w.reserved1.get() != 0
        {
            return None;
        }
        return Some((FsResponse::Refused(error_of(w.status)?), 0));
    }

    // Only a listing may set flags, and `FLAG_MORE` is the only defined bit.
    let listing = matches!(request, FsRequest::List { .. } | FsRequest::Search { .. });
    if w.flags & !FsRepWire::FLAG_MORE != 0 {
        return None;
    }
    if w.flags != 0 && !listing {
        return None;
    }

    let rep = match *request {
        FsRequest::OpenRoot { .. }
        | FsRequest::OpenAt { .. }
        | FsRequest::OpenGrant { .. }
        | FsRequest::Create { .. } => FsResponse::Opened {
            handle: Handle::from_bits(w.handle.get()),
            stable_object: crate::StableObjectId::new(w.reserved1.get())?,
            kind: kind_of(w.kind)?,
            subtype: crate::ObjectSubtype::from_bits(subtype),
            len: w.len.get(),
            revision: w.revision.get(),
        },
        FsRequest::Stat { .. } => FsResponse::Stat {
            object: ObjectId::from_bits(w.handle.get()),
            stable_object: crate::StableObjectId::new(w.reserved1.get())?,
            kind: kind_of(w.kind)?,
            subtype: crate::ObjectSubtype::from_bits(subtype),
            len: w.len.get(),
            revision: w.revision.get(),
            links: w.count.get(),
            grants: w.cursor.get(),
        },
        FsRequest::List { .. } => FsResponse::Listed {
            entries: w.count.get(),
            next: if w.flags & FsRepWire::FLAG_MORE != 0 {
                Some(w.cursor.get())
            } else {
                // No more pages, so no cursor — a stale one left set would be
                // a resume point a client might believe.
                if w.cursor.get() != 0 {
                    return None;
                }
                None
            },
        },
        FsRequest::ListGrants { .. } => FsResponse::GrantsListed {
            entries: w.count.get(),
        },
        FsRequest::SystemStatus { .. } => FsResponse::SystemStatusReported { len: w.count.get() },
        FsRequest::Read { .. } | FsRequest::ReadVersion { .. } => FsResponse::Transferred {
            count: w.count.get(),
        },
        FsRequest::WriteAt { .. } | FsRequest::Resize { .. } => FsResponse::Committed {
            revision: w.revision.get(),
        },
        FsRequest::Append { .. } => FsResponse::Appended {
            offset: w.len.get(),
            revision: w.revision.get(),
        },
        FsRequest::Grant { .. } | FsRequest::GrantScoped { .. } | FsRequest::GrantView { .. } => {
            FsResponse::Granted {
                grant: GrantId::from_bits(w.handle.get()),
            }
        }
        FsRequest::LendVersion { .. } | FsRequest::LendExecutable { .. } => {
            FsResponse::VersionLent {
                lease: crate::VersionLease::from_bits(w.handle.get()),
            }
        }
        FsRequest::CommitDerived { .. } => FsResponse::DerivedCommitted {
            parser: kore_svc_extract::ParserId(u16::try_from(w.cursor.get()).ok()?),
            parser_version: w.count.get(),
        },
        FsRequest::DerivedInfo { .. } => FsResponse::DerivedInfo {
            input_version: w.revision.get(),
            parser_version: w.count.get(),
        },
        FsRequest::Search { query, .. } => {
            let freshness = crate::FreshnessPolicy::from_wire(w.kind)?;
            let snapshot = w.handle.get();
            let index_generation = w.reserved1.get();
            if w.revision.get() == 0 || snapshot == 0 || freshness != query.freshness {
                return None;
            }
            if (freshness == crate::FreshnessPolicy::StructuralOnly) != (index_generation == 0) {
                return None;
            }
            let observation = w.revision.get();
            FsResponse::Searched {
                hits: w.count.get(),
                observation,
                snapshot,
                index_generation,
                freshness,
                next: if w.flags & FsRepWire::FLAG_MORE != 0 {
                    if w.cursor.get() == 0 {
                        return None;
                    }
                    Some(SearchCursor {
                        observation,
                        position: w.cursor.get(),
                    })
                } else {
                    if w.cursor.get() != 0 {
                        return None;
                    }
                    None
                },
                scanned: w.len.get() as u32,
                work: (w.len.get() >> 32) as u32,
            }
        }
        FsRequest::TakeAdmission => FsResponse::PeerAdmitted {
            peer: kore_rt::svc::PeerId::from_arrival(u16::try_from(w.handle.get()).ok()?)?,
        },
        FsRequest::AttachWindow { .. } => FsResponse::WindowAttached {
            max_transfer: w.count.get(),
        },
        FsRequest::RegisterSystemView { .. }
        | FsRequest::RegisterPrivateView { .. }
        | FsRequest::ApplyView { .. }
        | FsRequest::BindView { .. }
        | FsRequest::Close { .. }
        | FsRequest::Rename { .. }
        | FsRequest::Relocate { .. }
        | FsRequest::Link { .. }
        | FsRequest::Unlink { .. }
        | FsRequest::DetachEdge { .. }
        | FsRequest::Revoke { .. }
        | FsRequest::Sync { .. }
        | FsRequest::DeferCommits { .. }
        | FsRequest::SetSystemSeed { .. }
        | FsRequest::ReleaseVersion { .. }
        | FsRequest::AdmitPeer { .. }
        | FsRequest::RetirePeer { .. }
        | FsRequest::ConfirmRetired { .. }
        | FsRequest::ClosePeer => FsResponse::Done,
    };
    if !matches!(
        request,
        FsRequest::OpenRoot { .. }
            | FsRequest::OpenAt { .. }
            | FsRequest::OpenGrant { .. }
            | FsRequest::Create { .. }
            | FsRequest::Stat { .. }
            | FsRequest::Search { .. }
    ) && w.reserved1.get() != 0
    {
        return None;
    }
    Some((rep, 0))
}

/// Decode a reply **without** its request — only what is unambiguous.
///
/// The `SvcProtocol` hook has this shape, but one fixed record serves every
/// verb: `count` is a byte count for a `Read` and a link count for a `Stat`,
/// and nothing in the reply says which. So this reports the two shapes that
/// *are* self-describing — a refusal, and a success with no payload — and
/// refuses the rest rather than guessing. Callers holding the request use
/// [`decode_rep_for`], which is what the typed client does.
pub fn decode_rep_unambiguous(w: &FsRepWire) -> Option<(FsResponse, u8)> {
    if w.grants != 0 || w.reserved0.get() != 0 || w.reserved1.get() != 0 {
        return None;
    }
    if w.status == 0 || w.status > FsRepWire::STATUS_MAX {
        return None;
    }
    if w.status != FsRepWire::STATUS_OK {
        if w.kind != 0
            || w.flags != 0
            || w.count.get() != 0
            || w.handle.get() != 0
            || w.len.get() != 0
            || w.revision.get() != 0
            || w.cursor.get() != 0
        {
            return None;
        }
        return Some((FsResponse::Refused(error_of(w.status)?), 0));
    }
    // A payload-free success is `Done` and can be nothing else.
    if w.kind == 0
        && w.flags == 0
        && w.count.get() == 0
        && w.handle.get() == 0
        && w.len.get() == 0
        && w.revision.get() == 0
        && w.cursor.get() == 0
    {
        return Some((FsResponse::Done, 0));
    }
    None
}

/// Whether `response` is a shape `request` could have produced.
///
/// The request/reply rings are uncorrelated SPSC pairs, so this is the check
/// that a reply belongs to the call in flight rather than to the previous one.
pub fn reply_matches(request: &FsRequest, response: &FsResponse) -> bool {
    if matches!(response, FsResponse::Refused(_)) {
        // Any request may be refused.
        return true;
    }
    matches!(
        (request, response),
        (
            FsRequest::OpenRoot { .. }
                | FsRequest::OpenAt { .. }
                | FsRequest::OpenGrant { .. }
                | FsRequest::Create { .. },
            FsResponse::Opened { .. }
        ) | (FsRequest::Stat { .. }, FsResponse::Stat { .. })
            | (FsRequest::List { .. }, FsResponse::Listed { .. })
            | (
                FsRequest::ListGrants { .. },
                FsResponse::GrantsListed { .. }
            )
            | (
                FsRequest::SystemStatus { .. },
                FsResponse::SystemStatusReported { .. }
            )
            | (
                FsRequest::Read { .. } | FsRequest::ReadVersion { .. },
                FsResponse::Transferred { .. }
            )
            | (
                FsRequest::WriteAt { .. } | FsRequest::Resize { .. },
                FsResponse::Committed { .. }
            )
            | (FsRequest::Append { .. }, FsResponse::Appended { .. })
            | (
                FsRequest::Grant { .. }
                    | FsRequest::GrantScoped { .. }
                    | FsRequest::GrantView { .. },
                FsResponse::Granted { .. }
            )
            | (
                FsRequest::LendVersion { .. } | FsRequest::LendExecutable { .. },
                FsResponse::VersionLent { .. }
            )
            | (
                FsRequest::CommitDerived { .. },
                FsResponse::DerivedCommitted { .. }
            )
            | (
                FsRequest::DerivedInfo { .. },
                FsResponse::DerivedInfo { .. }
            )
            | (FsRequest::Search { .. }, FsResponse::Searched { .. })
            | (FsRequest::TakeAdmission, FsResponse::PeerAdmitted { .. })
            | (
                FsRequest::AttachWindow { .. },
                FsResponse::WindowAttached { .. }
            )
            | (
                FsRequest::RegisterSystemView { .. }
                    | FsRequest::RegisterPrivateView { .. }
                    | FsRequest::RegisterPrivateView { .. }
                    | FsRequest::ApplyView { .. }
                    | FsRequest::BindView { .. }
                    | FsRequest::Close { .. }
                    | FsRequest::Rename { .. }
                    | FsRequest::Relocate { .. }
                    | FsRequest::Link { .. }
                    | FsRequest::Unlink { .. }
                    | FsRequest::DetachEdge { .. }
                    | FsRequest::Revoke { .. }
                    | FsRequest::Sync { .. }
                    | FsRequest::DeferCommits { .. }
                    | FsRequest::SetSystemSeed { .. }
                    | FsRequest::ReleaseVersion { .. }
                    | FsRequest::AdmitPeer { .. }
                    | FsRequest::RetirePeer { .. }
                    | FsRequest::ConfirmRetired { .. }
                    | FsRequest::ClosePeer,
                FsResponse::Done
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::IntoBytes as _;

    /// The control verbs that carry no object must survive the round trip.
    ///
    /// `DeferCommits` is the reason this module exists: the service's own
    /// tests call `dispatch` with an `FsRequest` value and never cross the
    /// codec, so a request that encodes but does not decode passes every
    /// host test and then hangs the caller on the metal — the serve loop
    /// drops an unparseable record without replying, which the client
    /// cannot tell from a service that simply never answers.
    /// `Create` is the one request whose reserved byte carries data, and
    /// only there: every other verb still refuses a non-zero byte.
    #[test]
    fn create_carries_a_subtype_and_every_other_verb_keeps_reserved0_zero() {
        let stamped = FsRequest::Create {
            parent: crate::Handle::from_bits(1),
            name: crate::Name::new("tool").unwrap(),
            kind: crate::ObjectKind::File,
            subtype: crate::ObjectSubtype::EXECUTABLE_IMAGE,
        };
        let wire = encode_req(&stamped);
        assert_eq!(wire.reserved0, 1);
        assert_eq!(decode_req(&wire), Some(stamped));

        let mut other = encode_req(&FsRequest::Stat {
            handle: crate::Handle::from_bits(1),
        });
        other.reserved0 = 1;
        assert_eq!(
            decode_req(&other),
            None,
            "a subtype on a non-create verb is a foreign record"
        );
    }

    /// The replies that describe an object carry its subtype in the low byte
    /// of the reply's reserved word; nothing else may.
    #[test]
    fn opened_and_stat_replies_round_trip_the_subtype() {
        let open = FsRequest::OpenRoot {
            rights: crate::Rights::READ,
        };
        let opened = FsResponse::Opened {
            handle: crate::Handle::from_bits(2),
            stable_object: crate::StableObjectId::new(41).unwrap(),
            kind: crate::ObjectKind::File,
            subtype: crate::ObjectSubtype::EXECUTABLE_IMAGE,
            len: 4,
            revision: 1,
        };
        let wire = encode_rep(&opened, 0);
        assert_eq!(wire.reserved0.get(), 1);
        assert_eq!(decode_rep_for(&open, &wire), Some((opened, 0)));

        let stat = FsRequest::Stat {
            handle: crate::Handle::from_bits(2),
        };
        let described = FsResponse::Stat {
            object: crate::ObjectId::from_bits(4),
            stable_object: crate::StableObjectId::new(42).unwrap(),
            kind: crate::ObjectKind::File,
            subtype: crate::ObjectSubtype::from_bits(0x7f),
            len: 9,
            revision: 3,
            links: 1,
            grants: 0,
        };
        let wire = encode_rep(&described, 0);
        assert_eq!(decode_rep_for(&stat, &wire), Some((described, 0)));

        // Only one byte is data; the rest of the word stays reserved.
        let mut wide = wire;
        wide.reserved0 = U32::new(0x100);
        assert_eq!(decode_rep_for(&stat, &wide), None);

        // An attach answers with the advertised transfer limit, and a
        // service that says nothing decodes as zero: the floor applies.
        let attach = FsRequest::AttachWindow {
            slot: 3,
            bytes: 0x1_0000,
        };
        let attached = encode_rep(
            &FsResponse::WindowAttached {
                max_transfer: 0x1_0000,
            },
            0,
        );
        assert_eq!(
            decode_rep_for(&attach, &attached),
            Some((
                FsResponse::WindowAttached {
                    max_transfer: 0x1_0000
                },
                0
            ))
        );
        assert_eq!(
            decode_rep_for(&attach, &encode_rep(&FsResponse::Done, 0)),
            Some((FsResponse::WindowAttached { max_transfer: 0 }, 0))
        );

        // A reply that does not describe an object may not carry one.
        let done = encode_rep(&FsResponse::Done, 0);
        let mut stamped = done;
        stamped.reserved0 = U32::new(1);
        let close = FsRequest::Close {
            handle: crate::Handle::from_bits(2),
        };
        assert_eq!(decode_rep_for(&close, &done), Some((FsResponse::Done, 0)));
        assert_eq!(decode_rep_for(&close, &stamped), None);
    }

    /// A refusal carries a status and nothing else — a subtype on it is not
    /// a reply this service wrote.
    #[test]
    fn a_refusal_with_a_subtype_is_not_ours() {
        let stat = FsRequest::Stat {
            handle: crate::Handle::from_bits(2),
        };
        let mut refused = encode_rep(&FsResponse::Refused(FsError::NotFound), 0);
        assert!(decode_rep_for(&stat, &refused).is_some());
        refused.reserved0 = U32::new(1);
        assert_eq!(decode_rep_for(&stat, &refused), None);
    }

    #[test]
    fn control_verbs_round_trip() {
        for req in [
            FsRequest::DeferCommits { hold: true },
            FsRequest::DeferCommits { hold: false },
            FsRequest::SetSystemSeed { active: true },
            FsRequest::SetSystemSeed { active: false },
        ] {
            let wire = encode_req(&req);
            let back = decode_req(&wire).expect("decodes");
            assert_eq!(
                encode_req(&back).as_bytes(),
                wire.as_bytes(),
                "{req:?} did not survive the round trip"
            );
        }
    }

    /// Every refusal must survive the status codec, for the reason stated
    /// above: a value that encodes but does not decode passes every host test
    /// that calls `dispatch` directly, and then reaches the metal as a status
    /// the decoder's range check rejects — the caller sees no answer at all
    /// rather than the refusal the service actually sent.
    ///
    /// The `witness` match is exhaustive on purpose: a new [`FsError`] variant
    /// stops compiling here until it is named, which is the prompt to add it
    /// to the list below *and* to give it a status code.
    #[test]
    fn every_refusal_survives_the_status_codec() {
        fn witness(err: FsError) -> FsError {
            match err {
                FsError::NotFound
                | FsError::AlreadyExists
                | FsError::NotDirectory
                | FsError::IsDirectory
                | FsError::InvalidName
                | FsError::InvalidHandle
                | FsError::AccessDenied
                | FsError::AccessLost
                | FsError::Conflict
                | FsError::Cycle
                | FsError::NotEmpty
                | FsError::OutOfObjects
                | FsError::OutOfEdges
                | FsError::OutOfHandles
                | FsError::OutOfStorage
                | FsError::OutOfPlacements
                | FsError::OutOfVersionLeases
                | FsError::OutOfIndexSlots
                | FsError::Range
                | FsError::BudgetExceeded
                | FsError::IndexIncomplete
                | FsError::Unsupported
                | FsError::NoWindow
                | FsError::Busy
                | FsError::ProviderFailure(_) => err,
            }
        }

        for err in [
            FsError::NotFound,
            FsError::AlreadyExists,
            FsError::NotDirectory,
            FsError::IsDirectory,
            FsError::InvalidName,
            FsError::InvalidHandle,
            FsError::AccessDenied,
            FsError::AccessLost,
            FsError::Conflict,
            FsError::Cycle,
            FsError::NotEmpty,
            FsError::OutOfObjects,
            FsError::OutOfEdges,
            FsError::OutOfHandles,
            FsError::OutOfStorage,
            FsError::OutOfPlacements,
            FsError::OutOfVersionLeases,
            FsError::OutOfIndexSlots,
            FsError::Range,
            FsError::BudgetExceeded,
            FsError::IndexIncomplete,
            FsError::Unsupported,
            FsError::NoWindow,
            FsError::Busy,
        ] {
            let err = witness(err);
            let status = status_of(err);
            assert!(
                status != 0 && status <= FsRepWire::STATUS_MAX,
                "{err:?} encodes to {status}, which the decoder's range check rejects"
            );
            assert_eq!(
                error_of(status),
                Some(err),
                "{err:?} did not survive the status round trip"
            );
        }

        // The one deliberately lossy mapping, asserted rather than assumed:
        // the provider's own cause does not cross the boundary, and inventing
        // one back would be a lie (see `error_of`).
        let status = status_of(FsError::ProviderFailure(ProviderError::Transport));
        assert_eq!(status, FsRepWire::STATUS_PROVIDER_FAILURE);
        assert_eq!(
            error_of(status),
            Some(FsError::ProviderFailure(ProviderError::Media))
        );
    }

    /// A full extent table must not reach a client as "the volume is full".
    #[test]
    fn a_full_extent_table_keeps_its_own_status() {
        assert_ne!(
            status_of(FsError::OutOfPlacements),
            status_of(FsError::OutOfStorage)
        );
    }

    #[test]
    fn relationship_search_text_is_not_parsed_as_a_namespace_name() {
        let req = FsRequest::Search {
            handle: crate::Handle::from_bits(1),
            query: SearchQuery {
                content: Some(crate::SearchContent {
                    plane: crate::SearchPlane::Relationship,
                    term: crate::SearchTerm::new("object://target").unwrap(),
                }),
                freshness: crate::FreshnessPolicy::CurrentOnly,
                max_work: 123,
                ..SearchQuery::ALL
            },
            cursor: None,
            window_offset: 0,
        };
        let wire = encode_req(&req);
        let back = decode_req(&wire).expect("search request decodes");
        assert_eq!(encode_req(&back).as_bytes(), wire.as_bytes());
    }

    #[test]
    fn every_search_freshness_reply_survives_the_wire() {
        for freshness in [
            crate::FreshnessPolicy::StructuralOnly,
            crate::FreshnessPolicy::CurrentOnly,
            crate::FreshnessPolicy::BoundedStale,
        ] {
            let request = FsRequest::Search {
                handle: crate::Handle::from_bits(1),
                query: SearchQuery {
                    content: (freshness != crate::FreshnessPolicy::StructuralOnly).then_some(
                        crate::SearchContent {
                            plane: crate::SearchPlane::FullText,
                            term: crate::SearchTerm::new("answer").unwrap(),
                        },
                    ),
                    freshness,
                    max_index_lag: if freshness == crate::FreshnessPolicy::BoundedStale {
                        3
                    } else {
                        0
                    },
                    ..SearchQuery::ALL
                },
                cursor: None,
                window_offset: 0,
            };
            let response = FsResponse::Searched {
                hits: 1,
                observation: 7,
                snapshot: 11,
                index_generation: if freshness == crate::FreshnessPolicy::StructuralOnly {
                    0
                } else {
                    13
                },
                freshness,
                next: None,
                scanned: 2,
                work: 3,
            };
            let wire = encode_rep(&response, 0);
            let (decoded, grants) = decode_rep_for(&request, &wire)
                .unwrap_or_else(|| panic!("{freshness:?} reply must decode"));
            assert_eq!(decoded, response);
            assert_eq!(grants, 0);
        }
    }
}
