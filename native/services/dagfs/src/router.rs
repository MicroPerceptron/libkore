//! The two-volume filesystem router.
//!
//! A [`Handle`], [`GrantId`], or [`VersionLease`] is a generational slot name
//! inside exactly one [`FsService`](crate::FsService).  Once the durable row
//! owns both the conservative user volume and the rebuildable system volume,
//! those otherwise identical words need one bit of routing identity.  This
//! module owns that bit and nothing below it learns about the composition:
//! both services receive their original untagged words and run the same
//! authority checks as a one-volume service.
//!
//! The tag is the high bit of `SlotId::index`.  Every filesystem slot arena is
//! statically bounded far below `2^31`; preserving all 32 generation bits is
//! important because generation retirement is what makes stale authority fail
//! closed.  Flipping the tag is not an authority operation: the selected
//! engine still checks the pair-derived session against the named record.

use kore_abi::svc::{FsGrantEntryWire, FsSystemStatusWire};
use zerocopy::FromBytes;

use crate::{FsError, FsRequest, FsResponse, GrantId, Handle, VersionLease, window::DataWindow};

/// High bit of the packed `SlotId::index` word.
const SYSTEM_TAG: u64 = 1 << 63;

// `SlotId` packs its 32-bit index into the high word. Reserve that word's
// high bit for composition, and make a future arena expansion fail at build
// time instead of silently aliasing a native slot into the system engine.
const NATIVE_INDEX_LIMIT: usize = 1 << 31;
const _: () = assert!(crate::authority::MAX_HANDLES_TOTAL < NATIVE_INDEX_LIMIT);
const _: () = assert!(crate::bounds::MAX_AUTHORITY_EDGES < NATIVE_INDEX_LIMIT);
const _: () = assert!(crate::bounds::MAX_LENT_VERSIONS < NATIVE_INDEX_LIMIT);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Store {
    User,
    System,
}

/// A request after every opaque id has been stripped back to its native
/// engine-local form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoutedRequest {
    One(Store, FsRequest),
    /// Attachment and commit-deferral state is pair-local in each frontend.
    Both(FsRequest),
    /// Both authority tables contribute entries to one client window.
    ListGrants {
        window_offset: u32,
    },
    /// Both engines contribute one aggregate status record.
    SystemStatus {
        window_offset: u32,
    },
    /// Admin-only lifecycle testimony owned by the dual-volume composition.
    SystemSeed {
        active: bool,
    },
}

#[inline]
const fn split_bits(bits: u64) -> (Store, u64) {
    if bits & SYSTEM_TAG == 0 {
        (Store::User, bits)
    } else {
        (Store::System, bits & !SYSTEM_TAG)
    }
}

#[inline]
const fn join_bits(store: Store, bits: u64) -> u64 {
    debug_assert!(bits & SYSTEM_TAG == 0);
    match store {
        Store::User => bits,
        Store::System => bits | SYSTEM_TAG,
    }
}

#[inline]
const fn split_handle(handle: Handle) -> (Store, Handle) {
    let (store, bits) = split_bits(handle.to_bits());
    (store, Handle::from_bits(bits))
}

#[inline]
const fn split_grant(grant: GrantId) -> (Store, GrantId) {
    let (store, bits) = split_bits(grant.to_bits());
    (store, GrantId::from_bits(bits))
}

#[inline]
const fn split_lease(lease: VersionLease) -> (Store, VersionLease) {
    let (store, bits) = split_bits(lease.to_bits());
    (store, VersionLease::from_bits(bits))
}

fn same_store<T>(left: (Store, T), right: (Store, T)) -> Result<(Store, T, T), FsError> {
    if left.0 != right.0 {
        return Err(FsError::AccessDenied);
    }
    Ok((left.0, left.1, right.1))
}

/// Select one engine without trusting an engine name supplied separately
/// from the authority word.  A system-tagged request on the compatibility
/// one-volume posture is stale/forged authority, never an alias back to user.
pub(crate) fn route(request: FsRequest, has_system: bool) -> Result<RoutedRequest, FsError> {
    use FsRequest::*;

    let routed = match request {
        AttachWindow { .. } | DeferCommits { .. } => RoutedRequest::Both(request),
        ListGrants { window_offset } => RoutedRequest::ListGrants { window_offset },
        SystemStatus { window_offset } => RoutedRequest::SystemStatus { window_offset },
        SetSystemSeed { active } => RoutedRequest::SystemSeed { active },
        OpenRoot { .. }
        | RegisterSystemView { .. }
        | RegisterPrivateView { .. }
        | ApplyView { .. }
        | AdmitPeer { .. }
        | TakeAdmission
        | RetirePeer { .. }
        | ConfirmRetired { .. }
        | ClosePeer => RoutedRequest::One(Store::User, request),
        OpenGrant { grant, rights } => {
            let (store, grant) = split_grant(grant);
            RoutedRequest::One(store, OpenGrant { grant, rights })
        }
        OpenAt {
            parent,
            name,
            rights,
        } => {
            let (store, parent) = split_handle(parent);
            RoutedRequest::One(
                store,
                OpenAt {
                    parent,
                    name,
                    rights,
                },
            )
        }
        Close { handle } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(store, Close { handle })
        }
        Stat { handle } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(store, Stat { handle })
        }
        List {
            handle,
            cursor,
            window_offset,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                List {
                    handle,
                    cursor,
                    window_offset,
                },
            )
        }
        Create {
            parent,
            name,
            kind,
            subtype,
        } => {
            let (store, parent) = split_handle(parent);
            RoutedRequest::One(
                store,
                Create {
                    parent,
                    name,
                    kind,
                    subtype,
                },
            )
        }
        Read {
            handle,
            offset,
            window_offset,
            length,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                Read {
                    handle,
                    offset,
                    window_offset,
                    length,
                },
            )
        }
        WriteAt {
            handle,
            offset,
            window_offset,
            length,
            expected,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                WriteAt {
                    handle,
                    offset,
                    window_offset,
                    length,
                    expected,
                },
            )
        }
        Append {
            handle,
            window_offset,
            length,
            expected,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                Append {
                    handle,
                    window_offset,
                    length,
                    expected,
                },
            )
        }
        Resize {
            handle,
            length,
            expected,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                Resize {
                    handle,
                    length,
                    expected,
                },
            )
        }
        Rename { parent, old, new } => {
            let (store, parent) = split_handle(parent);
            RoutedRequest::One(store, Rename { parent, old, new })
        }
        Relocate {
            src_parent,
            name,
            dst_parent,
            new_name,
        } => {
            let (store, src_parent, dst_parent) =
                same_store(split_handle(src_parent), split_handle(dst_parent))?;
            RoutedRequest::One(
                store,
                Relocate {
                    src_parent,
                    name,
                    dst_parent,
                    new_name,
                },
            )
        }
        Link {
            parent,
            name,
            child,
        } => {
            let (store, parent, child) = same_store(split_handle(parent), split_handle(child))?;
            RoutedRequest::One(
                store,
                Link {
                    parent,
                    name,
                    child,
                },
            )
        }
        Unlink { parent, name } => {
            let (store, parent) = split_handle(parent);
            RoutedRequest::One(store, Unlink { parent, name })
        }
        DetachEdge { parent, name } => {
            let (store, parent) = split_handle(parent);
            RoutedRequest::One(store, DetachEdge { parent, name })
        }
        Grant {
            handle,
            destination,
            rights,
            display_name,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                Grant {
                    handle,
                    destination,
                    rights,
                    display_name,
                },
            )
        }
        GrantScoped {
            handle,
            destination,
            rights,
            quota,
            display_name,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                GrantScoped {
                    handle,
                    destination,
                    rights,
                    quota,
                    display_name,
                },
            )
        }
        GrantView {
            handle,
            rights,
            display_name,
            announced,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                GrantView {
                    handle,
                    rights,
                    display_name,
                    announced,
                },
            )
        }
        Revoke { grant } => {
            let (store, grant) = split_grant(grant);
            RoutedRequest::One(store, Revoke { grant })
        }
        Sync { handle } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(store, Sync { handle })
        }
        LendVersion {
            handle,
            window_offset,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                LendVersion {
                    handle,
                    window_offset,
                },
            )
        }
        LendExecutable {
            handle,
            window_offset,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                LendExecutable {
                    handle,
                    window_offset,
                },
            )
        }
        ReadVersion {
            lease,
            offset,
            window_offset,
            length,
        } => {
            let (store, lease) = split_lease(lease);
            RoutedRequest::One(
                store,
                ReadVersion {
                    lease,
                    offset,
                    window_offset,
                    length,
                },
            )
        }
        CommitDerived {
            lease,
            window_offset,
            length,
            output_window_bytes,
        } => {
            let (store, lease) = split_lease(lease);
            RoutedRequest::One(
                store,
                CommitDerived {
                    lease,
                    window_offset,
                    length,
                    output_window_bytes,
                },
            )
        }
        ReleaseVersion { lease } => {
            let (store, lease) = split_lease(lease);
            RoutedRequest::One(store, ReleaseVersion { lease })
        }
        DerivedInfo { handle, parser } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(store, DerivedInfo { handle, parser })
        }
        Search {
            handle,
            query,
            cursor,
            window_offset,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                Search {
                    handle,
                    query,
                    cursor,
                    window_offset,
                },
            )
        }
        BindView {
            handle,
            rights,
            mode,
            announced,
        } => {
            let (store, handle) = split_handle(handle);
            RoutedRequest::One(
                store,
                BindView {
                    handle,
                    rights,
                    mode,
                    announced,
                },
            )
        }
    };

    if matches!(routed, RoutedRequest::One(Store::System, _)) && !has_system {
        Err(FsError::InvalidHandle)
    } else {
        Ok(routed)
    }
}

/// Restore the selected engine on every opaque authority word a response
/// exposes.  Object ids remain untagged: they are correlation data and no
/// request uses one to select authority or an engine.
pub(crate) fn tag_response(store: Store, response: FsResponse) -> FsResponse {
    match response {
        FsResponse::Opened {
            handle,
            stable_object,
            kind,
            subtype,
            len,
            revision,
        } => FsResponse::Opened {
            handle: Handle::from_bits(join_bits(store, handle.to_bits())),
            stable_object,
            kind,
            subtype,
            len,
            revision,
        },
        FsResponse::Granted { grant } => FsResponse::Granted {
            grant: GrantId::from_bits(join_bits(store, grant.to_bits())),
        },
        FsResponse::VersionLent { lease } => FsResponse::VersionLent {
            lease: VersionLease::from_bits(join_bits(store, lease.to_bits())),
        },
        other => other,
    }
}

/// Add `store` to the grant ids already written into a merged listing.
pub(crate) fn tag_grant_entries(
    window: &mut DataWindow<'_>,
    window_offset: u32,
    first: usize,
    count: usize,
    store: Store,
) -> Result<(), FsError> {
    for index in first..first.checked_add(count).ok_or(FsError::Range)? {
        let step = index
            .checked_mul(FsGrantEntryWire::BYTES)
            .ok_or(FsError::Range)?;
        let at = usize::try_from(window_offset)
            .map_err(|_| FsError::Range)?
            .checked_add(step)
            .ok_or(FsError::Range)?;
        let at = u32::try_from(at).map_err(|_| FsError::Range)?;
        let mut entry =
            FsGrantEntryWire::read_from_bytes(window.source(at, FsGrantEntryWire::BYTES as u32)?)
                .map_err(|_| FsError::Range)?;
        entry.grant = join_bits(store, entry.grant.get()).into();
        window.put_grant_entry(window_offset, index, &entry)?;
    }
    Ok(())
}

pub(crate) fn grant_append_offset(window_offset: u32, entries: u32) -> Result<u32, FsError> {
    window_offset
        .checked_add(
            entries
                .checked_mul(FsGrantEntryWire::BYTES as u32)
                .ok_or(FsError::Range)?,
        )
        .ok_or(FsError::Range)
}

/// Read the status record just produced by one engine before the sibling
/// overwrites the same window span.
pub(crate) fn read_status(
    window: &DataWindow<'_>,
    window_offset: u32,
) -> Result<FsSystemStatusWire, FsError> {
    FsSystemStatusWire::read_from_bytes(
        window.source(window_offset, FsSystemStatusWire::BYTES as u32)?,
    )
    .map_err(|_| FsError::Range)
}

/// Fold the two engine snapshots into the service's one frozen status ABI.
/// Physical bytes describe their shared controller namespace and therefore
/// take the maximum; volume/occupancy and per-engine work counters sum.
pub(crate) fn fold_status(
    user: FsSystemStatusWire,
    system: FsSystemStatusWire,
) -> FsSystemStatusWire {
    use kore_abi::monitor::FilesystemStorageSnapshot;

    let add16 = |a: u16, b: u16| a.saturating_add(b).into();
    let add32 = |a: u32, b: u32| a.saturating_add(b).into();
    let add64 = |a: u64, b: u64| a.saturating_add(b).into();
    let user_storage = user.storage;
    let system_storage = system.storage;
    let storage = match (user_storage.available(), system_storage.available()) {
        (false, false) => FilesystemStorageSnapshot::EMPTY,
        (true, false) => user_storage,
        (false, true) => system_storage,
        (true, true) => {
            let block = user_storage.logical_block_bytes();
            if block == system_storage.logical_block_bytes() {
                let max_transfer = user_storage
                    .max_transfer_bytes()
                    .min(system_storage.max_transfer_bytes());
                let mut combined = FilesystemStorageSnapshot::new(
                    block,
                    user_storage
                        .physical_bytes()
                        .max(system_storage.physical_bytes()),
                    user_storage
                        .volume_bytes()
                        .saturating_add(system_storage.volume_bytes()),
                    user_storage
                        .occupied_bytes()
                        .saturating_add(system_storage.occupied_bytes()),
                    max_transfer,
                    user_storage.read_only() || system_storage.read_only(),
                    user_storage.flush() && system_storage.flush(),
                    user_storage.discard() && system_storage.discard(),
                );
                // One physical namespace: its layout facts are the same on
                // both sides, so either copy is the truth.
                combined = combined.with_layout_of(user_storage);
                let health = if user_storage.health().available() {
                    user_storage.health()
                } else {
                    system_storage.health()
                };
                if health.available() {
                    combined = combined.with_health(health);
                }
                combined
            } else {
                // One ABI record cannot truthfully encode two logical block
                // sizes. Keep the larger mounted namespace visible rather
                // than inventing a unit conversion for transfer granularity.
                if user_storage.volume_bytes() >= system_storage.volume_bytes() {
                    user_storage
                } else {
                    system_storage
                }
            }
        }
    };
    let note = if user.note.get() != 0 {
        user.note
    } else {
        system.note
    };
    FsSystemStatusWire {
        sessions: add16(user.sessions.get(), system.sessions.get()),
        peers: add16(user.peers.get(), system.peers.get()),
        views_private: add16(user.views_private.get(), system.views_private.get()),
        views_pinned: add16(user.views_pinned.get(), system.views_pinned.get()),
        windows: add16(user.windows.get(), system.windows.get()),
        flags: (user.flags.get() | system.flags.get()).into(),
        tree_objects: add32(user.tree_objects.get(), system.tree_objects.get()),
        tree_edges: add32(user.tree_edges.get(), system.tree_edges.get()),
        user_lifecycle: if user.user_lifecycle.get() != 0 {
            user.user_lifecycle
        } else {
            system.user_lifecycle
        },
        system_lifecycle: if system.system_lifecycle.get() != 0 {
            system.system_lifecycle
        } else {
            user.system_lifecycle
        },
        tree_bytes: add64(user.tree_bytes.get(), system.tree_bytes.get()),
        storage,
        note,
        commits: add64(user.commits.get(), system.commits.get()),
        reads: add64(user.reads.get(), system.reads.get()),
        writes: add64(user.writes.get(), system.writes.get()),
        flushes: add64(user.flushes.get(), system.flushes.get()),
        discards: add64(user.discards.get(), system.discards.get()),
        copied_bytes: add64(user.copied_bytes.get(), system.copied_bytes.get()),
        rollbacks: add64(user.rollbacks.get(), system.rollbacks.get()),
        late_completions: add64(user.late_completions.get(), system.late_completions.get()),
        orphan_reclaims: add64(user.orphan_reclaims.get(), system.orphan_reclaims.get()),
        reclaim_failures: add64(user.reclaim_failures.get(), system.reclaim_failures.get()),
        expired_waits: add64(user.expired_waits.get(), system.expired_waits.get()),
        slow_waits: add64(user.slow_waits.get(), system.slow_waits.get()),
        leaked_placements: add64(user.leaked_placements.get(), system.leaked_placements.get()),
        abandoned_reclaims: add64(
            user.abandoned_reclaims.get(),
            system.abandoned_reclaims.get(),
        ),
        readahead_hits: add64(user.readahead_hits.get(), system.readahead_hits.get()),
        readahead_reads: add64(user.readahead_reads.get(), system.readahead_reads.get()),
        submissions: add64(user.submissions.get(), system.submissions.get()),
        completions: add64(user.completions.get(), system.completions.get()),
        backpressure: add64(user.backpressure.get(), system.backpressure.get()),
        poll_calls: add64(user.poll_calls.get(), system.poll_calls.get()),
        device_waits: add64(user.device_waits.get(), system.device_waits.get()),
        route_parks: add64(user.route_parks.get(), system.route_parks.get()),
        route_fires: add64(user.route_fires.get(), system.route_fires.get()),
        deadline_waits: add64(user.deadline_waits.get(), system.deadline_waits.get()),
        spin_waits: add64(user.spin_waits.get(), system.spin_waits.get()),
        monitor_sleeps: add64(user.monitor_sleeps.get(), system.monitor_sleeps.get()),
        monitor_turns: add64(user.monitor_turns.get(), system.monitor_turns.get()),
        // One controller queue serves both regions: depth takes the max,
        // the histograms add.
        latency: user.latency.folded_with(system.latency),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FsEndowment, Name, ObjectKind, RamService, Rights, SearchCursor, SearchQuery, SessionId,
    };
    use kore_rt::service::SystemService;

    fn system_handle(bits: u64) -> Handle {
        Handle::from_bits(bits | SYSTEM_TAG)
    }

    #[test]
    fn roots_default_to_user_and_tagged_grants_select_system() {
        assert!(matches!(
            route(
                FsRequest::OpenRoot {
                    rights: Rights::READ
                },
                true
            ),
            Ok(RoutedRequest::One(Store::User, FsRequest::OpenRoot { .. }))
        ));
        let request = FsRequest::OpenGrant {
            grant: GrantId::from_bits(SYSTEM_TAG | 9),
            rights: Rights::READ,
        };
        assert_eq!(
            route(request, true),
            Ok(RoutedRequest::One(
                Store::System,
                FsRequest::OpenGrant {
                    grant: GrantId::from_bits(9),
                    rights: Rights::READ,
                }
            ))
        );
        assert_eq!(route(request, false), Err(FsError::InvalidHandle));
        assert_eq!(
            route(FsRequest::SetSystemSeed { active: true }, true),
            Ok(RoutedRequest::SystemSeed { active: true })
        );
    }

    #[test]
    fn every_returned_authority_word_keeps_its_engine() {
        let opened = tag_response(
            Store::System,
            FsResponse::Opened {
                handle: Handle::from_bits(7),
                stable_object: crate::StableObjectId::new(11).unwrap(),
                kind: ObjectKind::File,
                subtype: crate::ObjectSubtype::PLAIN,
                len: 2,
                revision: 3,
            },
        );
        assert!(matches!(
            opened,
            FsResponse::Opened { handle, .. } if handle == system_handle(7)
        ));
        assert_eq!(
            tag_response(
                Store::System,
                FsResponse::Granted {
                    grant: GrantId::from_bits(8)
                }
            ),
            FsResponse::Granted {
                grant: GrantId::from_bits(SYSTEM_TAG | 8)
            }
        );
        assert_eq!(
            tag_response(
                Store::System,
                FsResponse::VersionLent {
                    lease: VersionLease::from_bits(10)
                }
            ),
            FsResponse::VersionLent {
                lease: VersionLease::from_bits(SYSTEM_TAG | 10)
            }
        );
    }

    #[test]
    fn cross_engine_graph_operations_are_refused_before_dispatch() {
        let name = Name::new("x").unwrap();
        assert_eq!(
            route(
                FsRequest::Link {
                    parent: Handle::from_bits(1),
                    name,
                    child: system_handle(2),
                },
                true,
            ),
            Err(FsError::AccessDenied)
        );
        assert_eq!(
            route(
                FsRequest::Relocate {
                    src_parent: system_handle(1),
                    name,
                    dst_parent: Handle::from_bits(2),
                    new_name: Name::new("y").unwrap(),
                },
                true,
            ),
            Err(FsError::AccessDenied)
        );
    }

    #[test]
    fn the_system_tag_routes_handles_and_leases_but_not_search_cursors() {
        let request = FsRequest::Search {
            handle: system_handle(5),
            query: SearchQuery::ALL,
            cursor: Some(SearchCursor {
                observation: SYSTEM_TAG,
                position: 4,
            }),
            window_offset: 0,
        };
        let Ok(RoutedRequest::One(Store::System, FsRequest::Search { handle, cursor, .. })) =
            route(request, true)
        else {
            panic!("system search route");
        };
        assert_eq!(handle, Handle::from_bits(5));
        assert_eq!(cursor.unwrap().observation, SYSTEM_TAG);

        let request = FsRequest::ReadVersion {
            lease: VersionLease::from_bits(SYSTEM_TAG | 6),
            offset: 0,
            window_offset: 0,
            length: 1,
        };
        assert!(matches!(
            route(request, true),
            Ok(RoutedRequest::One(
                Store::System,
                FsRequest::ReadVersion { lease, .. }
            )) if lease == VersionLease::from_bits(6)
        ));
    }

    /// Adversarial form of the containment argument: a forged engine bit is
    /// only a route choice. The system engine receives the same raw slot word
    /// and must still validate the caller's pair-derived session. This test
    /// exercises the router half; authority's cross-session forged-handle
    /// tests exercise the second half at the table seam.
    #[test]
    fn flipping_the_tag_changes_only_the_selected_engine() {
        let raw = Handle::from_bits((3u64 << 32) | 7);
        let forged = Handle::from_bits(raw.to_bits() | SYSTEM_TAG);
        assert_eq!(split_handle(raw), (Store::User, raw));
        assert_eq!(split_handle(forged), (Store::System, raw));
    }

    #[test]
    fn a_forged_tag_still_cannot_cross_the_selected_engines_session_boundary() {
        let mut system = <RamService as SystemService>::bind(FsEndowment::UNSTAMPED);
        let owner = SessionId::new(0);
        let forger = SessionId::new(1);
        assert!(system.ensure_session(forger));
        let owned = system.engine().open_root(owner, Rights::READ).unwrap();
        let tagged = Handle::from_bits(join_bits(Store::System, owned.handle.to_bits()));
        let RoutedRequest::One(Store::System, FsRequest::Stat { handle }) =
            route(FsRequest::Stat { handle: tagged }, true).unwrap()
        else {
            panic!("tagged stat routes to system");
        };
        assert_eq!(
            system.dispatch(forger, &FsRequest::Stat { handle }, None),
            FsResponse::Refused(FsError::InvalidHandle)
        );
    }

    #[test]
    fn merged_grants_are_restamped_in_place() {
        let mut bytes = [0u8; FsGrantEntryWire::BYTES * 2];
        let mut window = DataWindow::new(&mut bytes);
        let first = FsGrantEntryWire {
            name_len: 0,
            rights: Rights::READ.bits(),
            kind: FsGrantEntryWire::KIND_GRANT,
            reserved0: 0,
            reserved1: 0.into(),
            grant: 3u64.into(),
            name: [0; kore_abi::svc::FS_NAME_BYTES],
        };
        window.put_grant_entry(0, 0, &first).unwrap();
        window.put_grant_entry(0, 1, &first).unwrap();
        tag_grant_entries(&mut window, 0, 1, 1, Store::System).unwrap();
        let user = FsGrantEntryWire::read_from_bytes(
            window.source(0, FsGrantEntryWire::BYTES as u32).unwrap(),
        )
        .unwrap();
        let system = FsGrantEntryWire::read_from_bytes(
            window
                .source(
                    FsGrantEntryWire::BYTES as u32,
                    FsGrantEntryWire::BYTES as u32,
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(user.grant.get(), 3);
        assert_eq!(system.grant.get(), SYSTEM_TAG | 3);
    }

    #[test]
    fn status_folding_sums_regions_but_not_the_shared_physical_namespace() {
        let zero = [0u8; FsSystemStatusWire::BYTES];
        let mut user = FsSystemStatusWire::read_from_bytes(&zero).unwrap();
        let mut system = user;
        user.flags = FsSystemStatusWire::FLAG_DURABLE.into();
        system.flags = FsSystemStatusWire::FLAG_DURABLE.into();
        user.sessions = 2u16.into();
        system.sessions = 3u16.into();
        user.tree_bytes = 7u64.into();
        system.tree_bytes = 11u64.into();
        user.user_lifecycle = FsSystemStatusWire::LIFECYCLE_CURRENT.into();
        user.system_lifecycle = FsSystemStatusWire::LIFECYCLE_REBUILT.into();
        user.route_parks = 9u64.into();
        let mut hot = [0u32; kore_abi::monitor::FilesystemLatencySnapshot::BUCKETS];
        hot[1] = 5;
        user.latency =
            kore_abi::monitor::FilesystemLatencySnapshot::new(4, 5, 15_000, 3_500, 6, hot);
        let mut cold = [0u32; kore_abi::monitor::FilesystemLatencySnapshot::BUCKETS];
        cold[5] = 1;
        system.latency =
            kore_abi::monitor::FilesystemLatencySnapshot::new(2, 1, 100_000, 100_000, 0, cold);
        user.storage = kore_abi::monitor::FilesystemStorageSnapshot::new(
            512, 1_000_000, 400_000, 100_000, 16_384, false, true, true,
        );
        system.storage = kore_abi::monitor::FilesystemStorageSnapshot::new(
            512, 1_000_000, 200_000, 50_000, 8_192, true, true, false,
        );
        let folded = fold_status(user, system);
        assert_eq!(folded.sessions.get(), 5);
        assert_eq!(folded.tree_bytes.get(), 18);
        assert_eq!(
            folded.user_lifecycle.get(),
            FsSystemStatusWire::LIFECYCLE_CURRENT
        );
        assert_eq!(
            folded.system_lifecycle.get(),
            FsSystemStatusWire::LIFECYCLE_REBUILT
        );
        assert_eq!(folded.route_parks.get(), 9);
        assert_eq!(folded.latency.max_outstanding(), 4);
        assert_eq!(folded.latency.samples(), 6);
        assert_eq!(folded.latency.max_ns(), 100_000);
        assert_eq!((folded.latency.bucket(1), folded.latency.bucket(5)), (5, 1));
        assert_eq!(folded.storage.physical_bytes(), 1_000_000);
        assert_eq!(folded.storage.volume_bytes(), 600_000);
        assert_eq!(folded.storage.occupied_bytes(), 150_000);
        assert_eq!(folded.storage.max_transfer_bytes(), 8_192);
        assert!(folded.storage.read_only());
        assert!(folded.storage.flush());
        assert!(!folded.storage.discard());
        assert!(folded.storage.validate());
    }
}
