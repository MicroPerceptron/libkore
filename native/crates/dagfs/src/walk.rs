//! The client-owned walk: a `/`-separated path becomes one bounded `OpenAt`
//! per component, refused past [`MAX_PATH_DEPTH`] rather than truncated.
//!
//! `OpenAt` resolves exactly one name component: paths are client-side
//! input, and every step extends the opened handle's edge proof by one
//! edge. The *syntax* of the path is [`crate::path`], so every client splits
//! paths identically and the splitting is covered by host tests a `no_main`
//! metal binary cannot host. What is written here is the walk itself, which
//! needs a live pair.

use crate::{
    Fs, FsError, FsRequest, FsResponse, GrantId, Handle, Name, ObjectKind, ObjectSubtype, Refusal,
    Revision, Rights, StableObjectId, Stat, bounds::MAX_PATH_DEPTH, path, read_grant_entry,
};

/// The three masks a verb walks with.
///
/// `READ` is in all of them because resolving a name *is* reading the
/// directory it is resolved in, so a mask without it cannot take a single
/// step. Past that each verb asks for exactly the right its leaf operation
/// needs and no more. The service independently enforces the rights of the
/// selected containment route or named resource grant.
///
/// `union`, not a `|` operator: `Rights` deliberately has no `BitOr`, so
/// that combining authority always reads as the deliberate act it is.
pub const WALK_READ: Rights = Rights::READ;
/// Creating, linking, detaching, moving beneath a directory.
pub const WALK_MUTATE: Rights = Rights::READ.union(Rights::MUTATE_NAMESPACE);
/// Writing a file's bytes.
pub const WALK_WRITE: Rights = Rights::READ.union(Rights::WRITE);

/// One opened object on this face.
///
/// Affine by construction — not `Copy`, not `Clone`, and `#[must_use]` —
/// so the only way to give it back is [`Fs::close`]. The handle-budget
/// discipline is then a property of the type rather than a comment on
/// every verb. What `Opened` reported at open time rides along, so a caller
/// that only needs the kind or the stable identity never spends a `Stat`.
#[must_use = "an open view holds one of the session's handles until Fs::close"]
#[derive(Debug)]
pub struct View {
    handle: Handle,
    stable: StableObjectId,
    kind: ObjectKind,
    subtype: ObjectSubtype,
    len: u64,
    revision: Revision,
}

/// A refused close retains the exact view for retry or owner teardown.
/// Dropping this value does not prove the remote handle was released.
#[must_use = "retain the view until close is acknowledged or the owner is retired"]
#[derive(Debug)]
pub struct CloseFailure {
    pub view: View,
    pub error: Refusal,
}

impl View {
    pub(crate) fn from_opened(response: FsResponse) -> Result<Self, Refusal> {
        match response {
            FsResponse::Opened {
                handle,
                stable_object,
                kind,
                subtype,
                len,
                revision,
            } => Ok(Self {
                handle,
                stable: stable_object,
                kind,
                subtype,
                len,
                revision,
            }),
            _ => Err(Refusal::Pair),
        }
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// Durable correlation identity. Two views with equal stable ids are two
    /// names for one object; the id itself grants nothing.
    pub fn stable(&self) -> StableObjectId {
        self.stable
    }

    pub fn kind(&self) -> ObjectKind {
        self.kind
    }

    pub fn is_directory(&self) -> bool {
        self.kind.is_dir()
    }

    /// What the object contains — data, never authority.
    pub fn subtype(&self) -> ObjectSubtype {
        self.subtype
    }

    /// The visible length at open time.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The committed revision at open time.
    pub fn revision(&self) -> Revision {
        self.revision
    }
}

/// Operations required by the common bounded path walker. Native Fs and
/// service-backed verification use the same lookup and attenuation logic.
#[allow(async_fn_in_trait)]
pub trait WalkClient {
    async fn request(&mut self, request: FsRequest) -> Result<FsResponse, Refusal>;
    async fn named_grant(&mut self, name: &Name) -> Result<Option<GrantId>, Refusal>;

    async fn close(&mut self, view: View) {
        let _ = self
            .request(FsRequest::Close {
                handle: view.handle,
            })
            .await;
    }
}

impl WalkClient for Fs<'_, '_> {
    async fn request(&mut self, request: FsRequest) -> Result<FsResponse, Refusal> {
        self.checked(&request).await
    }

    async fn named_grant(&mut self, name: &Name) -> Result<Option<GrantId>, Refusal> {
        Fs::named_grant(self, name).await
    }
}

/// Resolve one path while keeping containment names ahead of named grants.
/// A read-only namespace root does not limit an independent named grant's
/// rights; it still prevents writes through that root's containment edges.
pub async fn open_path<C: WalkClient>(
    client: &mut C,
    path: &[u8],
    rights: Rights,
) -> Result<View, Refusal> {
    let mut names = heapless::Vec::<Name, MAX_PATH_DEPTH>::new();
    for component in path::components(path) {
        names
            .push(Name::from_bytes(component).map_err(Refusal::Fs)?)
            .map_err(|_| Refusal::Fs(FsError::Range))?;
    }
    let mut skip_first = false;
    let mut current = match client.request(FsRequest::OpenRoot { rights }).await {
        Ok(opened) => View::from_opened(opened)?,
        Err(Refusal::Fs(FsError::AccessDenied)) if !names.is_empty() => {
            let root = View::from_opened(
                client
                    .request(FsRequest::OpenRoot {
                        rights: Rights::READ,
                    })
                    .await?,
            )?;
            // A readable containment entry shadows a same-named grant even
            // when it cannot carry the requested write/share rights. Only an
            // actual NotFound allows the independent named edge lookup.
            let probe = client
                .request(FsRequest::OpenAt {
                    parent: root.handle,
                    name: names[0],
                    rights: Rights::READ,
                })
                .await;
            client.close(root).await;
            match probe {
                Ok(opened) => {
                    client.close(View::from_opened(opened)?).await;
                    return Err(Refusal::Fs(FsError::AccessDenied));
                }
                Err(Refusal::Fs(FsError::NotFound)) => {}
                Err(error) => return Err(error),
            }
            let grant = client
                .named_grant(&names[0])
                .await?
                .ok_or(Refusal::Fs(FsError::NotFound))?;
            skip_first = true;
            View::from_opened(
                client
                    .request(FsRequest::OpenGrant { grant, rights })
                    .await?,
            )?
        }
        Err(error) => return Err(error),
    };
    for (depth, name) in names.into_iter().enumerate().skip(usize::from(skip_first)) {
        let next = client
            .request(FsRequest::OpenAt {
                parent: current.handle,
                name,
                rights,
            })
            .await;
        client.close(current).await;
        current = match next {
            Ok(opened) => View::from_opened(opened)?,
            Err(Refusal::Fs(FsError::NotFound)) if depth == 0 => {
                let grant = client
                    .named_grant(&name)
                    .await?
                    .ok_or(Refusal::Fs(FsError::NotFound))?;
                View::from_opened(
                    client
                        .request(FsRequest::OpenGrant { grant, rights })
                        .await?,
                )?
            }
            Err(error) => return Err(error),
        };
    }
    Ok(current)
}

impl<'w, 't> Fs<'w, 't> {
    /// Walk `path` from the task-visible root and hand back the final view.
    ///
    /// The first component may be either a containment edge beneath the
    /// private root or a named authority edge.  That distinction is entirely
    /// below the client interface: after the first step, traversal is the
    /// same bounded `OpenAt` walk in both cases. An empty path is the root.
    pub async fn open(&mut self, path: &[u8], rights: Rights) -> Result<View, Refusal> {
        open_path(self, path, rights).await
    }

    /// Enter a granted file or subtree root directly.
    pub async fn open_grant(&mut self, grant: GrantId, rights: Rights) -> Result<View, Refusal> {
        View::from_opened(
            self.checked(&FsRequest::OpenGrant { grant, rights })
                .await?,
        )
    }

    /// Close with an observable acknowledgement, retaining ownership on refusal.
    ///
    /// Retrying the returned view resumes an outstanding identical close rather
    /// than publishing a second request. Drive this future to completion: dropping
    /// it after publication is not acknowledged cleanup. A service refusal remains
    /// distinct from an uncertain transport outcome; neither fabricates success.
    pub async fn close_checked(&mut self, view: View) -> Result<(), CloseFailure> {
        let reply = self
            .client
            .call_exact(&FsRequest::Close {
                handle: view.handle,
            })
            .await;
        let error = match reply {
            Ok(FsResponse::Done) => return Ok(()),
            Ok(FsResponse::Refused(error)) => Refusal::Fs(error),
            _ => Refusal::Pair,
        };
        Err(CloseFailure { view, error })
    }

    /// Give a view back. The handle is gone from this client's reckoning
    /// either way, so a `Close` refusal has nothing behind it to act on.
    pub async fn close(&mut self, view: View) {
        self.release(view.handle).await;
    }

    /// Resolve `path`'s parent directory and validate its final component.
    ///
    /// The pair every namespace verb needs: the service names an edge by
    /// (parent handle, name), never by a path.
    pub async fn open_parent(
        &mut self,
        path: &[u8],
        rights: Rights,
    ) -> Result<(View, Name), Refusal> {
        let Some((prefix, last)) = path::split_parent(path) else {
            // The private root itself. It is not reached through a
            // containment edge, so there is no edge to create or remove.
            return Err(Refusal::Fs(FsError::InvalidName));
        };
        let name = Name::from_bytes(last).map_err(Refusal::Fs)?;
        let parent = self.open(prefix, rights).await?;
        Ok((parent, name))
    }

    /// Metadata for an open view, revalidated by the service on this call.
    pub async fn stat(&mut self, view: &View) -> Result<Stat, Refusal> {
        match self
            .checked(&FsRequest::Stat {
                handle: view.handle,
            })
            .await?
        {
            FsResponse::Stat {
                object,
                stable_object,
                kind,
                subtype,
                len,
                revision,
                links,
                grants,
            } => Ok(Stat {
                object,
                stable_object,
                kind,
                subtype,
                len,
                revision,
                links,
                grants,
            }),
            _ => Err(Refusal::Pair),
        }
    }

    /// Find a named authority edge in this task's view.
    pub async fn named_grant(&mut self, name: &Name) -> Result<Option<GrantId>, Refusal> {
        let FsResponse::GrantsListed { entries } = self
            .checked(&FsRequest::ListGrants { window_offset: 0 })
            .await?
        else {
            return Err(Refusal::Pair);
        };
        for index in 0..entries as usize {
            let entry = read_grant_entry(&self.window[..], 0, index).map_err(Refusal::Fs)?;
            if !entry.is_private_root && entry.display_name.as_ref() == Some(name) {
                return Ok(Some(entry.grant));
            }
        }
        Ok(None)
    }

    /// Discover the effective route rights without caching authority
    /// client-side. Each bit is probed through the service and every
    /// temporary handle is closed; a revoked grant therefore changes the
    /// next display immediately.
    pub async fn effective_rights(&mut self, path: &[u8]) -> Result<Rights, Refusal> {
        let mut rights = Rights::READ;
        for candidate in [Rights::WRITE, Rights::MUTATE_NAMESPACE, Rights::SHARE] {
            let requested = Rights::READ.union(candidate);
            match self.open(path, requested).await {
                Ok(view) => {
                    rights = rights.union(candidate);
                    self.close(view).await;
                }
                Err(Refusal::Fs(FsError::AccessDenied)) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(rights)
    }
}

/// The compact spelling of a right set: one letter per right, in the
/// order `r w n s` (read, write, mutate-namespace, share). The same
/// vocabulary [`rights_parse`] reads, so what a listing prints is what a
/// `share` accepts.
pub fn rights_label(rights: Rights) -> &'static [u8] {
    match rights.bits() {
        1 => b"r",
        3 => b"rw",
        5 => b"rn",
        7 => b"rwn",
        9 => b"rs",
        11 => b"rws",
        13 => b"rns",
        15 => b"rwns",
        _ => b"?",
    }
}

/// The inverse of [`rights_label`]: a subset of `rwns`, in any order, each
/// letter at most once. `READ` is implied — a route that cannot read
/// cannot take a single step — so `w` alone is `rw`. Anything else is
/// `None`, never a guess.
pub fn rights_parse(spelling: &[u8]) -> Option<Rights> {
    let mut rights = Rights::READ;
    let mut seen = Rights::NONE;
    for &letter in spelling {
        let right = match letter {
            b'r' => Rights::READ,
            b'w' => Rights::WRITE,
            b'n' => Rights::MUTATE_NAMESPACE,
            b's' => Rights::SHARE,
            _ => return None,
        };
        if seen.contains(right) {
            return None;
        }
        seen = seen.union(right);
        rights = rights.union(right);
    }
    Some(rights)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReadOnlyRoot {
        probe: Option<FsError>,
        requests: heapless::Vec<FsRequest, 8>,
    }

    impl WalkClient for ReadOnlyRoot {
        async fn request(&mut self, request: FsRequest) -> Result<FsResponse, Refusal> {
            self.requests.push(request).unwrap();
            let opened = |handle| FsResponse::Opened {
                handle: Handle::from_bits(handle),
                stable_object: StableObjectId::new(handle).unwrap(),
                kind: ObjectKind::Directory,
                subtype: ObjectSubtype::from_bits(0),
                len: 0,
                revision: 0,
            };
            match request {
                FsRequest::OpenRoot { rights } if rights == Rights::READ => Ok(opened(1)),
                FsRequest::OpenRoot { .. } => Err(Refusal::Fs(FsError::AccessDenied)),
                FsRequest::OpenAt { rights, .. } => {
                    assert_eq!(rights, Rights::READ);
                    match self.probe {
                        Some(error) => Err(Refusal::Fs(error)),
                        None => Ok(opened(2)),
                    }
                }
                FsRequest::Close { .. } => Ok(FsResponse::Done),
                _ => panic!("unexpected request"),
            }
        }
        async fn named_grant(&mut self, _: &Name) -> Result<Option<GrantId>, Refusal> {
            panic!("a present or uncertain containment edge cannot fall through to a grant")
        }
    }

    #[test]
    fn resource_path_walk_does_not_bypass_shadowed_or_uncertain_containment() {
        for (probe, expected) in [
            (None, FsError::AccessDenied),
            (Some(FsError::Busy), FsError::Busy),
            (Some(FsError::AccessLost), FsError::AccessLost),
            (Some(FsError::NotDirectory), FsError::NotDirectory),
        ] {
            let mut client = ReadOnlyRoot {
                probe,
                requests: heapless::Vec::new(),
            };
            assert!(
                matches!(crate::block_on(open_path(&mut client, b"input/child", WALK_WRITE)),
                Err(Refusal::Fs(error)) if error == expected)
            );
            assert!(client.requests.contains(&FsRequest::Close {
                handle: Handle::from_bits(1)
            }));
            assert_eq!(
                client.requests.contains(&FsRequest::Close {
                    handle: Handle::from_bits(2)
                }),
                probe.is_none()
            );
        }
    }

    #[test]
    fn resource_path_walk_validates_depth_and_names_before_opening() {
        let mut client = ReadOnlyRoot {
            probe: None,
            requests: heapless::Vec::new(),
        };
        assert!(matches!(
            crate::block_on(open_path(&mut client, b"../outside", WALK_WRITE)),
            Err(Refusal::Fs(FsError::InvalidName))
        ));
        let mut path = heapless::Vec::<u8, 1024>::new();
        for _ in 0..=MAX_PATH_DEPTH {
            path.extend_from_slice(b"a/").unwrap();
        }
        assert!(matches!(
            crate::block_on(open_path(&mut client, &path, WALK_WRITE)),
            Err(Refusal::Fs(FsError::Range))
        ));
        assert!(client.requests.is_empty());
    }

    #[test]
    fn rights_spellings_round_trip() {
        for spelling in [
            &b"r"[..],
            b"rw",
            b"rn",
            b"rwn",
            b"rs",
            b"rws",
            b"rns",
            b"rwns",
        ] {
            let rights = rights_parse(spelling).unwrap();
            assert_eq!(rights_label(rights), spelling);
        }
    }

    #[test]
    fn read_is_implied_and_letters_may_not_repeat() {
        assert_eq!(rights_parse(b""), Some(Rights::READ));
        assert_eq!(rights_parse(b"w"), Some(Rights::READ.union(Rights::WRITE)));
        assert_eq!(rights_parse(b"ww"), None);
        assert_eq!(rights_parse(b"x"), None);
        assert_eq!(rights_parse(b"ro"), None);
    }
}
