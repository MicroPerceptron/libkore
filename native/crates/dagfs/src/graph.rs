//! The containment graph, as a client mutates and reads it: create,
//! project (link), detach, move, list, search.
//!
//! The service names an edge by `(parent handle, name)` and never by a
//! path, so every verb here takes the parent as an open [`View`]. The
//! path-level conveniences ([`Fs::move_path`]) do the walk and hand the
//! service exactly that pair.

use kore_svc_extract::ParserId;

use crate::{
    DirEntry, Fs, FsError, FsRequest, FsResponse, Name, ObjectKind, ObjectSubtype, Refusal,
    Revision, SearchCursor, SearchHit, SearchQuery, View, WALK_MUTATE, WALK_READ, path,
    read_dir_entry, read_search_hit,
};

impl<'w, 't> Fs<'w, 't> {
    /// Create a new object beneath `parent`; the creator holds a view of it.
    pub async fn create(
        &mut self,
        parent: &View,
        name: Name,
        kind: ObjectKind,
    ) -> Result<View, Refusal> {
        self.create_typed(parent, name, kind, ObjectSubtype::PLAIN)
            .await
    }

    /// [`Fs::create`], stamping `subtype` — data the object carries for its
    /// life, never authority over it. A directory refuses any but `PLAIN`,
    /// and an attesting stamp needs its
    /// [`stamp_rights`](ObjectSubtype::stamp_rights) on `parent`'s route
    /// (`AccessDenied` otherwise).
    pub async fn create_typed(
        &mut self,
        parent: &View,
        name: Name,
        kind: ObjectKind,
        subtype: ObjectSubtype,
    ) -> Result<View, Refusal> {
        View::from_opened(
            self.checked(&FsRequest::Create {
                parent: parent.handle(),
                name,
                kind,
                subtype,
            })
            .await?,
        )
    }

    /// Add a second name for `child` beneath `parent` — a DAG projection,
    /// never a byte copy. A directory child is refused when the new edge
    /// would make `parent` reachable from it (`FsError::Cycle`).
    pub async fn project(
        &mut self,
        parent: &View,
        name: Name,
        child: &View,
    ) -> Result<(), Refusal> {
        self.checked(&FsRequest::Link {
            parent: parent.handle(),
            name,
            child: child.handle(),
        })
        .await?;
        Ok(())
    }

    /// Remove exactly one containment edge — the native graph operation. A
    /// populated directory is legal: the graph survives through any other
    /// containment or authority route, and an object no route reaches is
    /// reclaimed by the service.
    pub async fn detach(&mut self, parent: &View, name: Name) -> Result<(), Refusal> {
        self.checked(&FsRequest::DetachEdge {
            parent: parent.handle(),
            name,
        })
        .await?;
        Ok(())
    }

    /// Compatibility unlink: refuses a populated directory (`NotEmpty`).
    /// The POSIX `rm`/`rmdir` posture; [`Fs::detach`] is the native one.
    pub async fn unlink(&mut self, parent: &View, name: Name) -> Result<(), Refusal> {
        self.checked(&FsRequest::Unlink {
            parent: parent.handle(),
            name,
        })
        .await?;
        Ok(())
    }

    /// Rename within one directory.
    pub async fn rename(&mut self, parent: &View, old: Name, new: Name) -> Result<(), Refusal> {
        self.checked(&FsRequest::Rename {
            parent: parent.handle(),
            old,
            new,
        })
        .await?;
        Ok(())
    }

    /// Move one edge from `src` to `dst` under `new_name`, atomically.
    pub async fn relocate(
        &mut self,
        src: &View,
        name: Name,
        dst: &View,
        new_name: Name,
    ) -> Result<(), Refusal> {
        self.checked(&FsRequest::Relocate {
            src_parent: src.handle(),
            name,
            dst_parent: dst.handle(),
            new_name,
        })
        .await?;
        Ok(())
    }

    /// Move `src` to `dst` by path: a rename when both name the same
    /// directory, a relocation otherwise. Yields the old and new names.
    pub async fn move_path(&mut self, src: &[u8], dst: &[u8]) -> Result<(Name, Name), Refusal> {
        let (Some((src_prefix, _)), Some((dst_prefix, dst_last))) =
            (path::split_parent(src), path::split_parent(dst))
        else {
            return Err(Refusal::Fs(FsError::InvalidName));
        };
        let new_name = Name::from_bytes(dst_last).map_err(Refusal::Fs)?;
        let (src_parent, name) = self.open_parent(src, WALK_MUTATE).await?;
        let outcome = if path::same_directory(src_prefix, dst_prefix) {
            self.rename(&src_parent, name, new_name).await
        } else {
            match self.open(dst_prefix, WALK_MUTATE).await {
                Ok(dst_parent) => {
                    let moved = self
                        .relocate(&src_parent, name, &dst_parent, new_name)
                        .await;
                    self.close(dst_parent).await;
                    moved
                }
                Err(refusal) => Err(refusal),
            }
        };
        self.close(src_parent).await;
        outcome?;
        Ok((name, new_name))
    }

    /// One page of `dir`'s entries into the window, from `cursor`. Yields
    /// the entry count and the cursor to resume at (`None` when complete);
    /// read each entry with [`Fs::dir_entry`] before the next request.
    pub async fn list_page(
        &mut self,
        dir: &View,
        cursor: u32,
    ) -> Result<(u32, Option<u32>), Refusal> {
        match self
            .checked(&FsRequest::List {
                handle: dir.handle(),
                cursor,
                window_offset: 0,
            })
            .await?
        {
            // A resume cursor that does not advance would page forever.
            FsResponse::Listed {
                next: Some(next), ..
            } if next <= cursor => Err(Refusal::Pair),
            FsResponse::Listed { entries, next } => Ok((entries, next)),
            _ => Err(Refusal::Pair),
        }
    }

    /// Visit every entry of the directory at `path`, page by page, copying
    /// each entry out of the window before the next request may reuse it.
    pub async fn list_directory<F>(&mut self, path: &[u8], mut consume: F) -> Result<(), Refusal>
    where
        F: FnMut(DirEntry) -> Result<(), Refusal>,
    {
        let dir = self.open(path, WALK_READ).await?;
        let mut cursor = 0u32;
        let result = 'pages: loop {
            let (entries, next) = match self.list_page(&dir, cursor).await {
                Ok(page) => page,
                Err(error) => break Err(error),
            };
            for index in 0..entries as usize {
                let entry = match self.dir_entry(index) {
                    Ok(entry) => entry,
                    Err(error) => break 'pages Err(error),
                };
                if let Err(error) = consume(entry) {
                    break 'pages Err(error);
                }
            }
            let Some(next) = next else {
                break Ok(());
            };
            cursor = next;
        };
        self.close(dir).await;
        result
    }

    /// The `index`th entry of the page the last [`Fs::list_page`] landed.
    pub fn dir_entry(&self, index: usize) -> Result<DirEntry, Refusal> {
        read_dir_entry(&self.window[..], 0, index).map_err(Refusal::Fs)
    }

    /// One page of search hits beneath `anchor` into the window. Yields the
    /// hit count and the cursor to resume at; read each hit with
    /// [`Fs::search_hit`] before the next request.
    pub async fn search_page(
        &mut self,
        anchor: &View,
        query: SearchQuery,
        cursor: Option<SearchCursor>,
    ) -> Result<(u32, Option<SearchCursor>), Refusal> {
        match self
            .checked(&FsRequest::Search {
                handle: anchor.handle(),
                query,
                cursor,
                window_offset: 0,
            })
            .await?
        {
            FsResponse::Searched { hits, next, .. } => Ok((hits, next)),
            _ => Err(Refusal::Pair),
        }
    }

    /// The `index`th hit of the page the last [`Fs::search_page`] landed.
    pub fn search_hit(&self, index: usize) -> Result<SearchHit, Refusal> {
        read_search_hit(&self.window[..], 0, index).map_err(Refusal::Fs)
    }

    /// Whether `parser` has a committed batch for `file`'s current version:
    /// `Some((input_version, parser_version))`, or `None` when nothing is
    /// on record.
    pub async fn derived_info(
        &mut self,
        file: &View,
        parser: ParserId,
    ) -> Result<Option<(Revision, u32)>, Refusal> {
        match self
            .checked(&FsRequest::DerivedInfo {
                handle: file.handle(),
                parser,
            })
            .await
        {
            Ok(FsResponse::DerivedInfo {
                input_version,
                parser_version,
            }) => Ok(Some((input_version, parser_version))),
            Ok(_) => Err(Refusal::Pair),
            Err(Refusal::Fs(FsError::NotFound)) => Ok(None),
            Err(error) => Err(error),
        }
    }
}
