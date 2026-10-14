//! Custody of the one filesystem pair a guest's descriptors share.
//!
//! The pair carries **one request at a time**, and a request, once
//! published, owns the pair and its data window until its reply is taken —
//! a lost reply does not permit a replay. Three things in a WASI host would
//! break that if requests were issued from the caller's own future:
//!
//! - `wasi:io/poll` polls every listed pollable and **drops** the ones that
//!   were not ready, and `pollable.ready` polls once and drops. A file
//!   stream whose readiness is a read would abandon a published request.
//! - Several pollables in one `poll` may each want the pair.
//! - A descriptor dropped with its store (not through `drop`) has no async
//!   context to close its view in.
//!
//! So every request sequence runs as an **op** installed in [`FsHost`]'s one
//! slot, not in the caller: whoever polls the slot drives it, a caller that
//! is dropped leaves its op to finish on the next caller's turn, and a new
//! op is installed only once the slot is empty. A view whose last holder
//! was dropped synchronously is queued ([`Owned`]) and closed, in its own
//! op, before the next op is installed.
//!
//! All futures of one store are polled from the one future driving the
//! guest (`run`'s call), so whichever caller drives an op, the wake it
//! registers is that task's.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::future::poll_fn;
use core::pin::Pin;
use core::task::{Context, Poll};

use kore::fs::dag::GrantId;
use kore::fs::{Fs, FsError, Name, Refusal, Rights, View};
use kore::runtime::Pinned;

use super::walk::Steps;

/// The task's filesystem face, as a guest's host holds it.
pub(crate) type Face = Fs<'static, 'static>;

/// The pair, and what the host knows about its custody.
pub(crate) struct FsCell {
    /// Out only while an op holds it ([`Custody`]).
    fs: Cell<Option<Fs<'static, 'static>>>,
    /// A request's reply was lost (a transport failure, or an op dropped
    /// mid-request): a request may still own the pair and its window, so
    /// nothing is published or staged again.
    lost: Cell<bool>,
    /// Views whose last holder was dropped where no close could be awaited.
    orphans: RefCell<Vec<View>>,
}

impl FsCell {
    fn orphan(&self, view: View) {
        // Unreachable contention (nothing holds the queue across a call
        // that could drop a holder); should it happen, the view is left to
        // the service's teardown of this session rather than panic.
        if let Ok(mut orphans) = self.orphans.try_borrow_mut() {
            orphans.push(view);
        }
    }

    fn next_orphan(&self) -> Option<View> {
        self.orphans.try_borrow_mut().ok()?.pop()
    }
}

/// The pair, held by one op: the owner's `Fs`, one verb at a time.
///
/// Refuses every verb once a reply was lost; gives the pair back when the
/// op ends, and marks it lost if the op was dropped with a verb in flight.
pub(crate) struct Custody {
    cell: Rc<FsCell>,
    fs: Option<Fs<'static, 'static>>,
    outstanding: bool,
}

impl Custody {
    fn take(cell: Rc<FsCell>) -> Self {
        let fs = cell.fs.take();
        Self {
            cell,
            fs,
            outstanding: false,
        }
    }

    /// Run one of the owner's verbs — each is a single round trip — on the
    /// pair. A transport failure (`Refusal::Pair`) loses the pair.
    pub(crate) async fn call<R>(
        &mut self,
        verb: impl AsyncFnOnce(&mut Face) -> Result<R, Refusal>,
    ) -> Result<R, Refusal> {
        if self.cell.lost.get() {
            return Err(Refusal::Pair);
        }
        let Some(fs) = self.fs.as_mut() else {
            return Err(Refusal::Pair);
        };
        self.outstanding = true;
        let result = verb(fs).await;
        self.outstanding = false;
        if matches!(result, Err(Refusal::Pair)) {
            self.cell.lost.set(true);
        }
        result
    }

    /// The owner's acknowledged close. A refused close hands the view back
    /// (`CloseFailure`); there is nothing to retry it with here — the pair
    /// is lost, or the service has already forgotten the handle — so it is
    /// left to the session's teardown.
    pub(crate) async fn close(&mut self, view: View) {
        let _ = self
            .call(async move |fs: &mut Face| {
                fs.close_checked(view)
                    .await
                    .map_err(|failure| failure.error)
            })
            .await;
    }

    /// Own `view`, shared: when the last holder drops, its close is queued.
    pub(crate) fn own(&self, view: View) -> Shared {
        Rc::new(Owned {
            view: Some(view),
            cell: self.cell.clone(),
        })
    }
}

impl Steps for Custody {
    type View = View;

    async fn open_at(
        &mut self,
        parent: &View,
        name: Name,
        rights: Rights,
    ) -> Result<View, Refusal> {
        self.call(async move |fs: &mut Face| fs.open_at(parent, name, rights).await)
            .await
    }

    async fn close(&mut self, view: View) {
        Custody::close(self, view).await;
    }
}

impl Drop for Custody {
    fn drop(&mut self) {
        if self.outstanding {
            self.cell.lost.set(true);
        }
        if let Some(fs) = self.fs.take() {
            self.cell.fs.set(Some(fs));
        }
    }
}

/// A [`View`] the host owns. Dropping it queues its close.
pub(crate) struct Owned {
    view: Option<View>,
    cell: Rc<FsCell>,
}

/// A view held by a descriptor and the streams and listings made from it;
/// closed when the last of them lets go.
pub(crate) type Shared = Rc<Owned>;

impl Owned {
    pub(crate) fn view(&self) -> Result<&View, Refusal> {
        self.view
            .as_ref()
            .ok_or(Refusal::Fs(FsError::InvalidHandle))
    }

    /// Take the view to close it now; the queue is no longer involved.
    pub(crate) fn into_view(mut self) -> Option<View> {
        self.view.take()
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        if let Some(view) = self.view.take() {
            self.cell.orphan(view);
        }
    }
}

type Op = Pin<Box<dyn Future<Output = ()>>>;

/// Run `op` with the pair, and hand the pair back.
async fn with_pair<R, F>(cell: Rc<FsCell>, op: F) -> R
where
    F: AsyncFnOnce(&mut Custody) -> R,
{
    let mut custody = Custody::take(cell);
    op(&mut custody).await
}

/// The guest's filesystem: the pair and its one op slot.
pub(crate) struct FsHost {
    cell: Rc<FsCell>,
    slot: RefCell<Option<Op>>,
    /// The largest transfer, fixed at attach: what one stream write may
    /// carry.
    chunk: usize,
    /// Whether the task's named grants are offered as preopens of their
    /// names beside `.` ([`Filesystem::with_named_grants`]).
    grants: Cell<bool>,
}

impl FsHost {
    pub(crate) fn chunk(&self) -> usize {
        self.chunk
    }

    pub(crate) fn offers_grants(&self) -> bool {
        self.grants.get()
    }

    /// Drive the op in the slot; `Ready` once the slot is empty.
    fn poll_slot(&self, cx: &mut Context<'_>) -> Poll<()> {
        let Ok(mut slot) = self.slot.try_borrow_mut() else {
            // Unreachable: an op never reaches the slot. Yield rather than
            // panic.
            cx.waker().wake_by_ref();
            return Poll::Pending;
        };
        let Some(op) = slot.as_mut() else {
            return Poll::Ready(());
        };
        match op.as_mut().poll(cx) {
            Poll::Ready(()) => {
                *slot = None;
                Poll::Ready(())
            }
            Poll::Pending => Poll::Pending,
        }
    }

    /// Install `op` in the empty slot.
    fn install(&self, op: Op) {
        if let Ok(mut slot) = self.slot.try_borrow_mut() {
            *slot = Some(op);
        }
    }

    fn close_op(&self, view: View) -> Op {
        Box::pin(with_pair(
            self.cell.clone(),
            async move |pair: &mut Custody| pair.close(view).await,
        ))
    }

    /// Finish every op in flight and close every queued view.
    pub(crate) async fn settle(&self) {
        poll_fn(|cx| {
            loop {
                if self.poll_slot(cx).is_pending() {
                    return Poll::Pending;
                }
                match self.cell.next_orphan() {
                    Some(view) => self.install(self.close_op(view)),
                    None => return Poll::Ready(()),
                }
            }
        })
        .await
    }

    /// Run `op` with the pair once every earlier op has finished and every
    /// queued view is closed. Dropping this future before `op` is installed
    /// drops `op` unrun (queueing any view it held); after, `op` still runs
    /// to completion and its result is dropped (queueing any view it
    /// owned). `None` only if the result is gone, which the slot's
    /// discipline rules out.
    pub(crate) async fn exec<R, F>(&self, op: F) -> Option<R>
    where
        R: 'static,
        F: AsyncFnOnce(&mut Custody) -> R + 'static,
    {
        let out: Rc<Cell<Option<R>>> = Rc::new(Cell::new(None));
        let mut op = Some(op);
        poll_fn(|cx| {
            loop {
                if let Some(result) = out.take() {
                    return Poll::Ready(Some(result));
                }
                if self.poll_slot(cx).is_pending() {
                    return Poll::Pending;
                }
                if let Some(view) = self.cell.next_orphan() {
                    self.install(self.close_op(view));
                    continue;
                }
                let Some(op) = op.take() else {
                    return Poll::Ready(None);
                };
                let (cell, out) = (self.cell.clone(), out.clone());
                self.install(Box::pin(async move {
                    out.set(Some(with_pair(cell, op).await));
                }));
            }
        })
        .await
    }
}

/// A task's filesystem, given to a guest as one preopen, `.`: the task's
/// own root view (on Kore, the directory it was launched in). The guest
/// resolves paths only beneath it — never through the task's named grants
/// from there. An embedder that wants the guest to see those grants names
/// them ([`with_named_grants`](Self::with_named_grants)): each is then a
/// preopen of its own, under the name the task's store gives it, and the
/// guest resolves beneath it the same way.
///
/// `Send`, as a store's data must be; the pair is thread-bound and usable
/// only on the thread that pinned it.
pub struct Filesystem(Pinned<Rc<FsHost>>);

impl Filesystem {
    /// Give `fs` — this task's attached face (`kore::fs::Fs::attach`) —
    /// to a guest. `None` (the face dropped) when the thread has no runtime
    /// identity to pin it to, which a Kore task always has.
    pub fn new(fs: Fs<'static, 'static>) -> Option<Self> {
        let chunk = fs.chunk();
        let host = Rc::new(FsHost {
            cell: Rc::new(FsCell {
                fs: Cell::new(Some(fs)),
                lost: Cell::new(false),
                orphans: RefCell::new(Vec::new()),
            }),
            slot: RefCell::new(None),
            chunk,
            grants: Cell::new(false),
        });
        Pinned::new(host).ok().map(Self)
    }

    /// Offer the task's named grants to the guest too, each as a preopen
    /// under its store name beside `.`: a program launched as
    /// `shared/wasm/python.wasm` then finds `shared/wasm/lib/` where its
    /// launcher does, since `shared` is a grant the launcher's own walk
    /// resolves and the guest's one-step walk from `.` cannot. The
    /// embedder decides: a grant it holds for itself — a private data
    /// grant — is never offered unless it says so, and the grants offered
    /// are the task's, named as the task names them, with the rights the
    /// service answers for each.
    pub fn with_named_grants(self) -> Self {
        if let Some(host) = self.0.get() {
            host.grants.set(true);
        }
        self
    }

    pub(crate) fn host(&self) -> Option<&Rc<FsHost>> {
        self.0.get()
    }

    /// Finish what is in flight, close every view the guest's dropped
    /// descriptors held, and give the face back.
    ///
    /// Drop the guest's resource table first: a descriptor or stream still
    /// in it holds the host. Then the face is not given back (`None`); it
    /// is dropped with the last holder, and any view still queued then is
    /// left to the service's teardown of this session.
    pub async fn close(self) -> Option<Fs<'static, 'static>> {
        let host = self.0.into_inner().ok()?;
        host.settle().await;
        let host = Rc::into_inner(host)?;
        Rc::into_inner(host.cell)?.fs.into_inner()
    }
}

/// Widest first: the rights a preopen may carry, probed in order until
/// the service answers one — the task root's, and each named grant's.
const PREOPEN_RIGHTS: [Rights; 4] = {
    const MUTATE: Rights = Rights::READ.union(Rights::MUTATE_NAMESPACE);
    const WRITE: Rights = Rights::READ.union(Rights::WRITE);
    [MUTATE.union(Rights::WRITE), MUTATE, WRITE, Rights::READ]
};

/// The task's named grants, each opened with the widest rights the service
/// answers for it, in the order the store lists them: what
/// [`Filesystem::with_named_grants`] offers beside the root. The private
/// root is not among them (it is `.`), nor is a grant without a name (a
/// guest could not spell it). A grant the service refuses outright is left
/// out; one it refuses only at a width is tried at the next.
pub(crate) async fn preopen_grants(
    host: &FsHost,
) -> Result<Vec<(Name, GrantId, Shared, Rights)>, GrantsRefused> {
    host.exec(async |pair: &mut Custody| {
        // The listing lands in the pair's window, which the next request
        // reuses: read every entry before opening any.
        let named: Vec<(Name, GrantId)> = pair
            .call(async |fs: &mut Face| {
                let count = fs.shares().await?;
                let mut named = Vec::new();
                for index in 0..count as usize {
                    let entry = fs.share_entry(index)?;
                    if entry.is_private_root {
                        continue;
                    }
                    if let Some(name) = entry.display_name {
                        named.push((name, entry.grant));
                    }
                }
                Ok(named)
            })
            .await
            .map_err(GrantsRefused::Listing)?;
        let mut preopens = Vec::with_capacity(named.len());
        for (name, grant) in named {
            let mut refused = Refusal::Pair;
            for rights in PREOPEN_RIGHTS {
                let root = pair
                    .call(async move |fs: &mut Face| fs.open_grant(grant, rights).await)
                    .await;
                match root {
                    Ok(root) => {
                        preopens.push((name, grant, pair.own(root), rights));
                        refused = Refusal::Fs(FsError::AccessDenied);
                        break;
                    }
                    Err(error @ Refusal::Fs(FsError::AccessDenied)) => refused = error,
                    Err(error) => return Err(GrantsRefused::Grant(name, error)),
                }
            }
            if preopens.last().is_none_or(|(last, ..)| *last != name) {
                return Err(GrantsRefused::Grant(name, refused));
            }
        }
        Ok(preopens)
    })
    .await
    .unwrap_or(Err(GrantsRefused::Listing(Refusal::Pair)))
}

/// Why the task's named grants could not be offered.
pub(crate) enum GrantsRefused {
    /// The listing itself was refused (`Pair` when the pair is gone).
    Listing(Refusal),
    /// This grant could not be opened at any width: the service's refusal
    /// of the narrowest probe, or the first for a reason other than width.
    Grant(Name, Refusal),
}

/// The task root with the widest of the rights a preopen may carry; each
/// probe is the service's own answer. `Err` is the service's refusal of
/// the narrowest probe, or of the first one it refused for a reason other
/// than width; `Pair` when the pair is gone.
pub(crate) async fn preopen(host: &FsHost) -> Result<(Shared, Rights), Refusal> {
    host.exec(async |pair: &mut Custody| {
        let mut refused = Refusal::Pair;
        for rights in PREOPEN_RIGHTS {
            let root = pair
                .call(async move |fs: &mut Face| fs.open_root(rights).await)
                .await;
            match root {
                Ok(root) => return Ok((pair.own(root), rights)),
                Err(error @ Refusal::Fs(FsError::AccessDenied)) => refused = error,
                Err(error) => return Err(error),
            }
        }
        Err(refused)
    })
    .await
    .unwrap_or(Err(Refusal::Pair))
}
