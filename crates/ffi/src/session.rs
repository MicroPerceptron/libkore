use super::*;

/// Native owner for a foreign filesystem program.
///
/// Exactly one request can be outstanding. A cancelled `run` quarantines the
/// session: it can never issue another request or surrender its Fs for reuse.
/// The native data/ring mappings are task-lifetime mappings, not this struct's
/// scratch buffers. Cancellation/indeterminate transport therefore requires
/// retiring the enclosing task to reclaim authority; dropping this owner alone
/// is not acknowledged cleanup. Normal exit closes every retained view.
///
/// The foreign callback runs synchronously and must return promptly. This ABI
/// does not add an executor, a global root, or preemption of arbitrary C code.
pub struct Session<'w, 't> {
    fs: Fs<'w, 't>,
    context: KoreFsContext,
    views: [Option<View>; KORE_FS_HANDLE_CAPACITY],
}

#[derive(Debug)]
pub enum RunError {
    /// Cancellation or an uncertain transport outcome requires task retirement.
    Quarantined,
    /// An authenticated close refusal retained the view in this owner.
    Cleanup(Refusal),
}

impl<'w, 't> Session<'w, 't> {
    /// Consume the real native face. Identifier exhaustion returns it unchanged.
    /// Close any previously opened external views before handing off this face:
    /// this owner can clean up only views opened through its own callbacks.
    // Allocation-free failure returns the affine native owner intact. This
    // cold-path error exceeds Clippy's generic size threshold.
    #[allow(clippy::result_large_err)]
    pub fn new(fs: Fs<'w, 't>) -> Result<Self, Fs<'w, 't>> {
        let Some(context) = KoreFsContext::new() else {
            return Err(fs);
        };
        Ok(Self {
            fs,
            context,
            views: core::array::from_fn(|_| None),
        })
    }

    /// Run a callback program on the caller's existing native task/executor.
    /// Each callback sees the previous completion and may queue one new request.
    /// On an uncertain reply, its final callback sees `Indeterminate` and all
    /// attempted submissions fail with `Quarantined`. No mutation is replayed.
    ///
    /// # Safety
    /// `step` obeys KoreFsStep's pointer, serialization, and no-unwind contract.
    /// `user` remains valid and exclusively accessible as required by that
    /// callback until this future completes or is dropped. The callback must not
    /// retain the context/completion pointers or re-enter this owner.
    pub async unsafe fn run(
        &mut self,
        step: KoreFsStep,
        user: *mut c_void,
    ) -> Result<(), RunError> {
        if matches!(self.context.phase, Phase::InFlight | Phase::Quarantined) {
            return Err(RunError::Quarantined);
        }
        self.context.completion = KoreFsCompletion {
            abi_version: KORE_ABI_VERSION,
            ..Default::default()
        };
        let mut completed_turns = 0;
        loop {
            self.context.phase = Phase::Callback;
            self.context.command = None;
            // Copy the completion outside ctx: the callback may mutate ctx
            // through exports without aliasing an immutable reference into it.
            let completion = self.context.completion;
            // SAFETY: run's caller guarantees the foreign callback contract.
            unsafe {
                step(&mut self.context, &completion, user);
            }
            let Some(command) = self.context.command.take() else {
                return self.close_all().await;
            };
            // This durable phase precedes the first await. Dropping the future
            // leaves InFlight set and permanently prevents owner reuse.
            self.context.phase = Phase::InFlight;
            self.context.completion = KoreFsCompletion {
                abi_version: KORE_ABI_VERSION,
                operation: command.op,
                handle: if matches!(
                    command.op,
                    KORE_OP_OPEN
                        | KORE_OP_CREATE
                        | KORE_OP_SLEEP_FOR
                        | KORE_OP_SLEEP_UNTIL
                        | KORE_OP_YIELD
                ) {
                    KoreFsHandle::default()
                } else {
                    self.context.handle(command.slot)
                },
                ..Default::default()
            };
            if let Err(error) = self.execute(command).await {
                self.context.completion.status = match error {
                    Refusal::Pair => KoreStatus::Indeterminate,
                    _ => KoreStatus::Refused,
                };
                if let Refusal::Fs(FsError::ProviderFailure(provider)) = error {
                    self.context.completion.provider_error = provider_class(provider);
                }
                self.context.completion.error = match error {
                    Refusal::Fs(error) => classify(error),
                    _ => KoreFsError::Unknown,
                };
                if matches!(error, Refusal::Pair) {
                    self.context.phase = Phase::Quarantined;
                    let completion = self.context.completion;
                    // SAFETY: same callback contract; this is a final notification
                    // only, and the quarantined context rejects every submission.
                    unsafe {
                        step(&mut self.context, &completion, user);
                    }
                    return Err(RunError::Quarantined);
                }
            }
            cooperate(&mut completed_turns).await;
        }
    }

    async fn execute(&mut self, command: Command) -> Result<(), Refusal> {
        let slot = command.slot;
        match command.op {
            KORE_OP_SLEEP_FOR => kore::time::sleep_for(command.offset).await,
            KORE_OP_SLEEP_UNTIL => kore::time::sleep_until_ns(command.offset).await,
            KORE_OP_YIELD => kore::runtime::executor::yield_now().await,
            KORE_OP_OPEN => {
                let view = self
                    .fs
                    .open(&self.context.path[..command.length], command.rights)
                    .await?;
                self.install(command.destination, view);
            }
            KORE_OP_CREATE => {
                let view = self
                    .fs
                    .create(
                        self.views[slot].as_ref().unwrap(),
                        command.name.unwrap(),
                        command.kind,
                    )
                    .await?;
                self.install(command.destination, view);
            }
            KORE_OP_CLOSE => {
                let view = self.views[slot].take().unwrap();
                match self.fs.close_checked(view).await {
                    Ok(()) => self.context.retire(slot),
                    Err(failure) => {
                        self.views[slot] = Some(failure.view);
                        return Err(failure.error);
                    }
                }
            }
            _ => {
                let view = self.views[slot].as_ref().unwrap();
                match command.op {
                    KORE_OP_READ => {
                        let count = self
                            .fs
                            .read_at(
                                view,
                                command.offset,
                                &mut self.context.data[..command.length],
                            )
                            .await?;
                        self.context.completion.count = count as u64;
                    }
                    KORE_OP_WRITE => {
                        let count = self
                            .fs
                            .write_at(view, command.offset, &self.context.data[..command.length])
                            .await?;
                        self.context.completion.count = count as u64;
                    }
                    KORE_OP_STAT => {
                        let stat = self.fs.stat(view).await?;
                        let result = &mut self.context.completion;
                        result.length = stat.len;
                        result.revision = stat.revision;
                        result.stable_object = stat.stable_object.get();
                        result.kind = kind(stat.kind);
                    }
                    KORE_OP_RESIZE => {
                        self.context.completion.revision =
                            self.fs.resize(view, command.offset).await?;
                        self.context.completion.length = command.offset;
                    }
                    KORE_OP_SYNC => self.fs.sync(view).await?,
                    KORE_OP_UNLINK => self.fs.unlink(view, command.name.unwrap()).await?,
                    KORE_OP_DETACH => self.fs.detach(view, command.name.unwrap()).await?,
                    KORE_OP_LINK => {
                        self.fs
                            .project(
                                view,
                                command.name.unwrap(),
                                self.views[command.destination].as_ref().unwrap(),
                            )
                            .await?
                    }
                    KORE_OP_LIST => {
                        let (count, next) = self.fs.list_page(view, command.offset as u32).await?;
                        if count as usize > KORE_FS_ENTRY_CAPACITY {
                            return Err(Refusal::Pair);
                        }
                        for index in 0..count {
                            let entry = self.fs.dir_entry(index as usize)?;
                            let name = entry.name.as_bytes();
                            let out = &mut self.context.entries[index as usize];
                            *out = KoreFsEntry::default();
                            out.name[..name.len()].copy_from_slice(name);
                            out.name_length = name.len() as u32;
                            out.kind = kind(entry.kind);
                        }
                        self.context.completion.count = u64::from(count);
                        self.context.completion.next_cursor = next.unwrap_or(0);
                        self.context.completion.has_next = u32::from(next.is_some());
                    }
                    _ => unreachable!("only validated exports can queue commands"),
                }
            }
        }
        Ok(())
    }

    fn install(&mut self, slot: usize, view: View) {
        // The slot was reserved before publishing Open/Create, so no native
        // view can be received without capacity to retain it.
        assert!(self.views[slot].is_none());
        let result = &mut self.context.completion;
        result.length = view.len();
        result.revision = view.revision();
        result.stable_object = view.stable().get();
        result.kind = kind(view.kind());
        self.views[slot] = Some(view);
        self.context.slots[slot].occupied = true;
        self.context.completion.handle = self.context.handle(slot);
    }

    async fn close_all(&mut self) -> Result<(), RunError> {
        for slot in 0..self.views.len() {
            let Some(view) = self.views[slot].take() else {
                continue;
            };
            self.context.phase = Phase::InFlight;
            match self.fs.close_checked(view).await {
                Ok(()) => self.context.retire(slot),
                Err(failure) => {
                    self.views[slot] = Some(failure.view);
                    if matches!(failure.error, Refusal::Pair) {
                        self.context.phase = Phase::Quarantined;
                        return Err(RunError::Quarantined);
                    }
                    self.context.phase = Phase::Finished;
                    return Err(RunError::Cleanup(failure.error));
                }
            }
        }
        self.context.phase = Phase::Finished;
        Ok(())
    }
}

fn kind(kind: ObjectKind) -> u32 {
    match kind {
        ObjectKind::File => KORE_FS_FILE,
        ObjectKind::Directory => KORE_FS_DIRECTORY,
    }
}

fn classify(error: FsError) -> KoreFsError {
    use KoreFsError as C;
    match error {
        FsError::NotFound => C::NotFound,
        FsError::AlreadyExists => C::AlreadyExists,
        FsError::NotDirectory => C::NotDirectory,
        FsError::IsDirectory => C::IsDirectory,
        FsError::InvalidName => C::InvalidName,
        FsError::InvalidHandle => C::InvalidHandle,
        FsError::AccessDenied => C::AccessDenied,
        FsError::AccessLost => C::AccessLost,
        FsError::Conflict => C::Conflict,
        FsError::Cycle => C::Cycle,
        FsError::NotEmpty => C::NotEmpty,
        FsError::OutOfObjects => C::OutOfObjects,
        FsError::OutOfEdges => C::OutOfEdges,
        FsError::OutOfHandles => C::OutOfHandles,
        FsError::OutOfStorage => C::OutOfStorage,
        FsError::OutOfPlacements => C::OutOfPlacements,
        FsError::OutOfVersionLeases => C::OutOfVersionLeases,
        FsError::OutOfIndexSlots => C::OutOfIndexSlots,
        FsError::Range => C::Range,
        FsError::BudgetExceeded => C::BudgetExceeded,
        FsError::IndexIncomplete => C::IndexIncomplete,
        FsError::Unsupported => C::Unsupported,
        FsError::NoWindow => C::NoWindow,
        FsError::Busy => C::Busy,
        FsError::ProviderFailure(_) => C::ProviderFailure,
    }
}

fn provider_class(error: dag::ProviderError) -> KoreProviderError {
    use KoreProviderError as C;
    use dag::ProviderError as R;
    match error {
        R::Exhausted => C::Exhausted,
        R::PlacementsFull => C::PlacementsFull,
        R::UnknownKey => C::UnknownKey,
        R::Range => C::Range,
        R::TooLarge => C::TooLarge,
        R::Media => C::Media,
        R::Transport => C::Transport,
        R::Integrity => C::Integrity,
        R::Cancelled => C::Cancelled,
        R::Revoked => C::Revoked,
        R::Lost => C::Lost,
        R::Draining => C::Draining,
        R::ReadOnly => C::ReadOnly,
        R::Backpressure => C::Backpressure,
        R::Timeout => C::Timeout,
    }
}

// Even zero-length I/O can complete immediately. Bound consecutive callbacks
// so that such a foreign loop cannot monopolize one native executor poll.
const TURN_BUDGET: u32 = 8;
async fn cooperate(turns: &mut u32) {
    *turns += 1;
    if *turns == TURN_BUDGET {
        *turns = 0;
        kore::runtime::executor::yield_now().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };

    #[test]
    fn immediately_ready_operations_still_end_the_executor_poll() {
        let mut turns = 0;
        let mut cx = Context::from_waker(Waker::noop());
        for _ in 0..TURN_BUDGET - 1 {
            assert_eq!(
                pin!(cooperate(&mut turns)).as_mut().poll(&mut cx),
                Poll::Ready(())
            );
        }
        let mut yield_turn = pin!(cooperate(&mut turns));
        assert_eq!(yield_turn.as_mut().poll(&mut cx), Poll::Pending);
        assert_eq!(yield_turn.as_mut().poll(&mut cx), Poll::Ready(()));
    }
}
