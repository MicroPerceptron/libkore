//! WASI 0.2 for WebAssembly components on Kore.
//!
//! A stock `wasm32-wasip2` program — ordinary Rust `std`, nothing
//! Kore-specific — runs as a component linked against the
//! `wasi:cli/command` world, on an engine the caller configures for its
//! platform. Every import is a native owner's own implementation:
//!
//! - **`wasi:io` and the standard streams** are the process stream the
//!   launcher gave this task — [`stdio::ProcessStreams`] implements
//!   `wasmtime-wasi-io`'s stream and pollable traits on the native endpoint
//!   itself. No pump or copy sits between the guest and the rings.
//! - **`wasi:filesystem`** is the task's own filesystem face
//!   ([`Filesystem`], over `kore::fs`): one preopen, `.`, the task's root
//!   view, resolved component by component beneath the guest's descriptors
//!   and never through the task's named grants.
//! - **`wasi:clocks`** are `kore::time`; a deadline pollable sleeps on the
//!   Kore executor.
//! - **`wasi:random`** is libkore's seeded ChaCha generator.
//!
//! **Async without a bridge.** The `run` export is async, so the guest runs
//! on a Wasmtime fiber. A guest blocked on a stream, a deadline or a
//! filesystem request suspends its fiber, the pending future reaches the
//! Kore executor driving [`run`], and the executor parks the task on the
//! very post the stream, timer or pair named. No worker thread, no
//! `block_on`, no second executor.
//!
//! **What is linked.** `wasi:cli` (arguments, exit, stdio, terminals),
//! `wasi:clocks`, `wasi:filesystem`, `wasi:io`, `wasi:random`. A component
//! importing anything else — `wasi:sockets` today — is refused at
//! instantiation with the missing import named, never at run time.
//!
//! **Embedding.** The host state is the upstream `wasmtime-wasi` shape: a
//! [`WasiCtx`] beside the store's [`ResourceTable`], reached through
//! [`WasiView`]. A program with interfaces of its own keeps its own store
//! data, implements [`WasiView`] for it, calls [`add_to_linker`] beside its
//! own bindings' `add_to_linker`, and runs the guest with [`run_command`].
//! [`Wasi`] is the ready-made state [`run`] and [`run_with`] use.
#![no_std]

extern crate alloc;

pub mod bindings;
mod cli;
mod clocks;
pub mod filesystem;
mod random;
pub mod stdio;

/// The stock guest fixtures, `wasm32-wasip2` programs built by `build.rs`.
#[cfg(feature = "fixture")]
pub mod fixture {
    /// `guests/hello`: every line it prints is computed from an import, and
    /// it succeeds exactly when its arguments are `kore wasi`.
    pub static HELLO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wasi-hello.wasm"));
    /// `guests/files`: creates `wasi-files.txt` in its preopen, writes a
    /// line, reads it back, finds it in the listing, removes it, and prints
    /// each result; it succeeds exactly when every step did.
    pub static FILES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wasi-files.wasm"));
}

use alloc::string::String;
use alloc::vec::Vec;

use kore::process::MappedProcessStdioChild;
use kore::runtime::Pinned;
use kore::runtime::transport::TrapTransport;
use stdio::{ProcessStreams, StdioLost};
use wasmtime::component::{Component, HasData, Linker, ResourceTable};
use wasmtime::{Engine, Store};
use wasmtime_wasi_io::IoView;

pub use cli::Exit;
pub use filesystem::Filesystem;

/// The standard streams, shared on the task's thread (see
/// [`ProcessStreams`]).
pub type Streams = Pinned<ProcessStreams<TrapTransport>>;

/// One guest's WASI context: its arguments, generator, standard streams
/// and filesystem. Everything it hands the guest lives in the store's
/// [`ResourceTable`] beside it ([`WasiCtxView`]).
pub struct WasiCtx {
    pub(crate) args: Vec<String>,
    pub(crate) rng: kore::random::SystemRng,
    /// `None`: the stream getters trap.
    pub(crate) streams: Option<Streams>,
    /// `None`: `get-directories` is empty.
    pub(crate) filesystem: Option<Filesystem>,
}

impl WasiCtx {
    /// A context whose whole argument vector is `args`, program name
    /// first; no standard streams and no filesystem yet.
    pub fn new(args: Vec<String>) -> Self {
        Self {
            args,
            rng: kore::random::SystemRng::new(),
            streams: None,
            filesystem: None,
        }
    }

    /// The guest's standard streams.
    pub fn with_streams(mut self, streams: Streams) -> Self {
        self.streams = Some(streams);
        self
    }

    /// The guest's filesystem: `filesystem`'s root, as the preopen `.`.
    pub fn with_filesystem(mut self, filesystem: Filesystem) -> Self {
        self.filesystem = Some(filesystem);
        self
    }

    /// Take the standard streams back — to close them
    /// ([`ProcessStreams::close`]) once the guest's table is dropped.
    pub fn take_streams(&mut self) -> Option<Streams> {
        self.streams.take()
    }

    /// Take the filesystem back — to close it ([`Filesystem::close`]) once
    /// the guest's table is dropped.
    pub fn take_filesystem(&mut self) -> Option<Filesystem> {
        self.filesystem.take()
    }

    /// End the guest's access once its resource table is dropped: close
    /// every view its descriptors held (the face itself is dropped;
    /// [`take_filesystem`](Self::take_filesystem) first to keep it), then
    /// close the standard streams, after any line an outstanding input
    /// request still owes.
    pub async fn close(&mut self) -> Result<(), StdioLost> {
        if let Some(filesystem) = self.filesystem.take() {
            drop(filesystem.close().await);
        }
        match self.streams.take().map(Pinned::into_inner) {
            Some(Ok(streams)) => streams.close().await,
            _ => Ok(()),
        }
    }

    fn streams(&self) -> wasmtime::Result<&ProcessStreams<TrapTransport>> {
        self.streams
            .as_ref()
            .and_then(Pinned::get)
            .ok_or_else(|| wasmtime::format_err!("this guest has no process stream"))
    }
}

/// What every WASI host call works on: the context, and the table its
/// resources live in.
pub struct WasiCtxView<'a> {
    pub ctx: &'a mut WasiCtx,
    pub table: &'a mut ResourceTable,
}

/// Store data that carries a WASI guest. `IoView` is the table `wasi:io`
/// reaches its streams through; [`wasi`](Self::wasi) is the same table with
/// the context beside it.
pub trait WasiView: IoView + Send {
    fn wasi(&mut self) -> WasiCtxView<'_>;
}

/// The host traits' data: every interface is implemented on
/// [`WasiCtxView`].
struct HasWasi;

impl HasData for HasWasi {
    type Data<'a> = WasiCtxView<'a>;
}

/// The ready-made store data: a table and a context.
pub struct Wasi {
    pub table: ResourceTable,
    pub ctx: WasiCtx,
}

impl Wasi {
    pub fn new(ctx: WasiCtx) -> Self {
        Self {
            table: ResourceTable::new(),
            ctx,
        }
    }
}

impl IoView for Wasi {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

impl WasiView for Wasi {
    fn wasi(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

/// Add every interface Kore implements to `linker`, beside whatever the
/// embedder adds for its own.
pub fn add_to_linker<T: WasiView + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
    use bindings::wasi::{cli, clocks, filesystem, random};
    let l = linker;
    wasmtime_wasi_io::add_to_linker_async(l)?;
    cli::environment::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::exit::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::stdin::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::stdout::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::stderr::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::terminal_input::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::terminal_output::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::terminal_stdin::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::terminal_stdout::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    cli::terminal_stderr::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    clocks::monotonic_clock::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    clocks::wall_clock::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    filesystem::types::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    filesystem::preopens::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    random::random::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    random::insecure::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    random::insecure_seed::add_to_linker::<T, HasWasi>(l, T::wasi)?;
    Ok(())
}

/// How a guest's run ended.
#[derive(Debug)]
pub enum Outcome {
    /// `run` returned, or the guest called `exit`: its status.
    Exited(i32),
    /// The guest trapped, or an import refused.
    Trapped(wasmtime::Error),
    /// The process stream failed; the guest's output may be incomplete.
    StdioLost(StdioLost),
}

/// Why a guest could not start.
#[derive(Debug)]
pub enum RunError {
    /// The task has no runtime identity to pin its streams to.
    NoRuntime,
    /// The component imports something the linker does not provide, or
    /// linking failed.
    Instantiate(wasmtime::Error),
}

/// Instantiate `component` as a `wasi:cli/command` in `store` and run it
/// to completion — the runner for an embedder with store data of its own.
///
/// Nothing is closed here: the store still holds the guest's resources.
/// When the embedder is done with the store, it drops the store's
/// [`ResourceTable`] and then closes the context ([`WasiCtx::close`], or
/// [`WasiCtx::take_filesystem`] and [`WasiCtx::take_streams`] to keep
/// either). Await this on the Kore executor: the guest's waits park it.
pub async fn run_command<T: WasiView + 'static>(
    store: &mut Store<T>,
    component: &Component,
    linker: &Linker<T>,
) -> Result<Outcome, RunError> {
    let command = bindings::Command::instantiate_async(&mut *store, component, linker)
        .await
        .map_err(RunError::Instantiate)?;
    Ok(match command.wasi_cli_run().call_run(&mut *store).await {
        Ok(Ok(())) => Outcome::Exited(0),
        Ok(Err(())) => Outcome::Exited(1),
        Err(error) => match error.downcast_ref::<Exit>() {
            Some(Exit(status)) => Outcome::Exited(*status),
            None => Outcome::Trapped(error),
        },
    })
}

/// Run `component` as a `wasi:cli/command` with `ctx`, then close what the
/// guest held: its resources, the filesystem's views (the face is
/// dropped), and the standard streams.
pub async fn run_with(
    engine: &Engine,
    component: &Component,
    ctx: WasiCtx,
) -> Result<Outcome, RunError> {
    let mut linker = Linker::new(engine);
    add_to_linker(&mut linker).map_err(RunError::Instantiate)?;
    let mut store = Store::new(engine, Wasi::new(ctx));
    let outcome = run_command(&mut store, component, &linker).await;

    // The guest's resources die with its table — descriptors queue their
    // views' closes, stream resources release the endpoint — and then
    // the context closes, the stream after any line an outstanding input
    // request still owes.
    let Wasi { table, mut ctx } = store.into_data();
    drop(table);
    let closed = ctx.close().await;
    let outcome = outcome?;
    Ok(match closed {
        Err(lost) if !matches!(outcome, Outcome::Trapped(_)) => Outcome::StdioLost(lost),
        _ => outcome,
    })
}

/// Instantiate `component` as a `wasi:cli/command` and run it to completion,
/// its standard streams on `stdio` and no filesystem; then close the stream.
///
/// `args` is the guest's whole argument vector, program name first. Await
/// this on the Kore executor: the guest's waits park it. For a filesystem,
/// build a [`WasiCtx`] and use [`run_with`].
pub async fn run(
    engine: &Engine,
    component: &Component,
    args: Vec<String>,
    stdio: MappedProcessStdioChild,
) -> Result<Outcome, RunError> {
    let streams = Pinned::new(ProcessStreams::new(stdio)).map_err(|_| RunError::NoRuntime)?;
    run_with(engine, component, WasiCtx::new(args).with_streams(streams)).await
}

#[cfg(all(test, feature = "fixture"))]
mod tests {
    use super::*;

    fn resolves(fixture: &[u8]) -> Component {
        let engine = wasmtime::Engine::default();
        let component = Component::new(&engine, fixture).expect("fixture compiles");
        let mut linker = Linker::<Wasi>::new(&engine);
        add_to_linker(&mut linker).expect("Kore's interfaces link");
        if let Err(error) = linker.instantiate_pre(&component) {
            panic!("the fixture imports something Kore does not link: {error:?}");
        }
        component
    }

    /// Every import of a stock `wasm32-wasip2` program resolves against what
    /// Kore links, with the right types — the check instantiation performs,
    /// minus running it.
    #[test]
    fn the_stock_guest_resolves_against_kores_linker() {
        resolves(fixture::HELLO);
    }

    /// A program using `std::fs` imports `wasi:filesystem`, and those
    /// imports — async, resource drops included — resolve too.
    #[test]
    fn the_files_guest_resolves_with_the_filesystem() {
        let engine = wasmtime::Engine::default();
        let component = resolves(fixture::FILES);
        let imports: Vec<String> = component
            .component_type()
            .imports(&engine)
            .map(|(name, _)| String::from(name))
            .collect();
        for interface in ["wasi:filesystem/types@", "wasi:filesystem/preopens@"] {
            assert!(
                imports.iter().any(|name| name.starts_with(interface)),
                "the files fixture should import {interface}: {imports:?}"
            );
        }
    }

    /// An embedder's own store data carries WASI through `WasiView`.
    #[test]
    fn an_embedders_state_links_the_same_interfaces() {
        struct Embedder {
            wasi: Wasi,
            _own: u32,
        }
        impl IoView for Embedder {
            fn table(&mut self) -> &mut ResourceTable {
                &mut self.wasi.table
            }
        }
        impl WasiView for Embedder {
            fn wasi(&mut self) -> WasiCtxView<'_> {
                self.wasi.wasi()
            }
        }
        let engine = wasmtime::Engine::default();
        let component = Component::new(&engine, fixture::FILES).expect("fixture compiles");
        let mut linker = Linker::<Embedder>::new(&engine);
        add_to_linker(&mut linker).expect("Kore's interfaces link");
        linker
            .instantiate_pre(&component)
            .expect("an embedder's linker resolves the guest");
    }
}
