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
//! - **`wasi:clocks`** are `kore::time`; a deadline pollable sleeps on the
//!   Kore executor.
//! - **`wasi:random`** is libkore's seeded ChaCha generator.
//!
//! **Async without a bridge.** The `run` export is async, so the guest runs
//! on a Wasmtime fiber. A guest blocked on a stream or a deadline suspends
//! its fiber, the pending future reaches the Kore executor driving [`run`],
//! and the executor parks the task on the very post the stream or timer
//! named. No worker thread, no `block_on`, no second executor.
//!
//! **What is linked.** `wasi:cli` (arguments, exit, stdio, terminals),
//! `wasi:clocks`, `wasi:io`, `wasi:random`. A component importing anything
//! else — `wasi:filesystem` or `wasi:sockets` today — is refused at
//! instantiation with the missing import named, never at run time.
#![no_std]

extern crate alloc;

pub mod bindings;
mod cli;
mod clocks;
mod random;
pub mod stdio;

/// The stock guest fixture: `guests/hello`, a `wasm32-wasip2` program built by
/// `build.rs`. Every line it prints is computed from an import, and it
/// succeeds exactly when its arguments are `kore wasi`.
#[cfg(feature = "fixture")]
pub mod fixture {
    pub static HELLO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wasi-hello.wasm"));
}

use alloc::string::String;
use alloc::vec::Vec;

use kore::process::MappedProcessStdioChild;
use kore::runtime::Pinned;
use kore::runtime::transport::TrapTransport;
use stdio::{ProcessStreams, StdioLost};
use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::{Engine, Store};
use wasmtime_wasi_io::IoView;

pub use cli::Exit;

/// The standard streams, shared on the task's thread (see
/// [`ProcessStreams`]). `None` in a store built without a process stream:
/// its stream getters trap.
type Streams = Option<Pinned<ProcessStreams<TrapTransport>>>;

/// One guest's host state: its resources, arguments, generator and streams.
pub struct Wasi {
    table: ResourceTable,
    args: Vec<String>,
    rng: kore::random::SystemRng,
    streams: Streams,
}

impl Wasi {
    /// Host state for one guest: `args` is its whole argument vector,
    /// program name first; `streams` its standard streams.
    pub fn new(args: Vec<String>, streams: Streams) -> Self {
        Self {
            table: ResourceTable::new(),
            args,
            rng: kore::random::SystemRng::new(),
            streams,
        }
    }

    fn streams(&self) -> wasmtime::Result<&ProcessStreams<TrapTransport>> {
        self.streams
            .as_ref()
            .and_then(Pinned::get)
            .ok_or_else(|| wasmtime::format_err!("this guest has no process stream"))
    }
}

impl IoView for Wasi {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

/// Add every interface Kore implements to `linker`.
pub fn add_to_linker(linker: &mut Linker<Wasi>) -> wasmtime::Result<()> {
    use bindings::wasi::{cli, clocks, random};
    type Host = HasSelf<Wasi>;
    fn this(state: &mut Wasi) -> &mut Wasi {
        state
    }

    wasmtime_wasi_io::add_to_linker_async(linker)?;
    cli::environment::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::exit::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::stdin::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::stdout::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::stderr::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::terminal_input::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::terminal_output::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::terminal_stdin::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::terminal_stdout::add_to_linker::<Wasi, Host>(linker, this)?;
    cli::terminal_stderr::add_to_linker::<Wasi, Host>(linker, this)?;
    clocks::monotonic_clock::add_to_linker::<Wasi, Host>(linker, this)?;
    clocks::wall_clock::add_to_linker::<Wasi, Host>(linker, this)?;
    random::random::add_to_linker::<Wasi, Host>(linker, this)?;
    random::insecure::add_to_linker::<Wasi, Host>(linker, this)?;
    random::insecure_seed::add_to_linker::<Wasi, Host>(linker, this)?;
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
    /// The component imports something Kore does not link, or linking
    /// failed.
    Instantiate(wasmtime::Error),
}

/// Instantiate `component` as a `wasi:cli/command` and run it to completion,
/// its standard streams on `stdio`; then close the stream.
///
/// `args` is the guest's whole argument vector, program name first. Await
/// this on the Kore executor: the guest's waits park it.
pub async fn run(
    engine: &Engine,
    component: &Component,
    args: Vec<String>,
    stdio: MappedProcessStdioChild,
) -> Result<Outcome, RunError> {
    let streams = Pinned::new(ProcessStreams::new(stdio)).map_err(|_| RunError::NoRuntime)?;
    let mut linker = Linker::new(engine);
    add_to_linker(&mut linker).map_err(RunError::Instantiate)?;
    let mut store = Store::new(engine, Wasi::new(args, Some(streams)));
    let command = bindings::Command::instantiate_async(&mut store, component, &linker)
        .await
        .map_err(RunError::Instantiate)?;
    let outcome = match command.wasi_cli_run().call_run(&mut store).await {
        Ok(Ok(())) => Outcome::Exited(0),
        Ok(Err(())) => Outcome::Exited(1),
        Err(error) => match error.downcast_ref::<Exit>() {
            Some(Exit(status)) => Outcome::Exited(*status),
            None => Outcome::Trapped(error),
        },
    };

    // The guest's stream resources die with its table; then the stream is
    // closed, after any line an outstanding input request still owes.
    let Wasi { streams, table, .. } = store.into_data();
    drop(table);
    let closed = match streams.map(Pinned::into_inner) {
        Some(Ok(streams)) => streams.close().await,
        _ => Ok(()),
    };
    Ok(match closed {
        Err(lost) if !matches!(outcome, Outcome::Trapped(_)) => Outcome::StdioLost(lost),
        _ => outcome,
    })
}

#[cfg(all(test, feature = "fixture"))]
mod tests {
    use super::*;

    /// Every import of a stock `wasm32-wasip2` program resolves against what
    /// Kore links, with the right types — the check instantiation performs,
    /// minus running it.
    #[test]
    fn the_stock_guest_resolves_against_kores_linker() {
        let engine = wasmtime::Engine::default();
        let component = Component::new(&engine, fixture::HELLO).expect("fixture compiles");
        let mut linker = Linker::<Wasi>::new(&engine);
        add_to_linker(&mut linker).expect("Kore's interfaces link");
        if let Err(error) = linker.instantiate_pre(&component) {
            panic!("the fixture imports something Kore does not link: {error:?}");
        }
    }
}
