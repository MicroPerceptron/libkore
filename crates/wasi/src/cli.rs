//! `wasi:cli` — arguments, exit, the standard streams, terminals.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use wasmtime::component::Resource;
use wasmtime_wasi_io::streams::{DynInputStream, DynOutputStream};

use crate::Wasi;
use crate::bindings::wasi::cli::{
    environment, exit, stderr, stdin, stdout, terminal_input, terminal_output, terminal_stderr,
    terminal_stdin, terminal_stdout,
};

/// The status a guest asked to exit with, carried out of `run` as its error.
///
/// Upstream `wasmtime-wasi` spells it `I32Exit`; the runner downcasts it to
/// tell a requested exit from a trap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exit(pub i32);

impl fmt::Display for Exit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exit status {}", self.0)
    }
}

impl core::error::Error for Exit {}

/// A stream resource could not be pinned to this thread: the runtime is not
/// up. Unreachable once `run` has pinned the streams themselves.
fn no_runtime() -> wasmtime::Error {
    wasmtime::format_err!("no runtime identity to pin a stream to")
}

impl environment::Host for Wasi {
    /// kore has no environment block: a process's context is its
    /// capabilities, not ambient variables.
    fn get_environment(&mut self) -> wasmtime::Result<Vec<(String, String)>> {
        Ok(Vec::new())
    }

    fn get_arguments(&mut self) -> wasmtime::Result<Vec<String>> {
        Ok(self.args.clone())
    }

    /// No ambient working directory either; files arrive as preopens.
    fn initial_cwd(&mut self) -> wasmtime::Result<Option<String>> {
        Ok(None)
    }
}

impl exit::Host for Wasi {
    fn exit(&mut self, status: Result<(), ()>) -> wasmtime::Result<()> {
        Err(wasmtime::format_err!(Exit(if status.is_ok() {
            0
        } else {
            1
        })))
    }

    fn exit_with_code(&mut self, status_code: u8) -> wasmtime::Result<()> {
        Err(wasmtime::format_err!(Exit(i32::from(status_code))))
    }
}

impl stdin::Host for Wasi {
    fn get_stdin(&mut self) -> wasmtime::Result<Resource<DynInputStream>> {
        let stream = self.streams()?.stdin().ok_or_else(no_runtime)?;
        Ok(self.table.push(stream)?)
    }
}

impl stdout::Host for Wasi {
    fn get_stdout(&mut self) -> wasmtime::Result<Resource<DynOutputStream>> {
        let stream = self.streams()?.stdout().ok_or_else(no_runtime)?;
        Ok(self.table.push(stream)?)
    }
}

impl stderr::Host for Wasi {
    fn get_stderr(&mut self) -> wasmtime::Result<Resource<DynOutputStream>> {
        let stream = self.streams()?.stderr().ok_or_else(no_runtime)?;
        Ok(self.table.push(stream)?)
    }
}

// The process stream is a pipe to the launcher, which does the terminal's
// line discipline itself: the guest is not attached to a terminal, and says
// so, the way a program with redirected stdio does.
impl terminal_input::Host for Wasi {}
impl terminal_input::HostTerminalInput for Wasi {
    fn drop(&mut self, r: Resource<terminal_input::TerminalInput>) -> wasmtime::Result<()> {
        self.table.delete(r)?;
        Ok(())
    }
}
impl terminal_output::Host for Wasi {}
impl terminal_output::HostTerminalOutput for Wasi {
    fn drop(&mut self, r: Resource<terminal_output::TerminalOutput>) -> wasmtime::Result<()> {
        self.table.delete(r)?;
        Ok(())
    }
}
impl terminal_stdin::Host for Wasi {
    fn get_terminal_stdin(
        &mut self,
    ) -> wasmtime::Result<Option<Resource<terminal_input::TerminalInput>>> {
        Ok(None)
    }
}
impl terminal_stdout::Host for Wasi {
    fn get_terminal_stdout(
        &mut self,
    ) -> wasmtime::Result<Option<Resource<terminal_output::TerminalOutput>>> {
        Ok(None)
    }
}
impl terminal_stderr::Host for Wasi {
    fn get_terminal_stderr(
        &mut self,
    ) -> wasmtime::Result<Option<Resource<terminal_output::TerminalOutput>>> {
        Ok(None)
    }
}
