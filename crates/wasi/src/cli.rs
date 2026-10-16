//! `wasi:cli` — arguments, exit, the standard streams, terminals.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use wasmtime::component::Resource;
use wasmtime_wasi_io::streams::{DynInputStream, DynOutputStream};

use crate::WasiCtxView;
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

impl environment::Host for WasiCtxView<'_> {
    /// kore has no environment block: a process's context is its
    /// capabilities, not ambient variables.
    fn get_environment(&mut self) -> wasmtime::Result<Vec<(String, String)>> {
        Ok(Vec::new())
    }

    fn get_arguments(&mut self) -> wasmtime::Result<Vec<String>> {
        Ok(self.ctx.args.clone())
    }

    /// No ambient working directory either; files arrive as preopens.
    fn initial_cwd(&mut self) -> wasmtime::Result<Option<String>> {
        Ok(None)
    }
}

impl exit::Host for WasiCtxView<'_> {
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

impl stdin::Host for WasiCtxView<'_> {
    fn get_stdin(&mut self) -> wasmtime::Result<Resource<DynInputStream>> {
        let stream = self.ctx.streams()?.stdin().ok_or_else(no_runtime)?;
        Ok(self.table.push(stream)?)
    }
}

impl stdout::Host for WasiCtxView<'_> {
    fn get_stdout(&mut self) -> wasmtime::Result<Resource<DynOutputStream>> {
        let stream = self.ctx.streams()?.stdout().ok_or_else(no_runtime)?;
        Ok(self.table.push(stream)?)
    }
}

impl stderr::Host for WasiCtxView<'_> {
    fn get_stderr(&mut self) -> wasmtime::Result<Resource<DynOutputStream>> {
        let stream = self.ctx.streams()?.stderr().ok_or_else(no_runtime)?;
        Ok(self.table.push(stream)?)
    }
}

/// A guest's handle on its terminal input. The interface has no methods: holding
/// one *is* the answer to "is stdin a terminal?".
pub struct TerminalInput;

/// A guest's handle on a terminal output, likewise only its own answer.
pub struct TerminalOutput;

// The launcher does the line discipline: a terminal's stream arrives a line at
// a time, already edited and echoed. Whether its far end is a terminal is the
// embedder's declaration (`WasiCtx::with_terminal`): Kore's shell declares its
// interactive relay one, and a launcher that captures the output does not, so
// the guest is told the truth either way. All three streams share the one
// process stream, so they answer together.
impl terminal_input::Host for WasiCtxView<'_> {}
impl terminal_input::HostTerminalInput for WasiCtxView<'_> {
    fn drop(&mut self, r: Resource<TerminalInput>) -> wasmtime::Result<()> {
        self.table.delete(r)?;
        Ok(())
    }
}
impl terminal_output::Host for WasiCtxView<'_> {}
impl terminal_output::HostTerminalOutput for WasiCtxView<'_> {
    fn drop(&mut self, r: Resource<TerminalOutput>) -> wasmtime::Result<()> {
        self.table.delete(r)?;
        Ok(())
    }
}
impl terminal_stdin::Host for WasiCtxView<'_> {
    fn get_terminal_stdin(&mut self) -> wasmtime::Result<Option<Resource<TerminalInput>>> {
        if !self.ctx.terminal {
            return Ok(None);
        }
        Ok(Some(self.table.push(TerminalInput)?))
    }
}
impl terminal_stdout::Host for WasiCtxView<'_> {
    fn get_terminal_stdout(&mut self) -> wasmtime::Result<Option<Resource<TerminalOutput>>> {
        self.terminal_output()
    }
}
impl terminal_stderr::Host for WasiCtxView<'_> {
    fn get_terminal_stderr(&mut self) -> wasmtime::Result<Option<Resource<TerminalOutput>>> {
        self.terminal_output()
    }
}

impl WasiCtxView<'_> {
    /// stdout's or stderr's terminal, when the streams are a terminal's.
    fn terminal_output(&mut self) -> wasmtime::Result<Option<Resource<TerminalOutput>>> {
        if !self.ctx.terminal {
            return Ok(None);
        }
        Ok(Some(self.table.push(TerminalOutput)?))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::{Wasi, WasiCtx, WasiView};

    /// Each standard stream answers "is this a terminal?" exactly as the
    /// embedder declared, and a terminal handle the guest drops leaves the
    /// table.
    #[test]
    fn the_guest_is_told_the_terminal_its_embedder_declared() {
        for terminal in [false, true] {
            let mut wasi = Wasi::new(WasiCtx::new(Vec::new()).with_terminal(terminal));
            let mut view = wasi.wasi();
            let stdin = terminal_stdin::Host::get_terminal_stdin(&mut view).unwrap();
            let stdout = terminal_stdout::Host::get_terminal_stdout(&mut view).unwrap();
            let stderr = terminal_stderr::Host::get_terminal_stderr(&mut view).unwrap();
            assert_eq!(
                (stdin.is_some(), stdout.is_some(), stderr.is_some()),
                (terminal, terminal, terminal)
            );
            if let (Some(stdin), Some(stdout), Some(stderr)) = (stdin, stdout, stderr) {
                terminal_input::HostTerminalInput::drop(&mut view, stdin).unwrap();
                terminal_output::HostTerminalOutput::drop(&mut view, stdout).unwrap();
                terminal_output::HostTerminalOutput::drop(&mut view, stderr).unwrap();
            }
            assert!(wasi.table.is_empty());
        }
    }
}
