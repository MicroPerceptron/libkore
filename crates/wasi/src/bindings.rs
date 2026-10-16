//! Host bindings for the `wasi:cli/command` world, generated from the
//! vendored WASI 0.2 WIT (`wit/`), in the shape upstream `wasmtime-wasi`
//! generates them: `wasi:io` is `wasmtime-wasi-io`'s (its streams and
//! pollables are the traits kore implements), everything else is generated
//! here.
//!
//! Imports are trappable. `cli`, `clocks` and `random` are synchronous —
//! nothing in them blocks; what waits does so through a pollable. Every
//! `wasi:filesystem` import, resource drops included, is async: each is a
//! round trip to Kore's filesystem service, and the guest's fiber stays
//! suspended while the Kore executor parks on it. The `run` export is async,
//! so the guest runs on a fiber the kore executor drives.

wasmtime::component::bindgen!({
    path: "wit",
    world: "wasi:cli/command",
    imports: {
        "wasi:filesystem": async | trappable,
        default: trappable,
    },
    exports: { default: async },
    trappable_error_type: {
        "wasi:io/streams.stream-error" => wasmtime_wasi_io::streams::StreamError,
        "wasi:filesystem/types.error-code" => crate::filesystem::FsError,
    },
    with: {
        "wasi:io/poll": wasmtime_wasi_io::bindings::wasi::io::poll,
        "wasi:io/streams": wasmtime_wasi_io::bindings::wasi::io::streams,
        "wasi:io/error": wasmtime_wasi_io::bindings::wasi::io::error,
        "wasi:filesystem/types.descriptor": crate::filesystem::Descriptor,
        "wasi:filesystem/types.directory-entry-stream": crate::filesystem::DirectoryEntryStream,
        "wasi:cli/terminal-input.terminal-input": crate::cli::TerminalInput,
        "wasi:cli/terminal-output.terminal-output": crate::cli::TerminalOutput,
    },
    require_store_data_send: true,
});
