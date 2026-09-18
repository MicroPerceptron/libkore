//! Host bindings for the `wasi:cli/command` world, generated from the
//! vendored WASI 0.2 WIT (`wit/`), in the shape upstream `wasmtime-wasi`
//! generates them: `wasi:io` is `wasmtime-wasi-io`'s (its streams and
//! pollables are the traits kore implements), everything else is generated
//! here. Imports are synchronous and trappable — nothing in `cli`, `clocks`
//! or `random` blocks; what waits does so through a pollable — and the
//! `run` export is async, so the guest runs on a fiber the kore executor
//! drives.

wasmtime::component::bindgen!({
    path: "wit",
    world: "wasi:cli/command",
    imports: { default: trappable },
    exports: { default: async },
    trappable_error_type: {
        "wasi:io/streams.stream-error" => wasmtime_wasi_io::streams::StreamError,
    },
    with: {
        "wasi:io/poll": wasmtime_wasi_io::bindings::wasi::io::poll,
        "wasi:io/streams": wasmtime_wasi_io::bindings::wasi::io::streams,
        "wasi:io/error": wasmtime_wasi_io::bindings::wasi::io::error,
    },
    require_store_data_send: true,
});
