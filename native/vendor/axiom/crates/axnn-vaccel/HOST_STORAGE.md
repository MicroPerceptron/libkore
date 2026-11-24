# Host tensor backing and output bindings

`AxHetero` keeps the same `.ax` graph, placement model and `Accelerator` contract.
The execution adapter now retains typed host backing across partitions. This
change does not add device import/export or couple the portable `ax` crate to a
host surface, OS lease or graphics API.

The native compiler, fusion planner, typed storage and instruction executor now
live in [`ax-exec`](../ax-exec/README.md), shared with its `no_std` native program
API. `AxHetero` delegates native partitions to that same implementation; the
host provider/session layer itself has not become `no_std`.

## Start with automatic dispatch

```rust,ignore
let executable = engine.compile_auto_in(&graph, persistent_allocator)?;
// Bind the output you own; ordinary execution selects the optimized path.
engine.run_into_in(&executable, &inputs, &mut outputs, frame_allocator)?;
```

`compile_auto_in` reads the active accelerator's advertised TOSA target,
operators and dtype roles. It selects native CPU kernels where supported,
retains CPU TOSA for coverage gaps, and places accelerator work using the common
cost estimates. Boolean boundary bridges follow the provider's roles, with
their synthesized operators checked before admission. No CoreML/OpenVINO/XDNA
name switch or manual `cpu_tosa = false` is necessary. With no accelerator,
the same API compiles a CPU schedule. An unadvertised target is not admitted.

Ordinary compilation also selects the exact native fusions previously confined
to explicit preparation: MUL then ADD, three scaled planes then INTERLEAVE3,
and three CAST/SHIFT channels packed by OR. These apply inside mixed schedules
as well as CPU-only graphs. Partition outputs stay live; strict arithmetic
keeps operand order and separate rounding, without FMA contraction. Borrowed
ingress, retained typed storage and eligible direct provider readback are
automatic execution choices, not optimization flags. Inspect
`executable.native_plan_stats()` and `executable.schedule()` to see what ran.

Manual controls remain available: derive `engine.automatic_device_model()`,
adjust costs or policies, then call `compile_in`. Set `native_fusions = false`
for an unfused reference, or `prefer_native_cpu = false` to compare native and
CPU TOSA implementations using explicit cost calibration. Existing explicit
`AxDeviceModel::default()` models preserve that cost competition. Capability
masks, sparse policies and precision opt-ins remain available; automatic mode
does not invent ISA support or opt into reduced precision. Its costs are
heuristics, not measurements of the current host.

`PreparedNative` remains useful for a reusable arena and fixed single-native
schedule. Ordinary mixed execution now shares its fusion planner, but does not
automatically acquire its persistent arena. Allocators, resources and output
ownership must still be supplied by the caller.

For recurrent graphs, use `run_repeated_auto_in` with `AxFeedback` bindings.
It selects the existing resident provider loop when one provider partition can
keep all feedback in its buffers. Otherwise it moves retained output backing
into the next native/mixed invocation, restores non-feedback inputs and avoids
intermediate byte-vector materialization. Mixed provider boundaries still
transfer each iteration; final results use the byte-vector ABI. The selection
happens before execution, never by replaying a failed submission. Explicit
`run_repeated_in` remains available when provider residency is a requirement.

## Choose the completion interface

| API | Output ownership and copy behavior |
|---|---|
| `advance_retained_in` | Returns `AxTensor` handles. `as_bytes()` borrows the retained allocation; input passthroughs retain their input borrow. |
| `advance_into_in`, `run_into_in` | Accept packed or pitched `AxOutput` destinations. Native results write once from typed storage. Eligible terminal provider outputs read directly into the destination. |
| `advance_in`, `run_in` | Compatible owned `Vec<u8>` results. Existing byte storage moves; typed F32/I32 or borrowed results need final materialization. |
| `Session::poll_many_into` | Provider readback into caller-owned `ByteSink`s, with no intermediate host result vectors. |

For an existing compiled executable and frame backing:

```rust,ignore
let mut frame = engine.begin_in(&executable, &inputs, allocator)?;
loop {
    let mut output = [AxOutput::pitched(
        &mut mapping, offset, width * 4, height, stride,
    )?];
    match engine.advance_into_in(&mut frame, &mut output)? {
        AxInvocationStatus::Complete(()) => break,
        AxInvocationStatus::Progress(_) => { /* return to the event loop */ }
        AxInvocationStatus::Cancelled(_) => { /* do not publish */ break; }
    }
}
```

Outputs bind in `executable.output_ids()` order and carry raw graph ABI bytes.
`AxOutput` validates the range, overflow and pitch. The graph checks logical byte
lengths before advancing. It does not infer a pixel format or color space.
The complete runnable example is `tools/ax-surface-lab/src/hetero.rs`.

## What is eliminated

On little-endian hosts, aligned F32/I32 inputs and U8 inputs are borrowed by
native kernels. Unaligned F32/I32 inputs retain explicit, exact decoding. Native
outputs keep their typed vectors; provider readbacks allocate aligned typed
storage directly when the boundary representation matches. No vector is
reinterpreted with an incompatible allocation/deallocation layout.

A terminal provider can use direct output sinks when its full output set equals
the graph output set and all output transforms are identity. Boolean/precision
bridges, earlier graph outputs and passthroughs retain the general completion
path. Contiguous native kernels still need one write into a separately supplied
surface. Consumers that can use retained `AxTensor` backing avoid that write.
Big-endian execution retains explicit encoding/decoding; it was not run on the
Apple host used for this refactor.

## Lifetimes, failures and accounting

Output destinations are borrowed for an `advance_into_in` call, not pinned
across provider execution. Providers execute in their own buffers and never
retain these destinations. Publish only after `Complete(())`; provider readback
failure can leave partial bytes in a destination. Pending/failed/cancelled events
never start readback. Rejected event release preserves the pending token;
indeterminate release keeps the provider key unavailable for unsafe reuse.

An application still owns producer/consumer fencing, visibility and retirement.
This API does not grant a device mapping, lease authority or permission to reuse
an image that another consumer is reading. `MemoryDomain::Shared` is not an
external-buffer import mechanism.

`AxInvocation::host_copy_stats()` counts native ingress decoding, native boundary
encoding, and API output materialization. Its documented exclusions include
provider transfers and semantic kernel work. Instrument `write_buffer` and
`read_buffer` separately; these counters are payload lengths, not measured memory
bus traffic. The lab asserts zero native boundary copies on its aligned inputs
and reports the remaining typed-to-pitched write explicitly.
