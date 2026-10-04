# Pack-backed reads in the Lua surface

**Status: S-L0 through S-L3 and S-L6 are implemented; S-L4 and S-L5 are not.** The
implemented slices are described in the past tense below and their contract
lives in `filesystem.md`; everything from S-L4 on is still aspirational. No
part of this has run on metal — §7's acceptance is unmet until S-L5.

## 1. The question

Is the filesystem API the Lua surface uses compatible with the DagFS upgrade in
[MicroPerceptron/kore#1064][pr]?

**Yes, at the source level — and that is the problem.** It compiles unchanged,
runs unchanged, and reaches the pack-backed path exactly never.

[pr]: https://github.com/MicroPerceptron/kore/pull/1064

### Evidence

`#1064` is `ft/pack-tickets`, open, head `9b8c5d672` on main `fe9406151`.
`native/` here is pinned at Kore `9e4172bc2`. Syncing a scratch copy of this
repo to the PR head with `tools/sync-native.py` and compiling gives:

| Crate | Result |
|---|---|
| `kore-std --features fs` | clean |
| `kore-lua --all-features` | clean |
| `kore-ffi --all-features` | clean |

No owner package is added or removed — `kore-ring` was already in the export
closure — and the dagfs change is purely additive at the `Fs` facade:
`pub mod stor`, `RING_TARGET_RIGHTS`, `Fs::set_direct_polling`, and three
private fields on `Fs`. Every verb `crates/lua` calls keeps its signature.

## 2. Why it is inert

The ring engages on a size test. `Fs::read_at` streams through the facade ring
only when the destination is at least `FACADE_MIN_BYTES` (256 KiB); below that
it takes the pair's window exactly as today, and `read_range_by_ring` answers
`Ok(None)` with the note `"small read"`.

The Lua surface never presents a destination that large:

- `kore_lua.c` gives each open file a `KORE_LUA_BUFSIZE` (**4096**) read-ahead
  buffer, and `kore_lua_file_fill` asks the bridge for exactly that much.
- `f:read("a")` (`kore_lua_read_all`) and `f:read(n)` (`kore_lua_read_count`)
  both loop over `kore_lua_file_fill`, so they are 4 KiB at a time as well.
- `kore.fs.read_file` was built on `kore_lua_read_all`, so it was 4 KiB at a
  time too. (`io.lines(name)` reads *lines*, and `require` reads a module
  through `Loader::read_source` into a 64 KiB buffer — neither is a whole-file
  read and neither reaches the threshold. An earlier draft of this plan
  claimed both; it was wrong.)
- `crates/lua/src/files.rs::read` therefore allocates a 4096-byte `Vec` and
  calls `Fs::read_at` with it. `MAX_READ_BYTES` (1 MiB) is a ceiling the
  C side never approaches.

A 100 MiB file read by `kore.fs.read_file` is ~25,600 bridge round trips and
~25,600 pair round trips, before and after #1064 alike.

Exec does fund a ring for us: `augment_stor_ring_plan` runs for every launch
with `filesystem_requested`, moving `role::STOR_RING` and a 256 KiB
`role::STOR_TARGET` into the child. A Lua task gets a ring at founding and then
never submits to it.

## 3. Why it becomes a break, not just a missed win

`res/STORAGE_TRANSPORT.md` §9 lands **S8: "write grants and write consumers; the
window-copy verbs and `provider/block.rs` deleted."** The 4 KiB window verbs the
Lua surface is built on are scheduled for deletion. Staying chunk-at-a-time is
not a decision to forgo throughput; it is a decision to be ported under duress
later, by whoever is holding S8.

The same section rules out the lazy fix: *"`read_chunk` stays on the window: its
contract is one window-sized chunk per call, so a ring would have to stay
attached between calls — the lifetime §12's kernel fix must make safe first."*
A ring cannot be held open across a 4 KiB read-ahead refill. The buffer shape
has to change; there is no transparent option.

## 4. Three semantic deltas to absorb

Independent of the ring, #1064 changes what a refusal means.

1. **`read_at` no longer leaves the destination unchanged on refusal.** Its doc
   moves from *"The destination is unchanged on refusal"* to *"On refusal the
   destination may hold a prefix of the bytes."* `src/fs.rs::read_exact_at`
   documents `TransferError::completed` as the initialized prefix; once a ring
   read breaks off mid-stream that count undercounts what was written into
   `dst`. `docs/filesystem.md` ("Read errors report an initialized prefix")
   stops being true as written.
2. **`files.rs::read` discards its partial prefix.** The loop answers
   `refusal(error)` and drops `filled`. That was correct while a refusal left
   the destination untouched; it now throws away bytes that were delivered.
3. **No version is pinned across a Lua read.** `read_range_by_ring` pins one
   version per call. Lua's loop issues an independent call per 4 KiB, so
   `f:read("a")` can interleave bytes from several revisions under a concurrent
   writer. This hazard exists today on the window path; the point is that
   #1064 introduces the mechanism that would fix it, and the chunk loop is what
   stops us taking it.

## 5. The seam already exists

The facade ring is internal to `Fs`. Nothing here needs to name
`kore_dagfs::stor`, hold a `StorRing`, or manage a grant. Two verbs are the
whole interface, and **both already exist at the pinned revision with the exact
signatures #1064 keeps**:

```rust
pub async fn read_file<F>(&mut self, path: &[u8], consume: F) -> Result<(), Refusal>
    where F: FnMut(&[u8]) -> Result<(), Refusal>;   // S5d: whole file, by path
pub async fn read_at(&mut self, file: &View, offset: u64, dst: &mut [u8])
    -> Result<usize, Refusal>;                      // S5e: dst >= 256 KiB
```

So the work below is buildable, testable and mergeable **today against
`9e4172bc2`**, with no feature gate and no commented-out body. It is slower than
today's code by nothing and faster by nothing until `native/` syncs past #1064,
at which point the same source becomes pack-backed with no further edit.

That is what a seam should buy. The alternative — commented-out function bodies
— cannot be type-checked, cannot be tested, and rots against the API it is
waiting for; AGENTS.md already rules that "empty headers, placeholder allocators
and mock implementations are not completed language APIs."

The only genuinely unbuildable surface is observability (§6, S-L4), because
`Fs::facade_ring_note` and `Fs::facade_ring_files` do not exist before #1064.
That one gets a `#[cfg(feature = "pack-ring")]`, off by default — invisible to
the default build like a comment, but compiled in CI against a synced tree.

## 6. Slices

Each is its own PR, green on its own, in order. S-L1 through S-L3 need no
upstream change.

### S-L0 — guard the claim *(done)*
`kore_lua_read_geometry` exports the sizes the engine actually compiled, and
`streaming_smoke` in `tests/host_smoke.c` asserts them before anything else.
Three guards, because the failure is silent otherwise:

- the engine's compiled `KORE_LUA_BUFSIZE` (4096) and
  `KORE_LUA_FACADE_STREAM_MIN` (256 KiB), checked from the harness;
- `const _: () = assert!(...)` in `files.rs` binding `MAX_READ_BYTES` and
  `MAX_FILE_BYTES` to `FACADE_STREAM_BYTES`, so a cap lowered for an unrelated
  reason fails the build rather than the next read;
- the store records the largest destination any read asked for
  (`host_fs_max_read_want`), and the suite asserts a large read reached the
  threshold and a small one did not. That is the only thing a host build can
  observe about streaming — there is no device under it — and it is the
  property that was actually wrong.

Verified by negative control: stubbing `kore_lua_file_streams` to return 0
fails the suite at that assertion.

### S-L1 — a whole-file verb that is one call *(done)*
`OP_FS_READ_FILE` (30, past `OP_PROC_EXIT` rather than recycling the retired
22). `kore_lua_fs_read_file` is no longer open + `read_all` + close in C; it is
one bridge op, served on thread 0 by `files::read_file` over
`Fs::read_file(path, consume)`, appending into a `Vec` grown by `try_reserve`.

The cap went the recommended way: `MAX_FILE_BYTES` = 16 MiB, separate from
`MAX_READ_BYTES`, refused as **`range`** rather than `out-of-storage` — the
file is real and readable, and it is having all of it at once that is refused;
`out-of-storage` stays for an actual allocation failure, as in `files::read`.
`list` already spells an exceeded reply cap this way.

Reach is narrower than this plan first claimed: `kore.fs.read_file` and
anything built on it. See §2.

### S-L2 — large reads on the handle path *(done)*
Keep the 4 KiB inline buffer for interactive and small reads; bypass it when the
want is large.

- `kore_lua_file_stream` asks the host for a whole span straight into the
  `luaL_Buffer`; `kore_lua_file_streams` decides when, and both
  `kore_lua_read_count` and `kore_lua_read_all` go through them. A short answer
  is not an error — the loop continues — and once the remainder falls below the
  threshold the page buffer takes over again, so both halves run in one read.
- `KORE_LUA_STREAM_MIN` defaults to `KORE_LUA_FACADE_STREAM_MIN` but is
  separately `#define`d and overridable, so a test can lower the ask without
  lowering what the engine believes the facade promises.
- `kore_lua_read_all`: `open` already answers the file length beside the
  handle, but today it is only used to seed an append position and is then
  dropped. Carry it into `kore_lua_file` so `read_all` can ask for the whole
  remainder in one op rather than looping. It is a hint, not an authority — a
  short answer still ends the read, and a file that grew still needs the loop
  as a backstop.
- Do **not** grow `KORE_LUA_BUFSIZE`. It is a fixed array in every
  `kore_lua_file`, and `MAX_OPEN_FILES` is 256; a 256 KiB buffer would be
  64 MiB of per-VM arrays to serve a read pattern most files never use.

`files.rs::read` needs no change beyond S-L3: a larger `count` already flows
into a larger `dst`.

### S-L3 — partial-prefix semantics *(done)*
Make the three deltas in §4 true statements.

- `files.rs::read`: on refusal with `filled > 0` it answers the prefix, as
  `finish_write` already did for a short write. The caller's next call starts
  at `filled == 0` and meets the same refusal, so nothing is hidden — only
  deferred past the bytes that are real. Not covered by a test: `files::read`
  needs a live `Fs`, and this repo has no mock for one. The C-side half of the
  contract (a refusal after a partial read) is covered by the existing
  `read_failure_suite`.
- `src/fs.rs`: `TransferError::completed` is now documented as the
  acknowledged prefix on *both* directions — the distinction `write_all_at`
  already drew — with storage past it called clobbered rather than retained.
- `docs/filesystem.md`: the transfer-and-failure section says the same, and
  states the 256 KiB threshold as a property of the destination's size.

### S-L6 — the write path *(done)*
Not pack-backed — S8's write grants have not landed upstream, and nothing
here streams a write. Three defects the read work made visible, fixed with
what the store already owns:

- **`kore.fs.write_file` committed once per window.** It was open + a
  positional write + close, and `files::write` loops `Fs::write_at`, each
  call its own commit — one image/flush/root/flush chain per window of a
  save. It is now `OP_FS_SAVE` (31) over `Fs::write_bytes` under
  `hold_commits`, which is the store's own `BeginReplace` transaction: one
  commit for the save however many windows it takes. The release *is* the
  commit, so a refused release is the caller's answer even when every byte
  was acknowledged.
- **The open no longer truncates.** The replacement sets the whole
  contents, so truncating first was a wasted round trip — and it meant a
  save that then failed had already destroyed the previous contents. They
  now survive a refusal intact.
- **`f:write` guessed at a short write.** A short acknowledgement is a fact,
  not a refusal, and the old code turned any of them into `out-of-storage`
  — a reason it did not know. It now asks again from where the store
  stopped, so a real refusal arrives in the store's own vocabulary on the
  next call. Only a store that acknowledges *nothing* without refusing has
  no reason to give, and that is the one case spelled in C.

**Stated gap:** `kore_dagfs::Fs::replace_file` also publishes the index for
what it wrote, and `OP_FS_SAVE` does not. That verb is behind kore-dagfs's
`index` feature, which pulls `kore-svc-extract-worker-rust`; this crate
enables only `runtime`. Turning it on is an image-size decision for the Lua
target, not a side effect of this slice. Until then a save leaves the index
stale, and a later `CurrentOnly` search reports `IndexIncomplete` — the
designed signal, but worth knowing it is reachable from here.

### S-L4 — the witness (`pack-ring` feature, off by default)
A `kore.fs` introspection verb answering `Fs::facade_ring_note()` and
`Fs::facade_ring_files()`, so a Lua keytest can assert *"this read streamed"*
rather than inferring it from a stopwatch. `#[cfg(feature = "pack-ring")]`; CI
builds it against a tree synced to #1064, so it cannot rot.

### S-L5 — sync, flip, measure
Once #1064 merges: `tools/sync-native.py` to the merge commit, both `Cargo.lock`
files, `make fmt test check docs owners cross miri bindings-check`,
`tools/check-local.py`. Then enable `pack-ring` by default and measure on metal.
Per §4 of the doctrine this is the only acceptance criterion that counts —
simulation and argument are provisional.

## 7. Acceptance

- `kore.fs.read_file` on a file of at least 256 KiB reports
  `facade_ring_note() == "streamed"`.
- `f:read("a")` over the same file issues one `OP_FS_READ` bridge op, not
  *n*/4096 of them.
- A read refused mid-stream answers its delivered prefix, and the count it
  answers is the count the script can trust.
- Measured decode-path read throughput on Gigatron, against the same file read
  before the change. Not a benchmark on a host build.

## 8. What this plan does not do

It does not make writes pack-backed. S8 deletes the window-copy write verbs
too, and write grants are a separate upstream slice that has not landed;
positional writes stay chunk-at-a-time until they do. S-L6 fixes what could be
fixed without them — the commit count, the truncation, and the short-write
guess — and `OP_FS_SAVE` is the seam a write grant lands behind when it comes.

It does not add a Lua-visible ring, grant or target API — the facade owns
those, and exposing them would be a second authority over one ring.

It does not enable kore-dagfs's `index` feature. See S-L6's stated gap.
