# Lua on Kore

Lua 5.5, freestanding, over the task's own filesystem face. A script sees two
surfaces and they are one implementation: **`kore.*` is the store's vocabulary
spelled for Lua, and `io`/`os` is the same code with the capabilities
defaulted.** `io.open` and `kore.fs.open` return the *same* object; only the
failure spelling differs.

Which to reach for: `io`/`os` when a script is doing what a POSIX script does,
and `kore.*` when it needs something the store knows and POSIX cannot say — a
revision, a subtype, the rights it actually holds.

- [The capability default](#the-capability-default)
- [Capabilities in practice](#capabilities-in-practice)
- [Quick reference](#quick-reference)
- [`kore.fs`](#korefs)
- [The file object](#the-file-object)
- [`io` and `os`](#io-and-os)
- [`kore.time`](#koretime)
- [`kore.random`](#korerandom)
- [`kore.proc`](#koreproc)
- [`kore.has` and `kore.caps`](#korehas-and-korecaps)
- [`require`](#require)
- [Errors](#errors)
- [Absent, and why](#absent-and-why)
- [Not yet exposed](#not-yet-exposed)

## The capability default

DagFS authority is a right set on a route, not a mode bit on a file.

`kore.fs.open` says so directly — it takes a **rights spelling**, and
behaviour (create, truncate, append, exclusive) moves to an options table where
it belongs. The stock surface has no way to say any of that, so it says it by
convention instead: a mode string picks the rights, and the surface never asks
for more.

Both reach the same place. A mode string is translated to a spelling and a set
of options before it crosses, so this table is the whole of the convention:

| Spelling | Rights requested | On the parent |
|---|---|---|
| `"r"` | `READ` | — |
| `"w"` | `READ WRITE` + create + truncate | `READ MUTATE_NAMESPACE` to create |
| `"a"` | `READ WRITE` + create, writes at the end | `READ MUTATE_NAMESPACE` to create |
| `"r+"` | `READ WRITE`, does not create | — |
| `"w+"`, `"a+"` | `READ WRITE` + create | `READ MUTATE_NAMESPACE` to create |
| `"b"` | accepted, ignored — kore has no text mode to differ from |
| `remove`, `rename`, `mkdir` | — | `READ MUTATE_NAMESPACE` |

The *route* carries `READ` in every row; the *file object* still remembers what
the mode asked for, so `f:read` on a `"w"` file is `access-denied` rather than a
read the route would in fact have permitted. The mode governs the object, the
rights govern the route, and they are not the same question.

`kore.fs.open(path, "rw")` asks for both and gets an object that permits both,
which is the difference the two spellings exist to express.

**`READ` is always requested**, because a route that cannot read cannot take a
single step. Two rights are **never** requested by this surface: `SHARE` and
`BIND_EXECUTABLE`. That is the ceiling of everything below — no Lua script can
re-share a route or bind an executable through these verbs, whatever the task
itself holds.

`kore.fs.rights(path)` reports what the task actually has, which is not the
same question as what a mode string asked for.

## Capabilities in practice

### Four ways to hold one

A DagFS route is reached through a `View`, which is **affine**: it has one
owner, and using it after it moves is not a thing that may happen. Only one of
the four surfaces can hold one, because only one has a language that enforces
it.

| Surface | What you hold | What stops a second use | Rights you can name |
|---|---|---|---|
| `io` / `os` | a file object | `__close`/`__gc`, plus a generation check | none; inferred from the mode string |
| `kore.fs` | the same file object | the same | `r`, `w` today; `n`, `s` as their verbs land |
| C (`kore-ffi`) | a `KoreFsHandle` | the adapter, at run time | `uint32_t`; `KORE_FS_READ`, `KORE_FS_WRITE` |
| Rust (`kore::fs`) | the `View` itself | the borrow checker, at compile time | the whole `Rights` type, and `fs::dag` |

Lua and C are in the same position for the same reason: a Lua value can be
copied into a table or captured by a closure, and a C struct can be memcpy'd,
so neither language can be handed something affine. Both get a *handle* — an
index and a generation into a table thread 0 owns — and a stale one is
`invalid-handle` rather than a read of whatever took the slot. Rust is the only
surface that holds the capability itself, and `src/fs.rs` keeps a
`compile_fail` doctest to prove it.

No surface can forge authority: the store resolves every route through a
generational slot, so fabricated or stale bits fail there whatever language
produced them. What differs between the four is how much of the vocabulary you
can say, how much ceremony saying it costs, and how early a mistake is caught —
Rust at compile time, the other three at run time, with C additionally carrying
the pointer lifetimes the adapter has to preserve by hand.

### What `io` cannot say

`io` is not a smaller filesystem. It is the same one with the capability
questions answered for you by convention:

```lua
local f = assert(io.open("/var/log/app", "a"))   -- rights: READ WRITE, create, append
```

That line asks for a write route and gets one. What it cannot do is ask for
*less*, ask for *more*, or find out what it got.

```lua
-- Least authority, on purpose: a read route to a file this task may well
-- be able to write. `io.open(p, "r")` asks for the same thing, but the
-- rights are a consequence of the mode rather than the point of the call.
local f = assert(kore.fs.open("/etc/hosts", "r"))

-- What does this task actually hold here?
local held = assert(kore.fs.rights("/var/log"))
if not held.write then
  return nil, "this task was not endowed with write authority there"
end
```

`kore.fs` can express every route `io` can — the mapping is the table above —
and three things `io` has no spelling for: rights asked for by name, rights
reported back, and behaviour chosen separately from authority.

```lua
-- io's "w" is three decisions fused into one letter. Taken apart:
local f = assert(kore.fs.open(path, "rw", {create = true, truncate = true}))

-- ...so you can take only two of them. No io mode string means this.
local g = assert(kore.fs.open(path, "rw", {create = true}))  -- keep what is there
```

### Lifetime

A handle is retired deterministically or not at all. `<close>` is the idiom on
both surfaces, and `__gc` is the backstop:

```lua
do
  local f <close> = assert(kore.fs.open("/tmp/scratch", "rw", {create = true}))
  f:write(payload)
end   -- retired here, whether the block ended or threw
```

### Noticing someone else

`revision` moves on every commit, so a script can tell that a write landed
underneath it. Nothing on the `io` surface can see this:

```lua
local before = assert(kore.fs.stat("/conf")).revision
-- ... do work that depends on /conf ...
if assert(kore.fs.stat("/conf")).revision ~= before then
  return nil, "/conf changed under us"
end
```

`id` is the companion: volume-stable, so it answers "is this the same object"
where a path only answers "is this the same name".

## Quick reference

Every verb, both spellings. `—` means the surface has no spelling for it.

### Bytes

| What | `kore.*` | stock |
|---|---|---|
| Open a file | `kore.fs.open(path, rights [, options])` | `io.open(path, mode)` |
| Read a whole file | `kore.fs.read_file(path)` | — |
| Write a whole file | `kore.fs.write_file(path, bytes)` | — |
| Read from a handle | `f:read(...)` | `f:read(...)`, `io.read(...)` |
| Write to a handle | `f:write(...)` | `f:write(...)`, `io.write(...)` |
| Iterate lines | `f:lines(...)` | `io.lines([path, ...])` |
| Move the position | `f:seek(whence, offset)` | same |
| Durability barrier | `f:flush()` | `io.flush()` |
| Truncate or extend | `kore.fs.resize(file, bytes)` | — |
| Close | `f:close()` | `io.close([file])` |
| Is it closed? | `f:closed()` | — |

### Namespace and graph

| What | `kore.*` | stock |
|---|---|---|
| Does it exist? | `kore.fs.exists(path)` | — |
| The object's record | `kore.fs.stat(path)` | — |
| Names in a directory | `kore.fs.list(path)` | — |
| Edges in a directory | `kore.fs.entries(path)` | — |
| Rights held there | `kore.fs.rights(path)` | — |
| Make a directory | `kore.fs.mkdir(path)` | — |
| Remove a name | `kore.fs.remove(path)` | `os.remove(path)` |
| Rename or move | `kore.fs.rename(from, to)` | `os.rename(from, to)` |
| A temporary name | — | `os.tmpname()`, `io.tmpfile()` |

### Everything else

| What | `kore.*` | stock |
|---|---|---|
| Monotonic seconds | `kore.time.now()` — since boot | `os.clock()` — since this VM started |
| Wall seconds | `kore.time.wall()` | `os.time()` — integer |
| Exact nanoseconds | `kore.time.mono_ns()`, `kore.time.wall_ns()` | — |
| Sleep | `kore.time.sleep(seconds)` | — |
| Format a time | — | `os.date(format, [time])` |
| Random bytes | `kore.random.bytes(n)` | — |
| Random integer | `kore.random.integer(lo, hi)` | `math.random(lo, hi)` |
| Launch arguments | `kore.proc.args` | — |
| Terminate | — | `os.exit([code])` |
| What is provisioned | `kore.has(name)`, `kore.caps()` | — |

## `kore.fs`

Every verb answers `value, err` on failure, where `err` is a table (see
[Errors](#errors)). Paths resolve inside the task's endowed view; there is no
global root.

### `kore.fs.open(path [, rights [, options]]) -> file | nil, err`

The same object `io.open` answers, with the table failure spelling — but asked
for in the store's vocabulary rather than POSIX's.

```lua
local f = assert(kore.fs.open("/etc/hosts"))                       -- "r"
local g = assert(kore.fs.open("/var/log", "rw", {create = true}))
```

`rights` is a subset of the store's own `rwns` spelling, each letter at most
once, defaulting to `"r"`. `READ` is implied, so `"w"` is a readable write
route. **`n` and `s` are refused for now** — a route carrying authority nothing
here can exercise would be a claim this surface cannot make — and unlock with
the verbs that use them.

`options` are `create`, `truncate`, `append`, `exclusive`.

Everything that can only be a mistake **raises**: a malformed spelling, a right
that is not reachable, an unknown option key, `truncate` or `append` without
`w`, `exclusive` without `create`. A refusal from the store — `not-found`,
`access-denied` — is the other thing, and arrives as `nil, err`.

Rights alone do not create: a missing file is `not-found` until `create` is
asked for. Opening a directory is `is-directory`; use `mkdir` to make one and
`entries` or `list` to read one.

> **This moved.** `kore.fs.open(p, "r")` is unchanged, and `io.open` is
> untouched. Every other mode string now either means something different —
> `"w"` is a read-write route that does not create — or raises, because `a`,
> `+` and `b` are not rights letters. The argument error names the syntax.

### `kore.fs.read_file(path) -> string | nil, err`

A whole file in one host call — the host opens, reads and closes under **one
pinned version**, so the answer is one revision of the file rather than a seam
of several. Looping `f:read` cannot give you that: each call pins and releases
its own version, so a concurrent writer can leave the result a mix of revisions.

It is also the shape that *can* stream. A whole-file read of at least 256 KiB
hands the facade a destination large enough to carry by DMA through the task's
storage ring instead of a window at a time — on a native owner that has one.
The export this build pins does not yet; see `docs/pack-backed-reads.md`. The
verb does not change when it lands.

An empty file answers `""`, never `nil`. Bounded at 16 MiB, past which the
answer is `range` — the file is readable, it is having all of it at once that
is refused.

### `kore.fs.write_file(path, bytes) -> count | nil, err`

Replaces the file's whole contents, creating it if absent. One replacement
transaction under a held commit, so a large save is **one** commit rather than
one per window. The open deliberately does not truncate: a save that then fails
leaves the previous contents intact.

### `kore.fs.resize(file, bytes) -> true | nil, err`

Takes an **open file**, not a path — truncation is a handle operation. A
shortened file drags the stream position back with it, so the next write cannot
land past a hole nobody asked for.

### `kore.fs.stat(path) -> table | nil, err`

The object's whole record. None of it is authority — a stable id conveys none
and a subtype gates nothing.

| Field | Meaning |
|---|---|
| `size_bytes` | integer |
| `kind` | `"file"` or `"directory"` |
| `directory` | boolean, the same fact as a predicate |
| `revision` | monotonic, moves on every commit |
| `id` | volume-stable identity; survives a service restart |
| `links` | names pointing at this object |
| `grants` | grants outstanding on it |
| `subtype` | `"plain"`, `"executable-image"`, `"package-manifest"`, or **absent** |
| `subtype_bits` | the raw byte, always present |

`revision` is the field to remember if you intend to notice that a write landed
underneath you; `id` is the identity to remember where a path is not.

`subtype` is absent — not `"plain"` — when this build cannot name the byte. A
newer writer's subtype is data an older reader passes through intact, so
`subtype_bits` still reports it and a script can forward a value it cannot name.

### `kore.fs.entries(path) -> array | nil, err`

A directory's **edges**, where `list` answers its **names**. Each element is
`{name, kind, directory, subtype?, subtype_bits}` — the same spelling `stat`
uses. Telling a native module from a plain file costs one call rather than a
stat per name.

The target's object id is deliberately not here: it is a service-local slot key,
not stable across a restart, and a script that stored one as an identity would
be wrong in a way nothing would tell it. The stable one is `stat`'s.

An empty directory answers an empty array. A file answers `not-directory`.

### `kore.fs.rights(path) -> table | nil, err`

What this task may actually do there. Reports authority; confers none.

| Field | Meaning |
|---|---|
| `spelling` | the store's own compact form, a subset of `rwns` |
| `read`, `write`, `mutate_namespace`, `share` | booleans derived from the letters |

`spelling` is the contract: what a listing prints is what a `share` accepts, so
rights read here can be handed back later unchanged. The booleans are a
convenience over it, not a second vocabulary.

### `kore.fs.list(path) -> array | nil, err`

The names a directory holds, as an array of strings. The namespace question;
`entries` is the graph one.

### `kore.fs.exists(path) -> boolean`

The one verb with no failure spelling: a path that cannot be stat'd does not
exist, as far as this task is concerned.

### `kore.fs.mkdir(path) -> true | nil, err`

Makes the final component. An existing name is `already-exists`; a missing
parent is `not-found`.

### `kore.fs.remove(path)`, `kore.fs.rename(from, to) -> true | nil, err`

`remove` detaches the name. `rename` is the store's own move, including across
directories — not a copy and unlink, so the object keeps its identity and its
grants. Removing the root is `access-denied`.

## The file object

One userdata, whichever surface opened it. Seven stock methods plus `closed`.

| Method | Answers |
|---|---|
| `f:read(...)` | `"a"` whole rest, `"l"` line, `"L"` line with newline, `"n"` number, or a byte count |
| `f:write(...)` | the file, so writes chain |
| `f:lines(...)` | an iterator over the read formats |
| `f:seek([whence [, offset]])` | the new position; `"set"`, `"cur"`, `"end"` |
| `f:flush()` | `true` — a **durability barrier**, not a buffer drain |
| `f:setvbuf(mode)` | `true`; accepted and inert, kore has no stdio buffer to tune |
| `f:close()` | `true`; idempotent |
| `f:closed()` | boolean, without raising |

The position is the file object's, not the store's: DagFS addresses bytes
absolutely, and a stream position is a Lua idea layered on top. Reads ahead by
a page — so `f:read("l")` is a scan over bytes already here — and a read of at
least 256 KiB bypasses that buffer entirely, asking for the whole remainder at
once. A page-sized ask can never be large enough for the facade to stream,
however large the file is, so the decision has to be made on this side.

**Lifetime.** `__close` and `__gc` both retire the handle, with deterministic
close primary, so a to-be-closed local is the idiom:

```lua
do
  local f <close> = assert(io.open("/etc/hosts"))
  for line in f:lines() do print(line) end
end  -- retired here, whatever happened
```

Using a closed file **raises** — it is a programming error, not an expected
failure. A handle whose slot was recycled is `invalid-handle`, never an alias of
whatever took its place.

## `io` and `os`

Stock Lua, conforming, with the deviations below. Every member either conforms
fully or is [absent for a stated reason](#absent-and-why) — never present and
partial.

| Member | Deviation |
|---|---|
| `os.getenv(name)` | always `nil` — a kore task has no environment; authority arrives as capabilities, not inherited strings, and `nil` is exactly what an unset variable means |
| `os.date`, `os.time` | kore keeps one clock and it reads UTC, so `!` changes nothing and `isdst` is always `false`; the calendar is computed, not tabulated |
| `os.setlocale` | answers `"C"` and `nil` for anything else |
| `os.exit([code])` | **records** the code and unwinds by raising, so `<close>` handlers and C frames below the VM still run; the embedder reads the code afterwards |
| `os.tmpname()` | draws a name under `/tmp`, creates the file exclusively, then closes it — the POSIX behaviour the manual describes |
| `io.lines(name)` | answers four values, the last to-be-closed |
| `f:flush()` | a durability barrier; writes are already acknowledged before `f:write` returns |
| `os.clock()` | elapsed monotonic seconds **since this VM started**, not process CPU time — kore meters no per-process CPU, and a number that silently meant something else would be worse than a different origin |

## `kore.time`

Seconds are floats; anything else carries its unit in its name. A clock that
cannot be read answers `nil, why` — never a zero timestamp, which a caller would
read as 1970 rather than as "no reading".

| Verb | Answers |
|---|---|
| `kore.time.now()` | monotonic seconds, float |
| `kore.time.mono_ns()` | monotonic nanoseconds, integer |
| `kore.time.wall()` | wall seconds, float |
| `kore.time.wall_ns()` | wall nanoseconds, integer |
| `kore.time.sleep(seconds)` | `true`; parks the VM thread, never spins |

A double holds 53 bits of mantissa, so a wall reading keeps roughly 256 ns of
resolution. Use the `_ns` forms when every nanosecond matters.

## `kore.random`

A source, not a stream: every byte is a fresh CPU draw. A script wanting many
cheap values should use `math.random`, which this build seeds from the same
source at startup.

| Verb | Answers |
|---|---|
| `kore.random.bytes(n)` | `n` bytes as a string, `0 <= n <= 4096` |
| `kore.random.integer(lo, hi)` | a uniform integer in `[lo, hi]` |
| `kore.random.available()` | boolean |

`integer` masks and redraws rather than folding with a modulo, so every value is
equally likely and `kore.random.integer` and `math.random` agree on what uniform
means. A dead entropy source ends the loop with its refusal rather than spinning.

## `kore.proc`

`kore.proc.args` is a **lazy field**, not a call: a 1-based array split from the
launch bytes on ASCII whitespace, fetched once on first touch and cached on the
module table. The bridge is crossed exactly once per process.

Termination is `os.exit`, on the stock surface where a script already looks for
it.

## `kore.has` and `kore.caps`

Provisioning is honest: a submodule whose authority the launch did not convey is
**absent**, not present-and-always-failing. So `if kore.fs then` is the idiom.

- `kore.has(name) -> boolean`
- `kore.caps() -> array` — the module names in registration order, which is
  stable across boots where a table walk would not be.

An embedder's own libraries list themselves here; there is no second place to
remember.

## `require`

`require "<name>"` resolves against the task's own tree first, then `libs/`; a
source module `<name>.lua` wins over a native `<name>.so`. Source modules are
bounded at 64 KiB — larger sources belong in the store as several modules.

A native module is a store object stamped `EXECUTABLE_IMAGE` that is checked,
placed in the JIT arena, and linked against this executable's exported C API. It
must export `luaopen_<name>`.

## Errors

**One vocabulary, two spellings.** Both are derived from the same `FsError`, so
the surfaces cannot disagree about what happened — only about how to say it.

```lua
local f, err = kore.fs.open("/nowhere", "r")
-- err.code == "not-found", err.kind == "refused", tostring(err) == "not-found"

local g, why, errno = io.open("/nowhere", "r")
-- why == "not-found", errno == 2
```

`err.kind` is `"refused"` when the service named a reason and `"unavailable"`
when the call could not be made at all — no filesystem face, a broken ring. The
distinction matters: `"unavailable"` is not the file's fault.

Expected failures answer `nil, err`; programming errors raise. A missing file is
the first; using a closed file is the second.

| Code | errno | Code | errno |
|---|---|---|---|
| `not-found` | `ENOENT` 2 | `out-of-storage` | `ENOSPC` 28 |
| `already-exists` | `EEXIST` 17 | `out-of-handles` | `EMFILE` 24 |
| `not-directory` | `ENOTDIR` 20 | `range` | `ERANGE` 34 |
| `is-directory` | `EISDIR` 21 | `unsupported` | `ENOSYS` 38 |
| `invalid-name` | `EINVAL` 22 | `busy` | `EBUSY` 16 |
| `invalid-handle` | `EBADF` 9 | `cycle` | `ELOOP` 40 |
| `access-denied` | `EACCES` 13 | `not-empty` | `ENOTEMPTY` 39 |
| `access-lost` | `EBADF` 9 | anything else | `EIO` 5 |

A code with no POSIX analogue answers `EIO` — the honest "something below
failed". The kore spelling always carries the exact word.

## Absent, and why

Absent for a reason, never half-present.

| Member | Why |
|---|---|
| `os.execute`, `io.popen` | both launch a process, which is a service face the embedding program holds and this crate deliberately does not. An embedder adds them through its own library. |
| `string.format` | kore has no hosted `printf`, and a hand-written subset would accept format strings it silently did not conform to |
| `load`, `loadfile`, `dofile`, `string.dump` | this build accepts text chunks only; upstream Lua has no bytecode verifier |

Lua's base, coroutine, math, string, table and UTF-8 libraries are all present.

## Not yet exposed

**Aspirational — none of this is implemented.** Listed so the shape of the gap
is visible, not as a promise of spelling.

Unblocked, and wanted:

- `search` — the query vocabulary is settled in the native export
- grant **listing** (what this task has shared out) — reporting, like `rights`
- creating a file with a subtype, so a Lua installer can write what `require`
  loads
- holding commits across several writes, which `write_file` uses internally and
  no script can ask for
- `stat` on an open file, which the bridge already supports

Blocked upstream on MicroPerceptron/kore#986 and #991, which rule the capability
model for a userland authority domain — a spelling invented here first would
have to be retracted:

- creating a grant, revoking one, opening a view through one
- attenuating a view (projection)

Blocked on the `index` feature, an image-size decision for this target:

- index publication on save, and derived-version queries

See `docs/kore-fs-vocabulary.md` for the plan these sit in,
`docs/pack-backed-reads.md` for the storage-transport work they sit beside, and
`docs/filesystem.md` for the Rust facade underneath all of it.
