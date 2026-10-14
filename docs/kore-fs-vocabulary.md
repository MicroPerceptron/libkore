# `kore.fs` in the store's own vocabulary

**Status: V1 is implemented; V2 onward are not.** Implemented slices are
described in the past tense and their contract lives in `lua.md`. Nothing here
has run on metal.

## 1. The problem

`kore.fs` exists because `io` cannot say what the store knows. It has stopped
being true in one direction — `stat`, `entries` and `rights` answer the graph —
and is still false in the other: **the verbs that take authority still spell it
as POSIX.**

```lua
kore.fs.open(path, "w")   -- "write, and create, and truncate": three things, one letter
```

A mode string is a POSIX projection of a right set. DagFS authority is a right
set on a route, and the store already has a textual spelling for it —
`rights_label` prints what `rights_parse` reads, and `kore.fs.rights` has
answered in that spelling since the graph slice landed. So the surface reports
rights in one vocabulary and accepts them in another.

## 2. The rule

**Authority is a rights string. Behaviour is an options table. The surface never
offers a right it cannot exercise.**

The third clause is the one that bounds this work. A route carrying `SHARE` is
useless while nothing here can share, and a surface that accepts `s` would be
claiming a reach it does not have. So letters unlock as their verbs land, and
until then they are refused by name rather than silently ignored.

| Letter | Right | Reachable when |
|---|---|---|
| `r` | `READ` | now |
| `w` | `WRITE` | now |
| `n` | `MUTATE_NAMESPACE` | `link`/`unlink` on a handle exist (V2) |
| `s` | `SHARE` | `share` exists — blocked on kore#986/#991 |

`BIND_EXECUTABLE` has no letter: `rights_label` spells four bits and answers
`?` for anything else. That is the store's vocabulary, and this surface does not
extend it.

## 3. Two things that are not one thing

The rights string governs the **route**. The options table and the object's own
read/write permission govern the **handle**. They are not the same question, and
conflating them is what the mode string does:

- `io.open(p, "w")` opens a route with `READ WRITE` — a write route must be able
  to read to take a step — but hands back an object that refuses `f:read`.
- `kore.fs.open(p, "rw")` opens the same route and hands back an object that
  permits both, because that is what was asked for.

So `files::open` takes both: a packed rights spelling for the route, and flags
for the handle's permissions and the open's behaviour.

## 4. Slices

### V1 — `open` by rights *(done)*

```lua
kore.fs.open(path)                        -- "r"
kore.fs.open(path, "rw")
kore.fs.open(path, "rw", {create = true, truncate = true})
```

Options are `create`, `truncate`, `append`, `exclusive`. An unknown key, a
malformed spelling, or a combination that cannot mean anything — `truncate`
without `w`, `exclusive` without `create` — **raises**, because each is a
programming error and rule 1 divides those from expected failures.

The spelling crosses the bridge packed into one word as ASCII, and
`kore_dagfs::rights_parse` does the mapping on the far side. C re-implementing
letter-to-bit would be a second authority over one set of bits, which is the
mistake `rights` already avoids by deriving its booleans from the store's own
letters.

`io.open` converges onto the same path: its mode string is translated to a
spelling and a set of options in C, so the capability default documented in
`lua.md` lives in exactly one place. Nothing about `io` changes.

**The break.** `kore.fs.open(p, "r")` is unchanged — `"r"` means `READ` in both
vocabularies. Everything else moves, and every case fails loudly rather than
silently:

| Was | Now |
|---|---|
| `"w"` | parses as `READ WRITE`, does **not** create or truncate — a missing file is `not-found` |
| `"a"`, `"r+"`, `"w+"`, `"rb"` | raise: `a`, `+` and `b` are not rights letters |
| `"rn"`, `"rs"` | raise, by §2 — named as not yet reachable, not silently dropped |

The argument error names the syntax, so a script written against the old
spelling is told what to write instead.

### V2 — `create`, and the subtype

`create_typed` is unreachable, so a Lua installer cannot write the
`EXECUTABLE_IMAGE` objects that `require` loads. A `kore.fs.create(path, opts)`
taking `{kind = "file" | "directory", subtype = ...}` gives it a home, makes
`mkdir` the one-liner over it rather than a separate path, and is what unlocks
`n` in §2 once `link`/`unlink` join it.

### V3 — grant listing

`shares` and `share_entry` list this task's authority edges. **Reporting, not
conferring** — the same rule that let `stat` and `rights` land ahead of the
capability rulings. Grant *creation* stays blocked.

### V4 — `search`

The query vocabulary is settled in the pinned export (`SearchQuery`,
`SearchPlane`, `FreshnessPolicy`, `SearchScore`). It is the largest piece of the
store a script still cannot reach.

### V5 — batched commits

`write_file` holds commits internally and no script can ask for the same across
several writes. A scoped form — the hold released on the way out, whatever
happened — is the only shape worth offering, which makes it a to-be-closed
value rather than a pair of calls.

### V6 — the remaining small gaps

`stat` on an open handle (the bridge already supports it), `open_root`,
`bind_cwd`.

### Blocked

- Grant creation, revocation, opening a view through one, and projection:
  **kore#986** (grant boundary, route shape, rights vocabulary) and **kore#991**
  (whether a userland authority domain gets a generic capability type).
- Index publication on save and derived-version queries: kore-dagfs's `index`
  feature, which pulls `kore-svc-extract-worker-rust` — an image-size decision
  for this target.

## 5. Acceptance

- Every rights spelling the store accepts round-trips: what `kore.fs.rights`
  prints can be handed to `kore.fs.open`.
- A spelling this surface cannot exercise is refused by name, not ignored.
- `io`'s observable behaviour is unchanged across the convergence in V1 — its
  suite is the witness.
- Measured on metal, with the rest of the Lua surface. Nothing here is proven
  by a host build.
