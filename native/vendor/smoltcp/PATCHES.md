# kore patch to smoltcp 0.14.0

Source: crates.io `smoltcp` 0.14.0, repository
https://github.com/smoltcp-rs/smoltcp. Upstream 0BSD licence is retained
(`LICENSE-0BSD.txt`). No GPL source is included.

The published source records upstream commit
`cbfe05f835cfc75bf781addc1c0eb2e1a0df1ea3`; the downloaded `.crate` archive's
SHA-256 is
`b6f8b28ad56c6e35524a37dd492af5d1a47e31e1a4d175cd12f89c075f01980f`, which
matches the checksum crates.io records for that version. Not yanked at the
time of vendoring (2026-10-08).

## Deviations from upstream

**None.** This is an unmodified vendoring.

It is vendored rather than depended on from crates.io for one reason, stated
up front so the next person does not have to infer it: the design
(`res/INET_STACK.md` §5) takes smoltcp's **wire layer** and, later, its **TCP
state machine**, and writes ARP/IPv4/ICMP/UDP natively over them. That split
means kore will eventually diverge from upstream in specific, listed ways, and
a vendored tree with this file is where those divergences get recorded. Keeping
the vendoring at zero deviations today is the point: every future entry below
is then a deliberate act with a reason beside it, not an accumulated drift
nobody can separate from upstream.

When a deviation does land, record it here as `vendor/ieee80211/PATCHES.md`
does — the file and function touched, what changed, and the reference
justifying it — and keep it minimal and upstreamable so a rebase onto a new
release stays mechanical rather than archaeological.

## Why 0BSD makes this cheap, and what still costs

0BSD requires no attribution and imposes no conditions, so inlining carries no
licence obligation at all. What it does **not** remove is the merge burden:
every deviation is one more thing to reapply when upstream moves. That is the
cost this file exists to keep visible.

## Vetting (`kore-crate-selection` §2), recorded at adoption

| Bar | Result |
| --- | --- |
| `no_std` | **Yes.** Builds with `default-features = false`; upstream's own description is "designed for bare-metal, real-time systems without a heap". |
| Allocation posture | **Heapless-capable.** Buffers are caller-supplied through `managed::ManagedSlice`; `alloc` is an opt-in feature kore does not enable. |
| Build script | **Read before adoption.** `build.rs` reads `SMOLTCP_*` env vars and `CARGO_FEATURE_*`, and writes `pub const NAME: usize = N;` into `OUT_DIR/config.rs`. No network, no process execution, nothing written outside `OUT_DIR`. |
| Maintenance | 0.14.0 published 2026; ~1.86M recent downloads; the de-facto Rust bare-metal stack — Redox, Embassy and Hubris all build on it. |
| Licence | 0BSD — compatible, no attribution required. |
| Arch fit | Architecture-neutral; no per-arch code, so it belongs in the main dependency table rather than a `cfg` target table. |
| New transitive dependencies | **One:** `managed`. `bitflags`, `byteorder`, `cfg-if`, `heapless` and `log` already resolve in this tree, and smoltcp's `heapless 0.9` requirement matches the 0.9.3 already present. |

## What it provides, and what it does not

Checked against upstream `README.md` at this version rather than assumed, so
the gaps are known rather than discovered:

- **Present:** MSS negotiation, window scaling, out-of-order reassembly (4 or
  32 gaps), RTO from an RTT estimate with exponential backoff, delayed ACK,
  Nagle, zero-window probing, keep-alive, configurable user timeout,
  congestion control (CUBIC and Reno, optional).
- **Absent:** silly-window-syndrome avoidance, PLPMTUD, TCP timestamps. The
  urgent pointer is ignored. IPv4 options are silently ignored.

**Two corrections to an earlier version of this section**, both of which read
upstream's own `README.md` rather than the source, and both of which were
wrong in kore's favour:

- **SACK is half-present, not absent.** 0.14.0 implements the *receiver*
  side — `remote_has_sack` (`src/socket/tcp.rs:511`), SACK-permitted offered
  on every SYN (`:2608`), one reported block from the assembler
  (`:1497-1531`). What is missing is only *sender-side* RFC 6675 loss
  recovery; upstream issue #256 has been open since 2018 and calls that the
  hard half. So a kore peer advertises SACK and honours a peer's blocks; it
  does not yet use received blocks to drive retransmission.
- **IPv4 fragmentation and reassembly are gated behind
  `proto-ipv4-fragmentation`, which kore does not enable.** They were listed
  as present. We do not have them — `crates/inet`'s own `ip::input` refuses
  fragments outright, which is the honest posture while that stays true.

Sender-side SACK is the gap that matters at scale: on a lossy or high-BDP path
its absence costs real throughput. It costs nothing for the `ping google.com`
milestone or a LAN. The assessment of what adding it would
take and recommends against it.

## Features kore enables

`default-features = false`, and only what a slice actually needs. As of B2:
`medium-ethernet`, `proto-ipv4`, `proto-dhcpv4`, `proto-dns`, `socket-tcp`.

Note the shape of that list: the `proto-*` features gate **wire encodings**,
the `socket-*` features gate **implementations**. kore takes the first and
writes the second, so `proto-dhcpv4` and `proto-dns` are on while
`socket-dhcpv4` and `socket-dns` stay off — enabling those would pull in a
second implementation of something `crates/inet` owns
(`res/INET_STACK.md` §5.1). `socket-tcp` is the one deliberate exception, and
§6.1 of that document records why.

Deliberately **off**: `std`, `log`, `alloc`, `defmt`, `socket-dhcpv4`,
`socket-dns`, `socket-udp`, `proto-ipv6`, `proto-sixlowpan`,
`medium-ieee802154`.

## Modifications considered, and why none was taken

A ranked review of every change this tree might want in the vendored source
concluded with **zero deviations**, and the reasoning is kept here rather
than in a document of its own so it sits beside the thing it governs.

- **Sender-side SACK** — the only gap that costs real throughput at scale,
  and the only candidate that was ever close. Interior to the protosw
  boundary, so it is a patch whenever a measurement demands one, and it
  would be offered upstream rather than carried.
- **The two-poller problem** — the demux shim that lets one `Interface`
  serve a fixed vif ring. Better fixed on our side of the seam by moving
  the adapter to `Medium::Ip`, which also deletes a copy. Our rework, not
  theirs.
- **Socket-set panics** — `SocketSet::add` panics on a full borrowed set,
  and `get`/`get_mut`/`remove` panic on a retired handle. Closed kore-side
  by a budget type that owns the set privately and lends no `&mut`, so the
  panic branch is unreachable by call graph. Upstream unchanged.
- **Everything else** — wire encodings, congestion control, reassembly
  bounds — is either adequate or configurable.

The discipline this protects: an unmodified vendored tree can be re-pinned
by changing one commit hash. Every deviation makes that an exercise in
re-applying judgement, and the gaps above are not worth that.
