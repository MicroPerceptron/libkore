//! # kore-abi — the kore kernel <-> user ABI contract
//!
//! The narrow waist between the kernel and userland: the **types and
//! layouts that cross the boundary, and nothing else**. kore-abi carries no
//! kernel behavior — no epoch/MAC key, no closure operator, no retired-set,
//! no `derive`/`revoke`/`verify` logic. A userland program links kore-abi and
//! can *describe* requests (build session descriptors, read cap-slot
//! fields) but **cannot** invoke kernel authority operations, because that
//! logic is not in the crate it depends on. The kernel validates and
//! executes; kore-abi is only the contract they agree on.
//!
//! **Single source of truth.** The kernel's `schema` builds its typed
//! `Capability` view *on* these wire types, so the two sides can never
//! drift on a layout — a drift here would be a *security* bug, not a style
//! one.
//!
//! **Driver-first.** [`device`] (MMIO/IOMMU) and [`mem`] (DMA buffers) are
//! the priority surface: kore runs its drivers in userland, so the device
//! contract is what the ABI must serve first.
//!
//! **Two transports, split by what moves.** [`session`] is the *authority*
//! plane — the native transactional session-commit (`derive`/`revoke`/
//! `rendezvous`, `res/SYSCALL.md`). All capability movement is synchronous
//! and rides here; synchronous device control (map a BAR, map DMA, program
//! an MSI) is session verbs batched per-flow into one crossing, never ring
//! traffic. [`uring`] is the *data* plane — one io_uring-shaped SQ/CQ,
//! **byte-compatible with Linux io_uring** (issue #148) so liburing-style
//! I/O tooling ports by header-swap; it carries only I/O a holder performs
//! on authority it already has. A CQE carries a slot index + status, never
//! a capability — so no async authority state can exist, and `res/SYSCALL.md`
//! §3.3 (no general async ring in the authority path) holds by construction.
//!
//! **The wire freeze (#81).** The whole surface is frozen at [`ABI_VERSION`]:
//! every wire struct is size/offset-pinned by a `const` assertion (the
//! per-module `// Frozen wire pins` blocks), and post-freeze changes are
//! **additive only** — new op-code / status / flag values, new meanings for
//! reserved fields; never a reorder, resize, or field-type change. The
//! io_uring byte-compat pin above is part of the freeze: [`uring`]'s layouts
//! track the Linux wire, not a kore-private one.
//!
//! ## What lives here vs. the kernel
//!
//! | here (kore-abi) | kernel-only |
//! | --- | --- |
//! | cap/session/device/mem **layouts** | the epoch MAC key |
//! | rights + kind vocabularies | stamp **mint** / **verify** |
//! | pure field readers + (de)serialization | closure `Φ`, the retired-set |
//! | op-codes + status codes | `derive`/`revoke`/`rendezvous` **effects** |
#![no_std]
#![forbid(unsafe_code)]

pub mod cap;
pub mod clock;
pub mod command;
pub mod crash;
pub mod device;
pub mod execution;
pub mod founding;
pub mod guest;
pub mod image;
pub mod input_status;
pub mod machine;
pub mod mem;
pub mod monitor;
pub mod pkg;
pub mod pointer;
pub mod process;
pub mod processor_status;
pub mod session;
pub mod svc;
pub mod task;
pub mod uring;

/// **The frozen ABI version** (#81; see the crate header's wire-freeze
/// discipline). The wire surface —
/// [`cap`] (stamped slot record), [`session`] (descriptor + `Σ_min` op-codes +
/// status), [`device`] (driver ops), [`mem`] (alloc props), [`task`] (bootstrap
/// context + spawn args), [`image`] (loader wire: names, module descriptors,
/// the passive-ABI module page), [`uring`] (io_uring data-plane wire), [`clock`]
/// (published clock page + timeout submissions) — is **FROZEN**
/// at this version: every wire struct is
/// size/offset-pinned by a `const` assertion, and post-freeze changes are
/// **additive only** (new op-code / status / flag values, new meanings for
/// reserved fields — never a reorder, resize, or field-type change). Downstream
/// (userland runtime, driver processes, compute sidecar) builds against this
/// contract; a break here breaks all of them at once.
pub const ABI_VERSION: u16 = 1;
