//! # kore-rt — the kore runtime core
//!
//! Std-shaped ergonomics over the frozen [`kore_abi`] wire contracts — the
//! substrate every kore userland component builds on (`res/USERLAND.md`
//! §1.1): capability witnesses, the session builder, the heap engine, and
//! the service framework.  The concrete system services live one crate
//! each under `services/` (`kore-svc-{mm,swap,exec,image}`), and the
//! this crate is what all of them depend on, and it depends only on
//! `kore-abi`.
//!
//! - [`cap`] — **userland capability witnesses**: affine values (no `Copy`,
//!   no `Clone`) over the D1-stamped records in this task's arena.
//!   Possession of the token *is* the authority — `map` takes `(virt,
//!   token)` where Linux takes `(virt, phys)` — and consuming operations
//!   take tokens by value, so map-then-share and copy-instead-of-move are
//!   compile errors.  Sharing is peer-to-peer: attenuate your own token,
//!   move it over rendezvous.  The kernel is reached only through the
//!   [`cap::verbs`] seam — live possession reads on the metal target,
//!   typed refusals where the kernel arm has not landed.
//! - [`transport`] — **the transport seam** (`res/USERLAND.md` §2/§4):
//!   [`transport::Transport`], the one boundary every
//!   session-shaped kernel effect crosses — the trap transport, live on
//!   the x86 userland target, refusing typed everywhere else.
//! - [`executor`] — **the doorbell-muxed async executor** (`res/USERLAND.md`
//!   §3.1/§3.3, Phase A): single-threaded per task, heapless — a fixed slab
//!   of type-erased task records over per-task [`executor::TaskStorage`]
//!   statics (the embassy pattern), a cookie waker with no alloc and no
//!   `dyn`, and an idle path that is the runtime's ONE `commit_and_park`
//!   caller (consume → re-post → drain-all → park on the standing doorbell
//!   post, deadline folded in).  [`sleep_until`] rides its event-driven
//!   timer table.
//! - [`chan`] — **typed channel endpoints** (§4 pinned): [`ChanTx::send`]
//!   (the notice-mode kick, its own session, droppable under the §3.1
//!   order), [`ChanTx::transfer`] (the landed capability transfer), and
//!   [`ChanRx::recv`] (post-if-not-standing as its own commit; never
//!   parks).
//! - [`session::SessionBuilder`] — **staging batched syscalls**: the
//!   zero-alloc write side of the frozen descriptor ABI (one trap = one
//!   session of up to 64 SSA-chained invocations), plus the specialized
//!   flows that exploit it — [`flows::spawn_thread`](session::flows) (Flow
//!   1: 3 invocations, 1 crossing), [`flows::share_with`](session::flows)
//!   (peer sharing in one commit), [`flows::device_setup`](session::flows)
//!   (the whole driver bring-up in one trap), and
//!   [`flows::revoke_batch`](session::flows) (cluster teardown; the swap
//!   page-out precursor).
//! - [`heap::Heap`] — the process's **global-allocator engine**: a sharded,
//!   lock-free allocator (mimalloc-shape) generic over its
//!   [`MemSource`](heap::MemSource) backing.  kore-rt ships the engine,
//!   the seam, and the metal source (`heap::HeapSupplySource` — the
//!   founder-provisioned grow window every founded binary links).
//! - [`service::SystemService`] — the unified shape of a bootstrapped
//!   service (KRAPH §5.2's role bundle + a serve step): the endowment is a
//!   typed [`Bundle`](service::Bundle) resolved fail-closed from the
//!   bootstrap handle table (possession-shaped: locate, then *claim*).
//!   [`derive@Bundle`] makes each field's contextual role the single static
//!   declaration used for founding, readiness, and resolution; the sealed
//!   token type supplies the corresponding possession verb. Authority
//!   leaves only as [`Serve::grant`](service::Serve::grant)
//!   attenuations of held caps — delivered as tokens, never as reply
//!   geometry.
//!
//! **No mock world.**  There is exactly one kernel path: the
//! [`cap::verbs`] possession reads and the [`transport`] trap, live on
//! the userland target and typed refusals elsewhere.  Behavior is
//! proven where it runs — the QEMU boot gates — not against a host
//! model (the `cap::mock` layer was deleted by the 2026-07-24 ruling);
//! host tests cover what is genuinely host-computable: wire codecs,
//! staging, and pure data structures.

#![no_std]
#![feature(allocator_api)]
#![feature(never_type)]
#![feature(portable_simd)]
#![allow(async_fn_in_trait)]

// `alloc` rides with the [`heap`] module, which is the only non-test user
// of it in this crate — see the `heap` feature in `Cargo.toml`.  A
// `default-features = false` consumer (the linker stub) links the session
// and transport path with no allocator registered at all.
#[cfg(feature = "heap")]
extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

pub mod barrier;
pub mod block;
pub mod cap;
pub mod chan;
pub mod clock;
pub mod core_id;
pub mod crash;
pub mod entry;
pub mod event;
pub mod executor;
pub mod exit;
pub mod futex;
#[cfg(feature = "heap")]
pub mod heap;
pub mod hint;
pub mod lane;
pub mod mem_source;
pub mod pace;
pub mod process;
pub mod random;
pub mod room;
pub mod service;
pub mod session;
pub mod spmd;
pub mod stack_arena;
mod star;
pub mod startup;
pub mod svc;
pub mod termination;
pub mod thread;
pub mod thread_controller;
pub mod transport;
#[cfg(feature = "validate")]
pub mod validate;
pub mod wake;
pub mod worker;

pub use barrier::{Arrival, SpmdBarrier};
pub use block::{block_on, block_on_pinned, try_block_on};
pub use chan::{CapRx, ChanRx, ChanTx, Delivery, NoticeRx};
pub use entry::run_service;
pub use executor::{
    Executor, TaskStorage, enable_irq_parks, next_wake_until, retire_irq_parks, sleep_for,
    sleep_until,
};
pub use exit::ExitCode;
#[cfg(feature = "heap")]
pub use heap::Heap;
pub use service::SystemService;
pub use session::SessionBuilder;
pub use spmd::{Spmd, SpmdCtx, SpmdEntry, spawn_spmd};
pub use termination::Termination;
pub use transport::{Committed, Transport};

// The runtime attributes and authority-bundle derive
// (`res/USERLAND.md` §5.4/§3.3), resolved through this re-export — service
// crates never depend on the macro crate directly. `#[kore_rt::task]`'s
// expansion names `type_alias_impl_trait` (the consumer crate opts in; the
// workspace is nightly-pinned).
// `spmd` shares its name with the [`spmd`] module deliberately — the
// attribute lives in the macro namespace, the module in the type
// namespace, and one name for the one model is the point:
// `#[kore_rt::spmd(x, y, z)]` launches what `kore_rt::spmd` implements.
pub use kore_rt_macros::{Bundle, main, spmd, task};

pub mod tls;
