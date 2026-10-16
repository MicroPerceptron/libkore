//! Guest exit/control wire records — the **per-vCPU window vocabulary**
//! (#487/#481), frozen here because kernel and hypervisor are two
//! endpoints of one contract and can never drift on a layout.
//!
//! One page per vCPU carries **two SPSC rings** (`kore-ring`'s frozen
//! `SPSC_ABI = 1` format, one ring per direction):
//!
//! ```text
//! offset 0:                     exit ring    (kernel → hypervisor,
//!                                             capacity 2, GuestExitRecord)
//! offset EXIT_RING_BYTES:       control ring (hypervisor → kernel,
//!                                             capacity 4, GuestCtrlRecord)
//! ```
//!
//! The exit capacity is **2 by proof, not by tuning** (#487 §5): the
//! producer is the vCPU run loop, which parks after every publish, so at
//! most one ordinary exit plus one terminal record can ever be
//! outstanding.  `Full` on the exit ring is therefore a protocol
//! violation and the kernel's response is the fail-closed poisoned-window
//! retire — never a drop, never a wait.  The control capacity (4) is the
//! largest sane batch (`WriteState + Inject + Resume` and headroom); the
//! run loop drains it to empty at its top-of-loop safe point, so `Full`
//! there is ordinary backpressure on an awake producer.
//!
//! Records are **bytes and indices only** — no capability ever rides a
//! ring (the standing doctrine).  Every field is any-bit-valid on read:
//! the peer is untrusted, and a record must decode from arbitrary bytes
//! (unknown kinds refuse at the consumer, they do not trap).
//!
//! The register image rides as a fixed `[u64; GUEST_STATE_QWORDS]` in
//! the **executing architecture's own frozen wire order** — the kernel's
//! per-arch `state_to_wire`/`state_from_wire` seam uses the shared architecture
//! layout selected by [`arch`] so the instruction decoder cannot drift from it.
//! The record itself stays architecture-agnostic: the protocol cites
//! registers exclusively through the semantic roles in [`state_index`],
//! which every architecture's wire order is required to satisfy.  The
//! array is the contract, never a field list.
//!
//! ## The decode-assist tail (#765)
//!
//! A stage-2 fault names an *address*, not an instruction.  Servicing it
//! as MMIO needs the trapped instruction's width, direction and register
//! — facts the hardware hands over in three different shapes: AMD stores
//! the raw instruction bytes in the VMCB (decode assists), Intel stores
//! nothing and the kernel fetches the bytes through the guest's own page
//! tables, and Arm / RISC-V report the access **already decoded**
//! (`ESR_EL2.ISS` / `htinst`).  An exit record therefore carries whichever
//! shape the producing architecture has, tagged by [`GuestExitRecord::assist`]
//! (the [`assist`] roster): raw bytes in `insn`/`insn_len`, or a decoded
//! access in `access`/`data`.  The consumer's one job is to lift that into
//! a decoded access — running an ISA decoder only when handed bytes — so
//! an architecture whose hardware decodes for it never needs one.  No form
//! is mandatory: [`assist::NONE`] is an honest "no help", and the consumer
//! decides fail-closed.

// Keep architecture selection here, as in the kernel's platform::arch.
// Consumers use guest::arch; an x86 register name cannot leak into a
// build for another ISA through an unconditional module.
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "riscv64")]
pub mod riscv64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub use aarch64 as arch;
#[cfg(target_arch = "riscv64")]
pub use riscv64 as arch;
#[cfg(target_arch = "x86_64")]
pub use x86_64 as arch;

mod wire;
pub use wire::*;
