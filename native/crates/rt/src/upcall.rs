//! **Fault upcalls** — the program's half of
//! `kore_abi::session::DERIVE_KIND_FAULT_UPCALL` (the kernel's is
//! `runtime::sched::upcall`): a synchronous fault on a thread that armed
//! itself is delivered to the process's handler, on the thread's own upcall
//! stack, instead of ending the process.  This is what a runtime with its
//! own trap handling needs — a wasm engine whose guard regions make an
//! out-of-bounds access a page fault (`res/WASM_PLATFORM.md` §12).
//!
//! One [`Handler`] per process ([`install`]); one guarded stack and one
//! registration per thread that may fault into it ([`arm_thread`]).
//!
//! # The handler's contract
//!
//! The kernel enters this module's trampoline with the stack pointer at the
//! top of the thread's upcall stack and every register but five as the
//! fault left them; the trampoline calls the handler with those five: the
//! faulting `pc`, the frame pointer at the fault, whether the fault names
//! an address (`0` or `1`), the address, and the fault's
//! [`FaultKind`](kore_abi::crash::FaultKind) as its `u16`.
//!
//! - **To recover**, the handler transfers control to a recovery point on
//!   the thread's own stack, as Wasmtime's trap handler does with the state
//!   its entry trampoline saved.  It never returns then, and the thread is
//!   off the upcall stack from that jump on.
//! - **To decline**, it returns.  The trampoline then traps at its decline
//!   instruction, and the kernel ends the process with the fault as it was
//!   raised, not with that trap (`CaptureFlags::UPCALL_DECLINED`).
//!
//! A fault raised while on the upcall stack — the handler's own — is never
//! delivered: it ends the process, flagged as raised inside the upcall.  The
//! upcall stack's guard page counts as on it, so a handler that overflows
//! its stack ends the process rather than re-entering itself.

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::fiber_stack::GuardedStack;

/// The handler the trampoline calls: the faulting `pc`, the frame pointer
/// at the fault, whether the fault names an address (`0` or `1`), the
/// address, and the fault's kind (`FaultKind as u16`).
///
/// # Safety
///
/// It runs on the upcall stack at an arbitrary point of the interrupted
/// code: it may touch only state that code cannot have left inconsistent,
/// as a POSIX signal handler may.
pub type Handler =
    unsafe extern "C" fn(pc: usize, fp: usize, has_address: usize, address: usize, kind: usize);

/// Each armed thread's upcall stack: enough for a handler that walks the
/// faulting code's frames and records a trap (Wasmtime's own sigaltstack is
/// 64 KiB), with the guarded stack's unmapped page below it.
pub const STACK_BYTES: usize = 64 << 10;

/// The process's handler, as an address; `0` until [`install`].  Read by
/// the trampolines, which call through it.
static HANDLER: AtomicUsize = AtomicUsize::new(0);

/// Why the upcall machinery refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A different handler is already installed.
    HandlerTaken,
    /// [`arm_thread`] before [`install`]: a delivery would have nothing to
    /// call.
    NoHandler,
    /// No guarded stack: its window is full, the thread has no slot to hold
    /// its mapping, or the kernel refused the map.
    NoStack,
    /// The kernel refused the registration, or the thread holds no view of
    /// its own space to register through.
    Refused,
    /// Off the kore target: there is no kernel to deliver anything.
    Unsupported,
}

/// Install the process's fault handler.  Idempotent for the same handler;
/// a different one refuses rather than silently redirecting the threads
/// already armed.
pub fn install(handler: Handler) -> Result<(), Refusal> {
    let handler = handler as usize;
    match HANDLER.compare_exchange(0, handler, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Ok(()),
        Err(installed) if installed == handler => Ok(()),
        Err(_) => Err(Refusal::HandlerTaken),
    }
}

/// The calling thread's registration.  Dropping it — on the thread that
/// armed (the type is `!Send`) — clears the registration first, then
/// unmaps the stack, so no fault is ever delivered onto a stack that is
/// gone.
#[must_use = "dropping the registration disarms the thread at once"]
pub struct ArmedThread {
    stack: Option<GuardedStack>,
}

impl ArmedThread {
    /// The upcall stack's top: where a delivery sets the stack pointer.
    pub fn stack_top(&self) -> u64 {
        self.stack.as_ref().map_or(0, GuardedStack::top)
    }
}

impl Drop for ArmedThread {
    fn drop(&mut self) {
        // Clear first.  A refused clear keeps the stack mapped — leaked with
        // the registration that still names it — rather than unmapping
        // pages a delivery would land on.
        if register(0, 0, 0, 0) {
            drop(self.stack.take());
        } else {
            core::mem::forget(self.stack.take());
        }
    }
}

/// Arm the calling thread: map its upcall stack and register it, with this
/// module's trampoline as the entry.  [`install`] first.
pub fn arm_thread() -> Result<ArmedThread, Refusal> {
    if !cfg!(target_os = "none") {
        return Err(Refusal::Unsupported);
    }
    if HANDLER.load(Ordering::Acquire) == 0 {
        return Err(Refusal::NoHandler);
    }
    let stack = GuardedStack::new(STACK_BYTES).ok_or(Refusal::NoStack)?;
    // The registered range takes in the guard page: a handler overflowing
    // its stack faults there, and is still on it.
    let (lo, hi) = (stack.guard().start, stack.top());
    if !register(trampoline::entry(), trampoline::decline(), lo, hi) {
        return Err(Refusal::Refused);
    }
    Ok(ArmedThread { stack: Some(stack) })
}

/// Issue the registration: `entry == 0` (with everything else zero) clears
/// it.  The result slot is a borrowed scratch slot, left vacant by the
/// verb and handed straight back.
#[cfg(target_os = "none")]
fn register(entry: u64, decline: u64, stack_lo: u64, stack_hi: u64) -> bool {
    use crate::session::SessionBuilder;
    use crate::transport::{Transport, TrapTransport};
    use kore_abi::cap::Rights;
    use kore_abi::session::{DERIVE_KIND_FAULT_UPCALL, FaultUpcallReq, Status};

    let Some(space) = crate::entry::geometry::self_space_slot() else {
        return false;
    };
    let Some(scratch) = crate::entry::geometry::acquire_cap_slot() else {
        return false;
    };
    let mut req = [0u8; FaultUpcallReq::WIRE];
    for (i, word) in [entry, decline, stack_lo, stack_hi].into_iter().enumerate() {
        req[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    let mut b = SessionBuilder::new();
    let committed = b
        .derive_raw(
            space,
            Rights::empty(),
            DERIVE_KIND_FAULT_UPCALL,
            scratch,
            Some(&req),
        )
        .is_ok()
        && TrapTransport::new()
            .commit_now(&b)
            .is_ok_and(|done| done.status() == Status::Ok);
    crate::entry::geometry::release_cap_slot(scratch);
    committed
}

#[cfg(not(target_os = "none"))]
fn register(_entry: u64, _decline: u64, _stack_lo: u64, _stack_hi: u64) -> bool {
    false
}

/// The trampolines the kernel enters.  Each calls the installed handler
/// with the five argument registers exactly as delivered, and traps at its
/// decline instruction if the handler returns — or if none is installed,
/// which [`arm_thread`]'s order makes unreachable.
#[cfg(target_os = "none")]
mod trampoline {
    use super::HANDLER;

    pub(super) fn entry() -> u64 {
        upcall_entry as *const () as usize as u64
    }

    pub(super) fn decline() -> u64 {
        upcall_decline as *const () as usize as u64
    }

    /// x86-64: `rdi`..`r8` hold the arguments and `rsp` the stack's
    /// 16-aligned top, so the `call` leaves the handler the SysV entry
    /// alignment.
    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn upcall_entry() -> ! {
        core::arch::naked_asm!(
            "mov rax, qword ptr [rip + {handler}]",
            "test rax, rax",
            "jz 2f",
            "call rax",
            "2:",
            "jmp {decline}",
            handler = sym HANDLER,
            decline = sym upcall_decline,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn upcall_decline() -> ! {
        core::arch::naked_asm!("ud2")
    }

    /// AArch64: `x0`..`x4` hold the arguments; `x16` (IP0) carries the
    /// handler, which no argument occupies.
    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn upcall_entry() -> ! {
        core::arch::naked_asm!(
            "adrp x16, {handler}",
            "ldr x16, [x16, :lo12:{handler}]",
            "cbz x16, 2f",
            "blr x16",
            "2:",
            "b {decline}",
            handler = sym HANDLER,
            decline = sym upcall_decline,
        )
    }

    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn upcall_decline() -> ! {
        core::arch::naked_asm!("udf #0")
    }

    /// RISC-V: `a0`..`a4` hold the arguments; `t0` carries the handler.
    #[cfg(target_arch = "riscv64")]
    #[unsafe(naked)]
    unsafe extern "C" fn upcall_entry() -> ! {
        core::arch::naked_asm!(
            "lla t0, {handler}",
            "ld t0, 0(t0)",
            "beqz t0, 2f",
            "jalr t0",
            "2:",
            "tail {decline}",
            handler = sym HANDLER,
            decline = sym upcall_decline,
        )
    }

    #[cfg(target_arch = "riscv64")]
    #[unsafe(naked)]
    unsafe extern "C" fn upcall_decline() -> ! {
        core::arch::naked_asm!("unimp")
    }
}

#[cfg(not(target_os = "none"))]
mod trampoline {
    pub(super) fn entry() -> u64 {
        0
    }

    pub(super) fn decline() -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn first(_: usize, _: usize, _: usize, _: usize, _: usize) {}
    unsafe extern "C" fn second(_: usize, _: usize, _: usize, _: usize, _: usize) {}

    /// One handler per process: installing it again is a no-op, another
    /// one refuses, and off the kore target no thread arms.
    #[test]
    fn one_handler_per_process_and_no_arming_off_target() {
        assert_eq!(install(first), Ok(()));
        assert_eq!(install(first), Ok(()));
        assert_eq!(install(second), Err(Refusal::HandlerTaken));
        assert_eq!(arm_thread().err(), Some(Refusal::Unsupported));
    }
}
