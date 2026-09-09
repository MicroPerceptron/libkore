//! The x86-64 fiber switch — callee-saved push, SP swap, callee-saved
//! pop (the thread-trampoline module's own `naked_asm` discipline).
//!
//! # The frame, exactly
//!
//! A suspended fiber's stack top holds, descending:
//!
//! ```text
//! [ return address ]   ← where the resume's `ret` lands
//! [ rbx ] [ rbp ] [ r12 ] [ r13 ] [ r14 ] [ r15 ]
//! ```
//!
//! `switch` pushes that set, stores `rsp` to `*save`, adopts `to_sp`,
//! pops the set, `ret`s.  `seed` lays the same shape by hand with the
//! return address aimed at [`fiber_entry_thunk`] and the entry/arg pair
//! parked in the callee-saved slots (`r12` = entry, `r13` = arg) — the
//! thunk moves them into the C ABI and calls, so a fresh fiber and a
//! resumed one are indistinguishable to `switch`.  Caller-saved and
//! vector state are deliberately dead at a cooperative ring-3 switch
//! point (§13.3 rule 4); the SysV 16-alignment contract holds at the
//! thunk's `call` by construction.

use super::FiberSwitch;

/// The proven x86-64 sequence.
pub(crate) struct X64Switch;

/// Words in the suspended frame: 6 callee-saved + the return address.
const FRAME_WORDS: u64 = 7;

/// The fresh-fiber landing: unpack `(entry, arg)` from the seeded
/// callee-saved slots into the C ABI and enter the body, which never
/// returns.  Entered by the resume's `ret` with `rsp ≡ 0 mod 16` (the
/// seed's arithmetic), so the `call` leaves the entry the SysV
/// `≡ 8 mod 16` it expects.
#[unsafe(naked)]
extern "C" fn fiber_entry_thunk() -> ! {
    core::arch::naked_asm!(
        "mov rdi, r13", // arg
        "call r12",     // entry(arg) -> !
        "ud2",          // the entry's contract says never; make a lie loud
    )
}

/// The raw swap — a REAL call frame by construction (`naked`), because
/// the shape depends on it: on entry the stack top is this call's own
/// return address, so the pushed frame is `[ret][rbx..r15]` — exactly
/// what a later resume pops and `ret`s through, and exactly what
/// [`X64Switch::seed`] lays by hand for a fresh fiber.  An inlined
/// version would have no return address to capture.
#[unsafe(naked)]
extern "C" fn raw_switch(_save: *mut u64, _to_sp: u64) {
    core::arch::naked_asm!(
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "mov [rdi], rsp",
        "mov rsp, rsi",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "ret",
    )
}

impl FiberSwitch for X64Switch {
    fn available() -> bool {
        true
    }

    unsafe fn seed(stack_top: u64, entry: extern "C" fn(usize) -> !, arg: usize) -> u64 {
        let sp = stack_top - FRAME_WORDS * 8;
        let frame = sp as *mut u64;
        // SAFETY: the contract — exclusively owned writable stack with
        // room below `stack_top`.  Layout mirrors `switch`'s pops.
        unsafe {
            frame.write(0);                                  // r15 slot (popped first)
            frame.add(1).write(0);                           // r14
            frame.add(2).write(arg as u64);                  // r13 → the thunk's arg
            frame.add(3).write(entry as usize as u64);       // r12 → the thunk's entry
            frame.add(4).write(0);                           // rbp
            frame.add(5).write(0);                           // rbx
            frame.add(6).write(fiber_entry_thunk as usize as u64); // return address
        }
        sp
    }

    unsafe fn switch(save: *mut u64, to_sp: u64) {
        // The trait contract carries the whole safety argument; the
        // naked callee owns the frame shape.
        raw_switch(save, to_sp)
    }
}
