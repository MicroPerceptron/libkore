//! Frozen x86-64 register positions shared by the kernel and decoder.
pub const RAX: usize = 0;
pub const RBX: usize = 1;
pub const RCX: usize = 2;
pub const RDX: usize = 3;
pub const RSI: usize = 4;
pub const RDI: usize = 5;
pub const RBP: usize = 6;
pub const R8: usize = 7;
pub const R9: usize = 8;
pub const R10: usize = 9;
pub const R11: usize = 10;
pub const R12: usize = 11;
pub const R13: usize = 12;
pub const R14: usize = 13;
pub const R15: usize = 14;
pub const RIP: usize = 15;
pub const RSP: usize = 16;
pub const RFLAGS: usize = 17;
/// ModRM/REX GPR number to the frozen wire position.
pub const GPR_INDEX: [u8; 16] = [
    RAX as u8, RCX as u8, RDX as u8, RBX as u8, RSP as u8, RBP as u8, RSI as u8, RDI as u8,
    R8 as u8, R9 as u8, R10 as u8, R11 as u8, R12 as u8, R13 as u8, R14 as u8, R15 as u8,
];
const _: () = {
    assert!(RAX == super::state_index::IO_DATUM);
    assert!(RIP == super::state_index::IP);
    assert!(RFLAGS < super::GUEST_STATE_QWORDS);
};
