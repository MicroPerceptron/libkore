//! Versioned, bounded evidence for a terminated user process.
//!
//! All integers are little endian. A record is immutable after publication;
//! publication and disk durability are separate transport acknowledgements.
//! Missing registers are identified by a validity bitmap, never zero-filled
//! values presented as observations. No pointer in this record conveys authority.
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const VERSION: u16 = 2;
pub const MIN_BUFFER_BYTES: usize = 1024;
pub const MAX_FRAMES: usize = 16;
pub const MESSAGE_BYTES: usize = 256;
/// Stable panic outcome shared with the language runtime.
pub const PANIC_EXIT_STATUS: i32 = i32::MIN + 0x7f;
/// Outside the private service-rung band and application-reserved outcomes.
pub const FAULT_EXIT_STATUS: i32 = i32::MIN + 0x1_0000;

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    X86_64 = 62,
    Aarch64 = 183,
    Riscv64 = 243,
}
impl Architecture {
    pub const fn decode(value: u16) -> Option<Self> {
        match value {
            62 => Some(Self::X86_64),
            183 => Some(Self::Aarch64),
            243 => Some(Self::Riscv64),
            _ => None,
        }
    }
}
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind {
    Memory = 1,
    InvalidInstruction = 2,
    Arithmetic = 3,
    ExtendedStateDenied = 4,
    Protection = 5,
    Breakpoint = 6,
    Panic = 7,
}
impl FaultKind {
    pub const fn decode(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Memory),
            2 => Some(Self::InvalidInstruction),
            3 => Some(Self::Arithmetic),
            4 => Some(Self::ExtendedStateDenied),
            5 => Some(Self::Protection),
            6 => Some(Self::Breakpoint),
            7 => Some(Self::Panic),
            _ => None,
        }
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CaptureFlags: u16 {
        const ADDRESS_VALID = 1;
        const WRITE = 2;
        const EXECUTE = 4;
        const PRESENT = 8;
        const STACK_TRUNCATED = 16;
        const MESSAGE_TRUNCATED = 32;
        const REGISTERS_PARTIAL = 64;
        /// The parent-registered RAM buffer received the record before exit.
        const BUFFER_CAPTURED = 128;
    }
}

/// x86 registers follow DWARF numbering (RAX,RDX,RCX,RBX,RSI,RDI,RBP,RSP,
/// R8..R15); AArch64 uses X0..X30,SP; RISC-V uses X0..X31.
/// `pc`, `sp`, and `error` are always present. `address` requires ADDRESS_VALID.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CrashRecord {
    magic: [u8; 8],
    pub version: U16,
    pub architecture: U16,
    pub kind: U16,
    pub flags: U16,
    reserved_identity: U64,
    pub pc: U64,
    pub sp: U64,
    pub address: U64,
    pub error: U64,
    pub registers_valid: U64,
    pub registers: [U64; 32],
    pub frame_count: U32,
    pub message_len: U32,
    pub frames: [U64; MAX_FRAMES],
    pub message: [u8; MESSAGE_BYTES],
    /// Architecture exception vector/cause; zero for a software panic.
    pub syndrome: U64,
    task_name_len: u8,
    task_name: [u8; 63],
    pub task: crate::task::TaskId,
    pub task_id_quality: u8,
    reserved: [u8; 223],
}
impl CrashRecord {
    pub fn set_task_name(&mut self, name: &str) {
        let value = crate::task::TaskName::new(name).expect("bounded kernel task name");
        self.task_name.fill(0);
        self.task_name_len = value.as_str().len() as u8;
        self.task_name[..name.len()].copy_from_slice(name.as_bytes());
    }
    pub fn task_name(&self) -> &str {
        self.task_name
            .get(..self.task_name_len as usize)
            .and_then(|b| core::str::from_utf8(b).ok())
            .unwrap_or("")
    }
    pub const fn new(arch: Architecture, kind: FaultKind, pc: u64, sp: u64, error: u64) -> Self {
        Self {
            magic: *b"KORECRSH",
            version: U16::new(VERSION),
            architecture: U16::new(arch as u16),
            kind: U16::new(kind as u16),
            flags: U16::new(CaptureFlags::REGISTERS_PARTIAL.bits()),
            reserved_identity: U64::new(0),
            pc: U64::new(pc),
            sp: U64::new(sp),
            address: U64::new(0),
            error: U64::new(error),
            registers_valid: U64::new(0),
            registers: [U64::new(0); 32],
            frame_count: U32::new(0),
            message_len: U32::new(0),
            frames: [U64::new(0); MAX_FRAMES],
            message: [0; MESSAGE_BYTES],
            syndrome: U64::new(0),
            task_name_len: 0,
            task_name: [0; 63],
            task: crate::task::TaskId::from_bytes([0; 16]),
            task_id_quality: 0,
            reserved: [0; 223],
        }
    }
    pub fn set_register(&mut self, index: usize, value: u64) -> bool {
        let Some(register) = self.registers.get_mut(index) else {
            return false;
        };
        *register = U64::new(value);
        self.registers_valid = U64::new(self.registers_valid.get() | (1 << index));
        true
    }
    pub fn add_flags(&mut self, flags: CaptureFlags) {
        self.flags = U16::new(self.flags.get() | flags.bits());
    }
    pub fn set_address(&mut self, address: u64) {
        self.address = U64::new(address);
        self.add_flags(CaptureFlags::ADDRESS_VALID);
    }
    pub fn set_message(&mut self, message: &str) {
        let mut len = message.len().min(MESSAGE_BYTES);
        while !message.is_char_boundary(len) {
            len -= 1;
        }
        self.message.fill(0);
        self.message[..len].copy_from_slice(&message.as_bytes()[..len]);
        self.message_len = U32::new(len as u32);
        let mut flags = self.flags.get() & !CaptureFlags::MESSAGE_TRUNCATED.bits();
        if len < message.len() {
            flags |= CaptureFlags::MESSAGE_TRUNCATED.bits();
        }
        self.flags = U16::new(flags);
    }
    pub fn validate(&self) -> bool {
        self.magic == *b"KORECRSH"
            && self.version.get() == VERSION
            && Architecture::decode(self.architecture.get()).is_some()
            && FaultKind::decode(self.kind.get()).is_some()
            && CaptureFlags::from_bits(self.flags.get()).is_some()
            && self.registers_valid.get() >> 32 == 0
            && self.frame_count.get() as usize <= MAX_FRAMES
            && self.message_len.get() as usize <= MESSAGE_BYTES
            && self.reserved == [0; 223]
            && self.reserved_identity.get() == 0
            && crate::task::TaskIdQuality::decode(self.task_id_quality).is_some()
            && self.task_name_len <= 63
            && core::str::from_utf8(&self.task_name[..(self.task_name_len as usize).min(63)])
                .is_ok()
            && self.task_name[(self.task_name_len as usize).min(63)..]
                .iter()
                .all(|b| *b == 0)
            && self
                .message
                .get(..self.message_len.get() as usize)
                .is_some_and(|s| core::str::from_utf8(s).is_ok())
    }
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let record = Self::read_from_bytes(bytes.get(..MIN_BUFFER_BYTES)?).ok()?;
        record.validate().then_some(record)
    }
    pub fn encode_into(&self, bytes: &mut [u8]) -> bool {
        if !self.validate() {
            return false;
        }
        let Some(out) = bytes.get_mut(..MIN_BUFFER_BYTES) else {
            return false;
        };
        out.copy_from_slice(self.as_bytes());
        true
    }
}
const _: () = assert!(core::mem::size_of::<CrashRecord>() == MIN_BUFFER_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn name_roundtrips_and_rejects_invalid_utf8_and_padding() {
        let mut r = CrashRecord::new(Architecture::X86_64, FaultKind::Panic, 1, 2, 0);
        r.set_task_name("héllo");
        let mut bytes = [0; MIN_BUFFER_BYTES];
        assert!(r.encode_into(&mut bytes));
        assert_eq!(bytes[720], 6);
        assert_eq!(CrashRecord::decode(&bytes).unwrap().task_name(), "héllo");
        bytes[721] = 255;
        assert!(CrashRecord::decode(&bytes).is_none());
        assert!(r.encode_into(&mut bytes));
        bytes[783] = 1;
        assert!(CrashRecord::decode(&bytes).is_none());
    }
    #[test]
    fn refuses_short_unknown_and_corrupt_records() {
        let r = CrashRecord::new(Architecture::X86_64, FaultKind::Memory, 0x1234, 0x8000, 6);
        let mut bytes = [0; MIN_BUFFER_BYTES];
        assert!(r.encode_into(&mut bytes));
        assert_eq!(CrashRecord::decode(&bytes), Some(r));
        assert!(CrashRecord::decode(&bytes[..1023]).is_none());
        bytes[8] = 3;
        assert!(CrashRecord::decode(&bytes).is_none());
        assert!(r.encode_into(&mut bytes));
        bytes[1023] = 1;
        assert!(CrashRecord::decode(&bytes).is_none());
    }
    #[test]
    fn validity_distinguishes_zero_from_missing_and_text_is_utf8_bounded() {
        let mut r = CrashRecord::new(Architecture::Aarch64, FaultKind::Panic, 0, 0, 0);
        assert!(r.set_register(0, 0));
        assert!(!r.set_register(32, 9));
        assert_eq!(r.registers_valid.get(), 1);
        r.set_message(&"€".repeat(100));
        assert_eq!(r.message_len.get(), 255);
        assert!(r.validate());
        assert_ne!(r.flags.get() & CaptureFlags::MESSAGE_TRUNCATED.bits(), 0);
        r.set_message("short");
        assert_eq!(r.message_len.get(), 5);
        assert_eq!(r.flags.get() & CaptureFlags::MESSAGE_TRUNCATED.bits(), 0);
    }
}

#[cfg(test)]
mod transport_tests {
    use crate::{
        cap::{Cid, Rights},
        session::{CRASH_CHUNK_BYTES, CrashChunk},
    };
    #[test]
    fn chunks_cannot_smuggle_authority_or_claim_an_out_of_range_index() {
        let chunk = CrashChunk {
            index: 63,
            bytes: [0xff; CRASH_CHUNK_BYTES],
        };
        let record = chunk.encode();
        assert_eq!(CrashChunk::decode(&record), Some(chunk));
        let mut bad = record;
        bad.cid = Cid(1);
        assert!(CrashChunk::decode(&bad).is_none());
        let mut bad = record;
        bad.rights = Rights::ACCESS;
        assert!(CrashChunk::decode(&bad).is_none());
        let mut bad = record;
        bad.scope_rid = 64;
        assert!(CrashChunk::decode(&bad).is_none());
        let mut bad = record;
        bad.chain_len = 1;
        assert!(CrashChunk::decode(&bad).is_none());
    }
}
