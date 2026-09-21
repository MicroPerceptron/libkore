//! Parent-owned crash capture and a bounded plain/ANSI diagnostic renderer.
//!
//! Construct the buffer before START, collect before revoking the process
//! handle, then persist through the parent's existing filesystem authority.
//! The kernel reports immutable evidence; the parent owns presentation/storage.
use core::fmt::{self, Write};
use kore_abi::crash::{Architecture, CaptureFlags, CrashRecord, FaultKind, MIN_BUFFER_BYTES};

/// A prepared caller-owned destination. It may be a filesystem transfer window.
/// Construction validates capacity and clears prior evidence. No allocation.
pub struct CrashBuffer<'a> {
    bytes: &'a mut [u8],
    captured: bool,
}
impl<'a> CrashBuffer<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Option<Self> {
        if bytes.len() < MIN_BUFFER_BYTES {
            return None;
        }
        bytes[..MIN_BUFFER_BYTES].fill(0);
        Some(Self {
            bytes,
            captured: false,
        })
    }
    pub fn record(&self) -> Option<CrashRecord> {
        self.captured
            .then(|| CrashRecord::decode(self.bytes))
            .flatten()
    }
    /// Only initialized report bytes; extra capacity never exposes stale memory.
    pub fn captured_bytes(&self) -> Option<&[u8]> {
        self.captured.then_some(&self.bytes[..MIN_BUFFER_BYTES])
    }
    /// Collect after joining, while the owning process handle is still held.
    /// An ordinary exit returns false. Failures leave the buffer unpublished.
    #[cfg(target_os = "none")]
    pub fn collect(
        &mut self,
        task: &crate::cap::TaskToken,
    ) -> Result<bool, kore_abi::session::Status> {
        use crate::{SessionBuilder, Transport, transport::TrapTransport};
        use kore_abi::{
            cap::CapRecord,
            session::{CRASH_CHUNK_BYTES, CrashChunk, Status},
        };
        self.captured = false;
        let report = crate::entry::geometry::report_slot().ok_or(Status::Denied)?;
        for (index, out) in self.bytes[..MIN_BUFFER_BYTES]
            .chunks_exact_mut(CRASH_CHUNK_BYTES)
            .enumerate()
        {
            let mut session = SessionBuilder::new();
            session
                .read_crash(task.slot(), report, index as u32)
                .map_err(|_| Status::Malformed)?;
            let completion = TrapTransport::new()
                .commit_now(&session)
                .map_err(|_| Status::Malformed)?;
            match completion.status() {
                Status::Ok => {}
                Status::Exited => return Ok(false),
                status => return Err(status),
            }
            let pointer =
                crate::entry::geometry::slot_record_va(report).ok_or(Status::Malformed)?;
            // SAFETY: this thread's kernel-authored report alias after commit.
            let value = unsafe { core::ptr::read_volatile(pointer as *const CapRecord) };
            let chunk = CrashChunk::decode(&value).ok_or(Status::Malformed)?;
            if chunk.index as usize != index {
                return Err(Status::Malformed);
            }
            out.copy_from_slice(&chunk.bytes);
        }
        if CrashRecord::decode(self.bytes).is_none() {
            return Err(Status::Malformed);
        }
        self.captured = true;
        Ok(true)
    }
}

/// Register resident backing before starting the process. The lease range must
/// be unmapped and unclaimed. Registration excludes aliases until the process
/// is reaped; then the parent may map it or lend it to a filesystem writer.
#[cfg(target_os = "none")]
pub fn register(
    task: &crate::cap::TaskToken,
    lease: &crate::cap::StashedLease,
    offset: u64,
    length: u64,
) -> Result<(), kore_abi::session::Status> {
    use crate::{SessionBuilder, Transport, transport::TrapTransport};
    use kore_abi::session::Status;
    let mut session = SessionBuilder::new();
    session
        .bind_crash(
            task.slot(),
            lease.slot().ok_or(Status::Revoked)?,
            offset,
            length,
        )
        .map_err(|_| Status::Malformed)?;
    let result = TrapTransport::new()
        .commit_now(&session)
        .map_err(|_| Status::Malformed)?;
    match result.status() {
        Status::Ok => Ok(()),
        status => Err(status),
    }
}

/// Returning executables share the same recursion-safe diagnostic path.
pub fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    if crate::entry::begin_panic() {
        report_panic_info(info);
    }
    crate::thread::exit(crate::ExitCode::PANIC)
}

/// Best-effort software report. The entry panic handler has claimed recursion
/// protection before calling this; no formatter allocation or filesystem I/O.
#[cfg(target_os = "none")]
pub(crate) fn report_panic_info(info: &core::panic::PanicInfo<'_>) {
    use crate::{SessionBuilder, Transport, transport::TrapTransport};
    use kore_abi::task::role;
    struct Message {
        bytes: [u8; kore_abi::crash::MESSAGE_BYTES],
        len: usize,
    }
    impl Write for Message {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            let mut len = value.len().min(self.bytes.len() - self.len);
            while !value.is_char_boundary(len) {
                len -= 1;
            }
            self.bytes[self.len..self.len + len].copy_from_slice(&value.as_bytes()[..len]);
            self.len += len;
            if len == value.len() {
                Ok(())
            } else {
                Err(fmt::Error)
            }
        }
    }
    let Some(rt) = crate::entry::thread_context() else {
        return;
    };
    let Some(task) = rt.bootstrap().handle(role::SELF, 0) else {
        return;
    };
    let mut message = Message {
        bytes: [0; kore_abi::crash::MESSAGE_BYTES],
        len: 0,
    };
    let truncated = write!(message, "{info}").is_err();
    let Ok(message) = core::str::from_utf8(&message.bytes[..message.len]) else {
        return;
    };
    let mut session = SessionBuilder::new();
    if session
        .report_panic_with_truncation(task, message, truncated)
        .is_err()
    {
        return;
    }
    let _ = TrapTransport::new().commit_now(&session);
}
#[cfg(not(target_os = "none"))]
pub(crate) fn report_panic_info(_: &core::panic::PanicInfo<'_>) {}

/// Escape control characters in untrusted panic strings and source names.
fn text(out: &mut impl Write, value: &str) -> fmt::Result {
    for c in value.chars() {
        if c.is_control() {
            write!(out, "\\u{{{:x}}}", c as u32)?;
        } else {
            out.write_char(c)?;
        }
    }
    Ok(())
}

/// Presentation always preserves the raw evidence, including unknown registers.
/// Symbolication can add verified source annotations without changing the record.
pub fn render(record: &CrashRecord, out: &mut impl Write, color: bool) -> fmt::Result {
    if !record.validate() {
        return out.write_str("error[crash-record]: invalid or unsupported record\n");
    }
    let flags = CaptureFlags::from_bits_retain(record.flags.get());
    if record.task_id_quality != kore_abi::task::TaskIdQuality::HardwareOrFirmware as u8 {
        writeln!(
            out,
            "note: task ID seed quality: {:?}",
            kore_abi::task::TaskIdQuality::decode(record.task_id_quality).ok_or(fmt::Error)?
        )?;
    }
    let kind = FaultKind::decode(record.kind.get()).ok_or(fmt::Error)?;
    let title = match kind {
        FaultKind::Memory => "memory access fault",
        FaultKind::InvalidInstruction => "invalid instruction",
        FaultKind::Arithmetic => "arithmetic exception",
        FaultKind::ExtendedStateDenied => "extended-state access denied",
        FaultKind::Protection => "protection exception",
        FaultKind::Breakpoint => "unhandled breakpoint or debug trap",
        FaultKind::Panic => "Rust panic",
    };
    if color {
        out.write_str("\x1b[1;31m")?;
    }
    write!(out, "error[process-fault]: {title}")?;
    if color {
        out.write_str("\x1b[0m")?;
    }
    write!(out, "\n  = task: ")?;
    let name = record.task_name();
    for ch in (if name.is_empty() { "<unnamed>" } else { name }).chars() {
        for escaped in ch.escape_debug() {
            out.write_char(escaped)?;
        }
    }
    writeln!(out, " ({})", record.task)?;
    writeln!(
        out,
        "  = PC: {:#018x}  SP: {:#018x}",
        record.pc.get(),
        record.sp.get()
    )?;
    if flags.contains(CaptureFlags::ADDRESS_VALID) {
        if color {
            out.write_str("\x1b[1;31m")?;
        }
        writeln!(
            out,
            "  = address: {:#018x}  <- faulting address",
            record.address.get()
        )?;
        if color {
            out.write_str("\x1b[0m")?;
        }
    }
    write!(
        out,
        "  = cause: {:#x}  error: {:#x}",
        record.syndrome.get(),
        record.error.get()
    )?;
    if kind == FaultKind::Memory {
        let access = if flags.contains(CaptureFlags::EXECUTE) {
            "execute"
        } else if flags.contains(CaptureFlags::WRITE) {
            "write"
        } else {
            "read"
        };
        write!(out, "  user / {access}")?;
        if flags.contains(CaptureFlags::PRESENT) {
            out.write_str(" / protection")?;
        }
    }
    out.write_char('\n')?;
    let message = core::str::from_utf8(&record.message[..record.message_len.get() as usize])
        .map_err(|_| fmt::Error)?;
    if !message.is_empty() {
        out.write_str("  = message: ")?;
        text(out, message)?;
        out.write_char('\n')?;
    }
    let arch = Architecture::decode(record.architecture.get()).ok_or(fmt::Error)?;
    const X86: [&str; 16] = [
        "RAX", "RDX", "RCX", "RBX", "RSI", "RDI", "RBP", "RSP", "R8", "R9", "R10", "R11", "R12",
        "R13", "R14", "R15",
    ];
    out.write_str("\nregisters:\n")?;
    let mut columns = 0;
    for (i, register) in record.registers.iter().enumerate() {
        if record.registers_valid.get() & (1 << i) == 0 {
            continue;
        }
        match arch {
            Architecture::X86_64 => write!(out, "  {}", X86.get(i).copied().unwrap_or("unknown"))?,
            Architecture::Aarch64 if i == 31 => out.write_str("  SP")?,
            _ => write!(out, "  X{i}")?,
        }
        write!(out, ": {:#018x}", register.get())?;
        columns += 1;
        if columns % 3 == 0 {
            out.write_char('\n')?;
        }
    }
    if columns % 3 != 0 {
        out.write_char('\n')?;
    }
    if record.frame_count.get() != 0 {
        out.write_str("\nstack:\n")?;
        for (i, pc) in record.frames[..record.frame_count.get() as usize]
            .iter()
            .enumerate()
        {
            writeln!(out, "  {i:2}  {:#018x}", pc.get())?;
        }
    }
    if flags.intersects(
        CaptureFlags::STACK_TRUNCATED
            | CaptureFlags::MESSAGE_TRUNCATED
            | CaptureFlags::REGISTERS_PARTIAL,
    ) {
        out.write_str("\nnote: report contains partial evidence\n")?;
    }
    if flags.contains(CaptureFlags::BUFFER_CAPTURED) {
        out.write_str("note: parent RAM destination captured\n")?;
    }
    writeln!(
        out,
        "note: {title}; snapshot captured (durability reported separately)"
    )?;
    out.write_str("task: ")?;
    for ch in (if name.is_empty() { "<unnamed>" } else { name }).chars() {
        for escaped in ch.escape_debug() {
            out.write_char(escaped)?;
        }
    }
    writeln!(out, " ({})", record.task)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_and_untrusted_text_cannot_escape_the_report() {
        let mut record =
            CrashRecord::new(Architecture::X86_64, FaultKind::Panic, 0x1234, 0x5678, 0);
        record.set_task_name("hello\x1b[2J\nworker");
        record.set_message("oops\x1b[2J\nforged");
        let mut out = std::string::String::new();
        render(&record, &mut out, false).unwrap();
        assert!(out.contains("oops\\u{1b}[2J\\u{a}forged"));
        assert!(!out.contains('\x1b'));
        assert!(out.contains("hello\\u{1b}[2J\\nworker"));
        record.frame_count = zerocopy::byteorder::little_endian::U32::new(99);
        out.clear();
        render(&record, &mut out, false).unwrap();
        assert!(out.contains("invalid or unsupported"));
    }
    #[test]
    fn preparing_a_buffer_does_not_publish_old_bytes() {
        let mut small = [0; MIN_BUFFER_BYTES - 1];
        assert!(CrashBuffer::new(&mut small).is_none());
        let mut bytes = [0xff; MIN_BUFFER_BYTES];
        let buffer = CrashBuffer::new(&mut bytes).unwrap();
        assert!(buffer.record().is_none());
        assert!(buffer.captured_bytes().is_none());
    }
}
