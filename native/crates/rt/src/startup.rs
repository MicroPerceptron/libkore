//! Returning-program lifecycle, shared by language entry adapters.
//! A thread exit remains a thread operation; `finish` is the process-owner path.

use crate::ExitCode;

/// Physically drain peers and close future task admission. The report is
/// obtained through the caller's authenticated SELF capability, never a task ID.
#[cfg(target_os = "none")]
pub fn drain() -> Result<kore_abi::session::ProcessDrainReport, kore_abi::session::Status> {
    use crate::{SessionBuilder, Transport, transport::TrapTransport};
    use kore_abi::{cap::CapRecord, session::Status, task::role};
    let rt = crate::entry::thread_context().ok_or(Status::Denied)?;
    let self_handle = rt.bootstrap().handle(role::SELF, 0).ok_or(Status::Denied)?;
    let report = crate::entry::geometry::report_slot().ok_or(Status::Denied)?;
    let mut session = SessionBuilder::new();
    session
        .drain_process(self_handle, report)
        .map_err(|_| Status::Malformed)?;
    let completion = TrapTransport::new()
        .commit_now(&session)
        .map_err(|_| Status::Malformed)?;
    if completion.status() != Status::Ok {
        return Err(completion.status());
    }
    let pointer = crate::entry::geometry::slot_record_va(report).ok_or(Status::Malformed)?;
    // SAFETY: the completed invocation wrote this thread's kernel-authored
    // read-only report slot. No peer can write its arena alias.
    let record = unsafe { core::ptr::read_volatile(pointer as *const CapRecord) };
    kore_abi::session::ProcessDrainReport::decode(&record).ok_or(Status::Malformed)
}

/// Raise the process barrier a returning program ends behind, and return to the
/// caller, which is then the process's last running thread.
///
/// Every live worker is stopped without its cooperation, and task admission is
/// closed ([`drain`]). With `finalize`, the user finalizers then run, once per
/// process and only after a clean barrier: a forced one skips them, because
/// interrupted code can hold application locks or partial state.
///
/// This is [`finish`] without its exit, for a program whose last word is not an
/// [`ExitCode`]: a C program's status is the wire's `i32`, which its C library
/// commits itself once this returns. Only the process owner (the founded
/// thread) may drain; a barrier the kernel refuses ends the calling thread with
/// [`ExitCode::INTERNAL`], as [`finish`] does, since the process cannot be
/// ended cleanly and a status reported over running workers would be one the
/// process had not reached.
#[cfg(target_os = "none")]
pub fn barrier(finalize: bool) {
    use kore_abi::session::Status;
    let report = loop {
        match drain() {
            Ok(report) => break report,
            Err(Status::Conflict) => continue,
            Err(Status::WouldBlock) => {
                crate::pace::cede();
            }
            Err(_) => crate::thread::exit(ExitCode::INTERNAL),
        }
    };
    // Marked whether or not this end finalizes, so no later end can run them.
    let first = crate::entry::geometry::begin_process_exit();
    if finalize && first && !report.forced {
        crate::entry::geometry::run_finalizers();
    }
}

/// End a returning program. Live workers are stopped without their cooperation;
/// their stacks and shared storage are reclaimed by kernel task teardown.
/// User finalizers run only after a clean worker barrier ([`barrier`]).
#[cfg(target_os = "none")]
pub fn finish(code: ExitCode) -> ! {
    barrier(true);
    crate::thread::exit(code)
}

/// Host code cannot stand in for a founded process.
#[cfg(not(target_os = "none"))]
pub fn finish(_code: ExitCode) -> ! {
    crate::entry::refuse()
}
