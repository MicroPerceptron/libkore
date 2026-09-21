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

/// End a returning program. Live workers are stopped without their cooperation;
/// their stacks and shared storage are reclaimed by kernel task teardown.
/// User finalizers run only after a clean worker barrier. A forced barrier skips
/// them because interrupted code can hold application locks or partial state.
#[cfg(target_os = "none")]
pub fn finish(code: ExitCode) -> ! {
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
    if crate::entry::geometry::begin_process_exit() && !report.forced {
        crate::entry::geometry::run_finalizers();
    }
    crate::thread::exit(code)
}

/// Host code cannot stand in for a founded process.
#[cfg(not(target_os = "none"))]
pub fn finish(_code: ExitCode) -> ! {
    crate::entry::refuse()
}
