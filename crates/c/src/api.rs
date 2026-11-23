//! The C entry points `kore/program.h` declares: what an independent C program
//! on Kore adds to `kore.h` by linking `libkore.so`.
//!
//! Compiled on every target so the header has one source of declarations;
//! off Kore there is no program to host, and each entry refuses.

use core::ffi::c_void;

use kore_ffi::{KoreFsError, KoreStatus};

/// `kore.h`'s `KoreFsStep` as a C caller may pass it: possibly null.
pub type KoreFsStep = Option<kore_ffi::KoreFsStep>;

/// The stack `main` runs on: a guarded strip of the task's stack window,
/// committed whole when the program starts. A frame deeper than this faults
/// on the guard page below it, as a stack overflow does.
///
/// The size of the strip `tcc`, a large C program whose depth its input
/// decides, measured its need under Kore's runtime at and runs in; ISO C's
/// own guarantee of 127 nested blocks (C11 §5.2.4.1) needs a small part of
/// it. Larger strips come out of the 16 MiB window the program's SPMD gangs
/// share, and would be committed at every launch.
pub const KORE_MAIN_STACK_BYTES: usize = 1 << 20;

/// Run `step` as this program's filesystem session, through the filesystem
/// face its launch carried, rooted where it was launched (its working
/// directory). The callbacks and their pointers obey `KoreFsStep`'s contract:
/// each callback runs on the program's host thread, as on any native
/// executor, while the caller waits, so a callback must not depend on the
/// caller's thread-local state.
///
/// Returns `KORE_STATUS_OK` once the callback program ended and every view it
/// left open closed. `KORE_STATUS_QUARANTINED`: a reply was lost or
/// uncertain, and no session can run again in this program.
/// `KORE_STATUS_REFUSED`: a close was refused (`*error` names why, and the
/// view stays open), the launch carried no filesystem face
/// (`KORE_FS_ERROR_UNSUPPORTED`), or the caller is not the thread `main` runs
/// on, which alone waits for the host. `KORE_STATUS_CAPACITY`: the native
/// owner could not be formed. `KORE_STATUS_INVALID_ARGUMENT`: `step` is null.
/// `*error` is written whenever `error` is not null.
///
/// # Safety
/// `step` obeys `KoreFsStep`'s pointer, serialization and no-unwind contract;
/// `user` remains valid, and the callbacks' alone, until this returns. `error`
/// is null or valid for one aligned write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_run(
    step: KoreFsStep,
    user: *mut c_void,
    error: *mut KoreFsError,
) -> KoreStatus {
    let (status, cause) = match step {
        None => (KoreStatus::InvalidArgument, KoreFsError::None),
        // SAFETY: the caller's contract, forwarded whole.
        #[cfg(target_os = "none")]
        Some(step) => unsafe { crate::program::session(step, user) },
        // No program to host off Kore.
        #[cfg(not(target_os = "none"))]
        Some(_) => {
            let _ = user;
            (KoreStatus::Refused, KoreFsError::None)
        }
    };
    if !error.is_null() {
        // SAFETY: the caller's contract: null or valid for one aligned write.
        unsafe { error.write(cause) };
    }
    status
}

/// [`KoreFdInfo::kind`]: the memory supply the program's allocator grows
/// on. `mmap` answers a fresh, zeroed region of it.
pub const KORE_FD_MEMORY: u32 = 1;
/// [`KoreFdInfo::kind`]: a virtual device (`<kore/vdev.h>`). `mmap` answers
/// its register block and memory to lend it, `read` its interrupt, and
/// `write` rings its doorbell.
pub const KORE_FD_DEVICE: u32 = 2;
/// [`KoreFdInfo::access`]: the capability may be read.
pub const KORE_FD_READ: u32 = 1 << 0;
/// [`KoreFdInfo::access`]: the capability may be written.
pub const KORE_FD_WRITE: u32 = 1 << 1;

/// Bytes in a virtual device's register block: one page.
pub const KORE_VDEV_REGISTERS_BYTES: usize = 4096;
/// `mmap` offset on a virtual device's descriptor: the register block
/// ([`KoreVdevRegisters`]).
pub const KORE_VDEV_MAP_REGISTERS: i64 = 0;
/// `mmap` offset on a virtual device's descriptor: fresh, zeroed memory the
/// driver owns and lends the device. A device address is an offset into it.
/// The device takes a loan only in its reset state, so map it before the
/// first status write.
pub const KORE_VDEV_MAP_MEMORY: i64 = 4096;
/// [`KoreVdevRegisters::magic`]: `"kvdv"`, a Kore virtual device.
pub const KORE_VDEV_MAGIC: u32 = 0x7664_766b;
/// [`KoreVdevRegisters::version`]: this layout.
pub const KORE_VDEV_VERSION: u32 = 1;
/// [`KoreVdevRegisters::vendor_id`]: `"kore"`, a device Kore hosts.
pub const KORE_VDEV_VENDOR_KORE: u32 = 0x6572_6f6b;
/// [`KoreVdevRegisters::device_id`]: `"accl"`, the virtual accelerator, a
/// virtio-accel 1.0 device. Its name is `/kore/device/virtual/accel`.
pub const KORE_VDEV_ID_ACCEL: u32 = 0x6c63_6361;
/// Status: the driver has noticed the device (Virtio 1.3 section 2.1).
pub const KORE_VDEV_STATUS_ACKNOWLEDGE: u32 = 1;
/// Status: the driver knows how to drive it.
pub const KORE_VDEV_STATUS_DRIVER: u32 = 2;
/// Status: the driver is set up and the device may serve.
pub const KORE_VDEV_STATUS_DRIVER_OK: u32 = 4;
/// Status: feature negotiation is complete.
pub const KORE_VDEV_STATUS_FEATURES_OK: u32 = 8;
/// Status, written by the device: it cannot go on and must be reset.
pub const KORE_VDEV_STATUS_DEVICE_NEEDS_RESET: u32 = 64;
/// Status: the driver has given up on the device.
pub const KORE_VDEV_STATUS_FAILED: u32 = 128;
/// Feature bit 32, `VIRTIO_F_VERSION_1`: every device offers it and a
/// driver must accept it. Written out, because the header carries this
/// expression as it stands and `1 << 32` is not a 64-bit value in C.
pub const KORE_VDEV_F_VERSION_1: u64 = 0x1_0000_0000;
/// Bytes between the last register and the device configuration.
pub const KORE_VDEV_RESERVED_BYTES: usize = 168;
/// Bytes of the device configuration area, to the end of the block.
pub const KORE_VDEV_CONFIG_AREA_BYTES: usize = 3840;

/// The register block of a Kore virtual device: what `mmap` at
/// `KORE_VDEV_MAP_REGISTERS` answers, declared `volatile` by a driver.
///
/// Every word is little-endian and has one writer. A driver never writes a
/// device register, and what it writes to its own the device reads once, at
/// the step that uses it, when the driver rings the doorbell (a `write` of
/// one 32-bit word to the descriptor). The device answers by raising an
/// interrupt: it increments `interrupts`, and a `read` of one 32-bit word
/// from the descriptor returns once the count has moved.
///
/// Bringing the device up is Virtio 1.3 section 3.1.1 over these registers,
/// with the memory lent first (`KORE_VDEV_MAP_MEMORY`). A device just opened
/// is in its reset state, both status words zero; one that is not is reset
/// first, by writing 0 to `driver_status` and ringing. Then write
/// `ACKNOWLEDGE | DRIVER` and ring; write the accepted features, add
/// `FEATURES_OK`, ring, and check `device_status` carries it; lay the queue
/// out in the lent memory, write `queue_num`, the three addresses and
/// `queue_ready`; add `DRIVER_OK` and ring. A new value of `driver_status`
/// is always answered with an interrupt, and a step the device refused is
/// one its `device_status` does not carry. The value the device last saw
/// announces nothing and is not answered, so a driver waits only for what
/// it asked.
#[repr(C)]
pub struct KoreVdevRegisters {
    /// Device: `KORE_VDEV_MAGIC`.
    pub magic: u32,
    /// Device: `KORE_VDEV_VERSION`.
    pub version: u32,
    /// Device: which model this is (`KORE_VDEV_ID_ACCEL`).
    pub device_id: u32,
    /// Device: `KORE_VDEV_VENDOR_KORE`.
    pub vendor_id: u32,
    /// Device: the features it offers.
    pub device_features: u64,
    /// Driver: the features it accepts, a subset of the offer.
    pub driver_features: u64,
    /// Driver: the status it has reached (`KORE_VDEV_STATUS_*`).
    pub driver_status: u32,
    /// Device: the status it has accepted, and its own
    /// `KORE_VDEV_STATUS_DEVICE_NEEDS_RESET`.
    pub device_status: u32,
    /// Device: the largest queue it takes.
    pub queue_num_max: u32,
    /// Driver: the queue's size, a power of two between the longest chain
    /// the device takes (its configuration says how long) and
    /// `queue_num_max`.
    pub queue_num: u32,
    /// Driver: nonzero once the queue's size and addresses are written.
    pub queue_ready: u32,
    /// Device: changes whenever the configuration area does.
    pub config_generation: u32,
    /// Driver: the descriptor table's device address.
    pub queue_desc: u64,
    /// Driver: the available ring's device address.
    pub queue_driver: u64,
    /// Driver: the used ring's device address.
    pub queue_device: u64,
    /// Device: bytes of device configuration in `config`.
    pub config_bytes: u32,
    /// Device: interrupts raised, wrapping.
    pub interrupts: u32,
    /// Unused.
    pub reserved: [u8; KORE_VDEV_RESERVED_BYTES],
    /// Device: the device configuration.
    pub config: [u8; KORE_VDEV_CONFIG_AREA_BYTES],
}

// The declaration is `kore-vdev`'s register block, offset for offset, and
// the constants are its constants: a drift fails this build.
const _: () = {
    use core::mem::{offset_of, size_of};
    use kore_std::vdev::regs;
    macro_rules! at {
        ($($field:ident = $offset:expr),* $(,)?) => {
            $(assert!(offset_of!(KoreVdevRegisters, $field) as u64 == $offset);)*
        };
    }
    at! {
        magic = regs::MAGIC,
        version = regs::VERSION,
        device_id = regs::DEVICE_ID,
        vendor_id = regs::VENDOR_ID,
        device_features = regs::DEVICE_FEATURES,
        driver_features = regs::DRIVER_FEATURES,
        driver_status = regs::DRIVER_STATUS,
        device_status = regs::DEVICE_STATUS,
        queue_num_max = regs::QUEUE_NUM_MAX,
        queue_num = regs::QUEUE_NUM,
        queue_ready = regs::QUEUE_READY,
        config_generation = regs::CONFIG_GENERATION,
        queue_desc = regs::QUEUE_DESC,
        queue_driver = regs::QUEUE_DRIVER,
        queue_device = regs::QUEUE_DEVICE,
        config_bytes = regs::CONFIG_BYTES,
        interrupts = regs::INTERRUPTS,
        config = regs::CONFIG,
    }
    assert!(size_of::<KoreVdevRegisters>() as u64 == regs::BYTES);
    assert!(KORE_VDEV_REGISTERS_BYTES as u64 == regs::BYTES);
    assert!(KORE_VDEV_MAP_REGISTERS == crate::device::MAP_REGISTERS);
    assert!(KORE_VDEV_MAP_MEMORY == crate::device::MAP_MEMORY);
    assert!(KORE_VDEV_MAGIC == regs::MAGIC_VALUE);
    assert!(KORE_VDEV_VERSION == regs::VERSION_VALUE);
    assert!(KORE_VDEV_VENDOR_KORE == regs::VENDOR_KORE);
    assert!(KORE_VDEV_ID_ACCEL == crate::device::MODEL_ACCEL);
    assert!(KORE_VDEV_STATUS_ACKNOWLEDGE == regs::STATUS_ACKNOWLEDGE);
    assert!(KORE_VDEV_STATUS_DRIVER == regs::STATUS_DRIVER);
    assert!(KORE_VDEV_STATUS_DRIVER_OK == regs::STATUS_DRIVER_OK);
    assert!(KORE_VDEV_STATUS_FEATURES_OK == regs::STATUS_FEATURES_OK);
    assert!(KORE_VDEV_STATUS_DEVICE_NEEDS_RESET == regs::STATUS_DEVICE_NEEDS_RESET);
    assert!(KORE_VDEV_STATUS_FAILED == regs::STATUS_FAILED);
    assert!(KORE_VDEV_F_VERSION_1 == regs::F_VERSION_1);
};

/// What a descriptor names, as the calling thread holds it
/// ([`kore_fd_describe`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KoreFdInfo {
    /// What kind of capability it is: `KORE_FD_MEMORY` or `KORE_FD_DEVICE`.
    pub kind: u32,
    /// What the capability's own rights permit this thread, as
    /// `KORE_FD_READ | KORE_FD_WRITE` bits: the rights it was founded with,
    /// which the access `open` asked for can only narrow.
    pub access: u32,
    /// For `KORE_FD_MEMORY`, the supply's whole length in bytes: what the
    /// allocator and every mapping share, not what is still free. For
    /// `KORE_FD_DEVICE`, the register block's length.
    pub length: u64,
}

/// Describe the capability behind `fd`, a descriptor `open` answered
/// (`<fcntl.h>`): `open` finds a capability this program was founded holding
/// by its name, and this reads what was found. Nothing is granted or changed.
///
/// Capabilities are held per thread, so the answer is the calling thread's:
/// the same descriptor may describe a capability on the thread that opened
/// it and be refused on one founded without it.
///
/// Returns `KORE_STATUS_OK` with `*out` written.
/// `KORE_STATUS_INVALID_ARGUMENT`: `out` is null. `KORE_STATUS_STALE_HANDLE`:
/// `fd` is not an open descriptor. `KORE_STATUS_REFUSED`: the calling thread
/// holds no such capability, or this is not a Kore program. A refusal leaves
/// `*out` unchanged.
///
/// # Safety
/// `out` is null or valid for one aligned write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fd_describe(
    fd: core::ffi::c_int,
    out: *mut KoreFdInfo,
) -> KoreStatus {
    if out.is_null() {
        return KoreStatus::InvalidArgument;
    }
    let Ok((row, _)) = crate::fd::TABLE.get(fd) else {
        return KoreStatus::StaleHandle;
    };
    let Some(info) = describe(&crate::fd::NAMES[row]) else {
        return KoreStatus::Refused;
    };
    // SAFETY: the caller's contract: valid for one aligned write.
    unsafe { out.write(info) };
    KoreStatus::Ok
}

/// What the calling thread holds at `name`.
#[cfg(target_os = "none")]
fn describe(name: &crate::fd::Name) -> Option<KoreFdInfo> {
    use crate::fd::Kind;
    let held = crate::fd::holding(name)?;
    let (kind, length) = match name.kind {
        Kind::Memory => (
            KORE_FD_MEMORY,
            kore_std::runtime::cap::verbs::lease_geometry(held.slot)?.1,
        ),
        Kind::Device { .. } => (KORE_FD_DEVICE, kore_std::vdev::regs::BYTES),
    };
    Some(KoreFdInfo {
        kind,
        access: if held.readable { KORE_FD_READ } else { 0 }
            | if held.writable { KORE_FD_WRITE } else { 0 },
        length,
    })
}

/// Off Kore no thread holds a capability.
#[cfg(not(target_os = "none"))]
fn describe(_name: &crate::fd::Name) -> Option<KoreFdInfo> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn never(
        _: *mut kore_ffi::KoreFsContext,
        _: *const kore_ffi::KoreFsCompletion,
        _: *mut c_void,
    ) {
        panic!("a session ran with no program to host");
    }

    /// The headers carry a constant's expression as Rust wrote it, and C
    /// reads `1 << n` as an `int`: a shift that does not fit one is not the
    /// value Rust meant. `KORE_VDEV_F_VERSION_1` was `(1 << 32)` once, which
    /// a driver compiled as zero.
    #[test]
    fn no_header_constant_shifts_past_an_int() {
        for (name, header) in [
            (
                "kore/program.h",
                include_str!("../../../extern/c/include/kore/program.h"),
            ),
            ("kore.h", include_str!("../../../extern/c/include/kore.h")),
        ] {
            for line in header.lines().filter(|line| line.starts_with("#define ")) {
                let mut rest = line;
                while let Some(at) = rest.find("<<") {
                    rest = &rest[at + 2..];
                    let digits: std::string::String = rest
                        .trim_start()
                        .chars()
                        .take_while(char::is_ascii_digit)
                        .collect();
                    let shift: u32 = digits.parse().unwrap_or(u32::MAX);
                    assert!(shift < 31, "{name}: `{line}` shifts an int by {digits}");
                }
            }
        }
    }

    #[test]
    fn a_null_step_is_an_invalid_argument_and_the_error_is_written() {
        let mut error = KoreFsError::Busy;
        // SAFETY: a null step names no callback; `error` is valid.
        let status = unsafe { kore_fs_run(None, core::ptr::null_mut(), &mut error) };
        assert_eq!(status, KoreStatus::InvalidArgument);
        assert_eq!(error, KoreFsError::None);
    }

    #[test]
    fn with_no_program_to_host_a_session_is_refused_and_never_runs() {
        let mut error = KoreFsError::Busy;
        // SAFETY: `never` is never called; `error` is valid.
        let status = unsafe { kore_fs_run(Some(never), core::ptr::null_mut(), &mut error) };
        assert_eq!(status, KoreStatus::Refused);
        assert_eq!(error, KoreFsError::None);
        // A null error pointer is permitted.
        // SAFETY: as above.
        let status =
            unsafe { kore_fs_run(Some(never), core::ptr::null_mut(), core::ptr::null_mut()) };
        assert_eq!(status, KoreStatus::Refused);
    }

    #[test]
    fn describing_refuses_by_cause_and_leaves_the_answer_alone() {
        let untouched = KoreFdInfo {
            kind: 7,
            access: 7,
            length: 7,
        };
        let mut info = untouched;
        // SAFETY: null is permitted; `info` is valid.
        unsafe {
            assert_eq!(
                kore_fd_describe(3, core::ptr::null_mut()),
                KoreStatus::InvalidArgument
            );
            // The streams' numbers and an unopened one are not descriptors.
            for fd in [0, 1, 2, -1, 63, 1 << 20] {
                assert_eq!(kore_fd_describe(fd, &mut info), KoreStatus::StaleHandle);
            }
        }
        // An open descriptor, off Kore: no thread holds a capability.
        let access = crate::fd::Access::of(crate::fd::O_RDWR).unwrap();
        let fd = crate::fd::TABLE.claim(0, access).unwrap();
        // SAFETY: `info` is valid.
        assert_eq!(
            unsafe { kore_fd_describe(fd, &mut info) },
            KoreStatus::Refused
        );
        crate::fd::TABLE.release(fd).unwrap();
        assert_eq!(info, untouched);
    }
}
