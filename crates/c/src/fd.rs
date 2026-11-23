//! Descriptors: C's spelling for "a capability I hold".
//!
//! A program's authority arrives as capabilities at fixed places: its boot
//! block names each by a role and an index, and the capability itself sits in
//! the thread's arena. `open` is the readable way to say which one is meant.
//! It grants nothing and reaches no filesystem:
//!
//! - **A name is a row of one static table** ([`NAMES`]): a path-shaped
//!   string, the bootstrap coordinate it stands for, and what kind of
//!   capability lives there. A path that is no row is `ENOENT`.
//! - **A descriptor stores the row, never the capability.** Capabilities are
//!   held per thread on Kore, and a thread is a task: `main` runs on one, a
//!   `pthread` on another, and each has its own boot block. So every use of a
//!   descriptor resolves the row's coordinate in the *calling* thread's boot
//!   block. Descriptor numbers are the process's, as C expects, and no
//!   authority crosses a thread through one: a thread that was not founded
//!   holding the capability gets `ENXIO`, whoever opened the descriptor.
//! - **`close` forgets the number.** The capability was the program's before
//!   `open` and is after `close`; nothing is revoked.
//!
//! Descriptors 0, 1 and 2 are the standard streams' numbers and are never
//! handed out; the streams themselves are `<stdio.h>`'s.
//!
//! The table is plain data over atomics, so the host suite exercises the code
//! the target runs. Finding a row's capability in the boot block is the metal
//! half ([`holding`]).

use core::ffi::c_int;
use core::sync::atomic::{AtomicU32, Ordering};

use kore_std::abi::task::role;

/// `errno` values this module and [`crate::mmap`] report: musl's numbers,
/// which kore-c-shim's `errno` and the hosted `<errno.h>` share.
pub(crate) mod errno {
    use core::ffi::c_int;
    pub(crate) const ENOENT: c_int = 2;
    pub(crate) const EIO: c_int = 5;
    pub(crate) const ENXIO: c_int = 6;
    pub(crate) const EBADF: c_int = 9;
    pub(crate) const EAGAIN: c_int = 11;
    pub(crate) const ENOMEM: c_int = 12;
    pub(crate) const EACCES: c_int = 13;
    pub(crate) const EFAULT: c_int = 14;
    pub(crate) const EBUSY: c_int = 16;
    pub(crate) const EINVAL: c_int = 22;
    pub(crate) const EMFILE: c_int = 24;
    pub(crate) const ENOTSUP: c_int = 95;
}

/// What kind of capability a name stands for: which operations its
/// descriptor has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The memory supply the program's allocator grows on. `mmap` takes a
    /// region off it, through the runtime's own reservation so the two never
    /// meet ([`crate::mmap`]).
    Memory,
    /// A virtual device of the launch's virtual-device face, by its
    /// register-block id: registers and lent memory through `mmap`, the
    /// interrupt through `read`, the doorbell through `write`
    /// ([`crate::device`]).
    Device {
        /// The device model's id, the register block's `DEVICE_ID`.
        model: u32,
    },
}

/// One row: a name and the bootstrap coordinate it stands for.
#[derive(Debug)]
pub(crate) struct Name {
    /// The path `open` matches, byte for byte.
    pub(crate) path: &'static [u8],
    /// The boot block role of the capability.
    pub(crate) role: u16,
    /// The instance within that role.
    pub(crate) index: u16,
    /// What lives there.
    pub(crate) kind: Kind,
}

/// **The names.** Each is a coordinate every thread that holds the
/// capability finds it under, so a row is true on whichever thread asks.
///
/// `/kore/memory` is `role::HEAP@0`: the launch's memory supply, which
/// kore-rt endows every thread that may allocate with a share of
/// (`WorkerCap::heap`).
///
/// `/kore/device/virtual/accel` is the virtual accelerator on the launch's
/// virtual-device face. Its row's coordinate is the face's memory, whose
/// second page is the register block: the share of it this library founds
/// the program's threads holding is what a thread parks on the interrupt
/// through. The face's kick, beside it, is what a thread rings the doorbell
/// through ([`crate::device::KICK`]).
pub(crate) static NAMES: [Name; 2] = [
    Name {
        path: b"/kore/memory",
        role: role::HEAP,
        index: 0,
        kind: Kind::Memory,
    },
    Name {
        path: b"/kore/device/virtual/accel",
        role: role::SVC_RING,
        index: crate::device::FACE,
        kind: Kind::Device {
            model: crate::device::MODEL_ACCEL,
        },
    },
];

/// The row `path` names.
pub(crate) fn resolve(path: &[u8]) -> Option<usize> {
    NAMES.iter().position(|name| name.path == path)
}

/// What an `open` asked to do with the capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Access {
    pub(crate) read: bool,
    pub(crate) write: bool,
}

/// `<fcntl.h>`'s access modes, musl's values. They are the only flags `open`
/// takes: every other flag describes a file this namespace does not have.
pub(crate) const O_RDONLY: c_int = 0;
pub(crate) const O_WRONLY: c_int = 1;
pub(crate) const O_RDWR: c_int = 2;

impl Access {
    /// The access `flags` ask for, or `EINVAL` for any flag but an access mode.
    pub(crate) fn of(flags: c_int) -> Result<Self, c_int> {
        match flags {
            O_RDONLY => Ok(Self {
                read: true,
                write: false,
            }),
            O_WRONLY => Ok(Self {
                read: false,
                write: true,
            }),
            O_RDWR => Ok(Self {
                read: true,
                write: true,
            }),
            _ => Err(errno::EINVAL),
        }
    }

    /// Whether a capability that can be read (`ACCESS`) and, if `writable`,
    /// written (`MODIFY`) permits this access.
    pub(crate) fn within(self, readable: bool, writable: bool) -> bool {
        (!self.read || readable) && (!self.write || writable)
    }

    const fn bits(self) -> u32 {
        self.read as u32 | (self.write as u32) << 1
    }

    const fn from_bits(bits: u32) -> Self {
        Self {
            read: bits & 1 != 0,
            write: bits & 2 != 0,
        }
    }
}

/// The first number `open` hands out: 0, 1 and 2 are the standard streams'.
pub(crate) const FIRST: c_int = 3;

/// How many descriptors a process may hold open. A descriptor is one word
/// naming a row, and the rows are few, so this bounds a program that forgets
/// to `close` rather than one that needs many: sixty-four entries are one
/// 256-byte table, scanned whole only by an `open` that finds it full.
pub(crate) const CAPACITY: usize = 64;

/// A descriptor table: one word per descriptor, zero when free.
///
/// A live word is [`LIVE`], the access in two bits, and the row's index.
/// Claiming and releasing are one compare-and-swap each, so threads opening
/// and closing at once never share a lock, and a descriptor is never handed
/// to two of them.
pub(crate) struct Table<const N: usize> {
    words: [AtomicU32; N],
}

const LIVE: u32 = 1 << 31;
const ACCESS_SHIFT: u32 = 16;
const ROW_MASK: u32 = (1 << ACCESS_SHIFT) - 1;

// Every row's index fits its field.
const _: () = assert!(NAMES.len() <= ROW_MASK as usize + 1);

impl<const N: usize> Table<N> {
    pub(crate) const fn new() -> Self {
        Self {
            words: [const { AtomicU32::new(0) }; N],
        }
    }

    /// Open a descriptor for `row` with `access`: the lowest free number, as
    /// POSIX has it, or `EMFILE` when every one is taken.
    pub(crate) fn claim(&self, row: usize, access: Access) -> Result<c_int, c_int> {
        let word = LIVE | access.bits() << ACCESS_SHIFT | row as u32;
        for (at, slot) in self.words.iter().enumerate() {
            if slot
                .compare_exchange(0, word, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(FIRST + at as c_int);
            }
        }
        Err(errno::EMFILE)
    }

    /// The row and access behind `fd`, or `EBADF`.
    pub(crate) fn get(&self, fd: c_int) -> Result<(usize, Access), c_int> {
        let word = self.slot(fd)?.load(Ordering::Acquire);
        if word & LIVE == 0 {
            return Err(errno::EBADF);
        }
        Ok((
            (word & ROW_MASK) as usize,
            Access::from_bits(word >> ACCESS_SHIFT),
        ))
    }

    /// Close `fd`, or `EBADF` when it was not open.
    pub(crate) fn release(&self, fd: c_int) -> Result<(), c_int> {
        match self.slot(fd)?.swap(0, Ordering::AcqRel) {
            0 => Err(errno::EBADF),
            _ => Ok(()),
        }
    }

    fn slot(&self, fd: c_int) -> Result<&AtomicU32, c_int> {
        fd.checked_sub(FIRST)
            .and_then(|at| usize::try_from(at).ok())
            .and_then(|at| self.words.get(at))
            .ok_or(errno::EBADF)
    }
}

/// The process's descriptors.
pub(crate) static TABLE: Table<CAPACITY> = Table::new();

/// What the calling thread holds at a row's coordinate.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Holding {
    /// The capability's arena slot, a name this thread alone may use.
    pub(crate) slot: kore_std::abi::session::SlotRef,
    /// It carries `ACCESS`.
    pub(crate) readable: bool,
    /// It carries `MODIFY`.
    pub(crate) writable: bool,
}

/// **The capability `name` stands for, as the calling thread holds it**:
/// its slot from this thread's boot block, and the rights its record
/// carries. `None` when the thread was not founded holding one of that kind
/// there. The record is the kernel's, read through the arena's read-only
/// alias, and every later effect binds the slot again in the kernel, so this
/// is a description and never a grant.
#[cfg(target_os = "none")]
pub(crate) fn holding(name: &Name) -> Option<Holding> {
    use kore_std::abi::cap::{ResourceKind, Rights};
    use kore_std::runtime::cap::verbs;
    let slot = kore_std::runtime::entry::thread_context()?
        .bootstrap()
        .handle(name.role, name.index)?;
    let rights = verbs::capability_rights(slot, ResourceKind::MemLease)?;
    let writable = match name.kind {
        Kind::Memory => rights.contains(Rights::MODIFY),
        // What a thread writes of a device is its doorbell: a notice on the
        // face's kick, which is a capability beside the face's memory.
        Kind::Device { .. } => crate::device::holds_kick(),
    };
    Some(Holding {
        slot,
        readable: rights.contains(Rights::ACCESS),
        writable,
    })
}

/// `open`: a descriptor for the capability `path` names, with `flags`'
/// access.
///
/// `ENOENT`: no such name. `EINVAL`: a flag other than an access mode.
/// `ENXIO`: the calling thread holds no such capability. `EACCES`: it holds
/// one whose rights do not cover the access. `EMFILE`: no descriptor is free.
#[cfg(target_os = "none")]
pub(crate) fn open(path: &[u8], flags: c_int) -> Result<c_int, c_int> {
    let row = resolve(path).ok_or(errno::ENOENT)?;
    let access = Access::of(flags)?;
    let held = holding(&NAMES[row]).ok_or(errno::ENXIO)?;
    if !access.within(held.readable, held.writable) {
        return Err(errno::EACCES);
    }
    match NAMES[row].kind {
        Kind::Memory => TABLE.claim(row, access),
        // A device is opened as well as named: the service starts the model
        // and writes its register block.
        Kind::Device { model } => {
            crate::device::open(model)?;
            TABLE
                .claim(row, access)
                .inspect_err(|_| crate::device::close())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_is_a_row_or_nothing() {
        assert_eq!(resolve(b"/kore/memory"), Some(0));
        for path in [
            &b""[..],
            b"/",
            b"/kore",
            b"/kore/memory/",
            b"/kore/memory\0",
            b"kore/memory",
            b"/KORE/MEMORY",
            b"/dev/zero",
        ] {
            assert_eq!(resolve(path), None, "{:?}", core::str::from_utf8(path));
        }
        assert_eq!((NAMES[0].role, NAMES[0].index), (role::HEAP, 0));
        assert_eq!(resolve(b"/kore/device/virtual/accel"), Some(1));
        assert_eq!(resolve(b"/kore/device/virtual"), None);
        assert_eq!(resolve(b"/kore/device/virtual/accl"), None);
        assert_eq!(NAMES[1].role, role::SVC_RING);
        assert_eq!(
            NAMES[1].kind,
            Kind::Device {
                model: u32::from_le_bytes(*b"accl")
            }
        );
    }

    #[test]
    fn only_an_access_mode_is_a_flag() {
        assert_eq!(
            Access::of(O_RDONLY),
            Ok(Access {
                read: true,
                write: false
            })
        );
        assert_eq!(
            Access::of(O_WRONLY),
            Ok(Access {
                read: false,
                write: true
            })
        );
        assert_eq!(
            Access::of(O_RDWR),
            Ok(Access {
                read: true,
                write: true
            })
        );
        // The access modes are not bits: 3 is no mode, and neither is a mode
        // with O_CREAT (0o100), O_TRUNC (0o1000) or O_CLOEXEC (0o2000000).
        for flags in [3, O_RDWR | 0o100, O_WRONLY | 0o1000, 0o2000000, -1] {
            assert_eq!(Access::of(flags), Err(errno::EINVAL), "{flags:#o}");
        }
    }

    #[test]
    fn access_is_held_to_the_capabilitys_rights() {
        let read = Access::of(O_RDONLY).unwrap();
        let write = Access::of(O_WRONLY).unwrap();
        let both = Access::of(O_RDWR).unwrap();
        // (readable, writable) of the capability.
        assert!(read.within(true, false) && read.within(true, true));
        assert!(!read.within(false, true));
        assert!(write.within(false, true) && !write.within(true, false));
        assert!(both.within(true, true));
        assert!(!both.within(true, false) && !both.within(false, true));
    }

    #[test]
    fn the_lowest_free_number_is_handed_out_from_three() {
        let table = Table::<4>::new();
        let both = Access::of(O_RDWR).unwrap();
        let read = Access::of(O_RDONLY).unwrap();
        assert_eq!(table.claim(0, both), Ok(3));
        assert_eq!(table.claim(0, read), Ok(4));
        assert_eq!(table.claim(0, both), Ok(5));
        assert_eq!(table.get(4), Ok((0, read)));
        assert_eq!(table.release(4), Ok(()));
        assert_eq!(table.get(4), Err(errno::EBADF));
        // The hole is filled before a higher number is used.
        assert_eq!(table.claim(0, both), Ok(4));
        assert_eq!(table.claim(0, both), Ok(6));
        assert_eq!(table.claim(0, both), Err(errno::EMFILE));
        assert_eq!(table.get(6), Ok((0, both)));
    }

    #[test]
    fn a_number_that_is_not_open_is_ebadf() {
        let table = Table::<4>::new();
        // The streams' numbers, a negative, one past the table, and a free one.
        for fd in [0, 1, 2, -1, c_int::MIN, 7, c_int::MAX, 3] {
            assert_eq!(table.get(fd), Err(errno::EBADF), "{fd}");
            assert_eq!(table.release(fd), Err(errno::EBADF), "{fd}");
        }
        let fd = table.claim(0, Access::of(O_RDONLY).unwrap()).unwrap();
        assert_eq!(table.release(fd), Ok(()));
        // A second close is of a descriptor that is no longer open.
        assert_eq!(table.release(fd), Err(errno::EBADF));
    }

    #[test]
    fn the_row_and_access_survive_the_word() {
        let table = Table::<2>::new();
        let row = ROW_MASK as usize;
        for flags in [O_RDONLY, O_WRONLY, O_RDWR] {
            let access = Access::of(flags).unwrap();
            let fd = table.claim(row, access).unwrap();
            assert_eq!(table.get(fd), Ok((row, access)));
            table.release(fd).unwrap();
        }
    }

    #[test]
    fn threads_never_share_a_descriptor() {
        use std::collections::HashSet;
        use std::sync::Arc;
        let table = Arc::new(Table::<64>::new());
        let access = Access::of(O_RDWR).unwrap();
        let claims: Vec<_> = (0..8)
            .map(|_| {
                let table = Arc::clone(&table);
                std::thread::spawn(move || {
                    (0..8)
                        .map(|_| table.claim(0, access).unwrap())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let all: Vec<c_int> = claims
            .into_iter()
            .flat_map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(all.len(), 64);
        assert_eq!(all.iter().copied().collect::<HashSet<_>>().len(), 64);
        assert_eq!(table.claim(0, access), Err(errno::EMFILE));
    }
}
