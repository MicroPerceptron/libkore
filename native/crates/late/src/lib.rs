//! Late binding — `res/IMAGE.md` §8's "separate library consuming the
//! passive record, never a resurrection of the stub".
//!
//! A running task links a shared object out of the DagFS store into its
//! **own** address space and calls into it.  This is the `require` bridge
//! for an interpreter (`res/DAGFS.md` §4) and the seam any C engine's late
//! loading rides.  No new kernel authority is involved: the bytes come
//! through an immutable version lease the task's filesystem view already
//! grants; the executable authority is the task's own writable→executable
//! finalization (§6); the placement is a window the task owns.
//!
//! **Where native code comes from.**  A native module runs with the task's
//! whole authority, so it enters only with the provenance a native image
//! needs, and the loader — not each caller — holds that rule on two
//! independent counts:
//!
//! - *The route.*  `<x>.so` resolves only in the task's library view
//!   ([`LIBS_VIEW`]), entered as the granted authority edge it is — never by
//!   walking the path `libs/…`, which a `libs` child of the task's own root
//!   would answer first — and [`library_grant`] refuses that edge if it
//!   would let the task write or rename beneath it.  Never the task's own
//!   tree: a tree the task can write is exactly where native code must not
//!   come from, and a squatting `<x>.so` there must not shadow the library.
//! - *The stamp.*  The object must carry
//!   [`ObjectSubtype::EXECUTABLE_IMAGE`] *and* its bytes must pass
//!   [`check_image`].  The store takes that stamp — and any later change to
//!   the stamped bytes — only through `BIND_EXECUTABLE`
//!   (`ObjectSubtype::stamp_rights`), the install authority's right, which
//!   no task's home face carries.  So the stamp is the store's word that the
//!   installer wrote these bytes; it still grants the reader nothing, and a
//!   plain file that happens to parse as an ELF is refused.
//!
//! **Source modules search; native modules do not.**  A source module
//! `<x>.lua` runs inside the interpreter's VM with only what the VM grants,
//! so `Loader::read_source` walks the caller's [`Search`] — an ordered set
//! of path prefixes the task can already open, typically its own tree
//! (`""`) then `libs` — and "not found" is "no route".  `Loader::load_native`
//! takes no search list.
//!
//! **Placement.**  Modules go above everything the startup stub mapped
//! ([`ModulePage::high_water`]) inside `geom::LINK`, each at a 64 KiB
//! boundary, below the passive page itself.  A module's text+rodata run
//! becomes one arena generation published whole (`initialize_executable_at`
//! + `map_rx_at`), its data union one exact writable mapping at its linked
//! offset (`kore_ld::late::LateLayout`), both carved off the top of the
//! task's heap supply (`kore_rt::heap::reserve_from_top`).  Modules are
//! never unloaded. RELRO is sealed read-only after relocation and before
//! initializers run; a module with TLS, or a host with TLS, refuses.
//!
//! What is host-testable — names, paths, placement arithmetic, the host
//! module view over a descriptor and a page — lives beside the metal half
//! and is tested here; the arena and mapping half is proven by the boot
//! witness.

#![no_std]

extern crate alloc;

use kore_abi::image::{ModuleDescriptor, ModulePage, seg};
pub use kore_dagfs::LIBS_VIEW;
use kore_dagfs::{
    FsError, GrantEntry, GrantId, Name, Refusal, Rights, bounds::MAX_NAME_LEN,
    bounds::MAX_PATH_DEPTH,
};
use kore_ld::apply::ApplyError;
use kore_ld::check::CheckError;
use kore_ld::late::LateError;

/// Natively loaded modules one task may hold.  Each costs one arena
/// lease, one data lease, and a slot here; eight covers an interpreter's
/// realistic native set and keeps the table on the stack.
pub const MAX_LOADED_MODULES: usize = 8;
/// The largest object the loader stages: it is read whole into a heap
/// buffer before `check_image` (which needs the headers and every table),
/// then copied into its arena, so three copies of it must fit beside the
/// engine in the launch heap (2 MiB, `LAYOUT.heap_bytes`).
pub const MAX_MODULE_BYTES: usize = 256 * 1024;
/// Prefixes one search may name.
pub const MAX_SEARCH_PREFIXES: usize = 2;
/// Every late module starts on this boundary: page-multiple, and coarse
/// enough that a module's placement reads on a serial photo.
pub const LATE_BASE_ALIGN: u64 = 0x1_0000;
/// One page.
const PAGE: u64 = 4096;
/// A composed search path: `<prefix>/<stem>.<ext>`, bounded by the walk's
/// own limits (a prefix is at most `MAX_PATH_DEPTH - 1` components of
/// `MAX_NAME_LEN`).
pub const MAX_SEARCH_PATH: usize = MAX_PATH_DEPTH * (MAX_NAME_LEN + 1);

/// What a module is loaded as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleKind {
    /// Bytes for the interpreter to compile (`<stem>.lua`).
    Source,
    /// An ET_DYN linked into the task (`<stem>.so`).
    Native,
}

impl ModuleKind {
    /// The object-name suffix.
    pub const fn suffix(self) -> &'static [u8] {
        match self {
            Self::Source => b".lua",
            Self::Native => b".so",
        }
    }
}

/// Why a load did not happen.  Every variant is a typed refusal; the
/// loader never spins, retries, or partially links.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// The stem does not spell a store name once suffixed.
    Name,
    /// No prefix of the search list holds the object — for a native module,
    /// the task has no [`LIBS_VIEW`] edge or the library holds no
    /// `<stem>.so`.
    NotFound,
    /// The task's [`LIBS_VIEW`] edge would let it write bytes or names
    /// beneath it: a native module never comes from a route its loader can
    /// change.
    LibsWritable,
    /// The object is not stamped `EXECUTABLE_IMAGE`.
    NotExecutableImage,
    /// The object exceeds [`MAX_MODULE_BYTES`], or the caller's buffer.
    TooLarge,
    /// The filesystem refused.
    Fs(FsError),
    /// The filesystem face answered out of contract.
    Pair,
    /// `check_image` refused the bytes.
    Check(CheckError),
    /// The checked image cannot be late-linked.
    Plan(LateError),
    /// The host image carries TLS: a late scope cannot reproduce the
    /// thread layout the stub built.
    HostTls,
    /// The host cannot be described for a late scope; the payload names
    /// what was missing: 1 no `.kore.ld` block, 2 no valid passive module
    /// page at the fixed address, 3 the page's first record is not this
    /// executable, 4 no `LINK` window in the geometry, 5 the page address
    /// overflowed, 6 the page has no live segment (no high-water mark),
    /// 7 a host segment could not be viewed.
    NoRecord(u8),
    /// The LINK window has no room above the last placement.
    NoWindow,
    /// The heap supply could not carve the arena or data lease.
    NoSupply,
    /// A mapping (writable, executable) was refused.
    Map,
    /// The code could not be published to instruction fetch.
    Publish,
    /// The relocated module's read-only-after-relocation protection refused.
    Seal,
    /// Relocation refused.
    Link(ApplyError),
    /// An initializer array is outside the writable mapping or malformed.
    InitializerArray,
    /// An initializer does not name executable text in the loaded scope.
    Initializer(kore_ld::functions::FunctionError),
    /// Initializers exceed the shared startup/late structural ceiling.
    TooManyInitializers,
    /// [`MAX_LOADED_MODULES`] reached.
    TooMany,
}

impl From<Refusal> for LoadError {
    fn from(refusal: Refusal) -> Self {
        match refusal {
            Refusal::Fs(FsError::NotFound) => Self::NotFound,
            Refusal::Fs(error) => Self::Fs(error),
            _ => Self::Pair,
        }
    }
}

/// The ordered prefixes a *source* load may resolve under — each one a
/// directory the task can already open; `b""` is its own tree.  A native
/// load takes none: it resolves only through [`library_grant`].
#[derive(Clone, Copy, Debug)]
pub struct Search<'a> {
    pub prefixes: &'a [&'a [u8]],
}

/// Rights a native module's route must not carry: either would let the
/// loading task change what it is about to link.
const LIBRARY_ROUTE_FORBIDS: Rights = Rights::WRITE.union(Rights::MUTATE_NAMESPACE);

/// Admit the task's library edge — the grant its view lists under
/// [`LIBS_VIEW`], if any — as the one place native modules come from.
///
/// No edge is no route ([`LoadError::NotFound`]).  An edge that would let
/// this task write a byte or a name beneath it is refused
/// ([`LoadError::LibsWritable`]) rather than trusted for its name: the
/// system view is minted `READ` alone, so a writable `libs` is some other
/// edge wearing the name (a self-share where the view's install collided,
/// say).  `SHARE` and `BIND_EXECUTABLE` change nothing the loader reads and
/// are not this check's business.
pub fn library_grant(entry: Option<GrantEntry>) -> Result<GrantId, LoadError> {
    let entry = entry.ok_or(LoadError::NotFound)?;
    if entry.is_private_root || !entry.rights.intersection(LIBRARY_ROUTE_FORBIDS).is_empty() {
        return Err(LoadError::LibsWritable);
    }
    Ok(entry.grant)
}

/// `<stem><suffix>` as a store name, or `None` past [`MAX_NAME_LEN`] or
/// for a stem that is not a name (empty, `/`, `.`, `..`, NUL).
pub fn object_name(kind: ModuleKind, stem: &[u8]) -> Option<Name> {
    let suffix = kind.suffix();
    let mut buf = [0u8; MAX_NAME_LEN];
    let len = stem.len().checked_add(suffix.len())?;
    if stem.is_empty() || len > MAX_NAME_LEN {
        return None;
    }
    buf[..stem.len()].copy_from_slice(stem);
    buf[stem.len()..len].copy_from_slice(suffix);
    Name::from_bytes(&buf[..len]).ok()
}

/// `<prefix>/<name>` into `out` (`<name>` alone for an empty prefix).
pub fn compose_path<'b>(
    out: &'b mut [u8; MAX_SEARCH_PATH],
    prefix: &[u8],
    name: &Name,
) -> Option<&'b [u8]> {
    let name = name.as_bytes();
    let mut len = 0usize;
    if !prefix.is_empty() {
        if prefix.len() + 1 > out.len() {
            return None;
        }
        out[..prefix.len()].copy_from_slice(prefix);
        out[prefix.len()] = b'/';
        len = prefix.len() + 1;
    }
    let end = len.checked_add(name.len())?;
    if end > out.len() {
        return None;
    }
    out[len..end].copy_from_slice(name);
    Some(&out[..end])
}

/// The host image as the late scope's module 0: its base from the passive
/// page's first record (verified against the executable's own name), its
/// descriptor from `.kore.ld`.
#[derive(Clone, Copy, Debug)]
pub struct HostModule<'d> {
    pub base: u64,
    pub descriptor: &'d ModuleDescriptor,
}

impl<'d> HostModule<'d> {
    /// Resolve the host from the page and the executable's descriptor and
    /// name hash; `None` when record 0 is not the executable.
    pub fn from_page(
        page: &ModulePage,
        descriptor: &'d ModuleDescriptor,
        exe_name_hash: u64,
    ) -> Option<Self> {
        let record = page.records.first()?;
        if page.is_empty() || record.name_hash.get() != exe_name_hash {
            return None;
        }
        Some(Self {
            base: record.base.get(),
            descriptor,
        })
    }
}

/// Build a `ModuleView`'s read half for a module mapped at `base`: one
/// slice per text/rodata segment, `filesz` bytes at the segment's
/// address.  `slice_at(va, len)` is the caller's — a raw mapping on
/// metal, an owned buffer in tests — so this stays host-testable.
pub fn read_views<'a>(
    descriptor: &ModuleDescriptor,
    base: u64,
    mut slice_at: impl FnMut(u64, u64) -> Option<&'a [u8]>,
) -> Option<[Option<&'a [u8]>; kore_abi::image::MAX_SEGMENTS]> {
    let mut ro = [None; kore_abi::image::MAX_SEGMENTS];
    for (i, s) in descriptor.live_segs().iter().enumerate() {
        if s.kind != seg::KIND_TEXT && s.kind != seg::KIND_RODATA {
            continue;
        }
        let va = base.checked_add(s.vaddr.get())?;
        ro[i] = Some(slice_at(va, s.filesz.get())?);
    }
    Some(ro)
}

/// Round `v` up to a page.
pub const fn page_up(v: u64) -> Option<u64> {
    match v.checked_add(PAGE - 1) {
        Some(x) => Some(x & !(PAGE - 1)),
        None => None,
    }
}

#[cfg(target_os = "none")]
mod metal;
#[cfg(target_os = "none")]
pub use metal::{Loaded, Loader, ModuleId};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_names_take_the_kinds_suffix_and_stay_bounded() {
        assert_eq!(
            object_name(ModuleKind::Native, b"lua_demo")
                .unwrap()
                .as_bytes(),
            b"lua_demo.so"
        );
        assert_eq!(
            object_name(ModuleKind::Source, b"mod").unwrap().as_bytes(),
            b"mod.lua"
        );
        assert!(object_name(ModuleKind::Source, b"").is_none());
        assert!(object_name(ModuleKind::Source, b"a/b").is_none());
        // `.so` is three bytes, `.lua` four: the stem may use the rest.
        let exact_so = [b'x'; MAX_NAME_LEN - 3];
        assert!(object_name(ModuleKind::Native, &exact_so).is_some());
        assert!(object_name(ModuleKind::Source, &exact_so).is_none());
        let over = [b'x'; MAX_NAME_LEN - 2];
        assert!(object_name(ModuleKind::Native, &over).is_none());
    }

    fn libs_entry(rights: Rights, is_private_root: bool) -> GrantEntry {
        GrantEntry {
            grant: GrantId::from_bits(7),
            rights,
            is_private_root,
            display_name: Name::from_bytes(LIBS_VIEW.as_bytes()).ok(),
        }
    }

    /// The system library view is minted `READ` alone, and that is what a
    /// native module may come through.  Re-sharing or install authority on
    /// the edge changes nothing the loader reads.
    #[test]
    fn the_read_only_library_edge_is_where_native_modules_come_from() {
        for rights in [
            Rights::READ_ONLY,
            Rights::READ.union(Rights::SHARE),
            Rights::READ.union(Rights::BIND_EXECUTABLE),
        ] {
            assert_eq!(
                library_grant(Some(libs_entry(rights, false))),
                Ok(GrantId::from_bits(7))
            );
        }
    }

    /// A task with no library view has no route to native code — and a
    /// `libs` edge the task could write through, or its own tree, is never
    /// trusted for its name: that would load code from where the loader
    /// itself can put it.
    #[test]
    fn a_missing_or_writable_library_edge_is_refused() {
        assert_eq!(library_grant(None), Err(LoadError::NotFound));
        for rights in [
            Rights::READ.union(Rights::WRITE),
            Rights::READ.union(Rights::MUTATE_NAMESPACE),
            Rights::HOME,
            Rights::ALL,
        ] {
            assert_eq!(
                library_grant(Some(libs_entry(rights, false))),
                Err(LoadError::LibsWritable),
                "{rights:?}"
            );
        }
        assert_eq!(
            library_grant(Some(libs_entry(Rights::READ_ONLY, true))),
            Err(LoadError::LibsWritable),
            "the task's own root is never the library"
        );
    }

    #[test]
    fn paths_compose_under_a_prefix_or_stand_alone() {
        let name = object_name(ModuleKind::Native, b"demo").unwrap();
        let mut buf = [0u8; MAX_SEARCH_PATH];
        assert_eq!(compose_path(&mut buf, b"", &name), Some(&b"demo.so"[..]));
        assert_eq!(
            compose_path(&mut buf, b"libs", &name),
            Some(&b"libs/demo.so"[..])
        );
    }

    /// The late loader's placement *policy* — modules land on
    /// [`LATE_BASE_ALIGN`] and never reach the passive page. The
    /// arithmetic itself belongs to `kore_ld::place` and is covered by its
    /// own suite; this pins the boundary and the limit, which are ours.
    #[test]
    fn placement_rounds_to_the_late_boundary_and_stops_below_the_limit() {
        use kore_ld::place::{PlaceError, Placer};

        let limit = 0x5800_0000u64;
        let mut from_unaligned = Placer::new(0x5001_0001, limit - 0x5001_0001);
        assert_eq!(
            from_unaligned.choose(0x3000, LATE_BASE_ALIGN),
            Ok(0x5002_0000)
        );

        let mut from_aligned = Placer::new(0x5002_0000, limit - 0x5002_0000);
        assert_eq!(
            from_aligned.choose(0x3000, LATE_BASE_ALIGN),
            Ok(0x5002_0000)
        );

        let mut exact = Placer::new(0x57ff_0000, limit - 0x57ff_0000);
        assert_eq!(exact.choose(0x1_0000, LATE_BASE_ALIGN), Ok(0x57ff_0000));

        let mut overshoot = Placer::new(0x57ff_0000, limit - 0x57ff_0000);
        assert_eq!(
            overshoot.choose(0x1_0001, LATE_BASE_ALIGN),
            Err(PlaceError::Exhausted)
        );

        let mut ceiling = Placer::new(u64::MAX - 1, 1);
        assert_eq!(
            ceiling.choose(1, LATE_BASE_ALIGN),
            Err(PlaceError::Exhausted)
        );
    }

    #[test]
    fn the_host_is_record_zero_only_when_it_names_the_executable() {
        use kore_abi::image::{ImageName, ModuleRecord, ModuleSeg};
        use zerocopy::byteorder::little_endian::U64;
        let exe = ImageName::new("lua").unwrap();
        let mut page = ModulePage::new();
        let mut record = ModuleRecord {
            name_hash: U64::new(exe.fnv1a()),
            base: U64::new(0x5000_0000),
            entry: U64::new(0),
            seg_count: 1,
            kind: kore_abi::image::kind::EXECUTABLE,
            reserved: [0; 6],
            segs: [ModuleSeg::vacant(); kore_abi::image::MAX_SEGMENTS],
        };
        record.segs[0].vaddr = U64::new(0x5000_1000);
        record.segs[0].size = U64::new(0x3000);
        assert!(page.push(record));
        let descriptor = ModuleDescriptor::empty();
        let host = HostModule::from_page(&page, &descriptor, exe.fnv1a()).unwrap();
        assert_eq!(host.base, 0x5000_0000);
        assert!(HostModule::from_page(&page, &descriptor, 1).is_none());
        assert_eq!(page.high_water(), Some(0x5000_4000));
        assert!(HostModule::from_page(&ModulePage::new(), &descriptor, exe.fnv1a()).is_none());
    }
}

/// Shared initializer-admission diagnostic for native loader consumers.
pub use kore_ld::functions::FunctionError;
