//! The metal half: read a version out of the store, carve the two
//! windows, publish, map, relocate, and remember the module.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use kore_abi::image::{ImageName, ModuleDescriptor};
use kore_abi::task::layout::Window;
use kore_dagfs::{Fs, FsRequest, FsResponse, Name, ObjectSubtype, Refusal, View, WALK_READ};
use kore_ld::apply::{
    ModuleImage, ModuleView, SymbolScope, TlsLayout, apply_module, lookup_symbol,
};
use kore_ld::check::check_image;
use kore_ld::late::LateLayout;
use kore_ld::reloc::HostArch;
use kore_ld::store::LinkArena;
use kore_rt::cap::verbs;
use kore_rt::cap::{ExecutableCode, ExecutableMapping, MappedLeaseView, StashedLease};
use kore_rt::entry::Rt;

use crate::{
    HostModule, LoadError, MAX_LOADED_MODULES, MAX_MODULE_BYTES, MAX_SEARCH_PATH, ModuleKind,
    Search, compose_path, object_name, read_views,
};

/// Bytes the TLS layout's tables may carve for a scope of at most
/// `MAX_LOADED_MODULES + 1` TLS-free modules — a few words per module.
const LINK_ARENA_BYTES: usize = 1024;

/// A loaded module's index in the loader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleId(u8);

/// One linked module, alive for the task.
pub struct Loaded {
    name: Name,
    base: u64,
    code_vaddr: u64,
    descriptor: ModuleDescriptor,
    /// The published generation.  Leaked on purpose: a module is never
    /// unloaded (v1), and the R|X mapping below borrows it for `'static`.
    _code: &'static ExecutableCode,
    mapping: ExecutableMapping<'static>,
    data: Option<DataSpan>,
}

/// Owns the raw published allocation until the module enters the loader.
/// Its borrowed mappings must be declared later so rollback drops them first.
struct UnpublishedCode(*mut ExecutableCode);

impl Drop for UnpublishedCode {
    fn drop(&mut self) {
        // SAFETY: link obtains this pointer from Box::into_raw exactly once.
        // All borrowed mappings have already dropped; successful publication
        // forgets this guard and leaves the allocation alive for the task.
        unsafe { drop(Box::from_raw(self.0)) };
    }
}

struct DataSpan {
    _lease: StashedLease,
    view: MappedLeaseView,
    vaddr: u64,
    bounds_len: u64,
}

impl Loaded {
    fn view(&self) -> Option<ModuleView<'_>> {
        let code = self.mapping.as_ptr() as u64;
        let code_len = self.mapping.len() as u64;
        let ro = read_views(&self.descriptor, self.base, |va, len| {
            let rel = va.checked_sub(self.base)?.checked_sub(self.code_vaddr)?;
            if rel.checked_add(len)? > code_len {
                return None;
            }
            let ptr = (code + rel) as *const u8;
            // SAFETY: the R|X mapping covers `[code, code + code_len)` for
            // the module's life; the range was bounds-checked above.
            Some(unsafe { core::slice::from_raw_parts(ptr, len as usize) })
        })?;
        Some(ModuleView {
            base: self.base,
            descriptor: &self.descriptor,
            ro,
        })
    }
}

/// The task's late loader.
pub struct Loader {
    host_base: u64,
    host: &'static ModuleDescriptor,
    /// Where the next module lands. One `kore_ld` placer over the link
    /// window — the same type the startup stub places with, so module
    /// addresses come from one implementation rather than a second copy
    /// of the same rounding.
    placer: kore_ld::place::Placer,
    modules: heapless::Vec<Loaded, MAX_LOADED_MODULES>,
}

impl Loader {
    /// Read the passive page and the host descriptor; refuse a host with
    /// TLS (its thread layout cannot be reproduced for a late scope).
    pub fn new(rt: &Rt) -> Result<Self, LoadError> {
        let ld = rt.ld().ok_or(LoadError::NoRecord(1))?;
        let page = rt.modules().ok_or(LoadError::NoRecord(2))?;
        let host = HostModule::from_page(page, &ld.exe, ld.exe_name.fnv1a())
            .ok_or(LoadError::NoRecord(3))?;
        if ld.exe.tls.memsz.get() != 0 {
            return Err(LoadError::HostTls);
        }
        let (base, len) = rt
            .geom()
            .and_then(|g| g.window(kore_abi::task::geom::LINK))
            .ok_or(LoadError::NoRecord(4))?;
        let limit = kore_abi::image::module_page_va(base, len).ok_or(LoadError::NoRecord(5))?;
        // Above everything the stub placed — the records' segments AND the
        // unrecorded range the placement high water covers (the copied
        // template views, the static TLS block). The records alone put the
        // first module on the next 64 KiB line above the image, which is
        // inside that range once the image is large enough.
        let next = page
            .high_water()
            .ok_or(LoadError::NoRecord(6))?
            .max(page.placed_high_water(limit).unwrap_or(base))
            .max(base);
        Ok(Self {
            host_base: host.base,
            host: host.descriptor,
            placer: kore_ld::place::Placer::new(next, limit.saturating_sub(next)),
            modules: heapless::Vec::new(),
        })
    }

    /// A module already loaded under `stem`.
    pub fn find(&self, stem: &[u8]) -> Option<ModuleId> {
        let name = object_name(ModuleKind::Native, stem)?;
        self.modules
            .iter()
            .position(|m| m.name == name)
            .map(|i| ModuleId(i as u8))
    }

    /// Read `<stem>.lua` through the search list into `buf`.
    pub async fn read_source(
        fs: &mut Fs<'_, '_>,
        search: Search<'_>,
        stem: &[u8],
        buf: &mut [u8],
    ) -> Result<usize, LoadError> {
        let name = object_name(ModuleKind::Source, stem).ok_or(LoadError::Name)?;
        let view = open_in(fs, search, &name).await?;
        let mut len = 0usize;
        let result = loop {
            let count = match fs.read_chunk(&view, len as u64).await {
                Ok(0) => break Ok(len),
                Ok(count) => count,
                Err(error) => break Err(LoadError::from(error)),
            };
            let end = match len.checked_add(count) {
                Some(end) if end <= buf.len() => end,
                _ => break Err(LoadError::TooLarge),
            };
            buf[len..end].copy_from_slice(&fs.window()[..count]);
            len = end;
        };
        fs.close(view).await;
        result
    }

    /// Link `<stem>.so` from the search list into the task.
    pub async fn load_native(
        &mut self,
        fs: &mut Fs<'_, '_>,
        search: Search<'_>,
        stem: &[u8],
    ) -> Result<ModuleId, LoadError> {
        if let Some(id) = self.find(stem) {
            return Ok(id);
        }
        if self.modules.is_full() {
            return Err(LoadError::TooMany);
        }
        let name = object_name(ModuleKind::Native, stem).ok_or(LoadError::Name)?;
        let view = open_in(fs, search, &name).await?;
        let bytes = match Self::stage(fs, &view).await {
            Ok(bytes) => {
                fs.close(view).await;
                bytes
            }
            Err(error) => {
                fs.close(view).await;
                return Err(error);
            }
        };
        self.link(name, &bytes)
    }

    /// Stat, lend, read the whole version, release.
    async fn stage(fs: &mut Fs<'_, '_>, view: &View) -> Result<Vec<u8>, LoadError> {
        let stat = fs.stat(view).await?;
        if stat.subtype != ObjectSubtype::EXECUTABLE_IMAGE {
            return Err(LoadError::NotExecutableImage);
        }
        let len = usize::try_from(stat.len).map_err(|_| LoadError::TooLarge)?;
        if len > MAX_MODULE_BYTES {
            return Err(LoadError::TooLarge);
        }
        let lease = match fs
            .checked(&FsRequest::LendVersion {
                handle: view.handle(),
                window_offset: 0,
            })
            .await?
        {
            FsResponse::VersionLent { lease } => lease,
            _ => return Err(LoadError::Pair),
        };
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(len).is_err() {
            let _ = fs.checked(&FsRequest::ReleaseVersion { lease }).await;
            return Err(LoadError::TooLarge);
        }
        let result = loop {
            let remaining = len - bytes.len();
            if remaining == 0 {
                break Ok(());
            }
            let length = remaining.min(fs.chunk()) as u32;
            match fs
                .checked(&FsRequest::ReadVersion {
                    lease,
                    offset: bytes.len() as u64,
                    window_offset: 0,
                    length,
                })
                .await
            {
                Ok(FsResponse::Transferred { count }) => {
                    let count = count as usize;
                    if count == 0 || count > length as usize {
                        break Err(LoadError::Pair);
                    }
                    bytes.extend_from_slice(&fs.window()[..count]);
                }
                Ok(_) => break Err(LoadError::Pair),
                Err(error) => break Err(LoadError::from(error)),
            }
        };
        let _ = fs.checked(&FsRequest::ReleaseVersion { lease }).await;
        result.map(|()| bytes)
    }

    /// Check, plan, place, publish the code, map the data, relocate, run
    /// the initializers, remember.
    fn link(&mut self, name: Name, bytes: &[u8]) -> Result<ModuleId, LoadError> {
        let checked = check_image::<HostArch>(bytes).map_err(LoadError::Check)?;
        let plan = LateLayout::plan(&checked, MAX_MODULE_BYTES as u64, MAX_MODULE_BYTES as u64)
            .map_err(LoadError::Plan)?;
        // Placed on a copy: the window advances only when the module is
        // remembered, so a refused load leaves its span for the next one.
        let mut placer = self.placer;
        let base = placer
            .choose(plan.span, crate::LATE_BASE_ALIGN)
            .map_err(|_| LoadError::NoWindow)?;

        // The code run: one arena generation, published whole.
        let code_lease =
            kore_rt::heap::reserve_from_top(plan.code_len).ok_or(LoadError::NoSupply)?;
        let code_window =
            Window::try_new(base + plan.code_vaddr, plan.code_len).ok_or(LoadError::NoWindow)?;
        let copies = plan.code_copies;
        let (code, ()) = code_lease
            .initialize_executable_at(code_window, |w| {
                w.fill(0);
                for copy in copies.iter().flatten() {
                    let start = copy.file_off as usize;
                    let end = start.saturating_add(copy.len as usize).min(bytes.len());
                    if start >= end {
                        continue;
                    }
                    w.write_at(copy.dst as usize, &bytes[start..end])?;
                }
                Ok::<(), kore_rt::cap::CodeWriteError>(())
            })
            .map_err(|error| match error {
                kore_rt::cap::InitializeExecutableError::Initialize(_) => LoadError::Map,
                kore_rt::cap::InitializeExecutableError::Map(_) => LoadError::Map,
                _ => LoadError::Publish,
            })?;
        let rollback = UnpublishedCode(Box::into_raw(Box::new(code)));
        // SAFETY: the allocation is stable. On refusal, all borrows below
        // die before rollback; on success the module retains it for task life.
        let code: &'static ExecutableCode = unsafe { &*rollback.0 };
        let mapping = code.map_rx_at(code_window).map_err(|_| LoadError::Map)?;

        // The data union: one exact writable mapping at its linked offset.
        let data = match plan.data {
            None => None,
            Some(span) => {
                let lease = kore_rt::heap::reserve_from_top(span.len).ok_or(LoadError::NoSupply)?;
                let slot = lease.slot().ok_or(LoadError::NoSupply)?;
                let view = verbs::map_owned_lease_at_rw(slot, base + span.vaddr, span.len)
                    .ok_or(LoadError::Map)?;
                for copy in span.copies.iter().flatten() {
                    let start = copy.file_off as usize;
                    let end = start.saturating_add(copy.len as usize).min(bytes.len());
                    if start >= end {
                        continue;
                    }
                    // SAFETY: the mapping is `span.len` writable bytes the
                    // plan bounded every copy inside; the supply pages are
                    // fresh (never handed out before), so bss is zero.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            bytes[start..end].as_ptr(),
                            view.as_mut_ptr().add(copy.dst as usize),
                            end - start,
                        );
                    }
                }
                Some(DataSpan {
                    _lease: lease,
                    view,
                    vaddr: span.vaddr,
                    bounds_len: span.bounds_len,
                })
            }
        };

        let mut loaded = Loaded {
            name,
            base,
            code_vaddr: plan.code_vaddr,
            descriptor: checked.descriptor,
            _code: code,
            mapping,
            data,
        };
        self.relocate(&mut loaded, checked.relro)?;
        if let Some((off, len)) = checked.relro {
            let span = loaded.data.as_mut().ok_or(LoadError::Seal)?;
            let offset = off.checked_sub(span.vaddr).ok_or(LoadError::Seal)?;
            if !span.view.seal_read_only(offset as usize, len as usize) {
                return Err(LoadError::Seal);
            }
        }
        self.run_initializers(&loaded)?;
        self.modules.push(loaded).map_err(|_| LoadError::TooMany)?;
        core::mem::forget(rollback);
        self.placer = placer;
        Ok(ModuleId((self.modules.len() - 1) as u8))
    }

    /// Resolve the module against `[host, ...loaded, this]`.
    fn relocate(&self, module: &mut Loaded, relro: Option<(u64, u64)>) -> Result<(), LoadError> {
        let host_view = ModuleView {
            base: self.host_base,
            descriptor: self.host,
            ro: read_views(self.host, self.host_base, |va, len| {
                // SAFETY: the host's text/rodata are mapped R (or R|X) at
                // these addresses for the task's life (the stub's shares).
                Some(unsafe { core::slice::from_raw_parts(va as *const u8, len as usize) })
            })
            .ok_or(LoadError::NoRecord(7))?,
        };
        let mut views: heapless::Vec<ModuleView<'_>, { MAX_LOADED_MODULES + 2 }> =
            heapless::Vec::new();
        let _ = views.push(host_view);
        for m in &self.modules {
            let _ = views.push(m.view().ok_or(LoadError::Map)?);
        }
        let this_view = module.view().ok_or(LoadError::Map)?;
        let _ = views.push(this_view);
        let scope_index = views.len() - 1;
        let scope = SymbolScope { modules: &views };

        let mut arena_bytes = [MaybeUninit::<u8>::uninit(); LINK_ARENA_BYTES];
        let arena = LinkArena::new(&mut arena_bytes);
        let tls = TlsLayout::build::<HostArch>(&views, &arena).map_err(|_| LoadError::HostTls)?;

        let mut empty = [0u8; 0];
        let data: &mut [u8] = match module.data.as_ref() {
            Some(span) => {
                // SAFETY: the writable mapping spans `span.len >= bounds_len`
                // bytes owned by this module; nothing else references it.
                unsafe {
                    core::slice::from_raw_parts_mut(
                        span.view.as_mut_ptr(),
                        span.bounds_len as usize,
                    )
                }
            }
            None => &mut empty,
        };
        let mut image = ModuleImage {
            scope_index,
            view: this_view,
            data,
            relro,
        };
        apply_module::<HostArch, _>(&mut image, &scope, &tls).map_err(LoadError::Link)?;
        Ok(())
    }

    /// Validate and snapshot the whole relocated array before calling any code.
    fn run_initializers(&self, module: &Loaded) -> Result<(), LoadError> {
        let off = module.descriptor.dynz.init_array_off.get();
        let sz = module.descriptor.dynz.init_array_sz.get();
        if sz == 0 {
            return Ok(());
        }
        let span = module.data.as_ref().ok_or(LoadError::InitializerArray)?;
        let rel = off
            .checked_sub(span.vaddr)
            .ok_or(LoadError::InitializerArray)?;
        let end = rel.checked_add(sz).ok_or(LoadError::InitializerArray)?;
        if end > span.bounds_len {
            return Err(LoadError::InitializerArray);
        }
        // SAFETY: the complete array is inside the owned data mapping. This
        // borrow ends before any initializer may modify its source storage.
        let bytes = unsafe {
            core::slice::from_raw_parts(span.view.as_ptr().add(rel as usize), sz as usize)
        };
        let addresses = kore_ld::functions::function_addresses(bytes, |address| {
            kore_ld::functions::contains_text(self.host, self.host_base, address)
                || self
                    .modules
                    .iter()
                    .chain(core::iter::once(module))
                    .any(|loaded| {
                        kore_ld::functions::contains_text(&loaded.descriptor, loaded.base, address)
                    })
        })
        .map_err(LoadError::Initializer)?;
        let mut snapshot: heapless::Vec<u64, { kore_ld::link::MAX_INIT_FNS }> =
            heapless::Vec::new();
        for address in addresses {
            snapshot
                .push(address)
                .map_err(|_| LoadError::TooManyInitializers)?;
        }
        for entry in snapshot {
            // SAFETY: every non-sentinel target was checked against loaded
            // executable text before the first call; ELF initializers use this
            // ABI. The snapshot cannot be rewritten through the module array.
            let f: unsafe extern "C" fn() = unsafe { core::mem::transmute(entry as usize) };
            unsafe { f() };
        }
        Ok(())
    }

    /// An exported symbol's address in a loaded module.
    pub fn lookup(&self, id: ModuleId, symbol: &[u8]) -> Option<*const u8> {
        let module = self.modules.get(usize::from(id.0))?;
        let view = module.view()?;
        lookup_symbol(&view, symbol)
            .ok()
            .flatten()
            .map(|va| va as *const u8)
    }

    /// An exported entry inside this module's file-backed executable text.
    /// This proves executable placement, not its signature or ABI; callers
    /// still establish those before converting the pointer to a function.
    pub fn lookup_executable(&self, id: ModuleId, symbol: &[u8]) -> Option<*const u8> {
        let address = self.lookup(id, symbol)?;
        let module = self.modules.get(usize::from(id.0))?;
        kore_ld::functions::contains_text(&module.descriptor, module.base, address as u64)
            .then_some(address)
    }

    /// The image name this loader links against (diagnostics).
    pub fn host_name(rt: &Rt) -> Option<ImageName> {
        rt.ld().map(|ld| ld.exe_name)
    }
}

/// Open `name` under the first prefix that holds it.
async fn open_in(fs: &mut Fs<'_, '_>, search: Search<'_>, name: &Name) -> Result<View, LoadError> {
    let mut buf = [0u8; MAX_SEARCH_PATH];
    for prefix in search.prefixes.iter().take(crate::MAX_SEARCH_PREFIXES) {
        let path = compose_path(&mut buf, prefix, name).ok_or(LoadError::Name)?;
        match fs.open(path, WALK_READ).await {
            Ok(view) => return Ok(view),
            Err(Refusal::Fs(kore_dagfs::FsError::NotFound)) => continue,
            Err(error) => return Err(LoadError::from(error)),
        }
    }
    Err(LoadError::NotFound)
}
