//! Late binding: the plan a task follows to link a shared object into
//! its **own** address space after start — `res/IMAGE.md` §8's "separate
//! library consuming the record, never a resurrection of the stub".
//!
//! The startup stub shares an image's segments from the image service's
//! leases.  A late load has no such leases: the task holds the object's
//! bytes (read out of the DagFS store) and two things it can carve itself —
//! a writable→executable arena for the code (`JitArena`, §6a) and a plain
//! writable lease for the data.  This module is the pure arithmetic between
//! a [`CheckedImage`] and those two windows: which file bytes land where,
//! and which shapes are refused.  It touches no memory and holds no
//! capability, so it is host-testable against the same fixtures as the
//! check gate; the runtime half lives in `kore-late`.
//!
//! **Shape admitted.**  A `kind::LIBRARY` whose text and rodata segments
//! form the *code run* — the page-rounded span from the lowest text/rodata
//! `vaddr` to the highest end — and whose writable segments all lie
//! **above** that run (lld's `-z separate-loadable-segments` layout: R, RX,
//! RW).  The whole code run becomes one arena generation; the data union
//! ([`data_bounds`]) becomes one writable mapping at its linked offset, so
//! every PC-relative reference the linker fixed still lands.  Refused, not
//! guessed: executables (they have an entry and a startup contract), TLS
//! (the task's thread-pointer layout was closed at start — §7), a data
//! segment below or inside the code run, and images too large for the
//! bounds the caller passes.

use kore_abi::image::{MAX_SEGMENTS, ModuleDescriptor, kind, seg};

use crate::apply::data_bounds;
use crate::check::CheckedImage;
use crate::parse::PAGE_SIZE;

/// Why a checked image cannot be late-linked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LateError {
    /// Not a `kind::LIBRARY` — executables carry an entry and a startup
    /// contract the late path has no way to honor.
    NotLibrary,
    /// The image carries `PT_TLS`; a late object cannot join a thread
    /// layout that was built and installed at start.
    Tls,
    /// No text segment — nothing to publish.
    NoCode,
    /// A writable segment lies below or inside the code run.
    DataBelowCode,
    /// The code run exceeds `max_code`, or the image span exceeds
    /// `max_span`.
    TooLarge,
    /// The descriptor and carve plan disagree, or arithmetic overflowed.
    Layout,
}

/// One copy from the object's file bytes into a window: `dst` is relative
/// to the window's start, `file_off`/`len` name the source bytes.  A copy
/// past the end of the bytes the caller holds is zero-filled by the caller
/// (the carve windows are page-rounded; the file need not be).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy {
    pub dst: u64,
    pub file_off: u64,
    pub len: u64,
}

/// The writable half: the data union at its linked offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataSpan {
    /// Image-relative start of the union (`data_bounds`), page-aligned.
    pub vaddr: u64,
    /// Union length, page-rounded — the writable lease's length and the
    /// exact `ModuleImage::data` length [`crate::apply::apply_module`]
    /// demands is `bounds_len`, not this.
    pub len: u64,
    /// The exact union length the apply engine checks against.
    pub bounds_len: u64,
    /// Template copies (one per data segment with file bytes); the rest of
    /// the mapping is bss and stays zero.
    pub copies: [Option<Copy>; MAX_SEGMENTS],
}

/// The plan: where the code run and the data union sit relative to the
/// load base, and which bytes fill them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LateLayout {
    /// Image-relative start of the code run (page-aligned).
    pub code_vaddr: u64,
    /// Code run length (page-multiple) — the arena lease's length.
    pub code_len: u64,
    /// Text/rodata copies into the arena, `dst` relative to `code_vaddr`.
    pub code_copies: [Option<Copy>; MAX_SEGMENTS],
    /// The writable half, or `None` for an object with no data at all.
    pub data: Option<DataSpan>,
    /// Image-relative one-past-the-end of everything placed — what the
    /// next load base must clear.
    pub span: u64,
}

impl LateLayout {
    /// Derive the plan from a checked image.  `max_code` bounds the arena
    /// lease, `max_span` the whole placement; both are the caller's
    /// supply/window budgets, refused as [`LateError::TooLarge`].
    pub fn plan(checked: &CheckedImage, max_code: u64, max_span: u64) -> Result<Self, LateError> {
        let d: &ModuleDescriptor = &checked.descriptor;
        if d.kind != kind::LIBRARY {
            return Err(LateError::NotLibrary);
        }
        if d.tls.memsz.get() != 0 || d.live_segs().iter().any(|s| s.kind == seg::KIND_TLS) {
            return Err(LateError::Tls);
        }

        // The code run: every text/rodata segment, page-rounded.
        let mut code_start: Option<u64> = None;
        let mut code_end: u64 = 0;
        let mut has_text = false;
        for s in d.live_segs() {
            if s.kind != seg::KIND_TEXT && s.kind != seg::KIND_RODATA {
                continue;
            }
            has_text |= s.kind == seg::KIND_TEXT;
            let start = s.vaddr.get();
            let end = start
                .checked_add(s.memsz.get().max(s.filesz.get()))
                .ok_or(LateError::Layout)?;
            code_start = Some(code_start.map_or(start, |at| at.min(start)));
            code_end = code_end.max(end);
        }
        let (Some(code_start), true) = (code_start, has_text) else {
            return Err(LateError::NoCode);
        };
        let code_vaddr = align_down(code_start);
        let code_end = align_up(code_end)?;
        let code_len = code_end.checked_sub(code_vaddr).ok_or(LateError::Layout)?;
        if code_len > max_code {
            return Err(LateError::TooLarge);
        }

        // Copies into the arena: each text/rodata carve window at its
        // segment's offset from the run start.
        let mut code_copies = [None; MAX_SEGMENTS];
        for (i, s) in d.live_segs().iter().enumerate() {
            if s.kind != seg::KIND_TEXT && s.kind != seg::KIND_RODATA {
                continue;
            }
            let carve = checked
                .carve
                .get(i)
                .copied()
                .flatten()
                .ok_or(LateError::Layout)?;
            if carve.kind != s.kind {
                return Err(LateError::Layout);
            }
            let dst = s
                .vaddr
                .get()
                .checked_sub(code_vaddr)
                .ok_or(LateError::Layout)?;
            if dst.checked_add(carve.len).ok_or(LateError::Layout)? > code_len {
                return Err(LateError::Layout);
            }
            code_copies[i] = Some(Copy {
                dst,
                file_off: carve.file_off,
                len: carve.len,
            });
        }

        // The writable half, above the code run.
        let data = match data_bounds(d).map_err(|_| LateError::Layout)? {
            None => None,
            Some((start, bounds_len)) => {
                if start < code_end {
                    return Err(LateError::DataBelowCode);
                }
                if start % PAGE_SIZE != 0 {
                    return Err(LateError::Layout);
                }
                let len = align_up(bounds_len)?;
                let mut copies = [None; MAX_SEGMENTS];
                for (i, s) in d.live_segs().iter().enumerate() {
                    if s.kind != seg::KIND_DATA || s.filesz.get() == 0 {
                        continue;
                    }
                    let carve = checked
                        .carve
                        .get(i)
                        .copied()
                        .flatten()
                        .ok_or(LateError::Layout)?;
                    let dst = s.vaddr.get().checked_sub(start).ok_or(LateError::Layout)?;
                    if dst.checked_add(s.filesz.get()).ok_or(LateError::Layout)? > bounds_len {
                        return Err(LateError::Layout);
                    }
                    copies[i] = Some(Copy {
                        dst,
                        file_off: carve.file_off,
                        len: s.filesz.get(),
                    });
                }
                Some(DataSpan {
                    vaddr: start,
                    len,
                    bounds_len,
                    copies,
                })
            }
        };

        let span = match data {
            Some(ref span) => span.vaddr.checked_add(span.len).ok_or(LateError::Layout)?,
            None => code_end,
        };
        if span > max_span {
            return Err(LateError::TooLarge);
        }
        Ok(Self {
            code_vaddr,
            code_len,
            code_copies,
            data,
            span,
        })
    }
}

const fn align_down(v: u64) -> u64 {
    v & !(PAGE_SIZE - 1)
}

fn align_up(v: u64) -> Result<u64, LateError> {
    v.checked_add(PAGE_SIZE - 1)
        .map(align_down)
        .ok_or(LateError::Layout)
}
