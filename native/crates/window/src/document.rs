//! What kind of document an object is, as apps hand documents to each other.
//!
//! A dagfs object carries no content type; its kind is read from its name
//! (`kore_kit::kinds`). Handlers declare the kinds they open by class, not by
//! extension — the editor opens text and code, an image viewer images — so a
//! class is what crosses the wire: in `OpenObject`'s second word, a "Open
//! With" menu entry, and a dock row's handler mask. Codes are stable: a
//! class is added at the end, never renumbered.

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DocumentKind {
    /// No class known: the opener falls back to the catalogue's document app.
    #[default]
    Unknown = 0,
    /// Prose and notes: plain text, Markdown.
    Text = 1,
    /// Source and settings: Lua, Python, C, Rust, Zig, TOML, JSON, YAML.
    Code = 2,
    Image = 3,
    /// Page documents: PDF, word processor files.
    Document = 4,
    Spreadsheet = 5,
    /// Executable or binary formats (WebAssembly).
    Binary = 6,
}

impl DocumentKind {
    pub const ALL: [DocumentKind; 7] = [
        Self::Unknown,
        Self::Text,
        Self::Code,
        Self::Image,
        Self::Document,
        Self::Spreadsheet,
        Self::Binary,
    ];

    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The class `code` names; `None` for a code no class has.
    pub fn from_code(code: u64) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| u64::from(kind.code()) == code)
    }

    /// This class's bit in a [`KindMask`].
    pub const fn bit(self) -> u16 {
        1 << self as u8
    }
}

/// A set of [`DocumentKind`]s — the classes one handler opens. Bit `n` is
/// the class with code `n`; bits no class has are refused on the wire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindMask(u16);

impl KindMask {
    pub const NONE: Self = Self(0);
    const VALID: u16 = (1 << DocumentKind::ALL.len()) - 1;

    pub const fn of(kinds: &[DocumentKind]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < kinds.len() {
            bits |= kinds[i].bit();
            i += 1;
        }
        Self(bits)
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub fn from_bits(bits: u16) -> Option<Self> {
        (bits & !Self::VALID == 0).then_some(Self(bits))
    }

    pub const fn contains(self, kind: DocumentKind) -> bool {
        self.0 & kind.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_and_masks_refuse_unknown_bits() {
        for (code, kind) in DocumentKind::ALL.into_iter().enumerate() {
            assert_eq!(usize::from(kind.code()), code);
            assert_eq!(DocumentKind::from_code(code as u64), Some(kind));
        }
        assert_eq!(DocumentKind::from_code(DocumentKind::ALL.len() as u64), None);
        let editor = KindMask::of(&[DocumentKind::Text, DocumentKind::Code]);
        assert!(editor.contains(DocumentKind::Code) && !editor.contains(DocumentKind::Image));
        assert_eq!(KindMask::from_bits(editor.bits()), Some(editor));
        assert_eq!(KindMask::from_bits(1 << DocumentKind::ALL.len()), None);
    }
}
