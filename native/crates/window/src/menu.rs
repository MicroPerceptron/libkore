//! Context menus the compositor draws for a window (`ShowMenu`).
//!
//! A menu is data, not authority: the app describes it — labels, shortcuts,
//! states, one level of submenus — in its pair lease's menu plane, and the
//! compositor draws it above every window, runs hover, keys and dismissal,
//! and answers with the chosen item's id ([`crate::event::Event::Menu`]).
//! One system entry, [`flag::OPEN_WITH`], carries an object and its
//! [`DocumentKind`]
//! instead of children: the compositor fills it from the shell's handler
//! table, so an app never learns which other apps exist.
//!
//! The encoding is little-endian and closed: a header, then entries in
//! display order, each a fixed head followed by its text. A submenu's
//! children follow it directly at depth 1. [`Menu::parse`] refuses anything
//! the [`MenuBuilder`] cannot write — unknown flags, nonzero reserved bytes,
//! text that is not printable UTF-8, a second submenu level, duplicate ids,
//! trailing bytes — so the compositor validates a snapshot it copied out of
//! the plane, never the live plane the app can still write.

/// The menu plane: a page of the window pair lease after the event channel.
/// It bounds a whole encoded menu; the per-entry limits below bound each part.
pub const MENU_PLANE_BYTES: usize = 0x1000;
/// Top-level entries the compositor draws (separators included): the kit's
/// longest menu, the explorer's file menu, has fourteen.
pub const MAX_TOP_ENTRIES: usize = 16;
/// Children of one submenu.
pub const MAX_SUBMENU_ENTRIES: usize = 8;
/// Entries in one encoded menu, children included.
pub const MAX_ENTRIES: usize = 48;
/// UTF-8 bytes of one label ("Delete from All Folders…" is 26).
pub const LABEL_BYTES: usize = 48;
/// UTF-8 bytes of an entry's second line (the drop menu's "Stays in {folder}
/// as well").
pub const DETAIL_BYTES: usize = 64;
/// UTF-8 bytes of a shortcut hint ("Ctrl+Shift+L").
pub const SHORTCUT_BYTES: usize = 16;

use crate::document::DocumentKind;

const VERSION: u8 = 1;
const HEADER_BYTES: usize = 8;
const HEAD_BYTES: usize = 24;

/// Entry flags.
pub mod flag {
    /// Drawn, never chosen.
    pub const DISABLED: u8 = 1 << 0;
    /// A destructive command, drawn in the danger colour.
    pub const DANGER: u8 = 1 << 1;
    /// A check mark beside the label.
    pub const CHECKED: u8 = 1 << 2;
    /// A divider: no id, no text, no other flag.
    pub const SEPARATOR: u8 = 1 << 3;
    /// Opens the entries after it at depth 1; it is never chosen itself.
    pub const SUBMENU: u8 = 1 << 4;
    /// "Open With ▸": the compositor fills its submenu with the apps that
    /// handle the entry's object kind and opens the object in the one chosen.
    pub const OPEN_WITH: u8 = 1 << 5;
    pub(super) const ALL: u8 = DISABLED | DANGER | CHECKED | SEPARATOR | SUBMENU | OPEN_WITH;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuError {
    /// The menu does not fit the plane, or a text exceeds its limit.
    TooLarge,
    /// More entries than [`MAX_TOP_ENTRIES`], [`MAX_SUBMENU_ENTRIES`] or
    /// [`MAX_ENTRIES`].
    TooManyEntries,
    /// No top-level entry, or a submenu with no children.
    Empty,
    /// A version, flag, reserved byte or length the encoding does not have.
    Malformed,
    /// Text that is not UTF-8, or holds a control character.
    BadText,
    /// A child outside a submenu, a submenu inside a submenu, or a submenu
    /// left open.
    BadNesting,
    /// A choosable entry with id zero, or two entries with one id.
    BadId,
}

/// Where one text lives in the encoded bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    at: u16,
    len: u8,
}

impl Span {
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// One parsed entry. Texts are [`Span`]s into the bytes it was parsed from;
/// read them with [`text`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    /// What [`crate::event::Event::Menu`] reports when this entry is chosen;
    /// zero for separators and submenus.
    pub id: u16,
    pub flags: u8,
    /// 0 at the top level, 1 inside a submenu.
    pub depth: u8,
    pub label: Span,
    pub detail: Span,
    pub shortcut: Span,
    /// [`flag::OPEN_WITH`]'s object and its [`DocumentKind`] code; zero
    /// otherwise.
    pub object: u64,
    pub kind: u32,
    /// A submenu's child count; zero otherwise.
    pub children: u8,
}

impl Entry {
    pub fn has(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }
    /// "Open With"'s object class (parse admitted only known codes).
    pub fn document_kind(&self) -> DocumentKind {
        DocumentKind::from_code(u64::from(self.kind)).unwrap_or_default()
    }
    /// Whether choosing it ends the menu with its id: not a separator, a
    /// submenu or "Open With", and not disabled.
    pub fn choosable(&self) -> bool {
        self.flags & (flag::SEPARATOR | flag::SUBMENU | flag::OPEN_WITH | flag::DISABLED) == 0
    }
}

/// The text `span` names in `bytes`, the bytes [`Menu::parse`] accepted.
/// Bytes that changed since are read as no text, never as a panic.
pub fn text(bytes: &[u8], span: Span) -> &str {
    let at = usize::from(span.at);
    bytes
        .get(at..at + usize::from(span.len))
        .and_then(|b| core::str::from_utf8(b).ok())
        .unwrap_or("")
}

/// A validated menu: entries in display order, children after their parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Menu {
    entries: [Entry; MAX_ENTRIES],
    len: u8,
}

fn printable(bytes: &[u8]) -> Result<(), MenuError> {
    let text = core::str::from_utf8(bytes).map_err(|_| MenuError::BadText)?;
    if text.chars().any(char::is_control) {
        return Err(MenuError::BadText);
    }
    Ok(())
}

impl Menu {
    /// Validate `bytes`, an encoded menu (exactly its length, no more).
    pub fn parse(bytes: &[u8]) -> Result<Self, MenuError> {
        if bytes.len() > MENU_PLANE_BYTES {
            return Err(MenuError::TooLarge);
        }
        let header = bytes.get(..HEADER_BYTES).ok_or(MenuError::Malformed)?;
        if header[0] != VERSION || header[2..].iter().any(|&b| b != 0) {
            return Err(MenuError::Malformed);
        }
        let count = usize::from(header[1]);
        if count > MAX_ENTRIES {
            return Err(MenuError::TooManyEntries);
        }
        let mut menu = Self {
            entries: [Entry::default(); MAX_ENTRIES],
            len: 0,
        };
        let mut at = HEADER_BYTES;
        let (mut top, mut parent) = (0, None::<usize>);
        for index in 0..count {
            let head = bytes.get(at..at + HEAD_BYTES).ok_or(MenuError::Malformed)?;
            let word = |i: usize, n: usize| head[i..i + n].iter().rev().fold(0_u64, |w, &b| w << 8 | u64::from(b));
            let (id, flags, depth) = (word(0, 2) as u16, head[2], head[3]);
            let lengths = [head[4], head[5], head[6]];
            let (object, kind) = (word(8, 8), word(16, 4) as u32);
            if head[7] != 0 || word(20, 4) != 0 || flags & !flag::ALL != 0 || depth > 1 {
                return Err(MenuError::Malformed);
            }
            let limits = [LABEL_BYTES, DETAIL_BYTES, SHORTCUT_BYTES];
            if lengths.iter().zip(limits).any(|(&n, limit)| usize::from(n) > limit) {
                return Err(MenuError::TooLarge);
            }
            at += HEAD_BYTES;
            let mut spans = [Span::default(); 3];
            for (span, &len) in spans.iter_mut().zip(&lengths) {
                let text = bytes.get(at..at + usize::from(len)).ok_or(MenuError::Malformed)?;
                printable(text)?;
                *span = Span { at: at as u16, len };
                at += usize::from(len);
            }
            let entry = Entry {
                id,
                flags,
                depth,
                label: spans[0],
                detail: spans[1],
                shortcut: spans[2],
                object,
                kind,
                children: 0,
            };
            // The shape of each entry kind, exactly as the builder writes it.
            let separator = entry.has(flag::SEPARATOR);
            let submenu = entry.has(flag::SUBMENU);
            let open_with = entry.has(flag::OPEN_WITH);
            let shaped = if separator {
                flags == flag::SEPARATOR && id == 0 && lengths == [0; 3] && object == 0 && kind == 0
            } else if submenu {
                flags & !(flag::SUBMENU | flag::DISABLED) == 0
                    && id == 0
                    && depth == 0
                    && lengths[0] > 0
                    && lengths[2] == 0
                    && object == 0
                    && kind == 0
            } else if open_with {
                flags & !(flag::OPEN_WITH | flag::DISABLED) == 0
                    && depth == 0
                    && lengths[0] > 0
                    && lengths[2] == 0
                    && object != 0
                    && DocumentKind::from_code(u64::from(kind)).is_some()
            } else {
                id != 0 && lengths[0] > 0 && object == 0 && kind == 0
            };
            if !shaped {
                return Err(if id == 0 && !separator && !submenu && !open_with {
                    MenuError::BadId
                } else {
                    MenuError::Malformed
                });
            }
            if depth == 0 {
                if let Some(p) = parent.take() {
                    if menu.entries[p].children == 0 {
                        return Err(MenuError::Empty);
                    }
                }
                top += 1;
                if top > MAX_TOP_ENTRIES {
                    return Err(MenuError::TooManyEntries);
                }
                if submenu {
                    parent = Some(index);
                }
            } else {
                let Some(p) = parent else {
                    return Err(MenuError::BadNesting);
                };
                if submenu || open_with {
                    return Err(MenuError::BadNesting);
                }
                if usize::from(menu.entries[p].children) == MAX_SUBMENU_ENTRIES {
                    return Err(MenuError::TooManyEntries);
                }
                menu.entries[p].children += 1;
            }
            // Ids name one entry. Bounded by MAX_ENTRIES, a design constant.
            if id != 0 && menu.entries[..index].iter().any(|e| e.id == id) {
                return Err(MenuError::BadId);
            }
            menu.entries[index] = entry;
            menu.len += 1;
        }
        if parent.is_some_and(|p| menu.entries[p].children == 0) || top == 0 {
            return Err(MenuError::Empty);
        }
        if at != bytes.len() {
            return Err(MenuError::Malformed);
        }
        Ok(menu)
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries[..usize::from(self.len)]
    }

    /// The indices of the top-level entries, in order.
    pub fn top(&self) -> impl Iterator<Item = usize> + '_ {
        (0..usize::from(self.len)).filter(|&i| self.entries[i].depth == 0)
    }

    /// The children of the submenu at `index` (empty for any other entry).
    pub fn children(&self, index: usize) -> &[Entry] {
        let Some(entry) = self.entries().get(index) else {
            return &[];
        };
        let first = index + 1;
        &self.entries[first..first + usize::from(entry.children)]
    }

    /// The entry with `id`, if a choosable one has it.
    pub fn find(&self, id: u16) -> Option<&Entry> {
        self.entries().iter().find(|e| e.id == id && id != 0)
    }
}

/// One choosable entry for [`MenuBuilder::item`].
#[derive(Clone, Copy, Debug)]
pub struct Item<'s> {
    id: u16,
    label: &'s str,
    detail: &'s str,
    shortcut: &'s str,
    flags: u8,
}

impl<'s> Item<'s> {
    /// An entry reported as `id` (nonzero, unique in its menu).
    pub const fn new(id: u16, label: &'s str) -> Self {
        Self {
            id,
            label,
            detail: "",
            shortcut: "",
            flags: 0,
        }
    }
    /// A right-aligned shortcut hint, drawn faint.
    pub const fn shortcut(mut self, shortcut: &'s str) -> Self {
        self.shortcut = shortcut;
        self
    }
    /// A second line under the label.
    pub const fn detail(mut self, detail: &'s str) -> Self {
        self.detail = detail;
        self
    }
    pub const fn enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.flags &= !flag::DISABLED;
        } else {
            self.flags |= flag::DISABLED;
        }
        self
    }
    pub const fn checked(mut self, checked: bool) -> Self {
        if checked {
            self.flags |= flag::CHECKED;
        } else {
            self.flags &= !flag::CHECKED;
        }
        self
    }
    pub const fn danger(mut self) -> Self {
        self.flags |= flag::DANGER;
        self
    }
}

/// Encodes a menu into a plane, in display order. The first error sticks and
/// [`finish`](Self::finish) reports it; nothing reaches the compositor that
/// [`Menu::parse`] would refuse.
pub struct MenuBuilder<'b> {
    buf: &'b mut [u8],
    at: usize,
    count: usize,
    top: usize,
    /// The open submenu's head offset and child count.
    submenu: Option<(usize, usize)>,
    ids: [u16; MAX_ENTRIES],
    error: Option<MenuError>,
}

impl<'b> MenuBuilder<'b> {
    pub fn new(buf: &'b mut [u8]) -> Self {
        let error = (buf.len() < HEADER_BYTES).then_some(MenuError::TooLarge);
        Self {
            buf,
            at: HEADER_BYTES,
            count: 0,
            top: 0,
            submenu: None,
            ids: [0; MAX_ENTRIES],
            error,
        }
    }

    pub fn item(&mut self, item: Item<'_>) -> &mut Self {
        if item.id == 0 {
            return self.fail(MenuError::BadId);
        }
        if self.ids[..self.count].contains(&item.id) {
            return self.fail(MenuError::BadId);
        }
        let depth = u8::from(self.submenu.is_some());
        self.entry(item.id, item.flags, depth, [item.label, item.detail, item.shortcut], 0, 0)
    }

    pub fn separator(&mut self) -> &mut Self {
        let depth = u8::from(self.submenu.is_some());
        self.entry(0, flag::SEPARATOR, depth, ["", "", ""], 0, 0)
    }

    /// Start a submenu: the next entries, until
    /// [`end_submenu`](Self::end_submenu), are its children.
    pub fn submenu(&mut self, label: &str) -> &mut Self {
        if self.submenu.is_some() {
            return self.fail(MenuError::BadNesting);
        }
        let head = self.at;
        self.entry(0, flag::SUBMENU, 0, [label, "", ""], 0, 0);
        if self.error.is_none() {
            self.submenu = Some((head, 0));
        }
        self
    }

    pub fn end_submenu(&mut self) -> &mut Self {
        match self.submenu.take() {
            Some((_, 0)) => self.fail(MenuError::Empty),
            Some(_) => self,
            None => self.fail(MenuError::BadNesting),
        }
    }

    /// "Open With ▸" for `object` of `kind`, labelled by the app. `id`, if
    /// nonzero, is reported when the object is opened through it.
    pub fn open_with(&mut self, id: u16, label: &str, object: u64, kind: DocumentKind) -> &mut Self {
        if self.submenu.is_some() {
            return self.fail(MenuError::BadNesting);
        }
        if object == 0 || (id != 0 && self.ids[..self.count].contains(&id)) {
            return self.fail(if object == 0 { MenuError::Malformed } else { MenuError::BadId });
        }
        self.entry(id, flag::OPEN_WITH, 0, [label, "", ""], object, u32::from(kind.code()))
    }

    /// Seal the menu: its length in the plane, or the first error.
    pub fn finish(&mut self) -> Result<usize, MenuError> {
        if self.submenu.is_some() {
            self.fail(MenuError::BadNesting);
        }
        if self.top == 0 {
            self.fail(MenuError::Empty);
        }
        if let Some(error) = self.error {
            return Err(error);
        }
        self.buf[..HEADER_BYTES].copy_from_slice(&[VERSION, self.count as u8, 0, 0, 0, 0, 0, 0]);
        Ok(self.at)
    }

    fn fail(&mut self, error: MenuError) -> &mut Self {
        self.error.get_or_insert(error);
        self
    }

    fn entry(&mut self, id: u16, flags: u8, depth: u8, texts: [&str; 3], object: u64, kind: u32) -> &mut Self {
        if self.error.is_some() {
            return self;
        }
        let limits = [LABEL_BYTES, DETAIL_BYTES, SHORTCUT_BYTES];
        if texts.iter().zip(limits).any(|(t, limit)| t.len() > limit) {
            return self.fail(MenuError::TooLarge);
        }
        if texts.iter().any(|t| t.chars().any(char::is_control)) {
            return self.fail(MenuError::BadText);
        }
        if flags & flag::SEPARATOR == 0 && texts[0].is_empty() {
            return self.fail(MenuError::Malformed);
        }
        if self.count == MAX_ENTRIES {
            return self.fail(MenuError::TooManyEntries);
        }
        if depth == 0 {
            if self.top == MAX_TOP_ENTRIES {
                return self.fail(MenuError::TooManyEntries);
            }
        } else if let Some((_, children)) = self.submenu {
            if children == MAX_SUBMENU_ENTRIES {
                return self.fail(MenuError::TooManyEntries);
            }
        }
        let size = HEAD_BYTES + texts.iter().map(|t| t.len()).sum::<usize>();
        let Some(out) = self.buf.get_mut(self.at..self.at + size) else {
            return self.fail(MenuError::TooLarge);
        };
        let (head, mut tail) = out.split_at_mut(HEAD_BYTES);
        head.fill(0);
        head[0..2].copy_from_slice(&id.to_le_bytes());
        head[2] = flags;
        head[3] = depth;
        for (i, t) in texts.iter().enumerate() {
            head[4 + i] = t.len() as u8;
        }
        head[8..16].copy_from_slice(&object.to_le_bytes());
        head[16..20].copy_from_slice(&kind.to_le_bytes());
        for t in texts {
            let (dst, rest) = tail.split_at_mut(t.len());
            dst.copy_from_slice(t.as_bytes());
            tail = rest;
        }
        self.ids[self.count] = id;
        self.count += 1;
        self.at += size;
        if depth == 0 {
            self.top += 1;
        } else if let Some((head, children)) = self.submenu {
            self.submenu = Some((head, children + 1));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_menu(plane: &mut [u8]) -> usize {
        let mut m = MenuBuilder::new(plane);
        m.item(Item::new(1, "Open").shortcut("Enter"));
        m.open_with(2, "Open With", 0xfeed, DocumentKind::Code);
        m.separator();
        m.submenu("Sort By");
        m.item(Item::new(3, "Name").checked(true));
        m.item(Item::new(4, "Kind"));
        m.end_submenu();
        m.item(Item::new(5, "Delete from All Folders…").danger().detail("Every folder it is in"));
        m.item(Item::new(6, "Get Info").shortcut("Ctrl+I").enabled(false));
        m.finish().unwrap()
    }

    #[test]
    fn a_built_menu_parses_back_entry_for_entry() {
        let mut plane = [0_u8; MENU_PLANE_BYTES];
        let len = file_menu(&mut plane);
        let bytes = &plane[..len];
        let menu = Menu::parse(bytes).unwrap();
        let top: [usize; 6] = core::array::from_fn(|i| menu.top().nth(i).unwrap());
        assert_eq!(menu.top().count(), 6);
        assert_eq!(text(bytes, menu.entries()[top[0]].label), "Open");
        assert_eq!(text(bytes, menu.entries()[top[0]].shortcut), "Enter");
        let open_with = menu.entries()[top[1]];
        assert!(open_with.has(flag::OPEN_WITH) && !open_with.choosable());
        assert_eq!((open_with.id, open_with.object, open_with.kind), (2, 0xfeed, 2));
        assert!(menu.entries()[top[2]].has(flag::SEPARATOR));
        let sort = top[3];
        assert_eq!(menu.entries()[sort].children, 2);
        let children = menu.children(sort);
        assert_eq!(text(bytes, children[0].label), "Name");
        assert!(children[0].has(flag::CHECKED) && children[0].depth == 1);
        let delete = menu.find(5).unwrap();
        assert!(delete.has(flag::DANGER));
        assert_eq!(text(bytes, delete.detail), "Every folder it is in");
        assert!(!menu.find(6).unwrap().choosable());
        assert!(menu.find(0).is_none());
    }

    #[test]
    fn the_builder_refuses_what_the_parser_would() {
        let mut plane = [0_u8; MENU_PLANE_BYTES];
        let mut m = MenuBuilder::new(&mut plane);
        assert_eq!(m.finish(), Err(MenuError::Empty));
        let mut m = MenuBuilder::new(&mut plane);
        m.item(Item::new(0, "Zero"));
        assert_eq!(m.finish(), Err(MenuError::BadId));
        let mut m = MenuBuilder::new(&mut plane);
        m.item(Item::new(1, "A")).item(Item::new(1, "B"));
        assert_eq!(m.finish(), Err(MenuError::BadId));
        let mut m = MenuBuilder::new(&mut plane);
        m.submenu("Empty").end_submenu();
        assert_eq!(m.finish(), Err(MenuError::Empty));
        let mut m = MenuBuilder::new(&mut plane);
        m.submenu("A").submenu("B");
        assert_eq!(m.finish(), Err(MenuError::BadNesting));
        let mut m = MenuBuilder::new(&mut plane);
        m.item(Item::new(1, "Line\nbreak"));
        assert_eq!(m.finish(), Err(MenuError::BadText));
        let long = core::str::from_utf8(&[b'x'; LABEL_BYTES + 1]).unwrap();
        let mut m = MenuBuilder::new(&mut plane);
        m.item(Item::new(1, long));
        assert_eq!(m.finish(), Err(MenuError::TooLarge));
        let mut m = MenuBuilder::new(&mut plane);
        for id in 1..=MAX_TOP_ENTRIES as u16 + 1 {
            m.item(Item::new(id, "Item"));
        }
        assert_eq!(m.finish(), Err(MenuError::TooManyEntries));
        // The header and one entry's head and text: 8 + 24 + 4 bytes.
        let mut exact = [0_u8; 36];
        let mut m = MenuBuilder::new(&mut exact);
        m.item(Item::new(1, "Open"));
        assert_eq!(m.finish(), Ok(36));
        let mut small = [0_u8; 35];
        let mut m = MenuBuilder::new(&mut small);
        m.item(Item::new(1, "Open"));
        assert_eq!(m.finish(), Err(MenuError::TooLarge));
    }

    #[test]
    fn every_corrupted_byte_is_refused_or_still_a_valid_menu() {
        let mut plane = [0_u8; MENU_PLANE_BYTES];
        let len = file_menu(&mut plane);
        let good = plane;
        assert!(Menu::parse(&good[..len - 1]).is_err(), "a truncated menu");
        let mut longer = good;
        longer[len] = 0;
        assert_eq!(Menu::parse(&longer[..len + 1]), Err(MenuError::Malformed), "a trailing byte");
        // Any single flipped byte either refuses or yields a menu that still
        // keeps every invariant: no panic, spans in bounds, ids unique.
        for at in 0..len {
            for bit in 0..8 {
                let mut bytes = good;
                bytes[at] ^= 1 << bit;
                if let Ok(menu) = Menu::parse(&bytes[..len]) {
                    for (i, e) in menu.entries().iter().enumerate() {
                        for span in [e.label, e.detail, e.shortcut] {
                            assert!(usize::from(span.at) + usize::from(span.len) <= len);
                        }
                        assert!(e.id == 0 || menu.entries()[..i].iter().all(|o| o.id != e.id));
                    }
                }
            }
        }
    }

    #[test]
    fn a_child_must_follow_a_submenu() {
        let mut plane = [0_u8; MENU_PLANE_BYTES];
        let len = {
            let mut m = MenuBuilder::new(&mut plane);
            m.item(Item::new(1, "Open"));
            m.finish().unwrap()
        };
        // Make the only entry a depth-1 child with no parent.
        plane[HEADER_BYTES + 3] = 1;
        assert_eq!(Menu::parse(&plane[..len]), Err(MenuError::BadNesting));
    }
}
