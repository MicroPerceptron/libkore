//! Shortcuts as data (`res/KEYBOARD.md` §8): a [`Chord`] — one key and the
//! modifiers held with it — bound to an action in a [`Keymap`].
//!
//! A window's shortcuts are a table it reads, never a match over usages: the
//! same table answers what a key does, what its menu says the shortcut is,
//! and — for a terminal — which keys leave the shell's line at all
//! ([`crate::keysplit`]). A keymap the user edits (Settings ▸ Keyboard &
//! Pointer) replaces a table; nothing that reads one changes.
//!
//! Modifiers are side-agnostic: either Shift satisfies a chord's Shift. A
//! chord matches its modifiers exactly, so Command+Shift+C is not Command+C.
//! Locks never take part: Caps Lock changes what a letter types, not which
//! shortcut its key is.

use core::fmt;

use kore_abi::keyboard::{modifier, usage};

/// The modifiers a chord needs, left and right alike.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Mods(u8);

impl Mods {
    pub const NONE: Self = Self(0);
    pub const CTRL: Self = Self(1);
    pub const SHIFT: Self = Self(1 << 1);
    pub const ALT: Self = Self(1 << 2);
    /// The GUI key: Command on a Mac keyboard, the Super (Windows) key on a
    /// PC's.
    pub const COMMAND: Self = Self(1 << 3);

    /// The modifiers a report's bitmap holds.
    pub const fn of(modifiers: u8) -> Self {
        const fn held(modifiers: u8, mask: u8, bit: Mods) -> u8 {
            if modifiers & mask != 0 { bit.0 } else { 0 }
        }
        Self(
            held(modifiers, modifier::CTRL, Self::CTRL)
                | held(modifiers, modifier::SHIFT, Self::SHIFT)
                | held(modifiers, modifier::ALT, Self::ALT)
                | held(modifiers, modifier::GUI, Self::COMMAND),
        )
    }

    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// One key and the modifiers held with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Chord {
    key: u8,
    mods: Mods,
}

impl Chord {
    /// `key` (a Keyboard/Keypad-page usage) with `mods` held. `None` for no
    /// key or a modifier's own usage: a modifier is held with a chord, never
    /// its key.
    pub const fn new(key: u8, mods: Mods) -> Option<Self> {
        if key == 0 || modifier::bit(key).is_some() {
            return None;
        }
        Some(Self { key, mods })
    }

    pub const fn key(self) -> u8 {
        self.key
    }

    pub const fn mods(self) -> Mods {
        self.mods
    }

    /// Whether `key`, pressed with the report bitmap `modifiers` held, is
    /// this chord.
    pub const fn matches(self, key: u8, modifiers: u8) -> bool {
        self.key == key && Mods::of(modifiers).0 == self.mods.0
    }

    /// The chord as a menu or the settings page spells it: `Cmd+C`,
    /// `Shift+Page Up`.
    pub fn write_label(self, out: &mut impl fmt::Write) -> fmt::Result {
        for (bit, name) in [
            (Mods::CTRL, "Ctrl"),
            (Mods::ALT, "Alt"),
            (Mods::SHIFT, "Shift"),
            (Mods::COMMAND, "Cmd"),
        ] {
            if self.mods.contains(bit) {
                write!(out, "{name}+")?;
            }
        }
        match self.key {
            usage::A..=usage::Z => out.write_char(char::from(b'A' + (self.key - usage::A))),
            usage::DIGIT_1..=usage::DIGIT_9 => out.write_char(char::from(b'1' + (self.key - usage::DIGIT_1))),
            usage::DIGIT_0 => out.write_char('0'),
            usage::F1..=usage::F12 => write!(out, "F{}", self.key - usage::F1 + 1),
            key => out.write_str(match key {
                usage::ENTER => "Enter",
                usage::ESCAPE => "Esc",
                usage::BACKSPACE => "Backspace",
                usage::TAB => "Tab",
                usage::SPACE => "Space",
                usage::INSERT => "Insert",
                usage::HOME => "Home",
                usage::PAGE_UP => "Page Up",
                usage::DELETE => "Delete",
                usage::END => "End",
                usage::PAGE_DOWN => "Page Down",
                usage::RIGHT => "Right",
                usage::LEFT => "Left",
                usage::DOWN => "Down",
                usage::UP => "Up",
                _ => "Key",
            }),
        }
    }
}

/// The most bindings one keymap holds: a window's shortcuts, with room for a
/// user's own.
pub const MAX_BINDINGS: usize = 16;

/// A keymap is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeymapFull;

/// Chords bound to a window's actions: one action per chord, any number of
/// chords per action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keymap<A: Copy> {
    bindings: [Option<(Chord, A)>; MAX_BINDINGS],
}

impl<A: Copy + PartialEq> Default for Keymap<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Copy> Keymap<A> {
    pub const fn new() -> Self {
        Self {
            bindings: [None; MAX_BINDINGS],
        }
    }

    /// This keymap with `chord` bound to `action`, for a table built as a
    /// constant; a chord already bound or no room left is a build error.
    pub const fn with(mut self, chord: Option<Chord>, action: A) -> Self {
        let Some(chord) = chord else {
            panic!("a binding names a key");
        };
        let mut free = None;
        let mut index = 0;
        while index < MAX_BINDINGS {
            match self.bindings[index] {
                Some((bound, _)) if bound.key == chord.key && bound.mods.0 == chord.mods.0 => {
                    panic!("a chord is bound once");
                }
                None if free.is_none() => free = Some(index),
                _ => {}
            }
            index += 1;
        }
        let Some(free) = free else {
            panic!("the keymap is full");
        };
        self.bindings[free] = Some((chord, action));
        self
    }
}

impl<A: Copy + PartialEq> Keymap<A> {
    /// Bind `chord` to `action`, replacing what it was bound to.
    pub fn bind(&mut self, chord: Chord, action: A) -> Result<(), KeymapFull> {
        if let Some(bound) = self.bindings.iter_mut().flatten().find(|(bound, _)| *bound == chord) {
            bound.1 = action;
            return Ok(());
        }
        let free = self.bindings.iter_mut().find(|slot| slot.is_none()).ok_or(KeymapFull)?;
        *free = Some((chord, action));
        Ok(())
    }

    /// Leave `chord` unbound.
    pub fn unbind(&mut self, chord: Chord) {
        for slot in &mut self.bindings {
            if slot.is_some_and(|(bound, _)| bound == chord) {
                *slot = None;
            }
        }
    }

    /// What `key` pressed with the report bitmap `modifiers` held does.
    pub fn action(&self, key: u8, modifiers: u8) -> Option<A> {
        self.bindings
            .iter()
            .flatten()
            .find(|(chord, _)| chord.matches(key, modifiers))
            .map(|&(_, action)| action)
    }

    /// The chord a menu shows for `action`: the first bound to it.
    pub fn chord(&self, action: A) -> Option<Chord> {
        self.bindings
            .iter()
            .flatten()
            .find(|(_, bound)| *bound == action)
            .map(|&(chord, _)| chord)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::string::String;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Act {
        Copy,
        Paste,
    }

    const MAP: Keymap<Act> = Keymap::new()
        .with(Chord::new(usage::C, Mods::COMMAND), Act::Copy)
        .with(Chord::new(usage::V, Mods::COMMAND), Act::Paste)
        .with(Chord::new(usage::INSERT, Mods::SHIFT), Act::Paste);

    /// A chord matches its modifiers exactly, either side's, and no lock.
    #[test]
    fn a_chord_matches_its_modifiers_exactly_either_side() {
        assert_eq!(MAP.action(usage::C, modifier::LEFT_GUI), Some(Act::Copy));
        assert_eq!(MAP.action(usage::C, modifier::RIGHT_GUI), Some(Act::Copy));
        assert_eq!(MAP.action(usage::C, modifier::LEFT_GUI | modifier::LEFT_SHIFT), None);
        assert_eq!(MAP.action(usage::C, modifier::LEFT_CTRL), None);
        assert_eq!(MAP.action(usage::INSERT, modifier::RIGHT_SHIFT), Some(Act::Paste));
        assert_eq!(Chord::new(usage::LEFT_SHIFT, Mods::NONE), None, "a modifier is never a chord's key");
        assert_eq!(Chord::new(0, Mods::COMMAND), None);
    }

    /// Bindings change as data: rebinding replaces, unbinding removes, and a
    /// menu reads the chord an action is bound to.
    #[test]
    fn a_keymap_is_rebound_as_data() {
        let mut map = MAP;
        assert_eq!(map.chord(Act::Paste), Chord::new(usage::V, Mods::COMMAND));
        let ctrl_c = Chord::new(usage::C, Mods::CTRL.with(Mods::SHIFT)).unwrap();
        map.bind(ctrl_c, Act::Copy).unwrap();
        assert_eq!(map.action(usage::C, modifier::LEFT_CTRL | modifier::LEFT_SHIFT), Some(Act::Copy));
        map.bind(ctrl_c, Act::Paste).unwrap();
        assert_eq!(map.action(usage::C, modifier::LEFT_CTRL | modifier::LEFT_SHIFT), Some(Act::Paste));
        map.unbind(Chord::new(usage::V, Mods::COMMAND).unwrap());
        assert_eq!(map.action(usage::V, modifier::LEFT_GUI), None);
        assert_eq!(map.chord(Act::Paste), Chord::new(usage::INSERT, Mods::SHIFT));
        let mut full = Keymap::<Act>::new();
        for key in 0..MAX_BINDINGS as u8 {
            full.bind(Chord::new(usage::A + key, Mods::COMMAND).unwrap(), Act::Copy).unwrap();
        }
        assert_eq!(full.bind(Chord::new(usage::F1, Mods::NONE).unwrap(), Act::Copy), Err(KeymapFull));
    }

    /// A chord spells itself as menus show it.
    #[test]
    fn a_chord_labels_itself() {
        let label = |chord: Option<Chord>| {
            let mut text = String::new();
            chord.unwrap().write_label(&mut text).unwrap();
            text
        };
        assert_eq!(label(Chord::new(usage::C, Mods::COMMAND)), "Cmd+C");
        assert_eq!(label(Chord::new(usage::PAGE_UP, Mods::SHIFT)), "Shift+Page Up");
        assert_eq!(label(Chord::new(usage::DIGIT_0, Mods::CTRL.with(Mods::ALT))), "Ctrl+Alt+0");
        assert_eq!(label(Chord::new(usage::F1 + 11, Mods::NONE)), "F12");
    }
}
