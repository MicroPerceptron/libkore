//! The keys a terminal window keeps from its shell (`res/KEYBOARD.md` §8).
//!
//! A terminal's keys are its shell's line, except the chords that act on the
//! window itself: its [`TerminalKeys`] — by default Command+C to copy the
//! selection, Command+V to paste and Command+A to select all, as on a Mac, so
//! Control stays the shell's (Control+C interrupts), and Shift with Page Up,
//! Page Down, Home or End to scroll the window's history. With Command held
//! every key is the window's, bound or not: Command never types.
//!
//! A key belongs to whoever it was pressed for, until it is released: Command
//! let go while C is still held does not hand the shell a fresh `c`, and a
//! letter held down before Command was pressed stays the shell's. So the
//! split is a state machine over complete snapshots, one per source, run by
//! whatever stands between the keyboard and the line: input-stream for the
//! founded terminal, the terminal app for a program it hosts.

use kore_abi::keyboard::{modifier, usage};

use crate::shortcut::{Chord, Keymap, Mods};

/// Keys one snapshot lists.
pub const SNAPSHOT_KEYS: usize = 6;

/// What a terminal window does with a key it keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalAction {
    /// Put the selection on the clipboard.
    Copy,
    /// Type the clipboard into the line.
    Paste,
    /// Select every row held.
    SelectAll,
    /// Scroll history a page back, a page forward, to its oldest row, or
    /// back to the live grid.
    PageUp,
    PageDown,
    Top,
    Bottom,
}

/// A terminal window's keys: the shortcuts it acts on, and whether Command
/// keeps every key from the line. Both sides of the split read one table —
/// input-stream, which takes the keys out of the founded shell's line, and
/// the window, which acts on them — so a keymap the user edits changes both
/// at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalKeys {
    pub keymap: Keymap<TerminalAction>,
    /// With Command held, every key is the window's, bound or not.
    pub command_reserved: bool,
}

impl TerminalKeys {
    pub const DEFAULT: Self = Self {
        keymap: Keymap::new()
            .with(Chord::new(usage::C, Mods::COMMAND), TerminalAction::Copy)
            .with(Chord::new(usage::V, Mods::COMMAND), TerminalAction::Paste)
            .with(Chord::new(usage::A, Mods::COMMAND), TerminalAction::SelectAll)
            .with(Chord::new(usage::PAGE_UP, Mods::SHIFT), TerminalAction::PageUp)
            .with(Chord::new(usage::PAGE_DOWN, Mods::SHIFT), TerminalAction::PageDown)
            .with(Chord::new(usage::HOME, Mods::SHIFT), TerminalAction::Top)
            .with(Chord::new(usage::END, Mods::SHIFT), TerminalAction::Bottom),
        command_reserved: true,
    };

    /// Whether `key`, pressed with the report bitmap `modifiers` held, is the
    /// window's rather than the line's.
    pub fn owns(&self, key: u8, modifiers: u8) -> bool {
        if key == 0 || modifier::bit(key).is_some() {
            return false;
        }
        self.keymap.action(key, modifiers).is_some()
            || (self.command_reserved && Mods::of(modifiers).contains(Mods::COMMAND))
    }
}

impl Default for TerminalKeys {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Sources a split tells apart at once: input-stream's own bound on the
/// keyboards one terminal reads (its `PrevTable`). A further source shares
/// the last place, which forgets its holder's keys as it changes hands —
/// degraded the way every per-source table there is, never dropped.
const SOURCES: usize = 4;

/// What one source holds, by whom it was pressed for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Held {
    source: u8,
    window: [u8; SNAPSHOT_KEYS],
    shell: [u8; SNAPSHOT_KEYS],
    /// The window's view as last handed out: the modifiers and its keys.
    shown: (u8, [u8; SNAPSHOT_KEYS]),
}

impl Held {
    /// A source as the window first knows it: nothing held, no modifiers —
    /// the state a fresh window's keyboard starts in, so plain typing never
    /// reaches the window at all.
    const FRESH: Self = Self {
        source: 0,
        window: [0; SNAPSHOT_KEYS],
        shell: [0; SNAPSHOT_KEYS],
        shown: (0, [0; SNAPSHOT_KEYS]),
    };
}

/// One snapshot, split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// The keys the shell's line reads: the snapshot without the window's.
    pub shell: [u8; SNAPSHOT_KEYS],
    /// The keys the window reads, with the snapshot's modifiers, when what
    /// the window sees changed: a key of its own pressed or released, or a
    /// modifier, which it needs for a Shift-click.
    pub window: Option<[u8; SNAPSHOT_KEYS]>,
}

/// The keys of `keys` that `side` holds, in their order, packed to the front
/// as a keyboard reports them.
fn pack(keys: [u8; SNAPSHOT_KEYS], side: &[u8; SNAPSHOT_KEYS]) -> [u8; SNAPSHOT_KEYS] {
    let mut packed = [0; SNAPSHOT_KEYS];
    for (slot, key) in packed
        .iter_mut()
        .zip(keys.into_iter().filter(|key| *key != 0 && side.contains(key)))
    {
        *slot = key;
    }
    packed
}

/// The split's state, per source, under one window's keys.
pub struct KeySplit {
    keys: TerminalKeys,
    held: [Held; SOURCES],
    used: usize,
}

impl KeySplit {
    /// A split under `keys`: [`TerminalKeys::DEFAULT`] until the user's
    /// keymap names others.
    pub const fn new(keys: TerminalKeys) -> Self {
        Self {
            keys,
            held: [Held::FRESH; SOURCES],
            used: 0,
        }
    }

    /// `source`'s place in `held`, of which `used` are taken.
    fn place<'h>(held: &'h mut [Held; SOURCES], used: &mut usize, source: u8) -> &'h mut Held {
        let index = match held[..*used].iter().position(|held| held.source == source) {
            Some(index) => index,
            None => {
                let index = (*used).min(SOURCES - 1);
                *used = (*used + 1).min(SOURCES);
                held[index] = Held { source, ..Held::FRESH };
                index
            }
        };
        &mut held[index]
    }

    /// Split `source`'s complete snapshot: `modifiers` held, `keys` down.
    pub fn split(&mut self, source: u8, modifiers: u8, keys: [u8; SNAPSHOT_KEYS]) -> Split {
        let keys_owned = &self.keys;
        let held = Self::place(&mut self.held, &mut self.used, source);
        // A key no longer down is released, whoever held it.
        for owned in held.window.iter_mut().chain(held.shell.iter_mut()) {
            if *owned != 0 && !keys.contains(owned) {
                *owned = 0;
            }
        }
        // A key newly down is the owner's its press made it.
        for &key in keys.iter().filter(|&&key| key != 0) {
            if held.window.contains(&key) || held.shell.contains(&key) {
                continue;
            }
            let side = if keys_owned.owns(key, modifiers) {
                &mut held.window
            } else {
                &mut held.shell
            };
            // The two sides partition one snapshot's keys, so a free place is
            // always there.
            if let Some(free) = side.iter_mut().find(|slot| **slot == 0) {
                *free = key;
            }
        }
        let shell = pack(keys, &held.shell);
        let window = pack(keys, &held.window);
        let view = (modifiers, window);
        let changed = held.shown != view;
        held.shown = view;
        Split {
            shell,
            window: changed.then_some(window),
        }
    }

    /// A route boundary: whatever was held is forgotten, as the window
    /// forgets it on the focus change that made the boundary.
    pub fn reset(&mut self) {
        *self = Self::new(self.keys);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::keyboard::modifier::{GUI, LEFT_GUI, LEFT_SHIFT};

    const C: u8 = 0x06;
    const V: u8 = 0x19;
    const A: u8 = 0x04;

    fn keys_of(down: &[u8]) -> [u8; SNAPSHOT_KEYS] {
        keys(down)
    }

    fn keys(down: &[u8]) -> [u8; SNAPSHOT_KEYS] {
        let mut keys = [0; SNAPSHOT_KEYS];
        keys[..down.len()].copy_from_slice(down);
        keys
    }

    /// Command chords and Shift with the scrolling keys are the window's;
    /// everything else, Control chords included, is the shell's.
    #[test]
    fn the_window_keeps_command_chords_and_shifted_scrolling() {
        let keys = TerminalKeys::DEFAULT;
        assert!(keys.owns(C, LEFT_GUI));
        assert!(keys.owns(V, GUI));
        assert!(keys.owns(usage::PAGE_UP, LEFT_SHIFT));
        assert!(keys.owns(usage::END, LEFT_SHIFT | LEFT_GUI), "Command keeps every key");
        assert!(!keys.owns(usage::PAGE_UP, 0));
        assert!(!keys.owns(C, modifier::LEFT_CTRL));
        assert!(!keys.owns(A, LEFT_SHIFT));
        assert!(!keys.owns(usage::LEFT_GUI, LEFT_GUI), "a modifier's own usage is state");
        assert!(!keys.owns(0, LEFT_GUI));
        assert_eq!(keys.keymap.action(V, modifier::RIGHT_GUI), Some(TerminalAction::Paste));
        assert_eq!(keys.keymap.action(usage::HOME, LEFT_SHIFT), Some(TerminalAction::Top));
    }

    /// The split follows the keymap it was given: a user's binding takes its
    /// key from the line, and without Command reserved, unbound Command keys
    /// stay the line's.
    #[test]
    fn the_split_reads_the_keymap_it_was_given() {
        let mut keys = TerminalKeys::DEFAULT;
        keys.command_reserved = false;
        keys.keymap
            .bind(Chord::new(usage::INSERT, Mods::SHIFT).unwrap(), TerminalAction::Paste)
            .unwrap();
        assert!(keys.owns(usage::INSERT, LEFT_SHIFT));
        assert!(!keys.owns(usage::Z, LEFT_GUI));
        let mut split = KeySplit::new(keys);
        let paste = split.split(0, LEFT_SHIFT, keys_of(&[usage::INSERT]));
        assert_eq!((paste.shell, paste.window), (keys_of(&[]), Some(keys_of(&[usage::INSERT]))));
    }

    /// A key stays with whoever it was pressed for until it is released, and
    /// the window hears its own keys and every modifier change, nothing else.
    #[test]
    fn a_key_belongs_to_whoever_it_was_pressed_for() {
        let mut split = KeySplit::new(TerminalKeys::DEFAULT);
        // `a` held, then Command: the `a` stays the shell's, and no Command+A.
        let first = split.split(1, 0, keys(&[A]));
        assert_eq!((first.shell, first.window), (keys(&[A]), None), "plain typing is the line's alone");
        let chord = split.split(1, LEFT_GUI, keys(&[A]));
        assert_eq!((chord.shell, chord.window), (keys(&[A]), Some(keys(&[]))));
        // Command+C, with `a` still down.
        let copy = split.split(1, LEFT_GUI, keys(&[A, C]));
        assert_eq!((copy.shell, copy.window), (keys(&[A]), Some(keys(&[C]))));
        // Command let go while C is held: still the window's, no `c` typed.
        let late = split.split(1, 0, keys(&[A, C]));
        assert_eq!((late.shell, late.window), (keys(&[A]), Some(keys(&[C]))));
        // Released: C pressed again now is the shell's.
        let up = split.split(1, 0, keys(&[A]));
        assert_eq!((up.shell, up.window), (keys(&[A]), Some(keys(&[]))));
        let again = split.split(1, 0, keys(&[A, C]));
        assert_eq!((again.shell, again.window), (keys(&[A, C]), None), "nothing changed for the window");
        // Another source is its own.
        let other = split.split(2, LEFT_SHIFT, keys(&[usage::PAGE_UP]));
        assert_eq!((other.shell, other.window), (keys(&[]), Some(keys(&[usage::PAGE_UP]))));
        assert_eq!(split.split(1, 0, keys(&[A, C])).window, None);
    }

    /// A source past the table's bound takes the last place, forgetting the
    /// keys its holder had there rather than reading them as its own.
    #[test]
    fn a_source_past_the_bound_shares_the_last_place() {
        let mut split = KeySplit::new(TerminalKeys::DEFAULT);
        for source in 0..SOURCES as u8 {
            let _ = split.split(source, LEFT_GUI, keys(&[C]));
        }
        let stranger = split.split(9, 0, keys(&[C]));
        assert_eq!((stranger.shell, stranger.window), (keys(&[C]), None));
        let first = split.split(0, 0, keys(&[C]));
        assert_eq!((first.shell, first.window), (keys(&[]), Some(keys(&[C]))), "the first sources keep theirs");
    }

    /// A boundary forgets every key, and the window hears the next state.
    #[test]
    fn a_reset_forgets_the_holders() {
        let mut split = KeySplit::new(TerminalKeys::DEFAULT);
        let _ = split.split(0, LEFT_GUI, keys(&[V]));
        split.reset();
        let after = split.split(0, 0, keys(&[V]));
        assert_eq!((after.shell, after.window), (keys(&[V]), None));
    }
}
