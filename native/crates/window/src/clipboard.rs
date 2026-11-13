//! The seat's clipboard (`res/CLIPBOARD.md`): one text the compositor holds
//! for each seat, which every window of that seat can put text on and read
//! text from, and which no window can touch without the user asking.
//!
//! The text crosses a window's own pair lease, never a lent view: a piece at
//! a time through the request plane the window already maps
//! ([`crate::menu`]'s page), copied out by the compositor before it answers a
//! copy and copied in before it answers a paste. A lent view per copy or
//! paste would spend the window's map cursor, which no unmap gives back, and
//! carry a capability and a revocation for a few hundred bytes of text.
//!
//! **Gesture-gated.** Only the seat's focused window may begin a copy or a
//! paste, and only on a key press, button press or menu choice the
//! compositor delivered to it moments before — each such gesture begins one
//! copy or one paste. A window in the background can neither overwrite the
//! clipboard nor read it.

/// The most the clipboard holds, in bytes: a copy past it is refused, never
/// cut short. Equal to `kore_edit::CLIPBOARD_BYTES`, the bound every text
/// field already copies within, so any selection a field can copy fits.
pub const CLIPBOARD_BYTES: usize = 512 << 10;

/// The most one piece of a copy or a paste carries: the request plane whole.
pub const CHUNK_BYTES: usize = crate::menu::MENU_PLANE_BYTES;

const _: () = assert!(CLIPBOARD_BYTES <= u32::MAX as usize && CHUNK_BYTES <= u16::MAX as usize);

/// What a paste into a shell's line takes from the text (`res/CLIPBOARD.md`
/// §Shell): the printable ASCII before the first line break, and never a line
/// break, so a paste can never run a command the user has not read. Why the
/// rest was held back is told to the user, never silently dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellStop {
    /// The whole text was taken.
    Whole,
    /// The text goes on past a line break.
    LineBreak,
    /// A character the line cannot hold: a control byte, or a character
    /// outside ASCII.
    Unholdable,
    /// The line is full.
    Full,
}

impl ShellStop {
    /// What the user is told about the text held back: nothing when the
    /// whole text was typed.
    pub const fn describe(self) -> Option<&'static str> {
        match self {
            Self::Whole => None,
            Self::LineBreak => Some("typed the first line; the other lines were held back"),
            Self::Unholdable => {
                Some("stopped at a tab, control or non-ASCII character the line cannot hold")
            }
            Self::Full => Some("the line is full; the rest was held back"),
        }
    }
}

/// Scan `bytes`, the text from where the last scan stopped, for what a shell
/// paste may take: the length of the run it takes, and why it stops there.
/// `last` says `bytes` reaches the text's end; `room` bounds the run. `None`
/// means `bytes` ran out before the text did: scan the next piece with the
/// room that is left, even none, to learn what follows.
pub fn shell_run(bytes: &[u8], last: bool, room: usize) -> (usize, Option<ShellStop>) {
    let limit = bytes.len().min(room);
    let taken = bytes[..limit]
        .iter()
        .position(|&byte| !(b' '..=b'~').contains(&byte))
        .unwrap_or(limit);
    let stop = match bytes.get(taken) {
        Some(b'\r' | b'\n') => Some(ShellStop::LineBreak),
        Some(_) if taken < room => Some(ShellStop::Unholdable),
        Some(_) => Some(ShellStop::Full),
        None if last => Some(ShellStop::Whole),
        None => None,
    };
    (taken, stop)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shell paste takes printable ASCII up to the first line break, the
    /// first character the line cannot hold, or the line's room.
    #[test]
    fn a_shell_run_stops_before_what_the_line_cannot_take() {
        assert_eq!(shell_run(b"ls -la", true, 255), (6, Some(ShellStop::Whole)));
        assert_eq!(shell_run(b"ls -la", false, 255), (6, None));
        assert_eq!(shell_run(b"echo hi\nrm -rf /", true, 255), (7, Some(ShellStop::LineBreak)));
        assert_eq!(shell_run(b"a\r\nb", true, 255), (1, Some(ShellStop::LineBreak)));
        assert_eq!(shell_run(b"tab\there", true, 255), (3, Some(ShellStop::Unholdable)));
        assert_eq!(shell_run("caf\u{e9}".as_bytes(), true, 255), (3, Some(ShellStop::Unholdable)));
        assert_eq!(shell_run(b"abcdef", true, 4), (4, Some(ShellStop::Full)));
        // A run that fills the room exactly at the text's end took it whole;
        // short of the end, what follows decides.
        assert_eq!(shell_run(b"abcd", true, 4), (4, Some(ShellStop::Whole)));
        assert_eq!(shell_run(b"abcd", false, 4), (4, None));
        assert_eq!(shell_run(b"\nmore", true, 0), (0, Some(ShellStop::LineBreak)));
        assert_eq!(shell_run(b"more", true, 0), (0, Some(ShellStop::Full)));
        assert_eq!(shell_run(b"", true, 4), (0, Some(ShellStop::Whole)));
        // A line break at the room's edge is still a line break: the line
        // is not full, the text has another line.
        assert_eq!(shell_run(b"abcd\n", true, 4), (4, Some(ShellStop::LineBreak)));
    }
}
