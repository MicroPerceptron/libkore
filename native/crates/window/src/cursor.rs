//! The cursor a window asks for over its own content
//! ([`SurReqWire::OP_SET_CURSOR`]).
//!
//! What the pointer shows says what a press would do there, and only the app
//! knows that about its own pixels: text takes a caret, a control activates.
//! The compositor still chooses for the pixels it owns — chrome, the dock,
//! the desktop — and draws every shape from one family, so a window names a
//! shape and never supplies an image.

use kore_abi::svc::SurReqWire;

/// One cursor a window may ask for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Cursor {
    /// The standard arrow.
    #[default]
    Default,
    /// The I-beam: text a press would place a caret in.
    Text,
    /// The pointing hand: a control a press would activate.
    Pointer,
    /// The open hand: something a press would pick up.
    Grab,
    /// The closed hand: something being carried.
    Grabbing,
    /// The vertical resize arrow.
    ResizeNs,
    /// The horizontal resize arrow.
    ResizeEw,
    /// The diagonal resize arrow from top-left to bottom-right.
    ResizeNwse,
    /// The diagonal resize arrow from top-right to bottom-left.
    ResizeNesw,
    /// The hand under a no-entry badge: what a press here would not do.
    NotAllowed,
    /// The spinner: work under way while the window still answers.
    Progress,
}

impl Cursor {
    /// Every cursor, in wire-code order.
    pub const ALL: [Self; 11] = [
        Self::Default,
        Self::Text,
        Self::Pointer,
        Self::Grab,
        Self::Grabbing,
        Self::ResizeNs,
        Self::ResizeEw,
        Self::ResizeNwse,
        Self::ResizeNesw,
        Self::NotAllowed,
        Self::Progress,
    ];

    /// The [`SurReqWire`] `CURSOR_*` code this cursor rides as.
    pub const fn code(self) -> u64 {
        match self {
            Self::Default => SurReqWire::CURSOR_DEFAULT,
            Self::Text => SurReqWire::CURSOR_TEXT,
            Self::Pointer => SurReqWire::CURSOR_POINTER,
            Self::Grab => SurReqWire::CURSOR_GRAB,
            Self::Grabbing => SurReqWire::CURSOR_GRABBING,
            Self::ResizeNs => SurReqWire::CURSOR_RESIZE_NS,
            Self::ResizeEw => SurReqWire::CURSOR_RESIZE_EW,
            Self::ResizeNwse => SurReqWire::CURSOR_RESIZE_NWSE,
            Self::ResizeNesw => SurReqWire::CURSOR_RESIZE_NESW,
            Self::NotAllowed => SurReqWire::CURSOR_NOT_ALLOWED,
            Self::Progress => SurReqWire::CURSOR_PROGRESS,
        }
    }

    /// The cursor a wire code names; `None` for a code no cursor has, which
    /// the codec refuses rather than guessing.
    pub const fn from_code(code: u64) -> Option<Self> {
        Some(match code {
            SurReqWire::CURSOR_DEFAULT => Self::Default,
            SurReqWire::CURSOR_TEXT => Self::Text,
            SurReqWire::CURSOR_POINTER => Self::Pointer,
            SurReqWire::CURSOR_GRAB => Self::Grab,
            SurReqWire::CURSOR_GRABBING => Self::Grabbing,
            SurReqWire::CURSOR_RESIZE_NS => Self::ResizeNs,
            SurReqWire::CURSOR_RESIZE_EW => Self::ResizeEw,
            SurReqWire::CURSOR_RESIZE_NWSE => Self::ResizeNwse,
            SurReqWire::CURSOR_RESIZE_NESW => Self::ResizeNesw,
            SurReqWire::CURSOR_NOT_ALLOWED => Self::NotAllowed,
            SurReqWire::CURSOR_PROGRESS => Self::Progress,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cursor_round_trips_its_code_and_no_other_code_names_one() {
        for (index, cursor) in Cursor::ALL.into_iter().enumerate() {
            assert_eq!(cursor.code(), index as u64, "{cursor:?}");
            assert_eq!(Cursor::from_code(cursor.code()), Some(cursor));
        }
        for code in [Cursor::ALL.len() as u64, 255, u64::MAX] {
            assert_eq!(Cursor::from_code(code), None);
        }
    }
}
