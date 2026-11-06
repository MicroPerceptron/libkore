//! The notification record (`res/SCREENSHOTS.md`, Feedback).
//!
//! Data only and bounded, copied like a [`ShellAction`]: a title, a body and
//! an optional subject the notification is about. The record says nothing of
//! how it is shown. Its first sink is a line on the seat's terminal, which
//! the controller prints; the compositor's popup is a later sink behind the
//! same record.
//!
//! A source posts by handing its controller the encoded record. The capture
//! writer leaves it in its process output, which its controller reads when
//! it reaps the writer.
//!
//! [`ShellAction`]: crate::control::ShellAction

use crate::document::DocumentKind;

/// The longest title, in bytes of UTF-8.
pub const TITLE_BYTES: usize = 32;
/// The longest body, in bytes of UTF-8.
pub const BODY_BYTES: usize = 192;

const MAGIC: [u8; 4] = *b"KNT1";
/// Magic, title and body lengths, flags, subject kind, subject object.
const HEADER_BYTES: usize = 4 + 1 + 1 + 1 + 1 + 8;
/// The longest encoded record.
pub const RECORD_BYTES: usize = HEADER_BYTES + TITLE_BYTES + BODY_BYTES;
const FLAG_SUBJECT: u8 = 1;

/// What a notification is about: a filesystem object a click would open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Subject {
    /// Its stable identity (never zero), as `ShellAction::Open` carries it.
    pub object: u64,
    pub kind: DocumentKind,
}

/// One notification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Notification {
    title: [u8; TITLE_BYTES],
    title_len: u8,
    body: [u8; BODY_BYTES],
    body_len: u8,
    subject: Option<Subject>,
}

impl Notification {
    /// A notification, or `None` if `title` or `body` is too long or holds
    /// a control character, or the subject names no object. The text is
    /// display text only: a sink shows it, and nothing in it may steer the
    /// sink (a terminal line must not carry an escape sequence).
    pub fn new(title: &str, body: &str, subject: Option<Subject>) -> Option<Self> {
        if title.len() > TITLE_BYTES
            || body.len() > BODY_BYTES
            || title.chars().chain(body.chars()).any(char::is_control)
            || subject.is_some_and(|subject| subject.object == 0)
        {
            return None;
        }
        let mut record = Self {
            title: [0; TITLE_BYTES],
            title_len: title.len() as u8,
            body: [0; BODY_BYTES],
            body_len: body.len() as u8,
            subject,
        };
        record.title[..title.len()].copy_from_slice(title.as_bytes());
        record.body[..body.len()].copy_from_slice(body.as_bytes());
        Some(record)
    }

    pub fn title(&self) -> &str {
        // Validated UTF-8 at construction and at decode.
        core::str::from_utf8(&self.title[..usize::from(self.title_len)]).unwrap_or("")
    }

    pub fn body(&self) -> &str {
        core::str::from_utf8(&self.body[..usize::from(self.body_len)]).unwrap_or("")
    }

    pub const fn subject(&self) -> Option<Subject> {
        self.subject
    }

    /// Encode into `out`, answering the bytes written; `None` if `out` is
    /// shorter than the record.
    pub fn encode(&self, out: &mut [u8]) -> Option<usize> {
        let (title, body) = (usize::from(self.title_len), usize::from(self.body_len));
        let len = HEADER_BYTES + title + body;
        let out = out.get_mut(..len)?;
        out[..4].copy_from_slice(&MAGIC);
        out[4] = self.title_len;
        out[5] = self.body_len;
        out[6] = if self.subject.is_some() { FLAG_SUBJECT } else { 0 };
        let (kind, object) = self
            .subject
            .map_or((0, 0), |subject| (subject.kind.code(), subject.object));
        out[7] = kind;
        out[8..16].copy_from_slice(&object.to_le_bytes());
        out[HEADER_BYTES..HEADER_BYTES + title].copy_from_slice(&self.title[..title]);
        out[HEADER_BYTES + title..].copy_from_slice(&self.body[..body]);
        Some(len)
    }

    /// The record `bytes` holds exactly, or `None`: a wrong magic or length,
    /// text that is not UTF-8 or holds a control character, a subject flag
    /// without an object, or any stray byte.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_BYTES || bytes[..4] != MAGIC {
            return None;
        }
        let (title, body) = (usize::from(bytes[4]), usize::from(bytes[5]));
        if bytes.len() != HEADER_BYTES + title + body {
            return None;
        }
        let object = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let subject = match bytes[6] {
            0 if bytes[7] == 0 && object == 0 => None,
            FLAG_SUBJECT => Some(Subject {
                object,
                kind: DocumentKind::from_code(u64::from(bytes[7]))?,
            }),
            _ => return None,
        };
        Self::new(
            core::str::from_utf8(&bytes[HEADER_BYTES..HEADER_BYTES + title]).ok()?,
            core::str::from_utf8(&bytes[HEADER_BYTES + title..]).ok()?,
            subject,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_round_trips_exactly() {
        let subject = Subject {
            object: 0xfeed,
            kind: DocumentKind::Image,
        };
        for record in [
            Notification::new("Screenshot", "pictures/screenshots/a.png (1280×800, 1.2 MB)", Some(subject)),
            Notification::new("", "", None),
            Notification::new(&"t".repeat(TITLE_BYTES), &"b".repeat(BODY_BYTES), None),
        ] {
            let record = record.unwrap();
            let mut out = [0u8; RECORD_BYTES];
            let len = record.encode(&mut out).unwrap();
            assert_eq!(Notification::decode(&out[..len]), Some(record));
            assert_eq!(record.encode(&mut out[..len - 1]), None, "a short buffer");
        }
    }

    #[test]
    fn a_bad_record_is_refused() {
        assert_eq!(Notification::new(&"t".repeat(TITLE_BYTES + 1), "", None), None);
        assert_eq!(Notification::new("", &"b".repeat(BODY_BYTES + 1), None), None);
        let zero = Subject {
            object: 0,
            kind: DocumentKind::Image,
        };
        assert_eq!(Notification::new("", "", Some(zero)), None);
        for control in ["\x1b[2J", "a\nb", "\u{7f}", "\u{9b}31m"] {
            assert_eq!(Notification::new(control, "", None), None, "{control:?} as a title");
            assert_eq!(Notification::new("", control, None), None, "{control:?} as a body");
        }

        let record = Notification::new("Screenshot", "saved", None).unwrap();
        let mut out = [0u8; RECORD_BYTES];
        let len = record.encode(&mut out).unwrap();
        let good = &out[..len];
        let mutate = |at: usize, byte: u8| {
            let mut bad = [0u8; RECORD_BYTES];
            bad[..len].copy_from_slice(good);
            bad[at] = byte;
            Notification::decode(&bad[..len])
        };
        assert_eq!(mutate(0, b'X'), None, "magic");
        assert_eq!(mutate(4, 11), None, "title length");
        assert_eq!(mutate(6, 2), None, "unknown flag");
        assert_eq!(mutate(7, 3), None, "a kind without a subject");
        assert_eq!(mutate(9, 1), None, "an object without a subject");
        assert_eq!(mutate(HEADER_BYTES, 0xff), None, "not UTF-8");
        assert_eq!(mutate(HEADER_BYTES, 0x1b), None, "an escape");
        assert_eq!(Notification::decode(&good[..len - 1]), None, "truncated");
    }
}
