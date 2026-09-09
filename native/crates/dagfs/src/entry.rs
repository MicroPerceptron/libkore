use crate::{Name, ObjectId, ObjectKind, ObjectSubtype};

/// One entry from a directory listing, lifted out of the data window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub name: Name,
    pub object: ObjectId,
    pub kind: ObjectKind,
    /// Data, never authority (see [`ObjectSubtype`]).
    pub subtype: ObjectSubtype,
}
