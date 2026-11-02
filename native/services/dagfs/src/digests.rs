//! **Content digests, remembered** — the durable format 5 sidecar
//! (`res/STORAGE_TRANSPORT.md` §6).
//!
//! Lending a file version needs the BLAKE3 of exactly the bytes the lease
//! pins, and mount revalidation needs the same digest for every file behind
//! a stored search result. Both used to reread and rehash the whole file —
//! at every boot, for every catalogue member exec seeds, and for every
//! indexed file at mount. A revision's bytes never change, so its digest is
//! a fact worth keeping.
//!
//! [`DigestMemo`] keeps one digest per object, keyed by the revision it was
//! computed over, so a later revision can never be answered with an earlier
//! one's digest. Commits persist the memo into a sidecar extent the root
//! names, and mount loads it back.
//!
//! **The sidecar is derived data, never authority.** It is covered by its
//! own CRC and keyed by revision; a sidecar that fails its CRC, names a
//! revision the image no longer records, or is simply absent is ignored, and
//! the digest is computed again from the bytes. Nothing a sidecar says can
//! make a lease or a search result claim bytes the file does not hold —
//! the worst a stale or damaged sidecar can do is cost one rehash.

use alloc::vec::Vec;

use crc::{CRC_32_ISCSI, Crc};

use crate::StableObjectId;

const CRC: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);

/// Sidecar magic — "KOREDIG1".
pub const MAGIC: &[u8; 8] = b"KOREDIG1";
/// Header: magic, record count, CRC of the records.
pub const HEADER_BYTES: usize = 16;
/// One record: stable id, revision, digest.
pub const RECORD_BYTES: usize = 8 + 8 + 32;

/// One remembered digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    object: u64,
    revision: u64,
    digest: [u8; 32],
}

/// Why a sidecar was not loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarError {
    /// Not a sidecar, or a count past the capacity.
    Shape,
    /// The records fail their CRC.
    Crc,
}

/// The remembered digests, one per object, at most `capacity`.
pub struct DigestMemo {
    entries: Vec<Entry>,
    capacity: usize,
    dirty: bool,
}

impl DigestMemo {
    /// An empty memo for up to `capacity` objects.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
            dirty: false,
        }
    }

    /// The digest of `object` at `revision`, if remembered.
    pub fn get(&self, object: StableObjectId, revision: u64) -> Option<[u8; 32]> {
        self.entries
            .iter()
            .find(|entry| entry.object == object.get() && entry.revision == revision)
            .map(|entry| entry.digest)
    }

    /// Remember `digest` for `object` at `revision`, replacing whatever the
    /// object had. A full memo keeps what it has: forgetting costs a rehash,
    /// never a wrong answer.
    pub fn put(&mut self, object: StableObjectId, revision: u64, digest: [u8; 32]) {
        let key = object.get();
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.object == key) {
            if entry.revision != revision || entry.digest != digest {
                *entry = Entry {
                    object: key,
                    revision,
                    digest,
                };
                self.dirty = true;
            }
            return;
        }
        if self.entries.len() >= self.capacity || self.entries.try_reserve(1).is_err() {
            return;
        }
        self.entries.push(Entry {
            object: key,
            revision,
            digest,
        });
        self.dirty = true;
    }

    /// Keep only entries whose object still exists at the remembered
    /// revision (`current` answers an object's revision now, `None` for
    /// gone). Called before a commit encodes the memo.
    pub fn prune(&mut self, mut current: impl FnMut(StableObjectId) -> Option<u64>) {
        let before = self.entries.len();
        self.entries.retain(|entry| {
            StableObjectId::new(entry.object)
                .and_then(&mut current)
                .is_some_and(|revision| revision == entry.revision)
        });
        if self.entries.len() != before {
            self.dirty = true;
        }
    }

    /// Whether the memo changed since it was last encoded or loaded.
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the memo as not yet durable — a commit that encoded it failed,
    /// or a loaded sidecar was damaged and must be replaced.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Mark the memo durable without encoding it — its emptiness is what
    /// the next root records.
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Entries remembered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is remembered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes the encoded sidecar occupies.
    pub fn encoded_bytes(&self) -> usize {
        HEADER_BYTES + self.entries.len() * RECORD_BYTES
    }

    /// Encode into `dst` (at least [`encoded_bytes`](Self::encoded_bytes)
    /// long), sorted by object so the bytes are canonical, and mark the memo
    /// clean. Returns the encoded length.
    pub fn encode(&mut self, dst: &mut [u8]) -> Option<usize> {
        let len = self.encoded_bytes();
        let out = dst.get_mut(..len)?;
        self.entries.sort_unstable_by_key(|entry| entry.object);
        out[..8].copy_from_slice(MAGIC);
        out[8..12].copy_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for (index, entry) in self.entries.iter().enumerate() {
            let at = HEADER_BYTES + index * RECORD_BYTES;
            out[at..at + 8].copy_from_slice(&entry.object.to_le_bytes());
            out[at + 8..at + 16].copy_from_slice(&entry.revision.to_le_bytes());
            out[at + 16..at + 48].copy_from_slice(&entry.digest);
        }
        let crc = CRC.checksum(&out[HEADER_BYTES..len]);
        out[12..16].copy_from_slice(&crc.to_le_bytes());
        self.dirty = false;
        Some(len)
    }

    /// Load a sidecar's records, replacing the memo's contents and leaving
    /// it clean. On any error the memo is left empty — and the caller
    /// rehashes on demand, as it would with no sidecar at all.
    pub fn load(&mut self, src: &[u8]) -> Result<(), SidecarError> {
        self.entries.clear();
        self.dirty = false;
        let header = src.get(..HEADER_BYTES).ok_or(SidecarError::Shape)?;
        if &header[..8] != MAGIC {
            return Err(SidecarError::Shape);
        }
        let count = u32::from_le_bytes(header[8..12].try_into().expect("four bytes")) as usize;
        let crc = u32::from_le_bytes(header[12..16].try_into().expect("four bytes"));
        if count > self.capacity {
            return Err(SidecarError::Shape);
        }
        let records = src
            .get(HEADER_BYTES..HEADER_BYTES + count * RECORD_BYTES)
            .ok_or(SidecarError::Shape)?;
        if CRC.checksum(records) != crc {
            return Err(SidecarError::Crc);
        }
        self.entries
            .try_reserve(count)
            .map_err(|_| SidecarError::Shape)?;
        for record in records.chunks_exact(RECORD_BYTES) {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&record[16..48]);
            self.entries.push(Entry {
                object: u64::from_le_bytes(record[..8].try_into().expect("eight bytes")),
                revision: u64::from_le_bytes(record[8..16].try_into().expect("eight bytes")),
                digest,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> StableObjectId {
        StableObjectId::new(n).unwrap()
    }

    #[test]
    fn a_digest_answers_only_its_own_revision() {
        let mut memo = DigestMemo::new(4);
        assert!(!memo.dirty());
        memo.put(id(7), 3, [1; 32]);
        assert!(memo.dirty());
        assert_eq!(memo.get(id(7), 3), Some([1; 32]));
        assert_eq!(memo.get(id(7), 4), None, "a later revision is never answered");
        memo.put(id(7), 4, [2; 32]);
        assert_eq!(memo.get(id(7), 3), None, "one entry per object");
        assert_eq!(memo.get(id(7), 4), Some([2; 32]));
        assert_eq!(memo.len(), 1);
    }

    #[test]
    fn a_full_memo_forgets_rather_than_answers_wrongly() {
        let mut memo = DigestMemo::new(2);
        memo.put(id(1), 1, [1; 32]);
        memo.put(id(2), 1, [2; 32]);
        memo.put(id(3), 1, [3; 32]);
        assert_eq!(memo.len(), 2);
        assert_eq!(memo.get(id(3), 1), None);
    }

    #[test]
    fn prune_keeps_only_live_objects_at_their_remembered_revision() {
        let mut memo = DigestMemo::new(4);
        memo.put(id(1), 1, [1; 32]);
        memo.put(id(2), 5, [2; 32]);
        memo.put(id(3), 9, [3; 32]);
        let mut encoded = [0u8; 256];
        memo.encode(&mut encoded).unwrap();
        assert!(!memo.dirty());
        memo.prune(|object| match object.get() {
            1 => Some(1),
            2 => Some(6),
            _ => None,
        });
        assert!(memo.dirty());
        assert_eq!(memo.len(), 1);
        assert_eq!(memo.get(id(1), 1), Some([1; 32]));
    }

    #[test]
    fn a_sidecar_round_trips_and_a_damaged_one_loads_nothing() {
        let mut memo = DigestMemo::new(8);
        memo.put(id(9), 2, [9; 32]);
        memo.put(id(4), 7, [4; 32]);
        let mut bytes = [0u8; 512];
        let len = memo.encode(&mut bytes).unwrap();
        assert_eq!(len, HEADER_BYTES + 2 * RECORD_BYTES);

        let mut back = DigestMemo::new(8);
        back.load(&bytes[..len]).unwrap();
        assert_eq!(back.get(id(9), 2), Some([9; 32]));
        assert_eq!(back.get(id(4), 7), Some([4; 32]));
        assert!(!back.dirty());

        let mut flipped = bytes;
        flipped[HEADER_BYTES + 20] ^= 1;
        assert_eq!(back.load(&flipped[..len]), Err(SidecarError::Crc));
        assert!(back.is_empty(), "a damaged sidecar leaves nothing to trust");
        assert_eq!(back.load(&[0u8; 32]), Err(SidecarError::Shape));
        let mut small = DigestMemo::new(1);
        assert_eq!(small.load(&bytes[..len]), Err(SidecarError::Shape), "past capacity");
    }
}
