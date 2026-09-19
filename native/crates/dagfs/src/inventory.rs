//! Bounded discovery candidates emitted by a native filesystem command.
//! This record conveys observations only. A parent must successfully bind each
//! selected object through its held source before advertising it as authority.
use crate::{Name, ObjectKind, Rights};

pub const MAX_CANDIDATES: usize = 16;
pub const MAX_ROW_BYTES: usize = 5 + 2 * crate::bounds::MAX_NAME_LEN;
pub const REPLY_OVERHEAD: usize = 15;
pub const MAX_REPLY_BYTES: usize = REPLY_OVERHEAD + MAX_CANDIDATES * MAX_ROW_BYTES;
pub const COMMAND: &[u8] = b"--resource-inventory";
const HEX: &[u8; 16] = b"0123456789abcdef";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub name: Name,
    pub kind: ObjectKind,
    pub rights: Rights,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Candidates {
    /// A capacity observation, never a reservation or allocation witness.
    pub available_views: u8,
    pub more: bool,
    pub entries: heapless::Vec<Candidate, MAX_CANDIDATES>,
}

fn nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}
fn pair(bytes: &[u8]) -> Option<u8> {
    Some((nibble(*bytes.first()?)? << 4) | nibble(*bytes.get(1)?)?)
}

impl Candidates {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if usize::from(self.available_views) > crate::bounds::MAX_INVOCATION_VIEWS {
            return None;
        }
        let mut wire = heapless::Vec::<u8, MAX_REPLY_BYTES>::new();
        wire.extend_from_slice(b"KRES1 ").ok()?;
        wire.extend_from_slice(&[
            HEX[usize::from(self.available_views >> 4)],
            HEX[usize::from(self.available_views & 15)],
            b' ',
            if self.more { b'1' } else { b'0' },
            b'\n',
        ])
        .ok()?;
        for (index, entry) in self.entries.iter().enumerate() {
            if !entry.rights.contains(Rights::READ)
                || !entry.rights.is_subset_of(Rights::HOME)
                || self.entries[..index]
                    .iter()
                    .any(|prior| prior.name == entry.name)
            {
                return None;
            }
            wire.extend_from_slice(&[
                if entry.kind == ObjectKind::Directory {
                    b'd'
                } else {
                    b'f'
                },
                HEX[usize::from(entry.rights.bits() >> 4)],
                HEX[usize::from(entry.rights.bits() & 15)],
                b':',
            ])
            .ok()?;
            for byte in entry.name.as_bytes() {
                wire.extend_from_slice(&[HEX[usize::from(byte >> 4)], HEX[usize::from(byte & 15)]])
                    .ok()?;
            }
            wire.push(b'\n').ok()?;
        }
        wire.extend_from_slice(b"END\n").ok()?;
        output.get_mut(..wire.len())?.copy_from_slice(&wire);
        Some(wire.len())
    }

    pub fn decode(input: &[u8]) -> Option<Self> {
        let body = input.strip_prefix(b"KRES1 ")?;
        let available_views = pair(body)?;
        if usize::from(available_views) > crate::bounds::MAX_INVOCATION_VIEWS
            || body.get(2) != Some(&b' ')
            || body.get(4) != Some(&b'\n')
        {
            return None;
        }
        let more = match body.get(3)? {
            b'0' => false,
            b'1' => true,
            _ => return None,
        };
        let mut remaining = body.get(5..)?;
        let mut entries = heapless::Vec::new();
        while remaining != b"END\n" {
            let end = remaining.iter().position(|byte| *byte == b'\n')?;
            let row = &remaining[..end];
            let kind = match row.first()? {
                b'd' => ObjectKind::Directory,
                b'f' => ObjectKind::File,
                _ => return None,
            };
            let bits = pair(row.get(1..3)?)?;
            if bits & !Rights::HOME.bits() != 0
                || bits & Rights::READ.bits() == 0
                || row.get(3) != Some(&b':')
            {
                return None;
            }
            let encoded = row.get(4..)?;
            if encoded.len() % 2 != 0 {
                return None;
            }
            let mut name = heapless::Vec::<u8, { crate::bounds::MAX_NAME_LEN }>::new();
            for pair_bytes in encoded.chunks_exact(2) {
                name.push(pair(pair_bytes)?).ok()?;
            }
            let name = Name::from_bytes(&name).ok()?;
            if entries.iter().any(|entry: &Candidate| entry.name == name) {
                return None;
            }
            entries
                .push(Candidate {
                    name,
                    kind,
                    rights: Rights::from_bits_truncating(bits),
                })
                .ok()?;
            remaining = &remaining[end + 1..];
        }
        Some(Self {
            available_views,
            more,
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Candidates {
        let mut entries = heapless::Vec::new();
        entries
            .push(Candidate {
                name: Name::new("notes α").unwrap(),
                kind: ObjectKind::File,
                rights: Rights::READ.union(Rights::SHARE),
            })
            .unwrap();
        entries
            .push(Candidate {
                name: Name::new("output").unwrap(),
                kind: ObjectKind::Directory,
                rights: Rights::HOME,
            })
            .unwrap();
        Candidates {
            available_views: 7,
            more: true,
            entries,
        }
    }

    #[test]
    fn discovery_codec_roundtrips_names_kinds_rights_and_capacity_observations() {
        let candidates = sample();
        let mut wire = [0; MAX_REPLY_BYTES];
        let len = candidates.encode(&mut wire).unwrap();
        assert!(wire[..len].is_ascii());
        assert_eq!(Candidates::decode(&wire[..len]), Some(candidates));
        assert!(sample().encode(&mut wire[..len - 1]).is_none());
    }

    #[test]
    fn discovery_codec_requires_a_complete_unambiguous_versioned_record() {
        let mut wire = [0; MAX_REPLY_BYTES];
        let len = sample().encode(&mut wire).unwrap();
        for prefix in 0..len {
            assert!(Candidates::decode(&wire[..prefix]).is_none());
        }
        for invalid in [
            b"KRES2 07 0\nEND\n".as_slice(),
            b"KRES1 ff 0\nEND\n",
            b"KRES1 07 2\nEND\n",
            b"KRES1 07 0\nx01:61\nEND\n",
            b"KRES1 07 0\nf00:61\nEND\n",
            b"KRES1 07 0\nfff:61\nEND\n",
            b"KRES1 07 0\nf01:6\nEND\n",
            b"KRES1 07 0\nf01:\nEND\n",
            b"KRES1 07 0\nf01:61\nd0f:61\nEND\n",
            b"KRES1 07 0\nEND\nextra\n",
            b"fs: KRES1 07 0\nEND\n",
            b"KRES1 07 0\nf01:2e2e\nEND\n",
        ] {
            assert!(Candidates::decode(invalid).is_none(), "{invalid:?}");
        }
    }

    #[test]
    fn discovery_codec_bound_covers_maximum_width_candidates() {
        let mut candidates = Candidates {
            available_views: 0,
            more: false,
            entries: heapless::Vec::new(),
        };
        for index in 0..MAX_CANDIDATES {
            let mut name = [b'a'; crate::bounds::MAX_NAME_LEN];
            name[0] += index as u8;
            candidates
                .entries
                .push(Candidate {
                    name: Name::from_bytes(&name).unwrap(),
                    kind: ObjectKind::File,
                    rights: Rights::READ,
                })
                .unwrap();
        }
        let mut wire = [0; MAX_REPLY_BYTES];
        assert_eq!(candidates.encode(&mut wire), Some(MAX_REPLY_BYTES));
        assert_eq!(Candidates::decode(&wire), Some(candidates));
    }
}

#[cfg(feature = "runtime")]
impl crate::Fs<'_, '_> {
    /// Inspect a bounded prefix of the held root. Copy the directory entries
    /// before rights probes reuse the shared window. Later binders revalidate
    /// both kind and rights; these candidate names never select authority alone.
    pub async fn resource_candidates(
        &mut self,
        limit: usize,
    ) -> Result<Candidates, crate::Refusal> {
        use crate::{FsError, FsRequest, FsResponse, Refusal};
        if limit == 0 || limit > MAX_CANDIDATES {
            return Err(Refusal::Fs(FsError::Range));
        }
        let FsResponse::SystemStatusReported { len } = self
            .checked(&FsRequest::SystemStatus { window_offset: 0 })
            .await?
        else {
            return Err(Refusal::Pair);
        };
        if len as usize != kore_abi::svc::FsSystemStatusWire::BYTES {
            return Err(Refusal::Pair);
        }
        let status = crate::read_system_status(self.window(), 0).map_err(Refusal::Fs)?;
        let used = usize::from(status.views_private.get())
            .checked_add(usize::from(status.views_pinned.get()))
            .ok_or(Refusal::Pair)?;
        let available = crate::bounds::MAX_INVOCATION_VIEWS
            .checked_sub(used)
            .ok_or(Refusal::Pair)?;
        let root = self.open(b"/", Rights::READ).await?;
        let mut selected = heapless::Vec::<crate::DirEntry, MAX_CANDIDATES>::new();
        let mut cursor = 0;
        let mut more = false;
        let listing = loop {
            let (count, next) = match self.list_page(&root, cursor).await {
                Ok(page) => page,
                Err(error) => break Err(error),
            };
            let mut failure = None;
            for index in 0..count as usize {
                if selected.len() == limit {
                    more = true;
                    break;
                }
                match self.dir_entry(index) {
                    Ok(entry) => {
                        if selected.push(entry).is_err() {
                            failure = Some(Refusal::Fs(FsError::Range));
                            break;
                        }
                    }
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            if let Some(error) = failure {
                break Err(error);
            }
            if selected.len() == limit {
                more |= next.is_some();
                break Ok(());
            }
            match next {
                Some(next) => cursor = next,
                None => break Ok(()),
            }
        };
        self.close(root).await;
        listing?;
        let mut entries = heapless::Vec::new();
        for entry in selected {
            let mut rights = self.effective_rights(entry.name.as_bytes()).await?;
            if entry.kind == ObjectKind::File {
                rights =
                    rights.intersection(Rights::READ.union(Rights::WRITE).union(Rights::SHARE));
            }
            entries
                .push(Candidate {
                    name: entry.name,
                    kind: entry.kind,
                    rights,
                })
                .map_err(|_| Refusal::Fs(FsError::Range))?;
        }
        Ok(Candidates {
            available_views: available as u8,
            more,
            entries,
        })
    }
}
