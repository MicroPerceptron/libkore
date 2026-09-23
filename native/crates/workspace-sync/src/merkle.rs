//! Fixed-granule Merkle proofs for lazy host-backed file ranges.
//!
//! The manifest commits one root per file. A cache miss therefore needs only
//! one 4 KiB range plus at most eight sibling hashes for the frozen 1 MiB file
//! ceiling. Host source identities and paths never participate in the proof.

use heapless::Vec;

use crate::bounds::{MAX_FILE_BYTES, MAX_RANGE_PROOF_DEPTH, RANGE_BYTES};

const LEAF_DOMAIN: &[u8] = b"kore.workspace.range.leaf.v1\0";
const EMPTY_DOMAIN: &[u8] = b"kore.workspace.range.empty.v1\0";
const NODE_DOMAIN: &[u8] = b"kore.workspace.range.node.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RangeProof {
    pub file_len: u32,
    pub leaf_index: u16,
    pub leaf_count: u16,
    siblings: Vec<[u8; 32], MAX_RANGE_PROOF_DEPTH>,
}

impl RangeProof {
    #[must_use]
    pub fn siblings(&self) -> &[[u8; 32]] {
        self.siblings.as_slice()
    }

    pub fn from_parts(
        file_len: u32,
        leaf_index: u16,
        leaf_count: u16,
        siblings: &[[u8; 32]],
    ) -> Option<Self> {
        let expected = proof_depth(usize::from(leaf_count))?;
        if siblings.len() != expected || usize::from(leaf_index) >= usize::from(leaf_count) {
            return None;
        }
        Some(Self {
            file_len,
            leaf_index,
            leaf_count,
            siblings: Vec::from_slice(siblings).ok()?,
        })
    }
}

#[must_use]
pub fn range_root(content: &[u8]) -> Option<[u8; 32]> {
    if content.len() as u64 > MAX_FILE_BYTES {
        return None;
    }
    let count = leaf_count(content.len());
    let width = count.next_power_of_two();
    Some(subtree_hash(content, 0, width, count))
}

#[must_use]
pub fn build_range_proof(content: &[u8], leaf_index: u16) -> Option<RangeProof> {
    if content.len() as u64 > MAX_FILE_BYTES {
        return None;
    }
    let count = leaf_count(content.len());
    let index = usize::from(leaf_index);
    if index >= count {
        return None;
    }
    let width = count.next_power_of_two();
    let mut siblings = Vec::new();
    let mut node_start = index;
    let mut node_width = 1usize;
    while node_width < width {
        let sibling_start = if (node_start / node_width) & 1 == 0 {
            node_start.checked_add(node_width)?
        } else {
            node_start.checked_sub(node_width)?
        };
        siblings
            .push(subtree_hash(content, sibling_start, node_width, count))
            .ok()?;
        node_start &= !(node_width * 2 - 1);
        node_width = node_width.checked_mul(2)?;
    }
    Some(RangeProof {
        file_len: content.len() as u32,
        leaf_index,
        leaf_count: count as u16,
        siblings,
    })
}

#[must_use]
pub fn verify_range(root: [u8; 32], proof: &RangeProof, bytes: &[u8]) -> bool {
    if u64::from(proof.file_len) > MAX_FILE_BYTES {
        return false;
    }
    let count = leaf_count(proof.file_len as usize);
    let index = usize::from(proof.leaf_index);
    if count != usize::from(proof.leaf_count)
        || index >= count
        || proof.siblings.len() != proof_depth(count).unwrap_or(usize::MAX)
        || bytes.len() != range_len(proof.file_len as usize, index)
    {
        return false;
    }
    let mut hash = leaf_hash(index, bytes);
    let mut node = index;
    for (level, sibling) in proof.siblings.iter().enumerate() {
        hash = if node & 1 == 0 {
            node_hash(level, node >> 1, hash, *sibling)
        } else {
            node_hash(level, node >> 1, *sibling, hash)
        };
        node >>= 1;
    }
    hash == root
}

fn leaf_count(len: usize) -> usize {
    len.div_ceil(RANGE_BYTES).max(1)
}

fn range_len(file_len: usize, index: usize) -> usize {
    let start = index.saturating_mul(RANGE_BYTES);
    file_len.saturating_sub(start).min(RANGE_BYTES)
}

fn proof_depth(count: usize) -> Option<usize> {
    if count == 0 || count > (1usize << MAX_RANGE_PROOF_DEPTH) {
        return None;
    }
    Some(count.next_power_of_two().trailing_zeros() as usize)
}

fn leaf_hash(index: usize, bytes: &[u8]) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(LEAF_DOMAIN);
    hash.update(&(index as u16).to_le_bytes());
    hash.update(&(bytes.len() as u16).to_le_bytes());
    hash.update(bytes);
    *hash.finalize().as_bytes()
}

fn empty_hash(index: usize) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(EMPTY_DOMAIN);
    hash.update(&(index as u16).to_le_bytes());
    *hash.finalize().as_bytes()
}

fn node_hash(level: usize, index: usize, left: [u8; 32], right: [u8; 32]) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(NODE_DOMAIN);
    hash.update(&(level as u16).to_le_bytes());
    hash.update(&(index as u16).to_le_bytes());
    hash.update(&left);
    hash.update(&right);
    *hash.finalize().as_bytes()
}

fn subtree_hash(content: &[u8], start: usize, width: usize, count: usize) -> [u8; 32] {
    if width == 1 {
        if start >= count {
            return empty_hash(start);
        }
        let offset = start * RANGE_BYTES;
        let end = offset.saturating_add(RANGE_BYTES).min(content.len());
        return leaf_hash(start, &content[offset..end]);
    }
    let half = width / 2;
    let level = width.trailing_zeros() as usize - 1;
    node_hash(
        level,
        start / width,
        subtree_hash(content, start, half, count),
        subtree_hash(content, start + half, half, count),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_range_verifies_and_any_mutation_fails() {
        let bytes = (0..(RANGE_BYTES * 3 + 17))
            .map(|index| index as u8)
            .collect::<std::vec::Vec<_>>();
        let root = range_root(&bytes).unwrap();
        for index in 0..4u16 {
            let proof = build_range_proof(&bytes, index).unwrap();
            let start = usize::from(index) * RANGE_BYTES;
            let end = (start + RANGE_BYTES).min(bytes.len());
            let range = &bytes[start..end];
            assert!(verify_range(root, &proof, range));
            let mut corrupt = range.to_vec();
            if corrupt.is_empty() {
                corrupt.push(1);
            } else {
                corrupt[0] ^= 1;
            }
            assert!(!verify_range(root, &proof, &corrupt));
        }
    }

    #[test]
    fn empty_file_has_one_zero_length_provable_leaf() {
        let root = range_root(&[]).unwrap();
        let proof = build_range_proof(&[], 0).unwrap();
        assert!(verify_range(root, &proof, &[]));
        assert!(build_range_proof(&[], 1).is_none());
    }

    #[test]
    fn oversize_and_wrong_geometry_fail_closed() {
        let oversized = std::vec![0; MAX_FILE_BYTES as usize + 1];
        assert!(range_root(&oversized).is_none());
        let bytes = std::vec![1; RANGE_BYTES + 1];
        let root = range_root(&bytes).unwrap();
        let proof = build_range_proof(&bytes, 0).unwrap();
        assert!(!verify_range(root, &proof, &bytes[..RANGE_BYTES - 1]));
    }
}
