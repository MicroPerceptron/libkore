//! Native grant correlations. IDs are not bearer authority; DagFS checks the
//! caller's held view and the edge generation on every subsequent operation.
use crate::GrantId;

pub const RECEIPT_BYTES: usize = 25;
const PREFIX: &[u8; 8] = b"KGRANT1 ";
const HEX: &[u8; 16] = b"0123456789abcdef";

pub fn encode_id(grant: GrantId) -> [u8; 16] {
    let bits = grant.to_bits();
    core::array::from_fn(|index| HEX[((bits >> (4 * (15 - index))) & 15) as usize])
}

pub fn decode_id(bytes: &[u8]) -> Option<GrantId> {
    if bytes.len() != 16 {
        return None;
    }
    let mut bits = 0u64;
    for byte in bytes {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => return None,
        };
        bits = (bits << 4) | u64::from(digit);
    }
    Some(GrantId::from_bits(bits))
}

pub fn encode(grant: GrantId) -> [u8; RECEIPT_BYTES] {
    let mut bytes = [0; RECEIPT_BYTES];
    bytes[..8].copy_from_slice(PREFIX);
    bytes[8..24].copy_from_slice(&encode_id(grant));
    bytes[24] = b'\n';
    bytes
}

pub fn decode(bytes: &[u8]) -> Option<GrantId> {
    if bytes.len() != RECEIPT_BYTES || &bytes[..8] != PREFIX || bytes[24] != b'\n' {
        return None;
    }
    decode_id(&bytes[8..24])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grant_receipt_preserves_every_id_bit_and_requires_exact_framing() {
        for bits in [0, 1, 0x123456789abcdef0, u64::MAX] {
            let grant = GrantId::from_bits(bits);
            let bytes = encode(grant);
            assert_eq!(decode(&bytes), Some(grant));
            assert_eq!(decode_id(&encode_id(grant)), Some(grant));
            for len in 0..RECEIPT_BYTES {
                assert!(decode(&bytes[..len]).is_none());
            }
            let mut extra = [0; RECEIPT_BYTES + 1];
            extra[..RECEIPT_BYTES].copy_from_slice(&bytes);
            assert!(decode(&extra).is_none());
        }
        for bad in [
            b"KGRANT2 123456789abcdef0\n",
            b"KGRANT1 123456789abcdeF0\n",
            b"KGRANT1 123456789abcdef0 ",
        ] {
            assert!(decode(bad).is_none());
        }
        assert!(decode_id(b"123456789abcdef").is_none());
        assert!(decode_id(b"123456789abcdef00").is_none());
    }
}
