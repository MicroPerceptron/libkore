use crate::error::Error;
use crate::model::Header;
use crate::parse::expected_sparse_payload_len;

pub fn decompress_sparse_lz4_payload(
    header: Header,
    dirty_words_le: &[u8],
    compressed_payload: &[u8],
    raw_payload_len: usize,
    scratch: &mut [u8],
) -> Result<usize, Error> {
    let expected = expected_sparse_payload_len(header, dirty_words_le)?;
    if raw_payload_len != expected {
        return Err(Error::InvalidGeometry);
    }
    if scratch.len() < raw_payload_len {
        return Err(Error::ScratchTooSmall);
    }
    let dst = &mut scratch[..raw_payload_len];
    let written = lz4_flex::block::decompress_into(compressed_payload, dst)
        .map_err(|_| Error::DecompressFailed)?;
    if written != raw_payload_len {
        return Err(Error::InvalidGeometry);
    }
    Ok(written)
}
