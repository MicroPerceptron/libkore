#[inline]
pub(crate) fn read_u16_le(data: &[u8], off: usize) -> Option<u16> {
    let bytes = data.get(off..off.saturating_add(2))?;
    let arr = <[u8; 2]>::try_from(bytes).ok()?;
    Some(u16::from_le_bytes(arr))
}

#[inline]
pub(crate) fn read_u32_le(data: &[u8], off: usize) -> Option<u32> {
    let bytes = data.get(off..off.saturating_add(4))?;
    let arr = <[u8; 4]>::try_from(bytes).ok()?;
    Some(u32::from_le_bytes(arr))
}
