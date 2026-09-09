//! Authority-free window-controller command records.
//!
//! The shell remains the affine owner of every window session and
//! of its compositor controller pair.  A foreground `window` executable sees
//! only one bounded snapshot through `ProcessIo` and returns one inert action
//! description after exit.  Copying either record grants no authority: the
//! owner must authenticate the executable profile and revalidate the exact
//! `(slot, generation, peer)` handle before it may act.

use kore_abi::image::{ImageName, NAME_BYTES};

/// Version of both controller snapshot and command-output encodings.
pub const WINDOW_COMMAND_ABI: u8 = 2;
/// Nonzero slot identities representable by the controller command protocol.
pub const MAX_CONTROLLER_WINDOWS: usize = u8::MAX as usize;
/// Maximum human-readable bytes carried beside one requested action.
pub const MAX_CONTROLLER_COMMAND_TEXT: usize = 2048;

const SNAPSHOT_MAGIC: [u8; 4] = *b"KWNS";
const OUTPUT_MAGIC: [u8; 4] = *b"KWCO";
const SNAPSHOT_HEADER_BYTES: usize = 8;
const SNAPSHOT_ROW_BYTES: usize = 48 + NAME_BYTES;
/// Bounded messages page through the full handle space. Reserve enough output
/// for the worst row (name, four u64 values, exit, labels) and continuation.
pub const SNAPSHOT_PAGE_WINDOWS: usize = {
    let context_rows =
        (kore_abi::process::PROCESS_CONTEXT_BYTES - SNAPSHOT_HEADER_BYTES) / SNAPSHOT_ROW_BYTES;
    let text_rows = (MAX_CONTROLLER_COMMAND_TEXT - 80) / (NAME_BYTES + 180);
    if context_rows < text_rows {
        context_rows
    } else {
        text_rows
    }
};
const OUTPUT_HEADER_BYTES: usize = 24;

const ROW_HANDLE: u8 = 1 << 0;
const ROW_NAME: u8 = 1 << 1;
const ROW_EXIT: u8 = 1 << 2;
const ROW_ERROR: u8 = 1 << 3;
const ROW_KNOWN_FLAGS: u8 = ROW_HANDLE | ROW_NAME | ROW_EXIT | ROW_ERROR;

/// Why a controller snapshot or command result refused to encode/decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowCommandError {
    BufferTooSmall,
    TooManyWindows,
    DuplicateOrUnorderedSlot,
    InvalidField,
    InvalidText,
    InvalidEncoding,
}

/// Lifecycle posture exposed by the controller owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum WindowState {
    #[default]
    Empty = 0,
    Reserved = 1,
    Cached = 2,
    Running = 3,
    Closing = 4,
}

/// Compositor visibility/layout state copied into an authority-free snapshot.
///
/// The runtime controller maps this value to its protocol mode at the broker
/// boundary; keeping the command vocabulary independent avoids importing the
/// compositor transport into a heapless executable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ControllerMode {
    #[default]
    Normal = 0,
    Minimized = 1,
    Maximized = 2,
}

impl ControllerMode {
    fn from_wire(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Normal),
            1 => Some(Self::Minimized),
            2 => Some(Self::Maximized),
            _ => None,
        }
    }
}

#[cfg(feature = "runtime")]
impl From<crate::control::WindowMode> for ControllerMode {
    fn from(mode: crate::control::WindowMode) -> Self {
        match mode {
            crate::control::WindowMode::Normal => Self::Normal,
            crate::control::WindowMode::Minimized => Self::Minimized,
            crate::control::WindowMode::Maximized => Self::Maximized,
        }
    }
}

impl WindowState {
    fn from_wire(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Empty),
            1 => Some(Self::Reserved),
            2 => Some(Self::Cached),
            3 => Some(Self::Running),
            4 => Some(Self::Closing),
            _ => None,
        }
    }

    /// Stable display spelling used by the allocation-free system app.
    pub const fn label(self) -> &'static [u8] {
        match self {
            Self::Empty => b"empty",
            Self::Reserved => b"reserved",
            Self::Cached => b"cached",
            Self::Running => b"running",
            Self::Closing => b"closing",
        }
    }
}

/// Exact controller-local identity for one live window generation.
///
/// This is correlation data, not authority.  A controller must compare all
/// three fields against the entry it still owns before interpreting an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowHandle {
    /// One-based UI slot.
    pub slot: u8,
    /// Monotonic shell-local generation; zero is never live.
    pub generation: u64,
    /// Compositor-authored recyclable peer arrival; zero is never live.
    pub peer_arrival: u16,
}

impl WindowHandle {
    pub const fn is_valid(self) -> bool {
        self.slot != 0
            && self.slot as usize <= MAX_CONTROLLER_WINDOWS
            && self.generation != 0
            && self.peer_arrival != 0
    }
}

/// One validated slot in a controller-owned launch snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerWindow {
    pub slot: u8,
    pub state: WindowState,
    pub mode: ControllerMode,
    pub handle: Option<WindowHandle>,
    pub name: Option<ImageName>,
    pub surface: u64,
    pub presented: u64,
    pub exit: Option<i32>,
    pub error: Option<u64>,
}

impl ControllerWindow {
    fn validate(&self) -> bool {
        if self.slot == 0 || usize::from(self.slot) > MAX_CONTROLLER_WINDOWS {
            return false;
        }
        if let Some(handle) = self.handle {
            if !handle.is_valid()
                || handle.slot != self.slot
                || !matches!(self.state, WindowState::Running | WindowState::Closing)
                || self.name.is_none()
            {
                return false;
            }
        } else if self.state == WindowState::Running
            || self.surface != 0
            || self.presented != 0
            || self.mode != ControllerMode::Normal
        {
            return false;
        }
        if self.state == WindowState::Empty
            && (self.name.is_some() || self.exit.is_some() || self.error.is_some())
        {
            return false;
        }
        if self.error == Some(0) {
            return false;
        }
        self.name.as_ref().is_none_or(valid_name)
    }
}

/// Incremental encoder over a caller-owned ProcessIo context buffer.
pub struct ControllerSnapshotBuilder<'a> {
    bytes: &'a mut [u8],
    count: u8,
    len: usize,
    last_slot: u8,
}

impl<'a> ControllerSnapshotBuilder<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Result<Self, WindowCommandError> {
        if bytes.len() < SNAPSHOT_HEADER_BYTES {
            return Err(WindowCommandError::BufferTooSmall);
        }
        bytes[..SNAPSHOT_HEADER_BYTES].fill(0);
        bytes[..4].copy_from_slice(&SNAPSHOT_MAGIC);
        bytes[4] = WINDOW_COMMAND_ABI;
        Ok(Self {
            bytes,
            count: 0,
            len: SNAPSHOT_HEADER_BYTES,
            last_slot: 0,
        })
    }

    /// Append one row. Slots must be unique and strictly increasing so one
    /// semantic snapshot has exactly one byte representation.
    pub fn push(&mut self, row: ControllerWindow) -> Result<(), WindowCommandError> {
        if usize::from(self.count) >= SNAPSHOT_PAGE_WINDOWS {
            return Err(WindowCommandError::TooManyWindows);
        }
        if row.slot <= self.last_slot {
            return Err(WindowCommandError::DuplicateOrUnorderedSlot);
        }
        if !row.validate() {
            return Err(WindowCommandError::InvalidField);
        }
        let end = self
            .len
            .checked_add(SNAPSHOT_ROW_BYTES)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(WindowCommandError::BufferTooSmall)?;
        encode_row(&mut self.bytes[self.len..end], &row);
        self.len = end;
        self.count += 1;
        self.last_slot = row.slot;
        Ok(())
    }

    /// Seal and return the exact live prefix length.
    pub fn finish(self) -> usize {
        self.finish_page(None)
    }

    pub fn finish_page(self, next_slot: Option<u8>) -> usize {
        self.bytes[5] = self.count;
        self.bytes[6] = next_slot.unwrap_or(0);
        self.len
    }
}

/// Validated view over one immutable controller snapshot.
pub struct ControllerSnapshot<'a> {
    bytes: &'a [u8],
    count: u8,
}

impl<'a> ControllerSnapshot<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, WindowCommandError> {
        if bytes.len() < SNAPSHOT_HEADER_BYTES
            || bytes[..4] != SNAPSHOT_MAGIC
            || bytes[4] != WINDOW_COMMAND_ABI
            || bytes[7] != 0
        {
            return Err(WindowCommandError::InvalidEncoding);
        }
        let count = bytes[5];
        if usize::from(count) > SNAPSHOT_PAGE_WINDOWS {
            return Err(WindowCommandError::TooManyWindows);
        }
        let expected = SNAPSHOT_HEADER_BYTES
            .checked_add(usize::from(count) * SNAPSHOT_ROW_BYTES)
            .ok_or(WindowCommandError::InvalidEncoding)?;
        if bytes.len() != expected {
            return Err(WindowCommandError::InvalidEncoding);
        }
        let mut last_slot = 0;
        for index in 0..usize::from(count) {
            let row = decode_row(row_bytes(bytes, index)?)?;
            if row.slot <= last_slot {
                return Err(WindowCommandError::DuplicateOrUnorderedSlot);
            }
            last_slot = row.slot;
        }
        if bytes[6] != 0 && bytes[6] <= last_slot {
            return Err(WindowCommandError::InvalidEncoding);
        }
        Ok(Self { bytes, count })
    }

    pub fn next_slot(&self) -> Option<u8> {
        (self.bytes[6] != 0).then_some(self.bytes[6])
    }

    pub const fn len(&self) -> usize {
        self.count as usize
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn entries(&self) -> ControllerWindows<'a> {
        ControllerWindows {
            bytes: self.bytes,
            index: 0,
            count: self.count,
        }
    }

    pub fn find(&self, slot: u8) -> Option<ControllerWindow> {
        self.entries().find(|row| row.slot == slot)
    }
}

pub struct ControllerWindows<'a> {
    bytes: &'a [u8],
    index: u8,
    count: u8,
}

impl Iterator for ControllerWindows<'_> {
    type Item = ControllerWindow;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            return None;
        }
        let row = row_bytes(self.bytes, usize::from(self.index))
            .ok()
            .and_then(|bytes| decode_row(bytes).ok());
        self.index += 1;
        row
    }
}

/// One controller operation requested by the `window` executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WindowOperation {
    Close = 1,
    Minimize = 2,
    Maximize = 3,
    Restore = 4,
}

impl WindowOperation {
    fn from_wire(raw: u8) -> Option<Self> {
        match raw {
            1 => Some(Self::Close),
            2 => Some(Self::Minimize),
            3 => Some(Self::Maximize),
            4 => Some(Self::Restore),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowAction {
    pub operation: WindowOperation,
    pub handle: WindowHandle,
}

/// Validated child result. Text is inert display output; `action` must still
/// be authorized and revalidated by the controller owner after successful
/// child exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerCommandOutput<'a> {
    pub action: Option<WindowAction>,
    pub text: &'a [u8],
}

impl<'a> ControllerCommandOutput<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, WindowCommandError> {
        if bytes.len() < OUTPUT_HEADER_BYTES
            || bytes[..4] != OUTPUT_MAGIC
            || bytes[4] != WINDOW_COMMAND_ABI
            || bytes[7] != 0
            || bytes[20..24] != [0, 0, 0, 0]
        {
            return Err(WindowCommandError::InvalidEncoding);
        }
        let text_len = usize::from(read_u16(bytes, 18)?);
        if text_len > MAX_CONTROLLER_COMMAND_TEXT || bytes.len() != OUTPUT_HEADER_BYTES + text_len {
            return Err(WindowCommandError::InvalidEncoding);
        }
        let text = &bytes[OUTPUT_HEADER_BYTES..];
        if !valid_text(text) {
            return Err(WindowCommandError::InvalidText);
        }
        let action = if bytes[5] == 0 {
            if bytes[6] != 0 || read_u64(bytes, 8)? != 0 || read_u16(bytes, 16)? != 0 {
                return Err(WindowCommandError::InvalidEncoding);
            }
            None
        } else {
            let operation =
                WindowOperation::from_wire(bytes[5]).ok_or(WindowCommandError::InvalidEncoding)?;
            let handle = WindowHandle {
                slot: bytes[6],
                generation: read_u64(bytes, 8)?,
                peer_arrival: read_u16(bytes, 16)?,
            };
            if !handle.is_valid() {
                return Err(WindowCommandError::InvalidField);
            }
            Some(WindowAction { operation, handle })
        };
        Ok(Self { action, text })
    }
}

/// Encode one canonical result into the child's ProcessIo output buffer.
pub fn encode_command_output(
    bytes: &mut [u8],
    action: Option<WindowAction>,
    text: &[u8],
) -> Result<usize, WindowCommandError> {
    if text.len() > MAX_CONTROLLER_COMMAND_TEXT || text.len() > u16::MAX as usize {
        return Err(WindowCommandError::InvalidText);
    }
    if !valid_text(text) {
        return Err(WindowCommandError::InvalidText);
    }
    let len = OUTPUT_HEADER_BYTES
        .checked_add(text.len())
        .filter(|len| *len <= bytes.len())
        .ok_or(WindowCommandError::BufferTooSmall)?;
    bytes[..len].fill(0);
    bytes[..4].copy_from_slice(&OUTPUT_MAGIC);
    bytes[4] = WINDOW_COMMAND_ABI;
    if let Some(action) = action {
        if !action.handle.is_valid() {
            return Err(WindowCommandError::InvalidField);
        }
        bytes[5] = action.operation as u8;
        bytes[6] = action.handle.slot;
        write_u64(bytes, 8, action.handle.generation);
        write_u16(bytes, 16, action.handle.peer_arrival);
    }
    write_u16(bytes, 18, text.len() as u16);
    bytes[OUTPUT_HEADER_BYTES..len].copy_from_slice(text);
    Ok(len)
}

fn encode_row(bytes: &mut [u8], row: &ControllerWindow) {
    bytes.fill(0);
    bytes[0] = row.slot;
    bytes[1] = row.state as u8;
    bytes[2] = row.mode as u8;
    if let Some(handle) = row.handle {
        bytes[3] |= ROW_HANDLE;
        write_u16(bytes, 4, handle.peer_arrival);
        write_u64(bytes, 8, handle.generation);
    }
    write_u64(bytes, 16, row.surface);
    write_u64(bytes, 24, row.presented);
    if let Some(exit) = row.exit {
        bytes[3] |= ROW_EXIT;
        write_i32(bytes, 32, exit);
    }
    if let Some(error) = row.error {
        bytes[3] |= ROW_ERROR;
        write_u64(bytes, 40, error);
    }
    if let Some(name) = row.name {
        bytes[3] |= ROW_NAME;
        bytes[48..48 + NAME_BYTES].copy_from_slice(&name.0);
    }
}

fn decode_row(bytes: &[u8]) -> Result<ControllerWindow, WindowCommandError> {
    if bytes.len() != SNAPSHOT_ROW_BYTES
        || bytes[3] & !ROW_KNOWN_FLAGS != 0
        || bytes[6..8] != [0, 0]
        || bytes[36..40] != [0, 0, 0, 0]
    {
        return Err(WindowCommandError::InvalidEncoding);
    }
    let slot = bytes[0];
    let state = WindowState::from_wire(bytes[1]).ok_or(WindowCommandError::InvalidEncoding)?;
    let mode = ControllerMode::from_wire(bytes[2]).ok_or(WindowCommandError::InvalidEncoding)?;
    let handle = if bytes[3] & ROW_HANDLE != 0 {
        Some(WindowHandle {
            slot,
            generation: read_u64(bytes, 8)?,
            peer_arrival: read_u16(bytes, 4)?,
        })
    } else {
        if read_u16(bytes, 4)? != 0 || read_u64(bytes, 8)? != 0 {
            return Err(WindowCommandError::InvalidEncoding);
        }
        None
    };
    let name = if bytes[3] & ROW_NAME != 0 {
        let mut raw = [0u8; NAME_BYTES];
        raw.copy_from_slice(&bytes[48..48 + NAME_BYTES]);
        Some(ImageName(raw))
    } else {
        if bytes[48..48 + NAME_BYTES].iter().any(|byte| *byte != 0) {
            return Err(WindowCommandError::InvalidEncoding);
        }
        None
    };
    let exit = if bytes[3] & ROW_EXIT != 0 {
        Some(read_i32(bytes, 32)?)
    } else {
        if read_i32(bytes, 32)? != 0 {
            return Err(WindowCommandError::InvalidEncoding);
        }
        None
    };
    let error = if bytes[3] & ROW_ERROR != 0 {
        Some(read_u64(bytes, 40)?)
    } else {
        if read_u64(bytes, 40)? != 0 {
            return Err(WindowCommandError::InvalidEncoding);
        }
        None
    };
    let row = ControllerWindow {
        slot,
        state,
        mode,
        handle,
        name,
        surface: read_u64(bytes, 16)?,
        presented: read_u64(bytes, 24)?,
        exit,
        error,
    };
    row.validate()
        .then_some(row)
        .ok_or(WindowCommandError::InvalidField)
}

fn row_bytes(bytes: &[u8], index: usize) -> Result<&[u8], WindowCommandError> {
    let start = SNAPSHOT_HEADER_BYTES
        .checked_add(
            index
                .checked_mul(SNAPSHOT_ROW_BYTES)
                .ok_or(WindowCommandError::InvalidEncoding)?,
        )
        .ok_or(WindowCommandError::InvalidEncoding)?;
    bytes
        .get(start..start + SNAPSHOT_ROW_BYTES)
        .ok_or(WindowCommandError::InvalidEncoding)
}

fn valid_name(name: &ImageName) -> bool {
    let len = name.len();
    len != 0
        && name.0[len..].iter().all(|byte| *byte == 0)
        && core::str::from_utf8(&name.0[..len]).is_ok_and(|text| {
            text.chars()
                .all(|character| !character.is_control() && !character.is_whitespace())
        })
}

fn valid_text(bytes: &[u8]) -> bool {
    core::str::from_utf8(bytes).is_ok_and(|text| {
        text.chars()
            .all(|character| matches!(character, '\n' | '\t') || !character.is_control())
    })
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, WindowCommandError> {
    let raw: [u8; 2] = bytes
        .get(at..at + 2)
        .ok_or(WindowCommandError::InvalidEncoding)?
        .try_into()
        .map_err(|_| WindowCommandError::InvalidEncoding)?;
    Ok(u16::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], at: usize) -> Result<u64, WindowCommandError> {
    let raw: [u8; 8] = bytes
        .get(at..at + 8)
        .ok_or(WindowCommandError::InvalidEncoding)?
        .try_into()
        .map_err(|_| WindowCommandError::InvalidEncoding)?;
    Ok(u64::from_le_bytes(raw))
}

fn read_i32(bytes: &[u8], at: usize) -> Result<i32, WindowCommandError> {
    let raw: [u8; 4] = bytes
        .get(at..at + 4)
        .ok_or(WindowCommandError::InvalidEncoding)?
        .try_into()
        .map_err(|_| WindowCommandError::InvalidEncoding)?;
    Ok(i32::from_le_bytes(raw))
}

fn write_u16(bytes: &mut [u8], at: usize, value: u16) {
    bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_i32(bytes: &mut [u8], at: usize, value: i32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

const _: () = {
    assert!(
        SNAPSHOT_HEADER_BYTES + SNAPSHOT_PAGE_WINDOWS * SNAPSHOT_ROW_BYTES
            <= kore_abi::process::PROCESS_CONTEXT_BYTES
    );
    assert!(
        OUTPUT_HEADER_BYTES + MAX_CONTROLLER_COMMAND_TEXT
            <= kore_abi::process::PROCESS_OUTPUT_BYTES
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    fn running(slot: u8, generation: u64, peer_arrival: u16) -> ControllerWindow {
        ControllerWindow {
            slot,
            state: WindowState::Running,
            mode: ControllerMode::Normal,
            handle: Some(WindowHandle {
                slot,
                generation,
                peer_arrival,
            }),
            name: ImageName::new("aftershock"),
            surface: 71,
            presented: 12,
            exit: None,
            error: None,
        }
    }

    #[test]
    fn bounded_pages_preserve_high_slot_handles_and_continuation() {
        let mut bytes = [0; kore_abi::process::PROCESS_CONTEXT_BYTES];
        let mut builder = ControllerSnapshotBuilder::new(&mut bytes).unwrap();
        builder.push(running(200, 4, 0x1234)).unwrap();
        let len = builder.finish_page(Some(201));
        let page = ControllerSnapshot::parse(&bytes[..len]).unwrap();
        assert_eq!(page.find(200).unwrap().handle.unwrap().slot, 200);
        assert_eq!(page.next_slot(), Some(201));
        bytes[6] = 200;
        assert!(ControllerSnapshot::parse(&bytes[..len]).is_err());
        let mut builder = ControllerSnapshotBuilder::new(&mut bytes).unwrap();
        builder.push(running(255, 5, 0x1235)).unwrap();
        let len = builder.finish();
        let page = ControllerSnapshot::parse(&bytes[..len]).unwrap();
        assert!(page.find(255).unwrap().handle.unwrap().is_valid());
        assert_eq!(page.next_slot(), None);
    }

    #[test]
    fn snapshot_roundtrips_exact_handles_and_inert_rows() {
        let mut bytes = [0u8; 1536];
        let mut builder = ControllerSnapshotBuilder::new(&mut bytes).unwrap();
        let first = running(1, 9, 4);
        let second = ControllerWindow {
            slot: 2,
            state: WindowState::Cached,
            mode: ControllerMode::Normal,
            handle: None,
            name: ImageName::new("skylilt"),
            surface: 0,
            presented: 0,
            exit: Some(0),
            error: Some(17),
        };
        builder.push(first).unwrap();
        builder.push(second).unwrap();
        let len = builder.finish();
        let snapshot = ControllerSnapshot::parse(&bytes[..len]).unwrap();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot.find(1), Some(first));
        assert_eq!(snapshot.find(2), Some(second));
        assert_eq!(snapshot.find(3), None);
        let mut entries = snapshot.entries();
        assert_eq!(entries.next(), Some(first));
        assert_eq!(entries.next(), Some(second));
        assert_eq!(entries.next(), None);
    }

    #[test]
    fn snapshots_reject_ambiguous_or_stale_shapes() {
        let mut bytes = [0u8; 1536];
        let mut builder = ControllerSnapshotBuilder::new(&mut bytes).unwrap();
        builder.push(running(1, 2, 4)).unwrap();
        assert_eq!(
            builder.push(running(1, 3, 8)),
            Err(WindowCommandError::DuplicateOrUnorderedSlot)
        );
        let mut unsafe_name = running(2, 3, 8);
        unsafe_name.name = ImageName::new("line\nbreak");
        assert_eq!(
            builder.push(unsafe_name),
            Err(WindowCommandError::InvalidField)
        );
        let len = builder.finish();
        assert_eq!(
            ControllerSnapshot::parse(&bytes[..len + 1]).map(|_| ()),
            Err(WindowCommandError::InvalidEncoding)
        );

        let mut corrupt = bytes[..len].to_vec();
        corrupt[SNAPSHOT_HEADER_BYTES + 6] = 1;
        assert_eq!(
            ControllerSnapshot::parse(&corrupt).map(|_| ()),
            Err(WindowCommandError::InvalidEncoding)
        );
        let mut corrupt = bytes[..len].to_vec();
        corrupt[SNAPSHOT_HEADER_BYTES + 8..SNAPSHOT_HEADER_BYTES + 16]
            .copy_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            ControllerSnapshot::parse(&corrupt).map(|_| ()),
            Err(WindowCommandError::InvalidField)
        );
        let mut corrupt = bytes[..len].to_vec();
        corrupt[SNAPSHOT_HEADER_BYTES + 48 + 5] = 0;
        corrupt[SNAPSHOT_HEADER_BYTES + 48 + 6] = b'x';
        assert_eq!(
            ControllerSnapshot::parse(&corrupt).map(|_| ()),
            Err(WindowCommandError::InvalidField)
        );
    }

    #[test]
    fn command_output_roundtrips_and_rejects_unused_bits() {
        let handle = WindowHandle {
            slot: 2,
            generation: 44,
            peer_arrival: 8,
        };
        for operation in [
            WindowOperation::Close,
            WindowOperation::Minimize,
            WindowOperation::Maximize,
            WindowOperation::Restore,
        ] {
            let action = Some(WindowAction { operation, handle });
            let mut bytes = [0u8; 256];
            let len = encode_command_output(&mut bytes, action, b"accepted\n").unwrap();
            assert_eq!(
                ControllerCommandOutput::parse(&bytes[..len]),
                Ok(ControllerCommandOutput {
                    action,
                    text: b"accepted\n",
                })
            );
            bytes[20] = 1;
            assert_eq!(
                ControllerCommandOutput::parse(&bytes[..len]),
                Err(WindowCommandError::InvalidEncoding)
            );
        }
        let mut bytes = [0u8; 256];
        let len = encode_command_output(
            &mut bytes,
            Some(WindowAction {
                operation: WindowOperation::Close,
                handle,
            }),
            b"accepted\n",
        )
        .unwrap();
        bytes[5] = u8::MAX;
        assert_eq!(
            ControllerCommandOutput::parse(&bytes[..len]),
            Err(WindowCommandError::InvalidEncoding)
        );
        assert_eq!(
            encode_command_output(&mut bytes, None, b"terminal \x1b escape"),
            Err(WindowCommandError::InvalidText)
        );
    }

    #[test]
    fn no_action_has_one_canonical_zero_handle() {
        let mut bytes = [0u8; 128];
        let len = encode_command_output(&mut bytes, None, b"no windows\n").unwrap();
        assert_eq!(
            ControllerCommandOutput::parse(&bytes[..len]),
            Ok(ControllerCommandOutput {
                action: None,
                text: b"no windows\n",
            })
        );
        bytes[6] = 1;
        assert_eq!(
            ControllerCommandOutput::parse(&bytes[..len]),
            Err(WindowCommandError::InvalidEncoding)
        );
    }
}
