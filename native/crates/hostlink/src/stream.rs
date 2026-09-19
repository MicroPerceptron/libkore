//! Partial-read-safe stream framing.

use zerocopy::FromBytes;

use crate::{
    BULK_FRAME_MAX, CONTROL_FRAME_MAX, ENVELOPE_LEN, EnvelopeV1, HOSTLINK_MAGIC, SCHEMA_MAJOR_V1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    BadMagic,
    UnsupportedMajor,
    Reserved,
    InvalidLength,
    OutputTooSmall,
    AlreadyComplete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeProgress {
    pub consumed: usize,
    pub complete: bool,
}

/// Reassembles one frame from an arbitrary stream segmentation.
///
/// Payload storage belongs to the caller.  The same slice must be supplied
/// until `complete` is returned; after processing the frame, call [`reset`].
pub struct FramedDecoder<const MAX_FRAME: usize> {
    header: [u8; ENVELOPE_LEN],
    header_len: usize,
    envelope: Option<EnvelopeV1>,
    payload_len: usize,
}

pub type FrameDecoder = FramedDecoder<CONTROL_FRAME_MAX>;
pub type ControlFrameDecoder = FramedDecoder<CONTROL_FRAME_MAX>;
pub type BulkFrameDecoder = FramedDecoder<BULK_FRAME_MAX>;

impl<const MAX_FRAME: usize> Default for FramedDecoder<MAX_FRAME> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_FRAME: usize> FramedDecoder<MAX_FRAME> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            header: [0; ENVELOPE_LEN],
            header_len: 0,
            envelope: None,
            payload_len: 0,
        }
    }

    #[must_use]
    pub const fn envelope(&self) -> Option<&EnvelopeV1> {
        self.envelope.as_ref()
    }

    #[must_use]
    pub fn payload<'a>(&self, storage: &'a [u8]) -> Option<&'a [u8]> {
        self.envelope?;
        storage.get(..self.payload_len)
    }

    pub fn reset(&mut self) {
        self.header.fill(0);
        self.header_len = 0;
        self.envelope = None;
        self.payload_len = 0;
    }

    pub fn push(
        &mut self,
        input: &[u8],
        payload_storage: &mut [u8],
    ) -> Result<DecodeProgress, DecodeError> {
        if let Some(envelope) = self.envelope {
            let expected = envelope.payload_len.get() as usize;
            if self.payload_len == expected {
                return Err(DecodeError::AlreadyComplete);
            }
        }

        let mut consumed = 0;
        if self.header_len < ENVELOPE_LEN {
            let take = (ENVELOPE_LEN - self.header_len).min(input.len());
            self.header[self.header_len..self.header_len + take].copy_from_slice(&input[..take]);
            self.header_len += take;
            consumed += take;
            if self.header_len != ENVELOPE_LEN {
                return Ok(DecodeProgress {
                    consumed,
                    complete: false,
                });
            }

            let envelope = match EnvelopeV1::read_from_bytes(&self.header) {
                Ok(envelope) => envelope,
                Err(_) => {
                    self.reset();
                    return Err(DecodeError::InvalidLength);
                }
            };
            if envelope.magic.get() != HOSTLINK_MAGIC {
                self.reset();
                return Err(DecodeError::BadMagic);
            }
            if envelope.schema_major.get() != SCHEMA_MAJOR_V1 {
                self.reset();
                return Err(DecodeError::UnsupportedMajor);
            }
            if envelope.flags.get() != 0 || envelope.reserved.get() != 0 {
                self.reset();
                return Err(DecodeError::Reserved);
            }
            let Some(frame_len) = envelope.frame_len() else {
                self.reset();
                return Err(DecodeError::InvalidLength);
            };
            if frame_len > MAX_FRAME {
                self.reset();
                return Err(DecodeError::InvalidLength);
            }
            if payload_storage.len() < envelope.payload_len.get() as usize {
                self.reset();
                return Err(DecodeError::OutputTooSmall);
            }
            self.envelope = Some(envelope);
        }

        let expected = self
            .envelope
            .as_ref()
            .expect("header completion installs envelope")
            .payload_len
            .get() as usize;
        let take = (expected - self.payload_len).min(input.len() - consumed);
        payload_storage[self.payload_len..self.payload_len + take]
            .copy_from_slice(&input[consumed..consumed + take]);
        self.payload_len += take;
        consumed += take;

        Ok(DecodeProgress {
            consumed,
            complete: self.payload_len == expected,
        })
    }
}
