//! Frozen one-page worker result framing.
//!
//! The worker writes one report and exits. The owning broker observes task
//! termination before reading it, so publication needs no shared cursor or
//! lock. The nested normalized payload remains independently versioned by
//! [`crate::wire`].

use crate::{NormalizedResult, decode_result, encode_result};

pub const WORKER_REPORT_BYTES: usize = 4096;
const MAGIC: u32 = u32::from_le_bytes(*b"KEXW");
const VERSION: u16 = 1;
const HEAD: usize = 32;
const INPUT_MAGIC: u32 = u32::from_le_bytes(*b"KEXI");
const INPUT_HEAD: usize = 64;
const BROKER_MAGIC: u32 = u32::from_le_bytes(*b"KEXB");
const BROKER_HEAD: usize = 72;
const JOB_MAGIC: u32 = u32::from_le_bytes(*b"KEXJ");

pub const COVERAGE_JSON: u64 = 1 << 0;
pub const COVERAGE_C_JSON: u64 = 1 << 1;
pub const COVERAGE_XML: u64 = 1 << 2;
pub const COVERAGE_ARCHIVE: u64 = 1 << 3;
pub const COVERAGE_RASTER: u64 = 1 << 4;
pub const COVERAGE_FONT: u64 = 1 << 5;
pub const COVERAGE_DOCUMENT: u64 = 1 << 6;
/// Founded worker executed its deterministic malformed/truncation/recursion/
/// expansion/dimension rejection corpus within the same process boundary.
pub const COVERAGE_LIMIT_REJECTIONS: u64 = 1 << 7;
/// A separately founded broker selected the worker and validated its report.
pub const COVERAGE_BROKER_VALIDATED: u64 = 1 << 8;
pub const COVERAGE_CSV: u64 = 1 << 9;
pub const COVERAGE_PLAIN_TEXT: u64 = 1 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerReportError {
    Short,
    Framing,
    Status,
    Result,
}

/// Bounded dispatch decision emitted by the capability-confined broker.
///
/// It contains only source-neutral identity and registry data.  The process
/// that owns executable capabilities realizes this decision; the broker
/// never receives those capabilities or a filesystem namespace handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrokerDispatch {
    pub input: crate::InputIdentity,
    pub parser: crate::ParserId,
    pub parser_version: u32,
    pub ambiguous: bool,
}

impl BrokerDispatch {
    pub fn encode(self, dst: &mut [u8]) -> Result<usize, WorkerReportError> {
        if dst.len() < BROKER_HEAD {
            return Err(WorkerReportError::Short);
        }
        dst.fill(0);
        dst[0..4].copy_from_slice(&BROKER_MAGIC.to_le_bytes());
        dst[4..6].copy_from_slice(&VERSION.to_le_bytes());
        dst[6] = u8::from(self.ambiguous);
        dst[8..16].copy_from_slice(&self.input.source.to_le_bytes());
        dst[16..24].copy_from_slice(&self.input.version.to_le_bytes());
        dst[24..32].copy_from_slice(&self.input.len.to_le_bytes());
        dst[32..64].copy_from_slice(&self.input.digest);
        dst[64..66].copy_from_slice(&self.parser.0.to_le_bytes());
        dst[68..72].copy_from_slice(&self.parser_version.to_le_bytes());
        Ok(BROKER_HEAD)
    }

    pub fn decode(src: &[u8]) -> Result<Self, WorkerReportError> {
        if src.len() < BROKER_HEAD
            || u32::from_le_bytes(src[0..4].try_into().unwrap()) != BROKER_MAGIC
            || u16::from_le_bytes(src[4..6].try_into().unwrap()) != VERSION
            || src[6] > 1
            || src[7] != 0
            || src[66..68].iter().any(|byte| *byte != 0)
            || src[72..].iter().any(|byte| *byte != 0)
        {
            return Err(WorkerReportError::Framing);
        }
        Ok(Self {
            input: crate::InputIdentity {
                source: u64::from_le_bytes(src[8..16].try_into().unwrap()),
                version: u64::from_le_bytes(src[16..24].try_into().unwrap()),
                len: u64::from_le_bytes(src[24..32].try_into().unwrap()),
                digest: src[32..64].try_into().unwrap(),
            },
            parser: crate::ParserId(u16::from_le_bytes(src[64..66].try_into().unwrap())),
            parser_version: u32::from_le_bytes(src[68..72].try_into().unwrap()),
            ambiguous: src[6] != 0,
        })
    }
}

/// Format one broker-selected immutable worker job. Parser selection and
/// source identity are committed in the same bounded frame as the bytes; a
/// worker never guesses a parser from an extension or ambient filename.
pub fn encode_job(
    dispatch: BrokerDispatch,
    bytes: &[u8],
    dst: &mut [u8],
) -> Result<usize, WorkerReportError> {
    let end = BROKER_HEAD
        .checked_add(bytes.len())
        .ok_or(WorkerReportError::Short)?;
    if end > dst.len() || dispatch.input.len != bytes.len() as u64 {
        return Err(WorkerReportError::Short);
    }
    dispatch.encode(dst)?;
    dst[0..4].copy_from_slice(&JOB_MAGIC.to_le_bytes());
    dst[BROKER_HEAD..end].copy_from_slice(bytes);
    Ok(end)
}

/// Decode a complete broker-selected worker job and revalidate the immutable
/// byte identity before exposing either selection or input.
pub fn decode_job(src: &[u8]) -> Result<(BrokerDispatch, &[u8]), WorkerReportError> {
    if src.len() < BROKER_HEAD || u32::from_le_bytes(src[0..4].try_into().unwrap()) != JOB_MAGIC {
        return Err(WorkerReportError::Framing);
    }
    let mut header = [0u8; BROKER_HEAD];
    header.copy_from_slice(&src[..BROKER_HEAD]);
    header[0..4].copy_from_slice(&BROKER_MAGIC.to_le_bytes());
    let dispatch = BrokerDispatch::decode(&header)?;
    let len = usize::try_from(dispatch.input.len).map_err(|_| WorkerReportError::Short)?;
    let end = BROKER_HEAD
        .checked_add(len)
        .ok_or(WorkerReportError::Short)?;
    let bytes = src.get(BROKER_HEAD..end).ok_or(WorkerReportError::Short)?;
    if *blake3::hash(bytes).as_bytes() != dispatch.input.digest {
        return Err(WorkerReportError::Framing);
    }
    Ok((dispatch, bytes))
}

/// Format one immutable source page. The identity is supplied by the source
/// owner and rechecked against the bytes by the worker parser.
pub fn encode_input(
    identity: crate::InputIdentity,
    bytes: &[u8],
    dst: &mut [u8],
) -> Result<usize, WorkerReportError> {
    let end = INPUT_HEAD
        .checked_add(bytes.len())
        .ok_or(WorkerReportError::Short)?;
    if end > dst.len() || identity.len != bytes.len() as u64 {
        return Err(WorkerReportError::Short);
    }
    dst.fill(0);
    dst[0..4].copy_from_slice(&INPUT_MAGIC.to_le_bytes());
    dst[4..6].copy_from_slice(&VERSION.to_le_bytes());
    dst[8..16].copy_from_slice(&identity.source.to_le_bytes());
    dst[16..24].copy_from_slice(&identity.version.to_le_bytes());
    dst[24..32].copy_from_slice(&identity.len.to_le_bytes());
    dst[32..64].copy_from_slice(&identity.digest);
    dst[INPUT_HEAD..end].copy_from_slice(bytes);
    Ok(end)
}

pub fn decode_input(src: &[u8]) -> Result<(crate::InputIdentity, &[u8]), WorkerReportError> {
    if src.len() < INPUT_HEAD
        || u32::from_le_bytes(src[0..4].try_into().unwrap()) != INPUT_MAGIC
        || u16::from_le_bytes(src[4..6].try_into().unwrap()) != VERSION
        || src[6..8].iter().any(|byte| *byte != 0)
    {
        return Err(WorkerReportError::Framing);
    }
    let identity = crate::InputIdentity {
        source: u64::from_le_bytes(src[8..16].try_into().unwrap()),
        version: u64::from_le_bytes(src[16..24].try_into().unwrap()),
        len: u64::from_le_bytes(src[24..32].try_into().unwrap()),
        digest: src[32..64].try_into().unwrap(),
    };
    let len = usize::try_from(identity.len).map_err(|_| WorkerReportError::Short)?;
    let end = INPUT_HEAD
        .checked_add(len)
        .ok_or(WorkerReportError::Short)?;
    let bytes = src.get(INPUT_HEAD..end).ok_or(WorkerReportError::Short)?;
    if *blake3::hash(bytes).as_bytes() != identity.digest {
        return Err(WorkerReportError::Framing);
    }
    Ok((identity, bytes))
}

#[derive(Debug, PartialEq, Eq)]
pub struct WorkerReport {
    pub coverage: u64,
    pub result: NormalizedResult,
}

impl WorkerReport {
    pub fn encode(&self, dst: &mut [u8]) -> Result<usize, WorkerReportError> {
        if dst.len() < WORKER_REPORT_BYTES {
            return Err(WorkerReportError::Short);
        }
        dst[..WORKER_REPORT_BYTES].fill(0);
        let used = encode_result(&self.result, &mut dst[HEAD..WORKER_REPORT_BYTES])
            .map_err(|_| WorkerReportError::Result)?;
        dst[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        dst[4..6].copy_from_slice(&VERSION.to_le_bytes());
        dst[8..16].copy_from_slice(&self.coverage.to_le_bytes());
        dst[16..20].copy_from_slice(&(used as u32).to_le_bytes());
        // Commit marker last. The broker reads only after task termination,
        // but retaining a final marker makes torn output fail closed too.
        dst[6..8].copy_from_slice(&1u16.to_le_bytes());
        Ok(HEAD + used)
    }

    pub fn decode(src: &[u8]) -> Result<Self, WorkerReportError> {
        if src.len() < WORKER_REPORT_BYTES {
            return Err(WorkerReportError::Short);
        }
        if u32::from_le_bytes(src[0..4].try_into().unwrap()) != MAGIC
            || u16::from_le_bytes(src[4..6].try_into().unwrap()) != VERSION
            || src[20..HEAD].iter().any(|byte| *byte != 0)
        {
            return Err(WorkerReportError::Framing);
        }
        if u16::from_le_bytes(src[6..8].try_into().unwrap()) != 1 {
            return Err(WorkerReportError::Status);
        }
        let coverage = u64::from_le_bytes(src[8..16].try_into().unwrap());
        let len = u32::from_le_bytes(src[16..20].try_into().unwrap()) as usize;
        let end = HEAD.checked_add(len).ok_or(WorkerReportError::Framing)?;
        if end > WORKER_REPORT_BYTES || src[end..WORKER_REPORT_BYTES].iter().any(|b| *b != 0) {
            return Err(WorkerReportError::Framing);
        }
        let result = decode_result(&src[HEAD..end]).map_err(|_| WorkerReportError::Result)?;
        Ok(Self { coverage, result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InputIdentity, ParserId, ParserProvenance};

    #[test]
    fn one_page_report_round_trips_and_rejects_torn_publication() {
        let result = NormalizedResult::empty(ParserProvenance::deterministic(
            InputIdentity::of(1, 2, b"null"),
            ParserId(1),
            1,
        ));
        let expected = WorkerReport {
            coverage: COVERAGE_JSON | COVERAGE_ARCHIVE,
            result,
        };
        let mut page = [0xa5; WORKER_REPORT_BYTES];
        expected.encode(&mut page).unwrap();
        assert_eq!(WorkerReport::decode(&page).unwrap(), expected);
        page[6..8].fill(0);
        assert_eq!(WorkerReport::decode(&page), Err(WorkerReportError::Status));
    }

    #[test]
    fn immutable_input_identity_is_bound_to_exact_bytes() {
        let bytes = br#"{"same":true}"#;
        let identity = InputIdentity::of(9, 4, bytes);
        let mut page = [0xa5; 4096];
        encode_input(identity, bytes, &mut page).unwrap();
        assert_eq!(decode_input(&page).unwrap(), (identity, &bytes[..]));
        page[64] ^= 1;
        assert_eq!(decode_input(&page), Err(WorkerReportError::Framing));
    }

    #[test]
    fn broker_dispatch_round_trips_and_rejects_stale_trailing_bytes() {
        let bytes = br#"{\"broker\":true}"#;
        let expected = BrokerDispatch {
            input: InputIdentity::of(4, 9, bytes),
            parser: ParserId(2),
            parser_version: 7,
            ambiguous: true,
        };
        let mut page = [0xa5; WORKER_REPORT_BYTES];
        expected.encode(&mut page).unwrap();
        assert_eq!(BrokerDispatch::decode(&page).unwrap(), expected);
        page[100] = 1;
        assert_eq!(
            BrokerDispatch::decode(&page),
            Err(WorkerReportError::Framing)
        );
    }

    #[test]
    fn worker_job_binds_parser_selection_and_exact_bytes_in_one_frame() {
        let bytes = b"a,b\n1,2\n";
        let dispatch = BrokerDispatch {
            input: InputIdentity::of(7, 3, bytes),
            parser: crate::registry::CSV,
            parser_version: 1,
            ambiguous: false,
        };
        let mut page = [0xa5; WORKER_REPORT_BYTES];
        encode_job(dispatch, bytes, &mut page).unwrap();
        assert_eq!(decode_job(&page).unwrap(), (dispatch, &bytes[..]));
        page[BROKER_HEAD] ^= 1;
        assert_eq!(decode_job(&page), Err(WorkerReportError::Framing));
    }
}
