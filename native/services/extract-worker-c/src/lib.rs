//! C-backed adapter confined to the C worker image.

#![no_std]

#[cfg(test)]
extern crate std;

use heapless::String;
use kore_svc_extract::{
    registry::{C_JSON, Registry},
    schema::{
        Confidence, Fact, FactKind, FactValue, InputIdentity, NormalizedResult, ParserProvenance,
        ResultBuilder, SourceRange,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterError {
    Malformed,
    Truncated,
    InputLimit,
    OutputLimit,
}

unsafe extern "C" {
    fn kore_jsmn_summary(
        bytes: *const u8,
        len: usize,
        root_members: *mut u32,
        token_count: *mut u32,
    ) -> i32;
}

fn text<const N: usize>(value: &str) -> Result<String<N>, AdapterError> {
    String::try_from(value).map_err(|_| AdapterError::OutputLimit)
}

pub fn parse_json_c(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    let declaration = Registry::get(C_JSON).expect("built-in C JSON declaration");
    if input.len != bytes.len() as u64
        || input.len > declaration.limits.max_input
        || *blake3::hash(bytes).as_bytes() != input.digest
    {
        return Err(AdapterError::InputLimit);
    }
    let mut members = 0u32;
    let mut tokens = 0u32;
    // SAFETY: one immutable slice and two initialized out-pointers remain
    // live for the call. The audited wrapper retains no pointer and exposes
    // no foreign layout to Rust.
    let status =
        unsafe { kore_jsmn_summary(bytes.as_ptr(), bytes.len(), &mut members, &mut tokens) };
    if status != 0 || tokens == 0 {
        return Err(match status {
            -1 => AdapterError::OutputLimit,
            -3 => AdapterError::Truncated,
            _ => AdapterError::Malformed,
        });
    }
    let mut builder = ResultBuilder::new(ParserProvenance::deterministic(
        input,
        C_JSON,
        declaration.version,
    ));
    builder
        .fact(Fact {
            kind: FactKind::MediaType,
            key: text("media_type")?,
            value: FactValue::Utf8(text("application/json")?),
            confidence: Confidence::CERTAIN,
            source: None,
        })
        .map_err(|_| AdapterError::OutputLimit)?;
    builder
        .fact(Fact {
            kind: FactKind::JsonMemberCount,
            key: text("root_members")?,
            value: FactValue::Unsigned(u64::from(members)),
            confidence: Confidence::CERTAIN,
            source: Some(SourceRange::new(0, input.len)),
        })
        .map_err(|_| AdapterError::OutputLimit)?;
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_svc_extract_worker_rust::parse_json_rust;

    fn identity(bytes: &[u8]) -> InputIdentity {
        InputIdentity::of(1, 1, bytes)
    }

    #[test]
    fn rust_and_c_workers_normalize_the_same_fixture() {
        let bytes = br#"{"a":1,"b":[2,3]}"#;
        let rust = parse_json_rust(identity(bytes), bytes).unwrap();
        let c = parse_json_c(identity(bytes), bytes).unwrap();
        assert_eq!(rust.facts, c.facts);
        Registry::validate(&c, identity(bytes), 0).unwrap();
    }

    #[test]
    fn malformed_and_truncated_inputs_fail_closed() {
        for bytes in [&b"{"[..], &b"[1,"[..], &b"not json"[..]] {
            assert!(parse_json_c(identity(bytes), bytes).is_err());
        }
    }

    #[test]
    fn deterministic_mutation_corpus_stays_bounded_and_normalized() {
        let seed = br#"{"alpha":1,"nested":[true,null]}"#;
        for index in 0..seed.len() {
            for replacement in [0u8, 0xff, seed[index] ^ 0x80] {
                let mut bytes = std::vec::Vec::from(seed);
                bytes[index] = replacement;
                let input = identity(&bytes);
                if let Ok(result) = parse_json_c(input, &bytes) {
                    Registry::validate(&result, input, 0).unwrap();
                }
            }
        }
        for end in 0..seed.len() {
            let bytes = &seed[..end];
            let input = identity(bytes);
            if let Ok(result) = parse_json_c(input, bytes) {
                Registry::validate(&result, input, 0).unwrap();
            }
        }
    }

    #[test]
    #[ignore = "manual vendor benchmark; run with --release --ignored --nocapture"]
    fn benchmark_jsmn_adapter() {
        use std::time::Instant;
        const RUNS: u32 = 50_000;
        let bytes = br#"{"alpha":1,"nested":[true,null]}"#;
        let input = identity(bytes);
        let start = Instant::now();
        for _ in 0..RUNS {
            std::hint::black_box(parse_json_c(input, std::hint::black_box(bytes))).unwrap();
        }
        let ns = start.elapsed().as_nanos() / u128::from(RUNS);
        std::println!("extract-bench jsmn: {ns} ns/op ({} bytes)", bytes.len());
    }
}
