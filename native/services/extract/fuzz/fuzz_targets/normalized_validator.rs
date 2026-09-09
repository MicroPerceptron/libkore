#![no_main]

use heapless::String;
use kore_svc_extract::{
    Registry,
    registry::{ParserId, RUST_JSON},
    schema::{
        Confidence, Fact, FactKind, FactValue, InputIdentity, ParserProvenance, ResultBuilder,
        SourceRange, Thumbnail,
    },
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let input_bytes = data.get(..data.len().min(4096)).unwrap_or(data);
    let input = InputIdentity::of(0x5641_4c49, 1, input_bytes);
    let parser = data
        .first()
        .copied()
        .map_or(RUST_JSON, |id| ParserId(u16::from(id)));
    let mut provenance = ParserProvenance::deterministic(
        input,
        RUST_JSON,
        Registry::get(RUST_JSON).unwrap().version,
    );
    provenance.parser = parser;
    if data.get(1).is_some_and(|byte| byte & 1 != 0) {
        provenance.input.version = provenance.input.version.wrapping_add(1);
    }
    let mut result = ResultBuilder::new(provenance);
    let offset = data.get(2).copied().map_or(0, u64::from);
    let len = data.get(3).copied().map_or(input.len, u64::from);
    let _ = result.fact(Fact {
        kind: data.get(4).map_or(FactKind::JsonMemberCount, |byte| {
            if byte & 1 == 0 {
                FactKind::JsonMemberCount
            } else {
                FactKind::GlyphCount
            }
        }),
        key: String::try_from("fuzz").unwrap(),
        value: FactValue::Unsigned(input.len),
        confidence: Confidence::CERTAIN,
        source: Some(SourceRange::new(offset, len)),
    });
    let mut result = result.finish();
    if data.get(5).is_some_and(|byte| byte & 1 != 0) {
        let _ = result.thumbnails.push(Thumbnail {
            output_offset: u32::from(data.get(6).copied().unwrap_or(0)),
            output_len: u32::from(data.get(7).copied().unwrap_or(0)),
            width: u32::from(data.get(8).copied().unwrap_or(1)),
            height: u32::from(data.get(9).copied().unwrap_or(1)),
            format: 1,
        });
    }
    let _ = Registry::validate(&result, input, 64);
});
