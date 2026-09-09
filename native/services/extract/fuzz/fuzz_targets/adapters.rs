#![no_main]

use kore_svc_extract::{Registry, schema::InputIdentity};
use kore_svc_extract_worker_c::parse_json_c;
use kore_svc_extract_worker_rust::{
    parse_bmp, parse_csv, parse_epub, parse_json_rust, parse_plain_text, parse_tar, parse_ttf,
    parse_xml,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&selector, bytes)) = data.split_first() else {
        return;
    };
    // Keep each invocation within the registry's declared input authority.
    if bytes.len() > 1024 * 1024 {
        return;
    }
    let input = InputIdentity::of(0x4655_5a5a, 1, bytes);
    let result = match selector % 9 {
        0 => parse_json_rust(input, bytes).ok(),
        1 => parse_json_c(input, bytes).ok(),
        2 => parse_tar(input, bytes).ok(),
        3 => parse_bmp(input, bytes).ok(),
        4 => parse_ttf(input, bytes).ok(),
        5 => parse_epub(input, bytes).ok(),
        6 => parse_xml(input, bytes).ok(),
        7 => parse_csv(input, bytes).ok(),
        _ => parse_plain_text(input, bytes).ok(),
    };
    if let Some(result) = result {
        assert!(Registry::validate(&result, input, 64 * 1024).is_ok());
    }
});
