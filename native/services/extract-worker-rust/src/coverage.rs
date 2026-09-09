//! Deterministic in-process production fixtures.
//!
//! Host fuzzing remains broader. This compact set exists so a founded worker
//! proves that every initially accepted parser stack is actually linked,
//! runnable, and capable of producing a registry-valid normalized result.

use heapless::Vec;
use kore_svc_extract::{
    COVERAGE_ARCHIVE, COVERAGE_CSV, COVERAGE_DOCUMENT, COVERAGE_FONT, COVERAGE_JSON,
    COVERAGE_LIMIT_REJECTIONS, COVERAGE_PLAIN_TEXT, COVERAGE_RASTER, COVERAGE_XML, InputIdentity,
    Registry,
};

use crate::{
    parse_bmp, parse_csv, parse_epub, parse_json_rust, parse_plain_text, parse_tar, parse_ttf,
    parse_xml,
};

fn valid(
    bytes: &[u8],
    parser: fn(
        InputIdentity,
        &[u8],
    ) -> Result<kore_svc_extract::NormalizedResult, crate::AdapterError>,
) -> bool {
    let input = InputIdentity::of(0x5052_4f44, 1, bytes);
    parser(input, bytes)
        .and_then(|result| {
            Registry::validate(&result, input, 4096).map_err(|_| crate::AdapterError::Malformed)
        })
        .is_ok()
}

fn tar() -> [u8; 2048] {
    let mut out = [0u8; 2048];
    let data = b"hello";
    out[..9].copy_from_slice(b"hello.txt");
    out[100..108].copy_from_slice(b"0000644\0");
    out[108..116].copy_from_slice(b"0000000\0");
    out[116..124].copy_from_slice(b"0000000\0");
    out[124..136].copy_from_slice(b"00000000005\0");
    out[136..148].copy_from_slice(b"00000000000\0");
    out[148..156].fill(b' ');
    out[156] = b'0';
    out[257..263].copy_from_slice(b"ustar\0");
    out[263..265].copy_from_slice(b"00");
    let checksum: u32 = out[..512].iter().map(|byte| u32::from(*byte)).sum();
    let digits = [
        b'0' + ((checksum >> 15) & 7) as u8,
        b'0' + ((checksum >> 12) & 7) as u8,
        b'0' + ((checksum >> 9) & 7) as u8,
        b'0' + ((checksum >> 6) & 7) as u8,
        b'0' + ((checksum >> 3) & 7) as u8,
        b'0' + (checksum & 7) as u8,
        0,
        b' ',
    ];
    out[148..156].copy_from_slice(&digits);
    out[512..517].copy_from_slice(data);
    out
}

fn bmp() -> [u8; 58] {
    let mut out = [0u8; 58];
    out[0..2].copy_from_slice(b"BM");
    out[2..6].copy_from_slice(&58u32.to_le_bytes());
    out[10..14].copy_from_slice(&54u32.to_le_bytes());
    out[14..18].copy_from_slice(&40u32.to_le_bytes());
    out[18..22].copy_from_slice(&1i32.to_le_bytes());
    out[22..26].copy_from_slice(&1i32.to_le_bytes());
    out[26..28].copy_from_slice(&1u16.to_le_bytes());
    out[28..30].copy_from_slice(&24u16.to_le_bytes());
    out[34..38].copy_from_slice(&4u32.to_le_bytes());
    out
}

fn append<const N: usize>(out: &mut Vec<u8, N>, bytes: &[u8]) -> bool {
    out.extend_from_slice(bytes).is_ok()
}

fn epub_with_opf(opf: &[u8], advertised_opf: u32) -> Option<Vec<u8, 1024>> {
    let entries: [(&[u8], &[u8], u32); 2] = [
        (b"mimetype", b"application/epub+zip", 20),
        (b"OEBPS/content.opf", opf, advertised_opf),
    ];
    let mut out = Vec::<u8, 1024>::new();
    let mut central = Vec::<u8, 768>::new();
    for (name, data, advertised) in entries {
        let offset = out.len() as u32;
        for bytes in [
            &0x0403_4b50u32.to_le_bytes()[..],
            &20u16.to_le_bytes(),
            &0x0800u16.to_le_bytes(),
            &0u16.to_le_bytes(),
            &[0u8; 8],
            &(data.len() as u32).to_le_bytes(),
            &advertised.to_le_bytes(),
            &(name.len() as u16).to_le_bytes(),
            &0u16.to_le_bytes(),
            name,
            data,
        ] {
            if !append(&mut out, bytes) {
                return None;
            }
        }
        for bytes in [
            &0x0201_4b50u32.to_le_bytes()[..],
            &20u16.to_le_bytes(),
            &20u16.to_le_bytes(),
            &0x0800u16.to_le_bytes(),
            &0u16.to_le_bytes(),
            &[0u8; 8],
            &(data.len() as u32).to_le_bytes(),
            &advertised.to_le_bytes(),
            &(name.len() as u16).to_le_bytes(),
            &[0u8; 12],
            &offset.to_le_bytes(),
            name,
        ] {
            if !append(&mut central, bytes) {
                return None;
            }
        }
    }
    let central_offset = out.len() as u32;
    let central_len = central.len() as u32;
    if !append(&mut out, &central)
        || !append(&mut out, &0x0605_4b50u32.to_le_bytes())
        || !append(&mut out, &[0u8; 4])
        || !append(&mut out, &2u16.to_le_bytes())
        || !append(&mut out, &2u16.to_le_bytes())
        || !append(&mut out, &central_len.to_le_bytes())
        || !append(&mut out, &central_offset.to_le_bytes())
        || !append(&mut out, &0u16.to_le_bytes())
    {
        return None;
    }
    Some(out)
}

fn epub() -> Option<Vec<u8, 1024>> {
    let opf = b"<package><spine><itemref/></spine></package>";
    epub_with_opf(opf, opf.len() as u32)
}

// Public-domain synthetic font fixture used by the adapter regression tests.
const TTF: &[u8] = &[
    0, 1, 0, 0, 0, 7, 0, 64, 0, 2, 0, 48, 99, 109, 97, 112, 0, 9, 0, 118, 0, 0, 1, 0, 0, 0, 0, 44,
    103, 108, 121, 102, 241, 203, 102, 152, 0, 0, 1, 52, 0, 0, 0, 92, 104, 101, 97, 100, 242, 53,
    221, 248, 0, 0, 0, 124, 0, 0, 0, 54, 104, 104, 101, 97, 6, 97, 0, 202, 0, 0, 0, 180, 0, 0, 0,
    36, 104, 109, 116, 120, 4, 116, 0, 106, 0, 0, 0, 248, 0, 0, 0, 8, 108, 111, 99, 97, 0, 46, 0,
    20, 0, 0, 1, 44, 0, 0, 0, 6, 109, 97, 120, 112, 0, 5, 0, 11, 0, 0, 0, 216, 0, 0, 0, 32, 0, 1,
    0, 0, 0, 1, 0, 0, 245, 156, 41, 68, 95, 15, 60, 245, 0, 2, 3, 232, 0, 0, 0, 0, 180, 146, 244,
    0, 0, 0, 0, 0, 220, 47, 166, 92, 0, 6, 0, 0, 2, 88, 2, 188, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 4, 0, 254, 112, 0, 0, 2, 88, 0, 6, 255, 255, 2, 88, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 0, 11, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 88, 0, 100, 2, 28, 0, 6, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 12,
    0, 4, 0, 32, 0, 0, 0, 4, 0, 4, 0, 1, 0, 0, 0, 65, 255, 255, 0, 0, 0, 65, 255, 255, 255, 192, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 20, 0, 46, 0, 0, 0, 2, 0, 100, 0, 0, 2, 88, 2, 188, 0, 3, 0, 7, 0, 0,
    51, 17, 33, 17, 37, 33, 17, 33, 100, 1, 244, 254, 52, 1, 164, 254, 92, 2, 188, 253, 68, 40, 2,
    108, 0, 2, 0, 6, 0, 0, 2, 29, 2, 144, 0, 2, 0, 10, 0, 0, 19, 51, 3, 1, 19, 51, 19, 35, 39, 35,
    7, 173, 196, 99, 254, 248, 218, 96, 221, 89, 62, 239, 66, 1, 11, 1, 64, 253, 181, 2, 144, 253,
    112, 200, 200, 0,
];

/// Coverage bits for every adapter that produced a registry-valid result.
pub fn production_coverage() -> u64 {
    let mut bits = 0;
    let json = br#"{"alpha":1,"nested":[true,null]}"#;
    if valid(json, parse_json_rust) {
        bits |= COVERAGE_JSON;
    }
    let xml = br#"<?xml version="1.0"?><root><item/></root>"#;
    if valid(xml, parse_xml) {
        bits |= COVERAGE_XML;
    }
    let csv = b"name,value\nalpha,1\nbeta,2\n";
    if valid(csv, parse_csv) {
        bits |= COVERAGE_CSV;
    }
    let plain = b"alpha\nbeta\n";
    if valid(plain, parse_plain_text) {
        bits |= COVERAGE_PLAIN_TEXT;
    }
    let valid_tar = tar();
    if valid(&valid_tar, parse_tar) {
        bits |= COVERAGE_ARCHIVE;
    }
    let valid_bmp = bmp();
    if valid(&valid_bmp, parse_bmp) {
        bits |= COVERAGE_RASTER;
    }
    if valid(TTF, parse_ttf) {
        bits |= COVERAGE_FONT;
    }
    if let Some(epub) = epub()
        && valid(&epub, parse_epub)
    {
        bits |= COVERAGE_DOCUMENT;
    }
    let malformed_json = b"{\"unterminated\":";
    let malformed_csv = b"a,b\n\"unterminated,2";
    let binary_text = b"plain\0binary";
    let recursive_xml = b"<a><b><c><d><e><f><g><h><i><j><k><l><m><n><o><p><q/></p></o></n></m></l></k></j></i></h></g></f></e></d></c></b></a>";
    let truncated_tar = &tar()[..511];
    let mut absurd_bmp = bmp();
    absurd_bmp[18..22].copy_from_slice(&9000i32.to_le_bytes());
    let recursive_opf = b"<package><a><b><c><d><e><f><g><h><i><j><k><l><m><n><o><p><q/></p></o></n></m></l></k></j></i></h></g></f></e></d></c></b></a></package>";
    let recursive_epub = epub_with_opf(recursive_opf, recursive_opf.len() as u32);
    let bomb_epub = epub_with_opf(b"<package/>", 1_000_000);
    if parse_json_rust(InputIdentity::of(1, 1, malformed_json), malformed_json).is_err()
        && parse_csv(InputIdentity::of(1, 1, malformed_csv), malformed_csv).is_err()
        && parse_plain_text(InputIdentity::of(1, 1, binary_text), binary_text).is_err()
        && parse_xml(InputIdentity::of(1, 1, recursive_xml), recursive_xml).is_err()
        && parse_tar(InputIdentity::of(1, 1, truncated_tar), truncated_tar).is_err()
        && parse_bmp(InputIdentity::of(1, 1, &absurd_bmp), &absurd_bmp).is_err()
        && parse_ttf(InputIdentity::of(1, 1, &TTF[..32]), &TTF[..32]).is_err()
        && recursive_epub
            .as_deref()
            .is_some_and(|bytes| parse_epub(InputIdentity::of(1, 1, bytes), bytes).is_err())
        && bomb_epub
            .as_deref()
            .is_some_and(|bytes| parse_epub(InputIdentity::of(1, 1, bytes), bytes).is_err())
    {
        bits |= COVERAGE_LIMIT_REJECTIONS;
    }
    bits
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_production_fixture_reaches_its_real_adapter() {
        assert_eq!(
            super::production_coverage(),
            kore_svc_extract::COVERAGE_JSON
                | kore_svc_extract::COVERAGE_XML
                | kore_svc_extract::COVERAGE_ARCHIVE
                | kore_svc_extract::COVERAGE_RASTER
                | kore_svc_extract::COVERAGE_FONT
                | kore_svc_extract::COVERAGE_DOCUMENT
                | kore_svc_extract::COVERAGE_CSV
                | kore_svc_extract::COVERAGE_PLAIN_TEXT
                | kore_svc_extract::COVERAGE_LIMIT_REJECTIONS
        );
    }
}
