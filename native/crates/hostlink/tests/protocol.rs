use kore_hostlink::{
    DecodeError, EnvelopeFields, EnvelopeV1, FAMILY_LINK, FrameDecoder, INFERENCE_EVENT_HEADER_LEN,
    INFERENCE_INPUT_MAX, INFERENCE_PART_MAX, INFERENCE_REQUEST_HEADER_LEN, InferenceEventKind,
    InferenceEventView, InferenceRequestKind, InferenceRequestView, LinkEpoch, RequestId, RouteId,
    SHMEM_WAKE, SHMEM_WAKE_LEN, SHMEM_WAKE_VERSION, ShmemWakeV1, decode_inference_event,
    decode_inference_request, derive_shmem_epoch, link_envelope, shmem_wake_envelope_is_valid,
};
use zerocopy::{FromBytes, IntoBytes};

fn frame(payload: &[u8]) -> std::vec::Vec<u8> {
    let envelope = EnvelopeV1::new(EnvelopeFields {
        family: FAMILY_LINK,
        schema_minor: 0,
        kind: 7,
        epoch: LinkEpoch([1; 16]),
        route: RouteId([2; 16]),
        generation: 3,
        sequence: 0,
        request: RequestId([4; 16]),
        payload_len: payload.len() as u32,
    });
    let mut bytes = envelope.as_bytes().to_vec();
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn every_stream_split_reassembles_the_same_frame() {
    let bytes = frame(b"bounded hostlink");
    for split in 0..=bytes.len() {
        let mut decoder = FrameDecoder::new();
        let mut storage = [0; 64];
        let first = decoder.push(&bytes[..split], &mut storage).unwrap();
        if !first.complete {
            let second = decoder.push(&bytes[split..], &mut storage).unwrap();
            assert!(second.complete);
        }
        assert_eq!(decoder.payload(&storage), Some(&b"bounded hostlink"[..]));
    }
}

#[test]
fn reserved_bits_and_oversized_lengths_fail_closed() {
    let mut bytes = frame(&[]);
    // flags begins at byte 12.
    bytes[12] = 1;
    let mut decoder = FrameDecoder::new();
    assert_eq!(decoder.push(&bytes, &mut []), Err(DecodeError::Reserved));

    let mut bytes = frame(&[]);
    // payload_len begins at byte 80.
    bytes[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut decoder = FrameDecoder::new();
    assert_eq!(
        decoder.push(&bytes, &mut []),
        Err(DecodeError::InvalidLength)
    );
}

#[test]
fn decoder_never_consumes_bytes_from_the_following_frame() {
    let one = frame(b"one");
    let two = frame(b"two");
    let joined: std::vec::Vec<u8> = one.iter().chain(&two).copied().collect();
    let mut decoder = FrameDecoder::new();
    let mut storage = [0; 8];
    let progress = decoder.push(&joined, &mut storage).unwrap();
    assert!(progress.complete);
    assert_eq!(progress.consumed, one.len());
    assert_eq!(decoder.payload(&storage), Some(&b"one"[..]));
}

#[test]
fn a_rejected_frame_resets_without_panicking_on_the_next_frame() {
    let mut bad = frame(&[]);
    bad[0] ^= 0xff;
    let good = frame(b"recovered");
    let mut decoder = FrameDecoder::new();
    let mut storage = [0; 16];
    assert_eq!(decoder.push(&bad, &mut storage), Err(DecodeError::BadMagic));
    assert!(decoder.push(&good, &mut storage).unwrap().complete);
    assert_eq!(decoder.payload(&storage), Some(&b"recovered"[..]));
}

#[test]
fn shared_wake_codec_is_fixed_versioned_and_rejects_every_reserved_bit() {
    let wake = ShmemWakeV1::new(17).unwrap();
    assert_eq!(wake.as_bytes().len(), SHMEM_WAKE_LEN);
    assert_eq!(wake.publication(), Some(17));
    assert!(ShmemWakeV1::new(0).is_none());

    for byte in 0..8 {
        let mut corrupted = [0u8; SHMEM_WAKE_LEN];
        corrupted.copy_from_slice(wake.as_bytes());
        corrupted[byte] ^= 1;
        let decoded = ShmemWakeV1::read_from_bytes(&corrupted).unwrap();
        assert_eq!(decoded.publication(), None, "header byte {byte}");
    }
    let mut zero_publication = [0u8; SHMEM_WAKE_LEN];
    zero_publication.copy_from_slice(wake.as_bytes());
    zero_publication[8..].fill(0);
    assert_eq!(
        ShmemWakeV1::read_from_bytes(&zero_publication)
            .unwrap()
            .publication(),
        None
    );
    assert_eq!(SHMEM_WAKE_VERSION, 1);
}

#[test]
fn shared_wake_envelope_requires_exact_link_control_identity() {
    let epoch = LinkEpoch([9; 16]);
    let valid = link_envelope(
        SHMEM_WAKE,
        epoch,
        RouteId::ZERO,
        0,
        0,
        RequestId::ZERO,
        SHMEM_WAKE_LEN,
    );
    assert!(shmem_wake_envelope_is_valid(&valid, epoch));

    let mut variants = [valid; 8];
    variants[0].family.set(FAMILY_LINK + 1);
    variants[1].schema_minor.set(1);
    variants[2].kind.set(SHMEM_WAKE + 1);
    variants[3].route_id = RouteId([1; 16]);
    variants[4].task_generation.set(1);
    variants[5].sequence.set(1);
    variants[6].request_id = RequestId([1; 16]);
    variants[7].payload_len.set((SHMEM_WAKE_LEN + 1) as u32);
    for (index, envelope) in variants.iter().enumerate() {
        assert!(
            !shmem_wake_envelope_is_valid(envelope, epoch),
            "variant {index}"
        );
    }
    assert!(!shmem_wake_envelope_is_valid(&valid, LinkEpoch::ZERO));
    assert!(!shmem_wake_envelope_is_valid(&valid, LinkEpoch([10; 16])));
}

#[test]
fn shared_epoch_derivation_is_nonzero_deterministic_and_domain_sensitive() {
    assert_eq!(derive_shmem_epoch(LinkEpoch::ZERO), None);
    let one = derive_shmem_epoch(LinkEpoch([1; 16])).unwrap();
    assert_ne!(one, 0);
    assert_eq!(derive_shmem_epoch(LinkEpoch([1; 16])), Some(one));
    assert_ne!(derive_shmem_epoch(LinkEpoch([2; 16])), Some(one));
}

#[test]
fn inference_request_and_stream_events_are_versioned_bounded_and_exact() {
    let mut request = [0; INFERENCE_REQUEST_HEADER_LEN + INFERENCE_INPUT_MAX];
    let request_len = InferenceRequestView {
        kind: InferenceRequestKind::AgentTurn,
        turn: 17,
        max_output_tokens: 32,
        timeout_ticks: 99,
        input: b"private turn",
    }
    .encode(&mut request)
    .unwrap();
    let decoded = decode_inference_request(&request[..request_len]).unwrap();
    assert_eq!(decoded.kind, InferenceRequestKind::AgentTurn);
    assert_eq!(decoded.turn, 17);
    assert_eq!(decoded.input, b"private turn");
    assert!(decode_inference_request(&request[..request_len - 1]).is_none());
    request[7] = 1;
    assert!(decode_inference_request(&request[..request_len]).is_none());

    let large_input = [b'x'; INFERENCE_PART_MAX + 1];
    let large_len = InferenceRequestView {
        kind: InferenceRequestKind::BrokerPlan,
        turn: 18,
        max_output_tokens: 32,
        timeout_ticks: 99,
        input: &large_input,
    }
    .encode(&mut request)
    .unwrap();
    assert_eq!(
        decode_inference_request(&request[..large_len])
            .unwrap()
            .input,
        large_input
    );
    assert!(
        InferenceRequestView {
            kind: InferenceRequestKind::BrokerPlan,
            turn: 18,
            max_output_tokens: 32,
            timeout_ticks: 99,
            input: &[b'x'; INFERENCE_INPUT_MAX + 1],
        }
        .encode(&mut request)
        .is_none()
    );

    let mut event = [0; INFERENCE_EVENT_HEADER_LEN + INFERENCE_PART_MAX];
    let event_len = InferenceEventView {
        kind: InferenceEventKind::AssistantPart,
        terminal: false,
        part: 3,
        prompt_tokens: 5,
        completion_tokens: 2,
        turn: 17,
        payload: b"token",
    }
    .encode(&mut event)
    .unwrap();
    let decoded = decode_inference_event(&event[..event_len]).unwrap();
    assert!(!decoded.terminal);
    assert_eq!(decoded.part, 3);
    assert_eq!(decoded.payload, b"token");
    event[7] = 1;
    assert!(decode_inference_event(&event[..event_len]).is_none());
}
