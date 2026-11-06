//! Fail-closed block-provisioner codecs.

use kore_abi::svc::{BlockRepWire, BlockReqWire};
use kore_rt::svc::SvcProtocol;

use crate::{
    BlockRefusal, BlockRequest, BlockResponse, BlockService, CompletionMode, ProvisionRequest,
};

pub fn encode_req(request: &BlockRequest) -> BlockReqWire {
    let mut wire = BlockReqWire {
        op: 0,
        mode: 0,
        writable: 0,
        reserved: 0,
        depth: 0.into(),
        reserved2: 0.into(),
        buffer_bytes: 0.into(),
        reserved3: 0.into(),
    };
    match request {
        BlockRequest::Open(request) => {
            wire.op = BlockReqWire::OP_OPEN;
            wire.mode = match request.mode {
                CompletionMode::Poll => 0,
                CompletionMode::Interrupt => 1,
            };
            wire.writable = request.writable as u8;
            wire.depth = request.requested_queue_depth.into();
            wire.buffer_bytes = u32::try_from(request.requested_buffer_bytes)
                .unwrap_or(u32::MAX)
                .into();
        }
        BlockRequest::Status => wire.op = BlockReqWire::OP_STATUS,
        BlockRequest::Close => wire.op = 3,
    }
    wire
}

pub fn decode_req(wire: &BlockReqWire) -> Option<BlockRequest> {
    if wire.reserved != 0
        || wire.reserved2.get() != 0
        || wire.reserved3.get() != 0
        || wire.writable > 1
        || wire.mode > 1
    {
        return None;
    }
    Some(match wire.op {
        BlockReqWire::OP_OPEN => BlockRequest::Open(ProvisionRequest {
            writable: wire.writable != 0,
            mode: if wire.mode == 0 {
                CompletionMode::Poll
            } else {
                CompletionMode::Interrupt
            },
            requested_queue_depth: wire.depth.get(),
            requested_buffer_bytes: wire.buffer_bytes.get() as usize,
        }),
        BlockReqWire::OP_STATUS
            if wire.mode == 0
                && wire.writable == 0
                && wire.depth.get() == 0
                && wire.buffer_bytes.get() == 0 =>
        {
            BlockRequest::Status
        }
        3 if wire.mode == 0
            && wire.writable == 0
            && wire.depth.get() == 0
            && wire.buffer_bytes.get() == 0 =>
        {
            BlockRequest::Close
        }
        _ => return None,
    })
}

pub fn encode_rep(response: &BlockResponse, grants: u8) -> BlockRepWire {
    let (tag, refusal, available, channels, limit) = match *response {
        BlockResponse::Opened { channels, limit } => {
            (BlockRepWire::TAG_OPENED, 0, true, channels, limit)
        }
        BlockResponse::Status {
            available,
            channels,
            limit,
        } => (BlockRepWire::TAG_STATUS, 0, available, channels, limit),
        BlockResponse::Closed => (4, 0, false, 0, BlockService::LIMIT),
        BlockResponse::Refused(refusal) => (
            BlockRepWire::TAG_REFUSED,
            match refusal {
                BlockRefusal::Unavailable => BlockRepWire::REFUSE_UNAVAILABLE,
                BlockRefusal::Policy => BlockRepWire::REFUSE_POLICY,
                BlockRefusal::Exhausted => BlockRepWire::REFUSE_EXHAUSTED,
                BlockRefusal::Lost => BlockRepWire::REFUSE_LOST,
            },
            false,
            0,
            BlockService::LIMIT,
        ),
    };
    BlockRepWire {
        tag,
        refusal,
        grants,
        flags: available as u8,
        block_size: 0.into(),
        max_transfer_blocks: 0.into(),
        atomic_write_blocks: 0.into(),
        block_count: 0.into(),
        channels: channels.into(),
        limit: limit.into(),
        reserved: 0.into(),
    }
}

pub fn decode_rep(wire: &BlockRepWire) -> Option<(BlockResponse, u8)> {
    if wire.reserved.get() != 0
        || wire.block_size.get() != 0
        || wire.max_transfer_blocks.get() != 0
        || wire.atomic_write_blocks.get() != 0
        || wire.block_count.get() != 0
        || wire.flags > 1
    {
        return None;
    }
    let response = match (wire.tag, wire.refusal, wire.grants) {
        (BlockRepWire::TAG_OPENED, 0, 2) if wire.flags == 1 => BlockResponse::Opened {
            channels: wire.channels.get(),
            limit: wire.limit.get(),
        },
        (BlockRepWire::TAG_STATUS, 0, 0) => BlockResponse::Status {
            available: wire.flags == 1,
            channels: wire.channels.get(),
            limit: wire.limit.get(),
        },
        (4, 0, 0) if wire.flags == 0 && wire.channels.get() == 0 => BlockResponse::Closed,
        (BlockRepWire::TAG_REFUSED, refusal, 0) if wire.flags == 0 && wire.channels.get() == 0 => {
            BlockResponse::Refused(match refusal {
                BlockRepWire::REFUSE_UNAVAILABLE => BlockRefusal::Unavailable,
                BlockRepWire::REFUSE_POLICY => BlockRefusal::Policy,
                BlockRepWire::REFUSE_EXHAUSTED => BlockRefusal::Exhausted,
                BlockRepWire::REFUSE_LOST => BlockRefusal::Lost,
                _ => return None,
            })
        }
        _ => return None,
    };
    Some((response, wire.grants))
}

impl SvcProtocol for BlockService {
    type Request = BlockRequest;
    type Response = BlockResponse;
    type ReqWire = BlockReqWire;
    type RepWire = BlockRepWire;

    const REPLY_GRANTS: usize = 2;

    fn retain_grant(_index: usize) -> bool {
        true
    }

    fn reply_matches(request: &BlockReqWire, response: &BlockResponse, grants: u8) -> bool {
        matches!(
            (decode_req(request), response, grants),
            (Some(BlockRequest::Open(_)), BlockResponse::Opened { .. }, 2)
                | (Some(BlockRequest::Open(_)), BlockResponse::Refused(_), 0)
                | (Some(BlockRequest::Status), BlockResponse::Status { .. }, 0)
                | (Some(BlockRequest::Close), BlockResponse::Closed, 0)
                | (Some(BlockRequest::Close), BlockResponse::Refused(_), 0)
        )
    }

    fn encode_req(request: &BlockRequest) -> BlockReqWire {
        encode_req(request)
    }
    fn decode_req(wire: &BlockReqWire) -> Option<BlockRequest> {
        decode_req(wire)
    }
    fn encode_rep(response: &BlockResponse, grants: u8) -> BlockRepWire {
        encode_rep(response, grants)
    }
    fn decode_rep(wire: &BlockRepWire) -> Option<(BlockResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_round_trip_requires_exactly_two_grants() {
        let request = BlockRequest::Open(ProvisionRequest {
            writable: true,
            mode: CompletionMode::Poll,
            requested_queue_depth: 32,
            requested_buffer_bytes: 0x20_0000,
        });
        assert_eq!(decode_req(&encode_req(&request)), Some(request));
        let response = BlockResponse::Opened {
            channels: 1,
            limit: 1,
        };
        assert_eq!(decode_rep(&encode_rep(&response, 2)), Some((response, 2)));
        assert!(decode_rep(&encode_rep(&response, 1)).is_none());
    }
}
