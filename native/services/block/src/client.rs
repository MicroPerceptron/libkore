//! Typed client for the block control plane.
#![allow(async_fn_in_trait)]

use kore_rt::cap::{DeviceToken, SupplyToken, verbs};
use kore_rt::svc::SvcClient;
use kore_rt::transport::Transport;

use crate::{BlockRefusal, BlockRequest, BlockResponse, BlockService, ProvisionRequest};

pub struct OpenedChannel {
    pub device: DeviceToken,
    pub pool: SupplyToken,
}

pub trait BlockClient {
    async fn open(&mut self, request: ProvisionRequest) -> Result<OpenedChannel, BlockRefusal>;
    async fn status(&mut self) -> Result<(bool, u16, u16), BlockRefusal>;
    async fn close(&mut self) -> Result<(), BlockRefusal>;
}

pub struct RingBlock<'w, 't, T: Transport>(pub SvcClient<'w, 't, T, BlockService>);

impl<T: Transport> BlockClient for RingBlock<'_, '_, T> {
    async fn open(&mut self, request: ProvisionRequest) -> Result<OpenedChannel, BlockRefusal> {
        let mut reply = self
            .0
            .call(&BlockRequest::Open(request))
            .await
            .map_err(|_| BlockRefusal::Unavailable)?;
        match reply.response {
            BlockResponse::Opened { .. } => {}
            BlockResponse::Refused(refusal) => return Err(refusal),
            _ => return Err(BlockRefusal::Unavailable),
        }
        let device = reply.retained[0].take().ok_or(BlockRefusal::Unavailable)?;
        let pool = reply.retained[1].take().ok_or(BlockRefusal::Unavailable)?;
        let device = match verbs::claim_returned_device(device) {
            Ok(device) => device,
            Err(returned) => {
                reply.retained[0] = Some(returned);
                self.0.poison_retained(reply.retained);
                return Err(BlockRefusal::Unavailable);
            }
        };
        let pool = match verbs::claim_returned_supply(pool) {
            Ok(pool) => pool,
            Err(returned) => {
                reply.retained[1] = Some(returned);
                self.0.poison_retained(reply.retained);
                return Err(BlockRefusal::Unavailable);
            }
        };
        Ok(OpenedChannel { device, pool })
    }

    async fn status(&mut self) -> Result<(bool, u16, u16), BlockRefusal> {
        let reply = self
            .0
            .call(&BlockRequest::Status)
            .await
            .map_err(|_| BlockRefusal::Unavailable)?;
        match reply.response {
            BlockResponse::Status {
                available,
                channels,
                limit,
            } => Ok((available, channels, limit)),
            BlockResponse::Refused(refusal) => Err(refusal),
            _ => Err(BlockRefusal::Unavailable),
        }
    }

    async fn close(&mut self) -> Result<(), BlockRefusal> {
        let reply = self
            .0
            .call(&BlockRequest::Close)
            .await
            .map_err(|_| BlockRefusal::Unavailable)?;
        match reply.response {
            BlockResponse::Closed => Ok(()),
            BlockResponse::Refused(refusal) => Err(refusal),
            _ => Err(BlockRefusal::Unavailable),
        }
    }
}
