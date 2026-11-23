//! The C host's one network request at a time. Thread zero holds the two
//! process faces and owns the request until the vif has been closed.

use kore_inet::netinet::ping::{Heard, Ping};
use kore_inet::{Event, Identity, IpStack, VifIfnet};
use kore_std::abi::clock::Uptime;
use kore_std::runtime::cap::verbs;
use kore_std::runtime::entry::Rt;
use kore_std::runtime::transport::TrapTransport;
use kore_std::runtime::{clock, executor::sleep_for};
use kore_svc_mm::{MmClient, client::ProcessMm};
use kore_svc_net::client::{ProcessNet, WINDOW_RIGHTS, attach_process_face, format_window};
use kore_svc_net::window::WINDOW_BYTES;
use kore_svc_net::{EtherType, EtherTypeMatch, FrameFilter, NetClient, NetRequest, NetResponse};

use crate::api::KoreNetStatus as Status;

pub(crate) struct Host<'t> {
    network: Option<ProcessNet<'t>>,
    memory: Option<ProcessMm>,
}

impl<'t> Host<'t> {
    pub(crate) fn new(rt: &Rt, transport: &'t TrapTransport) -> Self {
        let network = attach_process_face(rt, transport);
        let memory = network
            .as_ref()
            .and_then(|_| kore_svc_mm::attach_process_face(rt));
        Self { network, memory }
    }

    pub(crate) async fn ping(&mut self, address: [u8; 4], timeout_ms: u32) -> Result<u64, Status> {
        let network = self.network.as_mut().ok_or(Status::Unavailable)?;
        let memory = self.memory.as_mut().ok_or(Status::Unavailable)?;
        let client = &mut network.client;
        let NetResponse::Identity {
            identity: Some(report),
        } = client.request(NetRequest::InetQuery).await
        else {
            return Err(Status::NoAddress);
        };
        let station = Identity::new(report.addr, report.prefix_len, report.gateway)
            .map_err(|_| Status::NoAddress)?;
        let (_, lease) = memory.client.supply(WINDOW_BYTES as u64).await;
        let lease = lease.ok_or(Status::Resource)?;
        let slot = lease.slot().ok_or(Status::Resource)?;
        let view = verbs::map_owned_lease_rw(slot).ok_or(Status::Resource)?;
        // SAFETY: this client owns the lease, and the service adopts only a
        // sublease after formatting. Both sides observe the window protocol.
        let window =
            unsafe { core::slice::from_raw_parts_mut(view.as_mut_ptr().cast(), WINDOW_BYTES) };
        format_window(window).map_err(|_| Status::Resource)?;
        let mut filter = FrameFilter::protocol(EtherType::IPV4);
        filter.ethertype =
            EtherTypeMatch::only(&[EtherType::IPV4, EtherType::ARP]).ok_or(Status::Resource)?;
        let opened = client
            .open_vif_staged(filter, |stage| {
                stage.sublease(&lease, WINDOW_RIGHTS, 0, WINDOW_BYTES as u64)
            })
            .await;
        let (vif, mac) = match opened {
            Ok(NetResponse::Opened { vif, mac }) => (vif, mac),
            Ok(_) => return Err(Status::Resource),
            Err(_) => {
                // Transfer may have moved. Leave its backing to process
                // teardown and refuse later requests on this face.
                core::mem::forget(view);
                core::mem::forget(lease);
                self.network = None;
                return Err(Status::Indeterminate);
            }
        };
        let result = match VifIfnet::attach(window, mac) {
            Ok(ifnet) => {
                let mut stack = IpStack::new(ifnet, station);
                let timeout_us = u64::from(timeout_ms) * 1_000;
                let start = now_us();
                let mut ping = Ping::new(
                    kore_std::driver::net::flow::Ipv4Addr(address),
                    0x6b43,
                    timeout_us,
                    timeout_us,
                    Some(1),
                );
                loop {
                    let (Some(start), Some(now)) = (start, now_us()) else {
                        break Err(Status::Unavailable);
                    };
                    if now.saturating_sub(start) >= timeout_us {
                        break Err(Status::Timeout);
                    }
                    if ping.send_due(&mut stack, now).is_err() {
                        break Err(Status::Unreachable);
                    }
                    let mut reply = None;
                    stack.input(now / 1_000, |event| {
                        if let Event::Icmp(icmp) = event {
                            match ping.absorb(icmp, now) {
                                Some(Heard::Reply(echo)) => reply = Some(Ok(echo.rtt_us)),
                                Some(Heard::Unreachable { .. }) if reply.is_none() => {
                                    reply = Some(Err(Status::Unreachable));
                                }
                                _ => {}
                            }
                        }
                    });
                    if stack.ifnet_mut().take_tx_kick() {
                        client.kick().await;
                    }
                    if let Some(reply) = reply {
                        break reply;
                    }
                    sleep_for(200_000).await;
                }
            }
            Err(_) => Err(Status::Resource),
        };
        if !matches!(client.close_vif(vif).await, NetResponse::Closed { vif: closed } if closed == vif)
        {
            core::mem::forget(view);
            core::mem::forget(lease);
            self.network = None;
            return Err(Status::Indeterminate);
        }
        result
    }
}

fn now_us() -> Option<u64> {
    clock::uptime()
        .ok()
        .map(Uptime::as_nanos)
        .filter(|ns| *ns != 0)
        .map(|ns| ns / 1_000)
}
