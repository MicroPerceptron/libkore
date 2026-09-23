//! Bounded SPSC retirement. These drains preserve cursor continuity; they do
//! not reinitialize shared memory while its native serving owner is alive.
use crate::HostLinkWire;
use kore_rt::svc::{ClientRings, ServeRings};

pub const BOOTSTRAP_AGENT_GENERATION: u64 = 1;

/// Stable structural pair for an agent slot. Pair three belongs to workspace.
pub const fn agent_pair_index(slot: u8) -> Option<usize> {
    if slot as usize >= crate::MAX_DYNAMIC_AGENT_ROUTES {
        None
    } else {
        Some(match slot {
            0 => 1,
            1 => 2,
            _ => slot as usize + 3,
        })
    }
}

pub(crate) fn successor_after_close(
    closed: Option<kore_hostlink::TaskGeneration>,
    next: kore_hostlink::TaskGeneration,
) -> bool {
    closed.and_then(|generation| generation.get().checked_add(1)) == Some(next.get())
}

/// Service-side drain after the broker has stopped the client and the route
/// owner has acknowledged closure. No reply is manufactured for abandoned
/// requests. False means the pair cannot be handed to a successor.
pub fn drain_retired_requests(rings: &mut ServeRings<'_, HostLinkWire>) -> bool {
    let count = rings.req.outstanding();
    if count > rings.req.capacity() {
        return false;
    }
    for _ in 0..count {
        if rings.req.pop().is_none() {
            return false;
        }
    }
    rings.req.outstanding() == 0
}

/// Parent-side drain after confirmed worker reap and the service close ACK.
/// The caller must be the only client consumer. A service-side close ACK
/// proves no new old-generation reply can race this drain.
pub fn drain_retired_replies(rings: &mut ClientRings<'_, HostLinkWire>) -> bool {
    let count = rings.rep.outstanding();
    if count > rings.rep.capacity() {
        return false;
    }
    for _ in 0..count {
        if rings.rep.pop().is_none() {
            return false;
        }
    }
    rings.rep.outstanding() == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ClientRequest, ClientResponse, HOSTLINK_REQUEST_CAPACITY, HOSTLINK_RESPONSE_CAPACITY,
    };
    use kore_rt::svc::{SvcProtocol, split_window};

    #[repr(align(64))]
    struct Window([u8; 4096]);

    #[test]
    fn retirement_drains_both_directions_without_resetting_live_ring_storage() {
        let mut window = Window([0; 4096]);
        let (mut client, mut service) = split_window::<HostLinkWire>(
            &mut window.0,
            HOSTLINK_REQUEST_CAPACITY,
            HOSTLINK_RESPONSE_CAPACITY,
        )
        .unwrap();
        for _ in 0..HOSTLINK_REQUEST_CAPACITY {
            client
                .req
                .push(HostLinkWire::encode_req(&ClientRequest::Poll))
                .unwrap();
        }
        for _ in 0..HOSTLINK_RESPONSE_CAPACITY {
            service
                .rep
                .push(HostLinkWire::encode_rep(&ClientResponse::Empty, 0))
                .unwrap();
        }
        assert!(drain_retired_requests(&mut service));
        assert_eq!(client.rep.outstanding(), HOSTLINK_RESPONSE_CAPACITY);
        assert!(drain_retired_replies(&mut client));
        assert!(drain_retired_requests(&mut service));
        assert!(drain_retired_replies(&mut client));
        // Existing producer/consumer owners remain attached; a successor's
        // records are exactly the first records visible after retirement.
        client
            .req
            .push(HostLinkWire::encode_req(&ClientRequest::Status))
            .unwrap();
        assert!(matches!(
            HostLinkWire::decode_req(&service.req.pop().unwrap()),
            Some(ClientRequest::Status)
        ));
        assert!(service.req.pop().is_none());
        let request = kore_hostlink::RequestId([9; 16]);
        service
            .rep
            .push(HostLinkWire::encode_rep(
                &ClientResponse::Accepted { request },
                0,
            ))
            .unwrap();
        assert!(
            matches!(HostLinkWire::decode_rep(&client.rep.pop().unwrap()), Some((ClientResponse::Accepted { request: id }, 0)) if id == request)
        );
        assert!(client.rep.pop().is_none());
    }

    #[test]
    fn successor_requires_the_exact_acknowledged_predecessor_without_wrap() {
        let generation = |value| kore_hostlink::TaskGeneration::new(value).unwrap();
        assert!(!successor_after_close(None, generation(2)));
        assert!(!successor_after_close(Some(generation(1)), generation(1)));
        assert!(!successor_after_close(Some(generation(1)), generation(3)));
        assert!(successor_after_close(Some(generation(1)), generation(2)));
        assert!(!successor_after_close(
            Some(generation(u64::MAX)),
            generation(1)
        ));
    }

    #[test]
    fn all_agent_pairs_are_distinct_and_skip_broker_and_workspace() {
        let mut seen = [false; crate::MAX_DYNAMIC_AGENT_ROUTES + 3];
        for slot in 0..crate::MAX_DYNAMIC_AGENT_ROUTES {
            let pair = agent_pair_index(slot as u8).unwrap();
            assert!(pair != 0 && pair != 3 && pair != crate::HOSTLINK_OWNER_PAIR);
            assert!(!seen[pair]);
            seen[pair] = true;
        }
        assert_eq!(
            agent_pair_index(crate::MAX_DYNAMIC_AGENT_ROUTES as u8),
            None
        );
        assert_eq!(agent_pair_index(u8::MAX), None);
    }
}
