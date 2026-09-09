//! Generic, bounded control vocabulary for a recyclable session acceptor.
//!
//! This is deliberately content-neutral. A founded owner uses pair zero as
//! its administration face, conveys one authenticated bundle component per
//! request, and the ordinary [`ServeHost`](super::ServeHost) installs the
//! completed pair. Dynamically admitted peers can only probe their structural
//! arrival identity or close themselves; wire data never selects that
//! identity.

use kore_abi::{
    session::SlotRef,
    task::{Bootstrap, Profile},
};

use crate::service::{BindError, Bundle, PeerClose, ReturnMode, Serve, SystemService};

use super::{
    PeerAdmissionRefusal, PeerAssemblyRefusal, PeerBundleAssembler, PeerBundlePart, PeerId,
    PeerMapPlan, SvcProtocol,
};

const MAGIC: u64 = 0x52 << 56;
const OP_PART: u64 = 1;
const OP_PROBE: u64 = 2;
const OP_CLOSE: u64 = 3;
const REP_ACCEPTED: u64 = 1;
const REP_PROBE: u64 = 2;
const REP_CLOSING: u64 = 3;
const REP_REFUSED: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecyclableRequest {
    Convey {
        part: PeerBundlePart,
        announced: SlotRef,
    },
    Probe {
        nonce: u32,
    },
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecyclableResponse {
    Accepted,
    Probed { nonce: u32, via: u16 },
    Closing,
    Refused(PeerAssemblyRefusal),
}

#[derive(Clone, Copy, Debug)]
pub struct NoRecyclableEndowment;

impl Bundle for NoRecyclableEndowment {
    const PROFILE: Profile = Profile::new("recyclable-session", &[]);

    fn resolve(_: &Bootstrap<'_>) -> Result<Self, BindError> {
        Ok(Self)
    }
}

/// A reusable acceptor/control service used by production owners that need a
/// generic recyclable-session face without importing domain policy.
pub struct RecyclableSessionAcceptor {
    assembler: PeerBundleAssembler,
    map: PeerMapPlan,
    last_admission: Option<Result<PeerId, PeerAdmissionRefusal>>,
    admissions: u64,
    admission_refusals: u64,
    probes: u64,
    last_probe_via: u16,
    probe_vias: [u16; super::MAX_SERVE_PAIRS],
    probe_via_count: usize,
    closes: u64,
    last_close: Option<(u16, PeerClose)>,
}

impl RecyclableSessionAcceptor {
    pub const fn new(map: PeerMapPlan) -> Self {
        Self {
            assembler: PeerBundleAssembler::new(0, true),
            map,
            last_admission: None,
            admissions: 0,
            admission_refusals: 0,
            probes: 0,
            last_probe_via: 0,
            probe_vias: [0; super::MAX_SERVE_PAIRS],
            probe_via_count: 0,
            closes: 0,
            last_close: None,
        }
    }

    pub fn take_admission(&mut self) -> Option<Result<PeerId, PeerAdmissionRefusal>> {
        self.last_admission.take()
    }

    pub const fn admissions(&self) -> u64 {
        self.admissions
    }

    pub const fn admission_refusals(&self) -> u64 {
        self.admission_refusals
    }

    pub const fn probes(&self) -> u64 {
        self.probes
    }

    pub const fn last_probe_via(&self) -> u16 {
        self.last_probe_via
    }

    pub fn saw_probe_from(&self, via: u16) -> bool {
        self.probe_vias[..self.probe_via_count].contains(&via)
    }

    pub const fn closes(&self) -> u64 {
        self.closes
    }

    pub const fn last_close(&self) -> Option<(u16, PeerClose)> {
        self.last_close
    }
}

impl SystemService for RecyclableSessionAcceptor {
    const NAME: &'static str = "recyclable-session";
    const RETURN_MODE: ReturnMode = ReturnMode::RetainedAny;
    type Endowment = NoRecyclableEndowment;

    fn bind(_: Self::Endowment) -> Self {
        Self::new(PeerMapPlan { base: 0, stride: 0 })
    }

    fn serve(&mut self, request: Self::Request, sv: &mut Serve<'_, Self::Response>) {
        match request {
            RecyclableRequest::Convey { part, announced } => {
                // Only the founder-wired administration face may change the
                // fan-in. A dynamic peer cannot convey a second peer merely
                // by naming the public correlation value.
                if sv.via() != 0 {
                    sv.reply(RecyclableResponse::Refused(
                        PeerAssemblyRefusal::WrongAuthority,
                    ));
                    return;
                }
                let Some(mut input) = sv.returned_input(announced) else {
                    sv.reply(RecyclableResponse::Refused(
                        PeerAssemblyRefusal::WrongAuthority,
                    ));
                    return;
                };
                let Some(capability) = input.capability() else {
                    sv.reply(RecyclableResponse::Refused(
                        PeerAssemblyRefusal::WrongAuthority,
                    ));
                    return;
                };
                if let Err(reason) = self.assembler.accepts(part, capability) {
                    sv.reply(RecyclableResponse::Refused(reason));
                    return;
                }
                let capability = input
                    .take()
                    .expect("validated returned owner remains present");
                let bundle = self
                    .assembler
                    .push(part, capability)
                    .expect("same synchronous validation cannot change");
                if let Some(bundle) = bundle {
                    sv.admit_peer(bundle, self.map);
                }
                sv.reply(RecyclableResponse::Accepted);
            }
            RecyclableRequest::Probe { nonce } => {
                self.probes = self.probes.saturating_add(1);
                self.last_probe_via = sv.via();
                if !self.saw_probe_from(sv.via()) && self.probe_via_count < self.probe_vias.len() {
                    self.probe_vias[self.probe_via_count] = sv.via();
                    self.probe_via_count += 1;
                }
                sv.reply(RecyclableResponse::Probed {
                    nonce,
                    via: sv.via(),
                });
            }
            RecyclableRequest::Close => {
                if sv.via() == 0 {
                    sv.reply(RecyclableResponse::Refused(
                        PeerAssemblyRefusal::WrongAuthority,
                    ));
                } else {
                    sv.reply(RecyclableResponse::Closing);
                    sv.close_peer(PeerClose::Orderly);
                }
            }
        }
    }

    fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        if result.is_ok() {
            self.admissions = self.admissions.saturating_add(1);
        } else {
            self.admission_refusals = self.admission_refusals.saturating_add(1);
        }
        self.last_admission = Some(result);
        true
    }

    fn peer_closed(&mut self, via: u16, reason: PeerClose) -> bool {
        self.closes = self.closes.saturating_add(1);
        self.last_close = Some((via, reason));
        true
    }
}

fn part_code(part: PeerBundlePart) -> u8 {
    match part {
        PeerBundlePart::Ring => 0,
        PeerBundlePart::Kick => 1,
        PeerBundlePart::ReturnRecv => 2,
        PeerBundlePart::Grant(index) => 0x80 | index,
    }
}

fn decode_part(code: u8) -> Option<PeerBundlePart> {
    Some(match code {
        0 => PeerBundlePart::Ring,
        1 => PeerBundlePart::Kick,
        2 => PeerBundlePart::ReturnRecv,
        value if value & 0x80 != 0 => PeerBundlePart::Grant(value & 0x7f),
        _ => return None,
    })
}

fn refusal_code(reason: PeerAssemblyRefusal) -> u8 {
    match reason {
        PeerAssemblyRefusal::Duplicate => 1,
        PeerAssemblyRefusal::GrantOutOfRange => 2,
        PeerAssemblyRefusal::UnexpectedReturn => 3,
        PeerAssemblyRefusal::WrongAuthority => 4,
    }
}

fn decode_refusal(code: u8) -> Option<PeerAssemblyRefusal> {
    Some(match code {
        1 => PeerAssemblyRefusal::Duplicate,
        2 => PeerAssemblyRefusal::GrantOutOfRange,
        3 => PeerAssemblyRefusal::UnexpectedReturn,
        4 => PeerAssemblyRefusal::WrongAuthority,
        _ => return None,
    })
}

impl SvcProtocol for RecyclableSessionAcceptor {
    type Request = RecyclableRequest;
    type Response = RecyclableResponse;
    type ReqWire = u64;
    type RepWire = u64;
    const REPLY_GRANTS: usize = 0;

    fn encode_req(request: &RecyclableRequest) -> u64 {
        MAGIC
            | match *request {
                RecyclableRequest::Convey { part, announced } => {
                    OP_PART
                        | (u64::from(part_code(part)) << 8)
                        | (u64::from(announced.index()) << 16)
                }
                RecyclableRequest::Probe { nonce } => OP_PROBE | (u64::from(nonce) << 8),
                RecyclableRequest::Close => OP_CLOSE,
            }
    }

    fn decode_req(wire: &u64) -> Option<RecyclableRequest> {
        if wire & 0xff00_0000_0000_0000 != MAGIC {
            return None;
        }
        Some(match wire & 0xff {
            OP_PART => RecyclableRequest::Convey {
                part: decode_part(((wire >> 8) & 0xff) as u8)?,
                announced: SlotRef::from_wire(((wire >> 16) & 0xffff_ffff) as u32),
            },
            OP_PROBE => RecyclableRequest::Probe {
                nonce: ((wire >> 8) & 0xffff_ffff) as u32,
            },
            OP_CLOSE if wire & 0x00ff_ffff_ffff_ff00 == 0 => RecyclableRequest::Close,
            _ => return None,
        })
    }

    fn encode_rep(response: &RecyclableResponse, grants: u8) -> u64 {
        debug_assert_eq!(grants, 0);
        MAGIC
            | match *response {
                RecyclableResponse::Accepted => REP_ACCEPTED,
                RecyclableResponse::Probed { nonce, via } => {
                    REP_PROBE | (u64::from(nonce) << 8) | (u64::from(via) << 40)
                }
                RecyclableResponse::Closing => REP_CLOSING,
                RecyclableResponse::Refused(reason) => {
                    REP_REFUSED | (u64::from(refusal_code(reason)) << 8)
                }
            }
    }

    fn decode_rep(wire: &u64) -> Option<(RecyclableResponse, u8)> {
        if wire & 0xff00_0000_0000_0000 != MAGIC {
            return None;
        }
        let response = match wire & 0xff {
            REP_ACCEPTED if wire & 0x00ff_ffff_ffff_ff00 == 0 => RecyclableResponse::Accepted,
            REP_PROBE => RecyclableResponse::Probed {
                nonce: ((wire >> 8) & 0xffff_ffff) as u32,
                via: ((wire >> 40) & 0xffff) as u16,
            },
            REP_CLOSING if wire & 0x00ff_ffff_ffff_ff00 == 0 => RecyclableResponse::Closing,
            REP_REFUSED if wire & 0x00ff_ffff_ffff_0000 == 0 => {
                RecyclableResponse::Refused(decode_refusal(((wire >> 8) & 0xff) as u8)?)
            }
            _ => return None,
        };
        Some((response, 0))
    }

    fn reply_matches(request: &u64, response: &RecyclableResponse, grants: u8) -> bool {
        if grants != 0 {
            return false;
        }
        match (Self::decode_req(request), response) {
            (Some(RecyclableRequest::Convey { .. }), RecyclableResponse::Accepted)
            | (Some(RecyclableRequest::Convey { .. }), RecyclableResponse::Refused(_))
            | (Some(RecyclableRequest::Close), RecyclableResponse::Closing)
            | (Some(RecyclableRequest::Close), RecyclableResponse::Refused(_)) => true,
            (
                Some(RecyclableRequest::Probe { nonce: requested }),
                RecyclableResponse::Probed { nonce, .. },
            ) => requested == *nonce,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_round_trip_and_request_correlation() {
        for request in [
            RecyclableRequest::Convey {
                part: PeerBundlePart::Ring,
                announced: SlotRef::from_wire(77),
            },
            RecyclableRequest::Convey {
                part: PeerBundlePart::Grant(3),
                announced: SlotRef::from_wire(88),
            },
            RecyclableRequest::Probe { nonce: 0x1234_5678 },
            RecyclableRequest::Close,
        ] {
            let wire = RecyclableSessionAcceptor::encode_req(&request);
            assert_eq!(RecyclableSessionAcceptor::decode_req(&wire), Some(request));
        }
        let request = RecyclableSessionAcceptor::encode_req(&RecyclableRequest::Probe { nonce: 9 });
        assert!(RecyclableSessionAcceptor::reply_matches(
            &request,
            &RecyclableResponse::Probed { nonce: 9, via: 12 },
            0,
        ));
        assert!(!RecyclableSessionAcceptor::reply_matches(
            &request,
            &RecyclableResponse::Probed { nonce: 8, via: 12 },
            0,
        ));
    }
}
