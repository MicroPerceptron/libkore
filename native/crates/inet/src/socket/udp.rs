//! **[`UdpSocket`]** — a datagram socket that is a view of one [`FlowCap`].
//!
//! # What is reused
//!
//! All of it. The demultiplex, the checksum, the ARP wait and the
//! reserve-forward output are [`IpStack`]'s, written once in
//! [`netinet`](crate::netinet) and exercised by B2's witnesses. This module
//! adds the socket *face* and the capability scoping, and no protocol
//! whatsoever.
//!
//! Note what is deliberately **not** reused: smoltcp's `socket-udp`. The
//! feature stays off for the reason `vendor/smoltcp/PATCHES.md` records —
//! the `proto-*` features gate encodings and the `socket-*` features gate
//! implementations, and kore takes the first and writes the second. Turning
//! `socket-udp` on would put a second UDP implementation beside the one
//! `netinet/udp.rs` owns and the DHCP client and resolver already ride. TCP
//! is the one deliberate exception (`res/INET_STACK.md` §6.1), and it is an
//! exception because the hard part of TCP is not the header.

use core::future::poll_fn;
use core::task::Poll;

use kore_driver::net::flow::{IpProto, Ipv4Addr};

use crate::flow::FlowCap;
use crate::netinet::{Event, IpStack, OutputError};

use super::{FlowFace, SocketRefusal, admits, local_of, peer_of};

/// A datagram socket scoped to one flow.
///
/// Its local endpoint *is* the one the authority pins, so there is no `bind`
/// that takes a port: binding happened at the mint, where the row could check
/// it. What is left here is to start delivering that port, which is a
/// bookkeeping step and not an authority one.
pub struct UdpSocket<F: FlowFace> {
    stack: IpStack<F>,
    local: (Ipv4Addr, u16),
    peer: Option<(Ipv4Addr, u16)>,
}

impl<F: FlowFace> UdpSocket<F> {
    /// Open a datagram socket over `flow`.
    ///
    /// Refuses rather than guessing when the authority is not socket-shaped:
    /// S0 grants nothing ([`SocketRefusal::Mediated`]), an unpinned rung has
    /// no endpoint ([`SocketRefusal::Unpinned`]), and an authority for some
    /// other protocol is not a UDP socket ([`SocketRefusal::OutsideFlow`]).
    pub fn open(flow: F) -> Result<Self, SocketRefusal> {
        let local = local_of(flow.cap())?;
        let peer = peer_of(flow.cap());
        // A connected authority is checked against itself here, which reads
        // like a tautology and is not: it is the one place that rejects a
        // capability minted for TCP being used to open a UDP socket, before
        // a datagram the classifier would never deliver the answer to has
        // been put on the wire.
        if let Some(p) = peer
            && !admits(flow.cap(), IpProto::UDP, local, p)
        {
            return Err(SocketRefusal::OutsideFlow);
        }
        let identity = *flow.identity();
        let mut stack = IpStack::new(flow, identity);
        // The engine's own two clients (DHCP, DNS) are already bound; this
        // adds the holder's. `false` means the binding table is full, which
        // is a composition bound rather than an authority question.
        if !stack.bind_udp(local.1) {
            return Err(SocketRefusal::Unpinned);
        }
        Ok(Self { stack, local, peer })
    }

    /// The endpoint this socket is.
    pub fn local_addr(&self) -> (Ipv4Addr, u16) {
        self.local
    }

    /// The peer the authority pins, when it pins one.
    pub fn peer_addr(&self) -> Option<(Ipv4Addr, u16)> {
        self.peer
    }

    /// The authority this socket is a view of.
    pub fn cap(&self) -> &FlowCap {
        self.stack.ifnet().cap()
    }

    /// The Tier 1 flow underneath, for a composition that drives the sweep.
    pub fn flow(&self) -> &F {
        self.stack.ifnet()
    }

    /// The Tier 1 flow underneath, mutably.
    pub fn flow_mut(&mut self) -> &mut F {
        self.stack.ifnet_mut()
    }

    /// Drain the interface once, discarding what this socket is not waiting
    /// for.
    ///
    /// Called from every future here before the operation it is attempting,
    /// and that ordering is load-bearing: the ARP reply that unblocks a send
    /// arrives on the same interface, so a send that never drained input
    /// could never learn the translation it is waiting for.
    pub fn sweep(&mut self) -> bool {
        let now = self.stack.ifnet().now_ms();
        self.stack.input(now, |_| {})
    }

    /// Send `data` to `peer`.
    ///
    /// Awaits the ARP translation and a transmit slot; both are ordinary
    /// transients rather than failures, and the ARP layer's own rate limit is
    /// what stops the retry becoming a broadcast storm.
    pub async fn send_to(
        &mut self,
        data: &[u8],
        peer: (Ipv4Addr, u16),
    ) -> Result<usize, SocketRefusal> {
        // Checked once, before anything is attempted: a peer outside the
        // authority is a caller mistake, and retrying it would turn a
        // mistake into a loop.
        if !admits(self.cap(), IpProto::UDP, self.local, peer) {
            return Err(SocketRefusal::OutsideFlow);
        }
        let port = self.local.1;
        poll_fn(|cx| {
            self.sweep();
            let now = self.stack.ifnet().now_ms();
            match self.stack.send_udp(port, peer, data, now) {
                Ok(()) => Poll::Ready(Ok(data.len())),
                // Not failures: no translation yet, or no slot yet. The
                // waker is nudged because nothing else will — Tier 1 offers
                // no readiness signal (see the module docs).
                Err(OutputError::Pending | OutputError::NoSlot) => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Err(e) => Poll::Ready(Err(SocketRefusal::Output(e))),
            }
        })
        .await
    }

    /// Send `data` to the peer the authority pins.
    ///
    /// Refused when it pins none — a listening authority has nobody to
    /// answer until someone has spoken.
    pub async fn send(&mut self, data: &[u8]) -> Result<usize, SocketRefusal> {
        let peer = self.peer.ok_or(SocketRefusal::Unpinned)?;
        self.send_to(data, peer).await
    }

    /// Await one datagram, copying it into `out`.
    ///
    /// Returns the bytes written and who sent them. A datagram longer than
    /// `out` is **truncated**, which is UDP's own contract and the one every
    /// `recvfrom` shim has to reproduce; the alternative — refusing it —
    /// would leave the caller with no way to learn how big a buffer to bring.
    pub async fn recv_from(
        &mut self,
        out: &mut [u8],
    ) -> Result<(usize, (Ipv4Addr, u16)), SocketRefusal> {
        let port = self.local.1;
        poll_fn(|cx| {
            let now = self.stack.ifnet().now_ms();
            let mut got = None;
            self.stack.input(now, |e| {
                if let Event::Udp {
                    from,
                    local_port,
                    payload,
                } = e
                    && local_port == port
                {
                    let n = payload.len().min(out.len());
                    out[..n].copy_from_slice(&payload[..n]);
                    got = Some((n, from));
                }
            });
            match got {
                Some(v) => Poll::Ready(Ok(v)),
                None => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        })
        .await
    }

    /// Await one datagram from the peer the authority pins.
    ///
    /// The filter the row installed already keeps anyone else's datagrams off
    /// this interface, so this is a convenience over
    /// [`recv_from`](Self::recv_from) and not a second check — stated plainly
    /// so nobody later removes the filter believing this stands in for it.
    pub async fn recv(&mut self, out: &mut [u8]) -> Result<usize, SocketRefusal> {
        self.peer.ok_or(SocketRefusal::Unpinned)?;
        let (n, _) = self.recv_from(out).await?;
        Ok(n)
    }
}
