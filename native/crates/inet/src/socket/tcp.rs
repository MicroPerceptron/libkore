//! **[`TcpStream`] and [`TcpListener`]** — a stream socket that is a view of
//! one [`FlowCap`].
//!
//! # What is reused
//!
//! The TCP state machine is smoltcp's, driven through [`TcpRow`], which is
//! driven through [`SmolDevice`](crate::smol::SmolDevice) over the Tier 1
//! flow. Not a line of retransmission, congestion control, window management
//! or reassembly is written here, and that is the point
//! (`res/INET_STACK.md` §5.1): the hard part of TCP is not the header, it is
//! everything that only fails at 3% loss on a real network.
//!
//! What this module adds is the kore-shaped, capability-scoped face: the
//! endpoints come from the authority, a peer outside it is refused by name,
//! and the lifecycle is async in the runtime's own level-triggered idiom.
//!
//! # Start and wait are separate, deliberately
//!
//! [`TcpStream::open`] puts the SYN in flight and returns; [`ready`] awaits
//! the handshake. [`connect`] is the two composed, and is what §6.2's sketch
//! shows. Keeping the halves reachable is not a convenience — it is what lets
//! a caller that owns the sweep (a test with a second station, a service
//! multiplexing several sockets) drive the wire between polls. A future that
//! owned the flow for the whole handshake would make the peer unreachable for
//! exactly as long as the peer is what it is waiting for.
//!
//! [`ready`]: TcpStream::ready
//! [`connect`]: TcpStream::connect

use core::future::poll_fn;
use core::task::Poll;

use kore_driver::net::flow::{IpProto, Ipv4Addr};

use crate::flow::FlowCap;
use crate::netinet::tcp::TcpError;
use crate::netinet::{TcpRow, TcpState, TcpStorage};

use super::{FlowFace, SocketRefusal, admits, local_of, peer_of};

/// A stream socket scoped to one flow.
///
/// Its local endpoint *is* the one the authority pins. Storage is borrowed
/// from the caller, as everywhere else in this crate: the receive and
/// transmit windows a connection needs are visible at the construction site
/// rather than hidden in an allocator that `crates/inet` does not have.
/// Whether a connection has settled enough to hand back, and how.
///
/// `Some(Ok)` is established, `Some(Err)` is settled-and-not-established,
/// and `None` means keep waiting. Both [`TcpStream::ready`] and
/// [`TcpListener::accept`] ask this rather than matching the state
/// themselves, because when they matched separately they disagreed — and the
/// disagreement was invisible, since neither path is reachable from a host
/// test without a whole `FlowFace`.
///
/// **`Closing` settles, it does not wait.** A peer that answers the SYN and
/// immediately FINs leaves a socket that will never be `Open`: nothing
/// afterwards moves a closing socket back, so a future that waits for one
/// self-wakes and stays pending for ever — a hung `connect` on the client
/// side and a wedged accept loop on the server's.
pub(crate) const fn establishment(state: TcpState) -> Option<Result<(), SocketRefusal>> {
    match state {
        TcpState::Open => Some(Ok(())),
        TcpState::Closed | TcpState::Closing => Some(Err(SocketRefusal::Reset)),
        TcpState::Idle | TcpState::Connecting => None,
    }
}

pub struct TcpStream<'s, F: FlowFace> {
    row: TcpRow<'s, F>,
    local: (Ipv4Addr, u16),
    peer: Option<(Ipv4Addr, u16)>,
}

impl<'s, F: FlowFace> TcpStream<'s, F> {
    /// Start a connection to `peer`. Does not wait.
    ///
    /// The SYN — or, first, the ARP request that has to precede it — is in
    /// flight when this returns. [`ready`](Self::ready) is what awaits the
    /// handshake.
    pub fn open(
        flow: F,
        storage: TcpStorage<'s>,
        peer: (Ipv4Addr, u16),
    ) -> Result<Self, SocketRefusal> {
        Self::open_with_seed(flow, storage, peer, 0)
    }

    /// Start a connection with a fresh seed from the caller's entropy source.
    pub fn open_with_seed(
        flow: F,
        storage: TcpStorage<'s>,
        peer: (Ipv4Addr, u16),
        seed: u64,
    ) -> Result<Self, SocketRefusal> {
        let local = local_of(flow.cap())?;
        if !admits(flow.cap(), IpProto::TCP, local, peer) {
            return Err(SocketRefusal::OutsideFlow);
        }
        let identity = *flow.identity();
        let now = flow.now_ms();
        let mut row = TcpRow::with_seed(flow, &identity, storage, now, seed)?;
        row.connect(peer, local.1, now)?;
        Ok(Self {
            row,
            local,
            peer: Some(peer),
        })
    }

    /// Start a connection and await the handshake.
    ///
    /// `res/INET_STACK.md` §6.2's sketch, with the one addition the heapless
    /// discipline forces: the caller brings the buffers.
    pub async fn connect(
        flow: F,
        storage: TcpStorage<'s>,
        peer: (Ipv4Addr, u16),
    ) -> Result<Self, SocketRefusal> {
        let mut stream = Self::open(flow, storage, peer)?;
        stream.ready().await?;
        Ok(stream)
    }

    /// The endpoint this socket is.
    pub fn local_addr(&self) -> (Ipv4Addr, u16) {
        self.local
    }

    /// The peer, once there is one.
    pub fn peer_addr(&self) -> Option<(Ipv4Addr, u16)> {
        self.peer
    }

    /// The authority this socket is a view of.
    pub fn cap(&self) -> &FlowCap {
        self.row.ifnet().cap()
    }

    /// The Tier 1 flow underneath.
    pub fn flow(&self) -> &F {
        self.row.ifnet()
    }

    /// The Tier 1 flow underneath, mutably.
    pub fn flow_mut(&mut self) -> &mut F {
        self.row.ifnet_mut()
    }

    /// What the connection is doing.
    pub fn state(&mut self) -> TcpState {
        self.row.state()
    }

    /// Drive the connection one turn: move whatever is ready, both
    /// directions.
    ///
    /// Every future here calls this on each poll, because Tier 1 offers no
    /// readiness signal and nothing else is going to.
    pub fn sweep(&mut self) -> bool {
        let now = self.row.ifnet().now_ms();
        self.row.poll(now)
    }

    /// Await the handshake.
    ///
    /// Ready the moment the connection is established; [`Closed`] before that
    /// is the peer refusing or resetting, which is reported as
    /// [`SocketRefusal::Reset`] rather than as a successful connection to
    /// nothing.
    ///
    /// [`Closed`]: TcpState::Closed
    pub async fn ready(&mut self) -> Result<(), SocketRefusal> {
        poll_fn(|cx| {
            self.sweep();
            if self.row.may_send() {
                return Poll::Ready(Ok(()));
            }
            match establishment(self.row.state()) {
                Some(verdict) => Poll::Ready(verdict),
                None => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        })
        .await
    }

    /// Wait for transmit room and queue one prefix of `data`.
    ///
    /// A pending poll has queued none of the caller's bytes, so a caller can
    /// cancel the wait before retrying. On success, only the returned prefix
    /// was accepted.
    pub async fn write(&mut self, data: &[u8]) -> Result<usize, SocketRefusal> {
        if data.is_empty() {
            return Ok(0);
        }
        poll_fn(|cx| {
            self.sweep();
            if self.row.state() == TcpState::Closed {
                return Poll::Ready(Err(SocketRefusal::Reset));
            }
            if !self.row.may_send() {
                return Poll::Ready(Err(SocketRefusal::Closed));
            }
            match self.row.send(data) {
                Ok(0) => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Ok(n) => {
                    self.sweep();
                    Poll::Ready(Ok(n))
                }
                Err(error) => Poll::Ready(Err(error.into())),
            }
        })
        .await
    }

    /// Write every byte of `data`, awaiting window as needed.
    ///
    /// Short writes are absorbed here rather than reported: a caller that had
    /// to loop on a partial write would be reimplementing this, and the one
    /// thing it could do differently — give up halfway — leaves a stream
    /// carrying a prefix of a message, which is worse than waiting.
    ///
    /// **Not cancel-safe**, and this is the one future here that is not: the
    /// count of bytes already queued lives in the future, so a caller that
    /// drops it part-way and calls again re-sends from the start. Everything
    /// else in this module re-reads the socket on every poll and therefore
    /// survives being dropped between turns. A write that fits the transmit
    /// window completes in a single poll, which is why the hazard is narrow —
    /// but it is real above the window and is named rather than left to be
    /// discovered as duplicated bytes.
    pub async fn write_all(&mut self, data: &[u8]) -> Result<(), SocketRefusal> {
        let mut sent = 0usize;
        poll_fn(|cx| {
            self.sweep();
            if sent == data.len() {
                return Poll::Ready(Ok(()));
            }
            if !self.row.may_send() {
                // The peer closed, or we did. The bytes still unsent will
                // never move, and saying so beats spinning on a half-closed
                // connection.
                return Poll::Ready(Err(SocketRefusal::Closed));
            }
            match self.row.send(&data[sent..]) {
                Ok(n) => {
                    sent += n;
                    if sent == data.len() {
                        // Queued in full. One more sweep is what puts it on
                        // the wire, and then there is nothing left to await —
                        // so it is done here rather than in a second poll the
                        // caller would have to come back for.
                        self.sweep();
                        return Poll::Ready(Ok(()));
                    }
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Err(e) => Poll::Ready(Err(SocketRefusal::Tcp(e))),
            }
        })
        .await
    }

    /// Await the transmit buffer draining into the network.
    ///
    /// Not the same as the peer having *read* the bytes — TCP cannot report
    /// that and neither will this. It is the point at which nothing of this
    /// write is still waiting on a local window.
    pub async fn flush(&mut self) -> Result<(), SocketRefusal> {
        poll_fn(|cx| {
            self.sweep();
            if !self.row.may_send() || !self.row.send_queued() {
                return Poll::Ready(Ok(()));
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        })
        .await
    }

    /// Await some bytes, copying them into `out`.
    ///
    /// `Ok(0)` is end of stream — the peer closed its half and nothing is
    /// left buffered. That is the POSIX answer and the one a shim over this
    /// has to reproduce; distinguishing it from "nothing yet" is what the
    /// await is for.
    pub async fn read(&mut self, out: &mut [u8]) -> Result<usize, SocketRefusal> {
        if out.is_empty() {
            return Ok(0);
        }
        poll_fn(|cx| {
            self.sweep();
            let receiving = self.row.may_recv();
            match self.row.recv(out) {
                Ok(0) if receiving => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Ok(n) => {
                    self.sweep();
                    Poll::Ready(Ok(n))
                }
                Err(TcpError::Refused) if self.row.state() == TcpState::Connecting => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Err(TcpError::Refused) => Poll::Ready(Err(SocketRefusal::Reset)),
                Err(error) => Poll::Ready(Err(error.into())),
            }
        })
        .await
    }

    /// Send FIN after queued data and wait for its acknowledgement.
    /// The caller may continue reading until the peer sends its own FIN.
    pub async fn finish_write(&mut self) -> Result<(), SocketRefusal> {
        self.row.close();
        poll_fn(|cx| {
            self.sweep();
            if self.row.write_finished() {
                Poll::Ready(Ok(()))
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await
    }

    /// Close this end and await the peer closing its own.
    ///
    /// Waits for **FIN both ways**, which is the thing a caller actually
    /// wants to know and the thing [`TcpState`] cannot say: it collapses six
    /// RFC states into [`Closing`], and the line between "our FIN is out" and
    /// "both were exchanged" runs through the middle of it. TIME_WAIT counts
    /// as closed here — no byte will move again, and holding a caller for two
    /// maximum segment lifetimes to tell it so would be theatre.
    ///
    /// [`Closing`]: TcpState::Closing
    pub async fn close(&mut self) -> Result<(), SocketRefusal> {
        self.row.close();
        poll_fn(|cx| {
            self.sweep();
            if self.row.peer_closed() {
                return Poll::Ready(Ok(()));
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        })
        .await
    }
}

/// A listening socket scoped to one flow.
///
/// # Why it yields exactly one connection
///
/// A vif carries exactly one [`FrameFilter`](kore_driver::net::FrameFilter),
/// and Tier 1 hands a socket exactly one vif. So the authority that admits an
/// inbound connection is the listener's own — peer end
/// [`Any`](kore_driver::net::flow::EndpointMatch::Any) — and there is no
/// second vif on which an accepted connection's narrower flow could be
/// enforced. `InetRow::open_flow` says the same thing from the other side:
/// a second flow on a bound vif is [`VifAlreadyBound`].
///
/// A `accept()` that returned a fresh independently-enforced socket would
/// therefore be lying about the isolation it provides. This one does not: the
/// listener *becomes* the connection, and a server wanting concurrency opens
/// a second vif. That is a Tier 1 limitation, recorded rather than papered
/// over.
///
/// [`VifAlreadyBound`]: crate::row::OpenRefusal::VifAlreadyBound
pub struct TcpListener<'s, F: FlowFace> {
    stream: TcpStream<'s, F>,
}

impl<'s, F: FlowFace> TcpListener<'s, F> {
    /// Listen on the endpoint the authority pins.
    ///
    /// No port argument, for the same reason [`UdpSocket`] has none: the port
    /// was bound at the mint, where the row could arbitrate it. Passing one
    /// here would be a second, unchecked opinion.
    ///
    /// [`UdpSocket`]: super::UdpSocket
    pub fn bind(flow: F, storage: TcpStorage<'s>) -> Result<Self, SocketRefusal> {
        let local = local_of(flow.cap())?;
        // A connected authority is not a listening one. Refused rather than
        // widened: listening on a peer-pinned flow would accept from exactly
        // one peer, which is a connection wearing a listener's name.
        if peer_of(flow.cap()).is_some() {
            return Err(SocketRefusal::OutsideFlow);
        }
        let identity = *flow.identity();
        let now = flow.now_ms();
        let mut row = TcpRow::new(flow, &identity, storage, now)?;
        row.listen(local.1)?;
        Ok(Self {
            stream: TcpStream {
                row,
                local,
                peer: None,
            },
        })
    }

    /// The endpoint this listener is.
    pub fn local_addr(&self) -> (Ipv4Addr, u16) {
        self.stream.local
    }

    /// The authority this listener is a view of.
    pub fn cap(&self) -> &FlowCap {
        self.stream.cap()
    }

    /// The Tier 1 flow underneath, mutably.
    pub fn flow_mut(&mut self) -> &mut F {
        self.stream.flow_mut()
    }

    /// Drive the listener one turn.
    pub fn sweep(&mut self) -> bool {
        self.stream.sweep()
    }

    /// Await an inbound connection, answering with the peer it came from.
    ///
    /// The peer is checked against the authority even though the classifier
    /// already was: the filter is the boundary, and this is the listener
    /// declining to serve a connection its own authority does not describe.
    /// On a correctly wired row the check never fires, and on a misconfigured
    /// one it is the difference between a refusal and a silent service.
    pub async fn accept(&mut self) -> Result<(Ipv4Addr, u16), SocketRefusal> {
        let local = self.stream.local;
        let peer = poll_fn(|cx| {
            self.stream.sweep();
            match establishment(self.stream.row.state()) {
                Some(Ok(())) => match self.stream.row.remote() {
                    Some(peer) => Poll::Ready(Ok(peer)),
                    // Established with no tuple is not a state smoltcp
                    // produces; reported rather than unwrapped, because a
                    // panic in a server's accept loop is the worst available
                    // answer.
                    None => Poll::Ready(Err(SocketRefusal::Reset)),
                },
                Some(Err(refusal)) => Poll::Ready(Err(refusal)),
                None => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        })
        .await?;
        if !admits(self.stream.cap(), IpProto::TCP, local, peer) {
            return Err(SocketRefusal::OutsideFlow);
        }
        self.stream.peer = Some(peer);
        Ok(peer)
    }

    /// The accepted connection.
    ///
    /// Borrowed rather than handed over, because there is only one and the
    /// listener is it.
    pub fn stream(&mut self) -> &mut TcpStream<'s, F> {
        &mut self.stream
    }

    /// The accepted connection, consuming the listener.
    pub fn into_stream(self) -> TcpStream<'s, F> {
        self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An established connection is handed back.
    #[test]
    fn open_settles_as_established() {
        assert_eq!(establishment(TcpState::Open), Some(Ok(())));
    }

    /// The two terminal states settle as a refusal — **`Closing` included**.
    ///
    /// This is the regression. `Closing` sat in the waiting arm, so a peer
    /// that FINned before the next poll left `ready` and `accept` self-waking
    /// for ever. Nothing moves a closing socket back to `Open`, so waiting on
    /// one is waiting on something that cannot happen.
    #[test]
    fn closing_settles_as_a_refusal_rather_than_waiting() {
        assert_eq!(
            establishment(TcpState::Closing),
            Some(Err(SocketRefusal::Reset))
        );
        assert_eq!(
            establishment(TcpState::Closed),
            Some(Err(SocketRefusal::Reset))
        );
    }

    /// The states that really are still in progress keep waiting.
    #[test]
    fn only_idle_and_connecting_wait() {
        assert_eq!(establishment(TcpState::Idle), None);
        assert_eq!(establishment(TcpState::Connecting), None);
    }

    /// Every state is accounted for: none both waits and settles, and none
    /// is left out. `ready` and `accept` share this, so a state added later
    /// cannot be handled differently by the two of them.
    #[test]
    fn every_state_has_exactly_one_verdict() {
        for state in [
            TcpState::Idle,
            TcpState::Connecting,
            TcpState::Open,
            TcpState::Closing,
            TcpState::Closed,
        ] {
            let settled = establishment(state).is_some();
            let waits = establishment(state).is_none();
            assert!(settled ^ waits, "{state:?} is neither or both");
        }
    }
}
