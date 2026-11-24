//! **TCP** — the one protocol this stack borrows whole.
//!
//! `res/INET_STACK.md` §5.1's split, at its sharpest: ARP, IPv4, ICMP, UDP,
//! DHCP and DNS are written in this crate over borrowed *encodings*; TCP is
//! the vendored implementation, driven through the [`SmolDevice`] adapter.
//! The reason is stated once and stands: the hard part of TCP is not the
//! header, it is RTO estimation, slow start, fast retransmit, delayed ACK,
//! window scaling and TIME_WAIT — none of which fail a unit test, and all of
//! which fail at 3% loss on a real network.
//!
//! # The composition cost, stated plainly
//!
//! smoltcp's `Interface` does its **own** ARP and its own IPv4. It is not a
//! protosw row that can sit beside [`icmp`](super::icmp) inside
//! [`IpStack`](super::IpStack) and share its neighbour cache, because it does
//! not consume datagrams — it consumes *frames*, and an interface is the one
//! thing a link can only have one poller of.
//!
//! So this row is a **separate composition over the same seam**, not a row
//! inside the other engine. A vif is driven either by [`IpStack`] or by
//! [`TcpRow`], and the two keep separate neighbour caches. That duplication is
//! real, and it is the honest shape of borrowing an implementation rather than
//! an encoding — §6.1 predicted it and this is it arriving.
//!
//! **What it costs today:** a client that wants both ICMP and TCP on one vif
//! cannot have them yet. **What fixes it:** the frame-level demux shim — a
//! device whose `receive` is fed by [`IpStack`]'s dispatch rather than by the
//! interface directly — which is a slice of its own and wants a consumer
//! before it is built.
//!
//! [`IpStack`]: super::IpStack

use kore_driver::net::flow::Ipv4Addr;
use smoltcp::iface::{Config, Interface, SocketHandle};
use smoltcp::socket::tcp;
use smoltcp::time::Instant;
use smoltcp::wire::{EthernetAddress, HardwareAddress, IpAddress, IpCidr};

use crate::ifnet::Ifnet;
use crate::netinet::to_ip;
use crate::row::Identity;
use crate::smol::{SMOL_SCRATCH_BYTES, SmolDevice, SocketBudget};

pub use smoltcp::iface::SocketStorage;

/// What a connection is doing, in kore's vocabulary.
///
/// A narrowing of smoltcp's own state enum: a caller wants to know whether it
/// can write, whether the peer is still there, and whether the attempt
/// failed — not which of eleven RFC states it is passing through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcpState {
    /// Nothing attempted.
    Idle,
    /// A handshake is in flight.
    Connecting,
    /// Established; data may move.
    Open,
    /// The peer closed, or we did; drain what is left.
    Closing,
    /// Closed, refused, or reset.
    Closed,
}

/// Why a connection could not be started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcpError {
    /// The storage handed in was too small.
    Storage,
    /// The row has no route to that peer.
    Unreachable,
    /// The socket is already in use.
    Busy,
    /// The peer refused, or the connection was reset.
    Refused,
    /// This vif's socket budget is spent, so no further endpoint can be
    /// opened on it.
    ///
    /// The budget is the socket storage *this* row borrows
    /// ([`SocketBudget`]), so the refusal reaches the holder that asked and
    /// nobody else. Upstream's answer to the same condition is a `panic!`
    /// (`vendor/smoltcp/src/iface/socket_set.rs:83`), which in a service is a
    /// denial of service rather than a diagnostic.
    SocketsExhausted,
    /// The row's handle names no live socket.
    ///
    /// Unreachable while a row adds exactly one socket at construction and
    /// never retires it — refused rather than unwrapped so a rung that gives
    /// the row a second connection cannot reintroduce upstream's
    /// `panic!("handle does not refer to a valid socket")` under our name.
    NoSocket,
}

/// Borrowed storage one connection needs.
///
/// Heapless like everything else here: the caller supplies every byte, so the
/// row's footprint is visible at its construction site rather than hidden in
/// an allocator.
pub struct TcpStorage<'a> {
    /// Receive buffer.
    pub rx: &'a mut [u8],
    /// Transmit buffer.
    pub tx: &'a mut [u8],
    /// Socket table backing.
    pub sockets: &'a mut [SocketStorage<'a>],
    /// The adapter's receive staging, at least [`SMOL_SCRATCH_BYTES`].
    pub scratch: &'a mut [u8],
}

/// One TCP connection over one interface.
pub struct TcpRow<'a, I: Ifnet> {
    device: SmolDevice<'a, I>,
    iface: Interface,
    sockets: SocketBudget<'a>,
    handle: SocketHandle,
}

impl<'a, I: Ifnet> TcpRow<'a, I> {
    /// Bind a row to an interface and an identity.
    ///
    /// The identity's address, prefix and gateway are installed into the
    /// borrowed interface, so smoltcp routes with the same facts the row
    /// arbitrated rather than a second configuration.
    pub fn new(
        net: I,
        identity: &Identity,
        storage: TcpStorage<'a>,
        now_ms: u64,
    ) -> Result<Self, TcpError> {
        Self::with_seed(net, identity, storage, now_ms, 0)
    }

    pub(crate) fn with_seed(
        net: I,
        identity: &Identity,
        storage: TcpStorage<'a>,
        now_ms: u64,
        seed: u64,
    ) -> Result<Self, TcpError> {
        // An empty socket slice is a storage mistake rather than a spent
        // budget: the caller gave the row nowhere to put its one connection.
        // A slice with slots in it that are all taken is the budget, and
        // `try_add` below names that case separately.
        if storage.scratch.len() < SMOL_SCRATCH_BYTES || storage.sockets.is_empty() {
            return Err(TcpError::Storage);
        }
        let hw = net.hwaddr();
        let mut device = SmolDevice::new(net, storage.scratch).ok_or(TcpError::Storage)?;

        let mut config = Config::new(HardwareAddress::Ethernet(EthernetAddress(hw.0)));
        config.random_seed = seed;
        let mut iface = Interface::new(config, &mut device, Instant::from_millis(now_ms as i64));

        iface.update_ip_addrs(|addrs| {
            let _ = addrs.push(IpCidr::new(
                IpAddress::Ipv4(to_ip(identity.addr())),
                identity.prefix_len(),
            ));
        });
        if let Some(gw) = identity.gateway() {
            let _ = iface.routes_mut().add_default_ipv4_route(to_ip(gw));
        }

        let mut sockets = SocketBudget::new(storage.sockets);
        let socket = tcp::Socket::new(
            tcp::SocketBuffer::new(storage.rx),
            tcp::SocketBuffer::new(storage.tx),
        );
        // Refused, never forced: the guard above only rules this out for the
        // slice this row was handed, and the one call site that could panic
        // upstream is this one.
        let handle = sockets
            .try_add(socket)
            .map_err(|_| TcpError::SocketsExhausted)?;

        Ok(Self {
            device,
            iface,
            sockets,
            handle,
        })
    }

    /// The interface this row drives.
    pub fn ifnet(&self) -> &I {
        self.device.ifnet()
    }

    /// The interface, mutably.
    pub fn ifnet_mut(&mut self) -> &mut I {
        self.device.ifnet_mut()
    }

    /// This vif's socket budget — what it was given and what it is using.
    pub fn sockets(&self) -> &SocketBudget<'a> {
        &self.sockets
    }

    /// This row's socket, if its handle still names one.
    fn socket(&mut self) -> Option<&mut tcp::Socket<'a>> {
        self.sockets.try_get_mut::<tcp::Socket>(self.handle)
    }

    /// What the connection is doing.
    pub fn state(&mut self) -> TcpState {
        use tcp::State as S;
        // No socket is nothing open, which is what `Closed` already says.
        let Some(socket) = self.socket() else {
            return TcpState::Closed;
        };
        match socket.state() {
            S::Closed => TcpState::Closed,
            S::Listen => TcpState::Idle,
            S::SynSent | S::SynReceived => TcpState::Connecting,
            S::Established => TcpState::Open,
            S::FinWait1
            | S::FinWait2
            | S::CloseWait
            | S::Closing
            | S::LastAck
            | S::TimeWait => TcpState::Closing,
        }
    }

    /// Start a connection to `peer` from `local_port`.
    pub fn connect(
        &mut self,
        peer: (Ipv4Addr, u16),
        local_port: u16,
        now_ms: u64,
    ) -> Result<(), TcpError> {
        let remote = (IpAddress::Ipv4(to_ip(peer.0)), peer.1);
        let cx = self.iface.context();
        let socket = self
            .sockets
            .try_get_mut::<tcp::Socket>(self.handle)
            .ok_or(TcpError::NoSocket)?;
        socket
            .connect(cx, remote, local_port)
            .map_err(|e| match e {
                tcp::ConnectError::InvalidState => TcpError::Busy,
                tcp::ConnectError::Unaddressable => TcpError::Unreachable,
            })?;
        self.poll(now_ms);
        Ok(())
    }

    /// Listen on `local_port`.
    pub fn listen(&mut self, local_port: u16) -> Result<(), TcpError> {
        self.socket()
            .ok_or(TcpError::NoSocket)?
            .listen(local_port)
            .map_err(|_| TcpError::Busy)
    }

    /// Drive the connection: move whatever is ready, both directions.
    ///
    /// Returns whether anything changed. A caller runs this on its sweep, the
    /// same drive-cycle posture the rest of the stack uses.
    pub fn poll(&mut self, now_ms: u64) -> bool {
        // The budget drives the sweep rather than lending its set out: a
        // `&mut SocketSet` in any other hand is a reachable `add`, and that
        // it has exactly one caller is why the panic cannot be entered.
        self.sockets.poll(
            &mut self.iface,
            &mut self.device,
            Instant::from_millis(now_ms as i64),
        ) == smoltcp::iface::PollResult::SocketStateChanged
    }

    /// Queue `data` for transmission, returning how much was taken.
    pub fn send(&mut self, data: &[u8]) -> Result<usize, TcpError> {
        self.socket()
            .ok_or(TcpError::NoSocket)?
            .send_slice(data)
            .map_err(|_| TcpError::Refused)
    }

    /// Take received data, returning how much was read.
    pub fn recv(&mut self, out: &mut [u8]) -> Result<usize, TcpError> {
        match self.socket().ok_or(TcpError::NoSocket)?.recv_slice(out) {
            Ok(n) => Ok(n),
            Err(tcp::RecvError::Finished) => Ok(0),
            Err(tcp::RecvError::InvalidState) => Err(TcpError::Refused),
        }
    }

    /// The peer this connection is with, once there is one.
    ///
    /// `None` before a handshake completes — which is also how a listener
    /// reports that nobody has arrived yet, so a caller has one question to
    /// ask rather than two.
    pub fn remote(&mut self) -> Option<(Ipv4Addr, u16)> {
        let endpoint = self.socket()?.remote_endpoint()?;
        match endpoint.addr {
            IpAddress::Ipv4(addr) => Some((crate::netinet::from_ip(addr), endpoint.port)),
        }
    }

    /// Whether the peer has closed its half.
    ///
    /// The narrowing [`TcpState`] deliberately cannot answer: it collapses
    /// six RFC states into [`TcpState::Closing`], and "we sent FIN" and "both
    /// FINs were exchanged" are on either side of that line. A close that
    /// waits for the wrong one either returns before the peer heard it or
    /// never returns at all, so the question is asked precisely here rather
    /// than approximated above.
    pub fn peer_closed(&mut self) -> bool {
        use tcp::State as S;
        // No socket is a closed connection, and `Closed` is already on the
        // true side of this match — so a caller waiting on the peer's FIN
        // stops waiting rather than never returning.
        let Some(socket) = self.socket() else {
            return true;
        };
        matches!(
            socket.state(),
            S::CloseWait | S::LastAck | S::Closing | S::TimeWait | S::Closed
        )
    }

    /// Whether data may be written now.
    pub fn may_send(&mut self) -> bool {
        self.socket().is_some_and(|s| s.may_send())
    }

    /// Whether data is waiting.
    pub fn may_recv(&mut self) -> bool {
        self.socket().is_some_and(|s| s.may_recv())
    }

    /// Whether bytes handed to [`send`](Self::send) are still queued locally.
    ///
    /// The transmit buffer's occupancy, not the unacknowledged window: this
    /// answers "has what I wrote left this machine", which is the only half
    /// of the question a sender can be told. Whether the peer *read* it is
    /// not a fact TCP carries.
    pub fn send_queued(&mut self) -> bool {
        self.socket().is_some_and(|s| s.send_queue() > 0)
    }

    /// Whether a locally initiated FIN has been acknowledged, or the stream ended.
    pub(crate) fn write_finished(&mut self) -> bool {
        use tcp::State as S;
        self.socket()
            .is_none_or(|s| matches!(s.state(), S::FinWait2 | S::TimeWait | S::Closed))
    }

    /// Close this end of the connection.
    pub fn close(&mut self) {
        if let Some(socket) = self.socket() {
            socket.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
    use kore_driver::net::{EtherType, FrameHead, MacAddr};

    const MY_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const GW_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);
    const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
    const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
    const WAN: Ipv4Addr = Ipv4Addr([142, 250, 72, 46]);

    #[repr(align(64))]
    struct Aligned<const N: usize>([u8; N]);

    fn identity() -> Identity {
        Identity::new(ME, 24, Some(GW)).expect("valid")
    }

    /// Declare every buffer a row borrows as a local, then build the row over
    /// them.
    ///
    /// They have to be locals rather than fields of a fixture struct:
    /// `TcpStorage<'a>` ties the socket slice and its contents to one
    /// lifetime, so a struct holding them forces `'static` and nothing in a
    /// test is static.
    macro_rules! row {
        ($name:ident) => {
            let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
            let mut scratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
            let mut rx = [0u8; 1024];
            let mut tx = [0u8; 1024];
            let mut sockets = [SocketStorage::EMPTY; 1];
            let net = MockIfnet::new(&mut ifscratch.0, MY_MAC).expect("scratch");
            let mut $name = TcpRow::new(
                net,
                &identity(),
                TcpStorage {
                    rx: &mut rx,
                    tx: &mut tx,
                    sockets: &mut sockets,
                    scratch: &mut scratch.0,
                },
                0,
            )
            .expect("builds");
        };
    }

    /// Storage smaller than the adapter needs is refused at construction, not
    /// discovered when a frame arrives.
    #[test]
    fn undersized_storage_is_refused() {
        let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut rx = [0u8; 1024];
        let mut tx = [0u8; 1024];
        let mut sockets = [SocketStorage::EMPTY; 1];
        let mut tiny = [0u8; 8];
        let net = MockIfnet::new(&mut ifscratch.0, MY_MAC).expect("scratch");
        let built = TcpRow::new(
            net,
            &identity(),
            TcpStorage {
                rx: &mut rx,
                tx: &mut tx,
                sockets: &mut sockets,
                scratch: &mut tiny,
            },
            0,
        );
        assert!(matches!(built.err(), Some(TcpError::Storage)));
    }

    /// A fresh row has attempted nothing.
    #[test]
    fn a_fresh_row_is_closed() {
        row!(row);
        assert_eq!(row.state(), TcpState::Closed);
        assert!(!row.may_send());
    }

    /// **The B3 witness.** A connect to an off-link peer starts the exchange
    /// and puts a frame on the wire — the borrowed TCP driving kore's own
    /// interface through the adapter, which is what A1b was for.
    ///
    /// The *first* frame is an ARP request, not a SYN: smoltcp resolves the
    /// gateway before it can address the segment, exactly as this stack's own
    /// IP output does. Asserting that rather than a SYN is asserting what the
    /// protocol actually does.
    #[test]
    fn a_connect_reaches_the_wire() {
        row!(row);
        row.connect((WAN, 80), 49152, 0).expect("starts");
        assert_eq!(row.state(), TcpState::Connecting);

        let (sent, et) = row.ifnet().last_sent().expect("something went out");
        let head = FrameHead::parse(sent).expect("well formed");
        assert_eq!(
            et,
            EtherType::ARP,
            "the gateway is resolved before the segment can be addressed"
        );
        assert_eq!(head.dst, MacAddr::BROADCAST);
        assert_eq!(head.src, MY_MAC);
    }

    /// Once the gateway answers, the SYN itself reaches the wire — addressed
    /// to the gateway's station address, carrying the peer's IP.
    #[test]
    fn the_syn_follows_the_arp_reply() {
        use smoltcp::wire::{ArpOperation, ArpPacket, ArpRepr};

        row!(row);
        row.connect((WAN, 80), 49152, 0).expect("starts");

        let repr = ArpRepr::EthernetIpv4 {
            operation: ArpOperation::Reply,
            source_hardware_addr: EthernetAddress(GW_MAC.0),
            source_protocol_addr: to_ip(GW),
            target_hardware_addr: EthernetAddress(MY_MAC.0),
            target_protocol_addr: to_ip(ME),
        };
        let n = repr.buffer_len();
        let mut body = [0u8; 64];
        {
            let mut p = ArpPacket::new_unchecked(&mut body[..n]);
            repr.emit(&mut p);
        }
        let mut f = [0u8; 128];
        FrameHead {
            dst: MY_MAC,
            src: GW_MAC,
            ethertype: EtherType::ARP,
        }
        .write(&mut f)
        .expect("header fits");
        f[14..14 + n].copy_from_slice(&body[..n]);
        row.ifnet_mut()
            .inject(&f[..14 + n], EtherType::ARP)
            .expect("injects");

        for t in 1..8 {
            row.poll(t);
            if let Some((sent, EtherType::IPV4)) = row.ifnet().last_sent() {
                let head = FrameHead::parse(sent).expect("well formed");
                assert_eq!(
                    head.dst, GW_MAC,
                    "off-link traffic goes to the gateway's station address"
                );
                let ip = smoltcp::wire::Ipv4Packet::new_checked(&sent[14..]).expect("parses");
                assert_eq!(
                    crate::netinet::from_ip(ip.dst_addr()),
                    WAN,
                    "addressed to the peer, routed via the gateway"
                );
                assert_eq!(ip.next_header(), smoltcp::wire::IpProtocol::Tcp);
                return;
            }
        }
        panic!("no SYN reached the wire");
    }

    /// Listening is a different lifecycle than connecting, and the row
    /// reports it as such.
    #[test]
    fn a_listener_is_idle_until_someone_arrives() {
        row!(row);
        row.listen(8080).expect("listens");
        assert_eq!(row.state(), TcpState::Idle);
        assert!(!row.may_recv(), "nothing has arrived");
    }

    /// A row given nowhere to put a socket is refused at construction, with
    /// the storage mistake named as such rather than as a spent budget.
    #[test]
    fn a_row_with_no_socket_storage_is_refused() {
        let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut scratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
        let mut rx = [0u8; 1024];
        let mut tx = [0u8; 1024];
        let mut sockets: [SocketStorage; 0] = [];
        let net = MockIfnet::new(&mut ifscratch.0, MY_MAC).expect("scratch");
        let built = TcpRow::new(
            net,
            &identity(),
            TcpStorage {
                rx: &mut rx,
                tx: &mut tx,
                sockets: &mut sockets,
                scratch: &mut scratch.0,
            },
            0,
        );
        assert!(matches!(built.err(), Some(TcpError::Storage)));
    }

    /// **The budget is per-vif.** A row's sockets are the storage its own
    /// caller handed it, nothing can raise that at runtime, and there is no
    /// pool between rows — so one row sitting at its ceiling leaves the other
    /// with everything it was given. That is the property that keeps a full
    /// set from becoming one holder's lever on another.
    #[test]
    fn one_rows_spent_budget_leaves_another_rows_alone() {
        row!(a);
        row!(b);

        assert_eq!(a.sockets().budget(), 1);
        assert_eq!(a.sockets().live(), 1);
        assert!(
            a.sockets().is_full(),
            "a row sized for one connection is at its ceiling once it holds it"
        );

        a.connect((WAN, 80), 49152, 0).expect("a connects");
        b.connect((WAN, 443), 49153, 0)
            .expect("and b connects on its own budget, untouched by a's");
        assert_eq!(b.sockets().live(), 1);
    }

    /// A second connect on a socket already in use is refused rather than
    /// quietly replacing the first.
    #[test]
    fn a_socket_in_use_refuses_a_second_connect() {
        row!(row);
        row.connect((WAN, 80), 49152, 0).expect("starts");
        assert_eq!(
            row.connect((WAN, 443), 49153, 1).err(),
            Some(TcpError::Busy)
        );
    }
}
