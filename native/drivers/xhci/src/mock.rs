//! **`MockXhciDevice`** — a fake xHC in process memory, the host-proof for
//! the whole driver core (the `MockNvmeDevice` lesson: prove semantics on the
//! host, prove the seam on hardware).
//!
//! It is a [`Regs`] register file with **side effects**: writing `USBCMD`
//! drives the halt/reset/run transitions the bring-up waits on; ringing
//! doorbell 0 makes it walk the command ring (honoring the Link TRB + cycle
//! bit) and post Command Completion events into the interrupter-0 event ring
//! with the correct cycle bit — exactly what a controller does, so the core's
//! rings, doorbell discipline, and completion correlation are all exercised.
//!
//! DMA is process memory: [`MockPool`] hands out page-aligned leaked
//! allocations with `iova == cpu pointer`, so the mock dereferences a ring's
//! IOVA directly.

use core::cell::RefCell;

use kore_abi::device::IovaAddr;
use kore_driver::seam::{Dma, DmaPool, Regs, Regs64};

// Register-block offsets the mock lays out (arbitrary but self-consistent;
// the driver reads them back from CAPLENGTH/RTSOFF/DBOFF).
const OP: usize = 0x40;
const RT: usize = 0x600;
const DB: usize = 0x800;
/// Register file: 0x1000 bytes = 1024 u32 (covers CAP/op/port/runtime/db).
const REGS_U32: usize = 0x400;
/// Ring depth — must match the controller's `RING_TRBS`.
const RING_TRBS: u16 = 256;

// TRB Type discriminants (xHCI 1.2 Table 6-91) the mock decodes/produces.
const TYPE_LINK: u8 = 6;
const TYPE_ENABLE_SLOT: u8 = 9;
const TYPE_DISABLE_SLOT: u8 = 10;
const TYPE_NOOP_CMD: u8 = 23;
const TYPE_CMD_COMPLETION: u32 = 33;
/// Completion Code: Success.
const CC_SUCCESS: u8 = 1;

// Register offsets within the blocks (mirror `crate::regs`).
use crate::regs::{
    CONFIG, CRCR, CRCR_CA, CRCR_CRR, DCBAAP, ERDP, ERDP_EHB, ERSTBA, ERSTSZ, IMAN, IMAN_IE,
    IMAN_IP, INTR_STRIDE, USBCMD, USBCMD_HCRST, USBCMD_INTE, USBCMD_RS, USBSTS, USBSTS_EINT,
    USBSTS_HCH,
};
/// Interrupter register sets the mock models (matches its advertised
/// MaxIntrs) — enough to prove per-core event-ring routing.
const MOCK_INTRS: usize = 8;

// TRB Type discriminants on the transfer ring (xHCI Table 6-91).
const TYPE_NORMAL: u8 = 1;
const TYPE_SETUP: u8 = 2;
const TYPE_DATA: u8 = 3;
const TYPE_STATUS: u8 = 4;
const TYPE_ADDRESS_DEVICE: u8 = 11;
const TYPE_CONFIGURE_ENDPOINT: u8 = 12;
const TYPE_EVALUATE_CONTEXT: u8 = 13;
const TYPE_RESET_ENDPOINT: u8 = 14;
const TYPE_SET_TR_DEQUEUE: u8 = 16;
const TYPE_TRANSFER_EVENT: u32 = 32;
/// Full-speed Protocol Speed ID (PORTSC speed field).
const PSI_FULL: u32 = 1;
/// Protocol Speed ID 3 = High speed — what a 512-byte bulk endpoint needs.
const PSI_HIGH: u32 = 3;

// PORTSC bit positions the mock drives (xHCI 5.4.8).
const PORTSC_CCS: u32 = 1 << 0;
const PORTSC_PED: u32 = 1 << 1;
const PORTSC_PR: u32 = 1 << 4;
const PORTSC_SPEED_SHIFT: u32 = 10;
const PORTSC_CSC: u32 = 1 << 17;
const PORTSC_PRC: u32 = 1 << 21;
/// Every RW1C change bit (CSC, PEC, WRC, OCC, PRC, PLC, CEC).
const PORTSC_CHANGES: u32 =
    (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20) | (1 << 21) | (1 << 22) | (1 << 23);

/// What device (if any) sits on a mock root-hub port / slot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceKind {
    None,
    Keyboard,
    Mouse,
    /// A CDC Ethernet (ECM) network device — a control interface carrying the
    /// functional descriptors, and a data interface whose **altsetting 1**
    /// holds the bulk pair.
    Ecm,
}

#[derive(Clone, Copy)]
struct InterruptPipe {
    base: u64,
    dci: u8,
    deq: u16,
    ccs: bool,
    halted: bool,
    pending: Option<([u8; 8], usize)>,
}

impl InterruptPipe {
    const EMPTY: Self = Self {
        base: 0,
        dci: 0,
        deq: 0,
        ccs: true,
        halted: false,
        pending: None,
    };
}

/// The mock's per-slot device state, established at Address Device (EP0) and
/// Configure Endpoint (the interrupt IN endpoint).
#[derive(Clone, Copy)]
struct SlotDevice {
    kind: DeviceKind,
    port: u8,
    /// The EP0 transfer ring base (from the input context's EP0 dequeue ptr).
    ep0_base: u64,
    /// The mock's EP0 dequeue index + consumer cycle state.
    ep0_deq: u16,
    ep0_ccs: bool,
    ep0_mps: u16,
    ep0_halted: bool,
    ep0_needs_dequeue: bool,
    /// Per-DCI rings/buffers, not one global mouse/keyboard slot.
    interrupt: [InterruptPipe; crate::device::MAX_ENDPOINTS],
    /// The bulk IN endpoint's ring base + DCI (0 = not configured).
    bin_base: u64,
    bin_dci: u8,
    bin_deq: u16,
    bin_ccs: bool,
    /// The bulk OUT endpoint's ring base + DCI (0 = not configured).
    bout_base: u64,
    bout_dci: u8,
    bout_deq: u16,
    bout_ccs: bool,
}

impl SlotDevice {
    const EMPTY: Self = Self {
        kind: DeviceKind::None,
        port: 0,
        ep0_base: 0,
        ep0_deq: 0,
        ep0_ccs: true,
        ep0_mps: 0,
        ep0_halted: false,
        ep0_needs_dequeue: false,
        interrupt: [InterruptPipe::EMPTY; crate::device::MAX_ENDPOINTS],
        bin_base: 0,
        bin_dci: 0,
        bin_deq: 0,
        bin_ccs: true,
        bout_base: 0,
        bout_dci: 0,
        bout_deq: 0,
        bout_ccs: true,
    };
}

/// The MAC the mock's ECM device publishes, as the twelve ASCII hex digits a
/// CDC Ethernet string descriptor carries.
pub const ECM_MAC: [u8; 6] = [0x40, 0x01, 0x02, 0x03, 0x04, 0x05];
/// The string index the Ethernet Networking functional descriptor names.
const ECM_MAC_STRING_INDEX: u8 = 3;
/// Largest frame the mock's emulated wire carries (Ethernet II, `ETH_FRAME_LEN`).
pub const MOCK_FRAME_MAX: usize = 1514;
/// `bConfigurationValue` of the CDC-ECM configuration.
pub const ECM_CONFIG_VALUE: u8 = 1;
/// `bConfigurationValue` of the RNDIS configuration.
pub const ECM_RNDIS_CONFIG_VALUE: u8 = 2;
/// Descriptor **index** of the CDC-ECM configuration — deliberately not its
/// value, which is 1.
pub const ECM_CONFIG_INDEX: u8 = 1;

/// The mock's CDC-ECM device descriptor (18 B) — **two** configurations, the
/// NetChip IDs QEMU's `usb-net` reports.
const ECM_DEVICE_DESC: [u8; 18] = [
    18, 1, 0x00, 0x02, 0x02, 0x00, 0x00, 64, 0x25, 0x05, 0xa2, 0xa4, 0, 0, 1, 2, 3, 2,
];

/// The mock's **RNDIS** configuration (48 B) — configuration *index 0*, whose
/// `bConfigurationValue` is **2**.
///
/// This exists to be rejected, and its position is the whole point.  QEMU's
/// `usb-net` really does list RNDIS first and CDC-ECM second, with the
/// values crossed over — so a driver that takes "the first configuration"
/// gets RNDIS, and one that assumes "configuration 1 is at index 0" reads
/// the wrong descriptor entirely.  A mock with only the ECM config would let
/// both bugs pass.
const ECM_RNDIS_CONFIG_DESC: [u8; 48] = [
    9,
    2,
    48,
    0,
    2,
    ECM_RNDIS_CONFIG_VALUE,
    0,
    0xC0,
    50, // configuration, value 2
    9,
    4,
    0,
    0,
    1,
    0x02,
    0x02,
    0xFF,
    0, // interface 0 — Comms / ACM / vendor (RNDIS)
    7,
    5,
    0x81,
    0x03,
    0x10,
    0,
    9, //       EP1 IN, interrupt
    9,
    4,
    1,
    0,
    2,
    0x0A,
    0x00,
    0x00,
    0, // interface 1 — CDC Data, bulk on alt 0
    7,
    5,
    0x82,
    0x02,
    0x40,
    0,
    0, //       EP2 IN, bulk, 64 B
    7,
    5,
    0x02,
    0x02,
    0x40,
    0,
    0, //       EP2 OUT, bulk, 64 B
];

/// The mock's **CDC-ECM** configuration (80 B) — configuration *index 1*,
/// whose `bConfigurationValue` is **1**.
///
/// Deliberately the awkward shape a real device has, because it is the shape
/// that catches bring-up bugs: the functional descriptors sit between the
/// control interface and its notification endpoint, the data interface's
/// **altsetting 0 carries no endpoints at all**, and the bulk pair only
/// appears on altsetting 1.  Endpoint addresses and the 64-byte bulk packet
/// size are QEMU's.
const ECM_CONFIG_DESC: [u8; 80] = [
    9,
    2,
    80,
    0,
    2,
    ECM_CONFIG_VALUE,
    0,
    0xC0,
    50, // configuration, value 1
    9,
    4,
    0,
    0,
    1,
    0x02,
    0x06,
    0x00,
    0, // interface 0 alt 0 — Comms / ECM
    5,
    0x24,
    0x00,
    0x10,
    0x01, //          CS_INTERFACE / Header
    5,
    0x24,
    0x06,
    0x00,
    0x01, //          CS_INTERFACE / Union: master 0, slave 1
    13,
    0x24,
    0x0F,
    ECM_MAC_STRING_INDEX,
    0,
    0,
    0,
    0,
    0xEA,
    0x05,
    0,
    0,
    0, //                                  CS_INTERFACE / Ethernet Networking
    7,
    5,
    0x81,
    0x03,
    0x10,
    0,
    9, //       EP1 IN, interrupt — notifications, 16 B
    9,
    4,
    1,
    0,
    0,
    0x0A,
    0x00,
    0x00,
    0, // interface 1 alt 0 — no endpoints
    9,
    4,
    1,
    1,
    2,
    0x0A,
    0x00,
    0x00,
    0, // interface 1 alt 1 — the bulk pair
    7,
    5,
    0x82,
    0x02,
    0x40,
    0,
    0, //       EP2 IN, bulk, 64 B
    7,
    5,
    0x02,
    0x02,
    0x40,
    0,
    0, //       EP2 OUT, bulk, 64 B
];

/// The MAC string descriptor (26 B): twelve ASCII hex digits, UTF-16LE.
const ECM_MAC_STRING: [u8; 26] = [
    26, 3, b'4', 0, b'0', 0, b'0', 0, b'1', 0, b'0', 0, b'2', 0, b'0', 0, b'3', 0, b'0', 0, b'4',
    0, b'0', 0, b'5', 0,
];

/// The same descriptor with one digit that is not hex — the shape of a real
/// dongle whose MAC string does not decode.
///
/// The driver refuses rather than substituting: a made-up address would put a
/// station on the wire under an identity nobody chose.  That refusal is a
/// *bring-up* failure on a port that really does carry an Ethernet function,
/// which is a different fact from an empty port, and this is what lets a test
/// prove the two are reported differently.
const ECM_MAC_STRING_BAD: [u8; 26] = [
    26, 3, b'4', 0, b'0', 0, b'0', 0, b'1', 0, b'0', 0, b'2', 0, b'0', 0, b'3', 0, b'0', 0, b'4',
    0, b'0', 0, b'Z', 0,
];

// ── Canned boot-device descriptors (the bytes the mock returns over EP0) ──

/// A boot keyboard's device descriptor (18 B): QEMU-shaped IDs, MPS0 = 8.
const KBD_DEVICE_DESC: [u8; 18] = [
    18, 1, 0x00, 0x02, 0, 0, 0, 8, 0x27, 0x06, 0x01, 0x00, 0, 0, 0, 0, 0, 1,
];
/// A boot keyboard's configuration blob (34 B): config + interface
/// (3/1/1 = HID/boot/keyboard) + HID descriptor (skipped) + interrupt IN EP.
const KBD_CONFIG_DESC: [u8; 34] = [
    9, 2, 34, 0, 1, 1, 0, 0xA0, 50, // configuration
    9, 4, 0, 0, 1, 3, 1, 1, 0, //       interface (HID, boot, keyboard)
    9, 0x21, 0x11, 0x01, 0, 1, 0x22, 63, 0, // HID descriptor (skipped)
    7, 5, 0x81, 0x03, 8, 0, 10, //          endpoint 1 IN, interrupt, 8 B
];
/// A boot mouse's device descriptor (18 B).
const MOUSE_DEVICE_DESC: [u8; 18] = [
    18, 1, 0x00, 0x02, 0, 0, 0, 8, 0x27, 0x06, 0x02, 0x00, 0, 0, 0, 0, 0, 1,
];
/// A boot mouse's configuration blob (34 B): interface protocol 2 = mouse.
const MOUSE_CONFIG_DESC: [u8; 34] = [
    9, 2, 34, 0, 1, 1, 0, 0xA0, 50, // configuration
    9, 4, 0, 0, 1, 3, 1, 2, 0, //       interface (HID, boot, mouse)
    9, 0x21, 0x11, 0x01, 0, 1, 0x22, 52, 0, // HID descriptor (skipped)
    7, 5, 0x81, 0x03, 4, 0, 10, //          endpoint 1 IN, interrupt, 4 B
];

struct State {
    regs: [u32; REGS_U32],
    /// Command ring: the mock's dequeue index + consumer cycle state.
    cmd_deq: u16,
    cmd_ccs: bool,
    /// Event ring: the mock's enqueue index + producer cycle state.
    /// Per-interrupter event-ring enqueue index + producer cycle state.
    event_enq: [u16; MOCK_INTRS],
    event_pcs: [bool; MOCK_INTRS],
    /// Next slot id handed out by Enable Slot.
    next_slot: u8,
    /// Per-slot device state (index by slot id; slot 0 unused).
    slots: [SlotDevice; 33],
    /// When set, doorbells are ignored (no completions posted) — the
    /// abort-on-timeout proof.
    swallow: bool,
    /// Per-interrupter acknowledgement step: 0=pending/unstarted,
    /// 1=USBSTS.EINT cleared, 2=IMAN.IP cleared.
    ack_step: [u8; MOCK_INTRS],
    ack_count: [u32; MOCK_INTRS],
    bad_ack_order: bool,
    /// `DCCTRL` offset of an emulated Debug Capability (0 = none): `DCR`
    /// tracks `DCE` there unless `dbc_sticky` models an engine that stays.
    dbc_ctrl: usize,
    dbc_sticky: bool,
    /// Frames queued for the bulk IN endpoint, oldest first.
    rx: alloc::collections::VecDeque<alloc::vec::Vec<u8>>,
    /// Frames the mock captured off the bulk OUT endpoint, in order.
    tx: alloc::vec::Vec<alloc::vec::Vec<u8>>,
    /// When set, a frame written to bulk OUT is queued straight back onto the
    /// bulk IN side — the emulated wire looped on itself, which is how a
    /// round trip is proved without a second device.
    loopback: bool,
    /// When set, the ECM device's MAC string descriptor comes back
    /// undecodable ([`ECM_MAC_STRING_BAD`]) — a connected Ethernet function
    /// that refuses partway up its bring-up ladder.
    bad_mac: bool,
    /// Class requests seen on EP0, as `(bRequest, wValue, wIndex)` — so a
    /// test can assert a device really was told to change packet filter or
    /// alternate setting, rather than assuming the call reached the wire.
    class_requests: alloc::vec::Vec<(u8, u16, u16)>,
    /// The OUT data stage of each class request, as `(wIndex, bytes)` —
    /// what a device was actually told, not just that it was asked.
    class_payloads: alloc::vec::Vec<(u16, alloc::vec::Vec<u8>)>,
    /// While set, EP0 doorbells are noted but not processed: a control
    /// transfer the device has not answered yet (a keyboard NAKing its
    /// output report), released with [`MockXhciDevice::release_ep0`].
    hold_ep0: bool,
    /// Opt-in descriptor/class-request fixtures. Static slices and a single
    /// exact request match keep regressions bounded, not a USB bus simulator.
    hid_config: Option<&'static [u8]>,
    hid_request_error: Option<(u8, u8, u16, u8)>, // port, request, interface, CC
    hid_mps0: u8,
    device_prefix_len: usize,
    ep0_evaluations: u32,
    ep0_recoveries: u32,
    command_failure: Option<(u8, u8)>,
}

/// A fake xHC controller behind the [`Regs`] seam.
pub struct MockXhciDevice {
    st: RefCell<State>,
}

impl MockXhciDevice {
    /// A controller advertising 32 slots, 4 ports, 8 interrupters, 32-byte
    /// contexts, 2 scratchpad buffers, no extended capabilities.
    pub fn new() -> Self {
        let mut regs = [0u32; REGS_U32];
        // CAPLENGTH = OP, HCIVERSION 1.10 in the high half.
        regs[0x00 / 4] = 0x0110_0000 | (OP as u32);
        // HCSPARAMS1: MaxSlots 32, MaxIntrs 8, MaxPorts 4.
        regs[0x04 / 4] = (4 << 24) | (8 << 8) | 32;
        // HCSPARAMS2: ERST Max 4; Max Scratchpad Lo = 2.
        regs[0x08 / 4] = (4 << 4) | (2 << 27);
        // HCCPARAMS1: AC64 = 1, CSZ = 0 (32-byte contexts), no xECP.
        regs[0x10 / 4] = 1;
        // DBOFF / RTSOFF.
        regs[0x14 / 4] = DB as u32;
        regs[0x18 / 4] = RT as u32;
        // PAGESIZE: 4 KiB pages supported (bit 0).
        regs[(OP + 0x08) / 4] = 1;
        // USBSTS: halted (HCH) at power-on, CNR clear.
        regs[(OP + USBSTS) / 4] = USBSTS_HCH;

        // Root ports: port 1 has a keyboard, port 2 a mouse — both connected,
        // Full speed, connect-change latched, not yet enabled (a reset
        // enables them, as a real USB2 port).
        let connected = PORTSC_CCS | PORTSC_CSC | (PSI_FULL << PORTSC_SPEED_SHIFT);
        regs[(OP + 0x400) / 4] = connected; // port 1
        regs[(OP + 0x410) / 4] = connected; // port 2

        Self {
            st: RefCell::new(State {
                regs,
                cmd_deq: 0,
                cmd_ccs: true,
                event_enq: [0; MOCK_INTRS],
                event_pcs: [true; MOCK_INTRS],
                next_slot: 0,
                slots: [SlotDevice::EMPTY; 33],
                swallow: false,
                ack_step: [0; MOCK_INTRS],
                ack_count: [0; MOCK_INTRS],
                rx: alloc::collections::VecDeque::new(),
                tx: alloc::vec::Vec::new(),
                loopback: false,
                bad_mac: false,
                class_requests: alloc::vec::Vec::new(),
                class_payloads: alloc::vec::Vec::new(),
                hold_ep0: false,
                hid_config: None,
                hid_request_error: None,
                hid_mps0: 8,
                device_prefix_len: 8,
                ep0_evaluations: 0,
                ep0_recoveries: 0,
                command_failure: None,
                bad_ack_order: false,
                dbc_ctrl: 0,
                dbc_sticky: false,
            }),
        }
    }

    /// A controller that ignores doorbells — bring-up still works, but a
    /// command never completes (the abort path proof).
    pub fn new_swallowing() -> Self {
        let d = Self::new();
        d.st.borrow_mut().swallow = true;
        d
    }

    /// A controller advertising **64-byte contexts** (HCCPARAMS1.CSZ = 1 — the
    /// layout many real Intel parts use), for the CSZ=1 enumeration proof.
    pub fn new_csz64() -> Self {
        let d = Self::new();
        d.st.borrow_mut().regs[0x10 / 4] |= 1 << 2;
        d
    }

    /// A single root device with caller-supplied, static HID descriptors.
    pub fn with_hid_config(config: &'static [u8]) -> Self {
        let this = Self::new();
        {
            let mut st = this.st.borrow_mut();
            st.hid_config = Some(config);
            st.regs[(OP + 0x410) / 4] = PSI_FULL << PORTSC_SPEED_SHIFT;
        }
        this
    }

    /// Fail exactly this class request; a STALL really halts EP0 until the
    /// driver resets it and republishes the next dequeue pointer.
    pub fn fail_hid_request(&self, port: u8, request: u8, interface: u16, code: u8) {
        self.st.borrow_mut().hid_request_error = Some((port, request, interface, code));
    }

    pub fn set_hid_mps0(&self, mps: u8) {
        self.st.borrow_mut().hid_mps0 = mps;
    }

    pub fn set_device_prefix_len(&self, length: usize) {
        self.st.borrow_mut().device_prefix_len = length.min(8);
    }

    pub fn ep0_evaluations(&self) -> u32 {
        self.st.borrow().ep0_evaluations
    }
    pub fn ep0_recoveries(&self) -> u32 {
        self.st.borrow().ep0_recoveries
    }

    pub fn fail_command(&self, command: u8, code: u8) {
        self.st.borrow_mut().command_failure = Some((command, code));
    }

    pub fn clear_command_failure(&self) {
        self.st.borrow_mut().command_failure = None;
    }

    /// The slot id most recently handed out by Enable Slot (test probe).
    pub fn last_slot(&self) -> u8 {
        self.st.borrow().next_slot
    }

    /// Whether `intr` has completed at least one full
    /// EINT→IP→ERDP/EHB acknowledgement sequence with both pending bits clear.
    pub fn interrupt_acknowledged(&self, intr: u16) -> bool {
        let st = self.st.borrow();
        let i = intr as usize;
        if i >= MOCK_INTRS || st.ack_count[i] == 0 || st.bad_ack_order {
            return false;
        }
        let iman = RT + 0x20 + i * INTR_STRIDE + IMAN;
        st.regs[(OP + USBSTS) / 4] & USBSTS_EINT == 0 && st.regs[iman / 4] & IMAN_IP == 0
    }

    /// Connect a CDC-ECM network device on root port 3.
    ///
    /// A separate constructor rather than a default, so every existing HID
    /// test keeps enumerating exactly the two ports it always did.
    pub fn new_with_ecm() -> Self {
        let this = Self::new();
        {
            let mut st = this.st.borrow_mut();
            // High speed: a 512-byte bulk max packet size is a high-speed
            // value, and bringing it up at full speed would be a lie the
            // driver could not detect.
            st.regs[(OP + 0x420) / 4] = PORTSC_CCS | PORTSC_CSC | (PSI_HIGH << PORTSC_SPEED_SHIFT);
        }
        this
    }

    /// Connect a CDC-ECM device whose MAC string descriptor does not decode.
    ///
    /// The port is populated exactly as [`new_with_ecm`](Self::new_with_ecm)
    /// populates it — this device really is an Ethernet function — and the
    /// bring-up ladder refuses at the MAC string.  That distinction is the
    /// whole point: an operator told "no adaptor" moves a cable, and this
    /// device is already in the right port.
    pub fn new_with_ecm_bad_mac() -> Self {
        let this = Self::new_with_ecm();
        this.st.borrow_mut().bad_mac = true;
        this
    }

    /// **Queue a frame** for the bulk IN endpoint.  Delivered as soon as the
    /// driver has a receive buffer posted; held otherwise, so a test may
    /// inject before or after arming.
    pub fn inject_frame(&self, slot: u8, frame: &[u8]) {
        // Say so at the source rather than truncating downstream: a test that
        // injects more than the wire carries is asking for behaviour this
        // double does not model, and a silent short frame would look like a
        // driver bug.
        assert!(
            frame.len() <= MOCK_FRAME_MAX,
            "injected frame is {} bytes; this wire carries at most {MOCK_FRAME_MAX}",
            frame.len()
        );
        let mut st = self.st.borrow_mut();
        st.rx.push_back(frame.to_vec());
        Self::deliver_bulk_in(&mut st, slot);
    }

    /// Frames the mock took off the bulk OUT endpoint, oldest first.
    pub fn transmitted(&self) -> alloc::vec::Vec<alloc::vec::Vec<u8>> {
        self.st.borrow().tx.clone()
    }

    /// Loop the emulated wire back on itself: a frame written to bulk OUT is
    /// queued straight onto the bulk IN side.  This is how a full transmit →
    /// receive round trip is proved without a second device.
    pub fn set_loopback(&self, on: bool) {
        self.st.borrow_mut().loopback = on;
    }

    /// Class requests seen on EP0, as `(bRequest, wValue, wIndex)`.
    pub fn class_requests(&self) -> alloc::vec::Vec<(u8, u16, u16)> {
        self.st.borrow().class_requests.clone()
    }

    /// The OUT data stages of the class requests seen on EP0, as
    /// `(wIndex, bytes)`, oldest first.
    pub fn class_payloads(&self) -> alloc::vec::Vec<(u16, alloc::vec::Vec<u8>)> {
        self.st.borrow().class_payloads.clone()
    }

    /// Stop answering control transfers: an EP0 doorbell leaves its TD
    /// staged, as a device NAKing every attempt would.
    pub fn hold_ep0(&self) {
        self.st.borrow_mut().hold_ep0 = true;
    }

    /// Answer control transfers again, starting with whatever `slot` left
    /// staged while held.
    pub fn release_ep0(&self, slot: u8) {
        let mut st = self.st.borrow_mut();
        st.hold_ep0 = false;
        Self::process_ep0(&mut st, slot);
    }

    /// **Inject an input report** on `slot`'s interrupt endpoint — the mock's
    /// stand-in for a keypress / mouse move.  Delivered immediately if the
    /// endpoint is already armed, else held until the next arm (so either
    /// order works).  Something QEMU cannot do unattended, which is exactly
    /// why the decode path is host-proven here.
    pub fn inject_report(&self, slot: u8, report: &[u8]) {
        let dci = self
            .st
            .borrow()
            .slots
            .get(usize::from(slot))
            .and_then(|device| device.interrupt.iter().find(|pipe| pipe.dci != 0))
            .map_or(0, |pipe| pipe.dci);
        self.inject_report_at(slot, dci, report);
    }

    /// Inject onto the named composite interface pipe, never a sibling.
    pub fn inject_report_at(&self, slot: u8, dci: u8, report: &[u8]) {
        let mut st = self.st.borrow_mut();
        if slot == 0 || slot as usize >= st.slots.len() {
            return;
        }
        let mut buf = [0u8; 8];
        let n = report.len().min(8);
        buf[..n].copy_from_slice(&report[..n]);
        let Some(pipe) = st.slots[slot as usize]
            .interrupt
            .iter_mut()
            .find(|pipe| pipe.dci == dci && dci != 0)
        else {
            return;
        };
        pipe.pending = Some((buf, n));
        Self::deliver_report(&mut st, slot, dci);
    }

    /// Finish the current interrupt TD with an error and halt only its pipe.
    pub fn fail_interrupt_at(&self, slot: u8, dci: u8, code: u8) {
        let mut st = self.st.borrow_mut();
        let pipe = st.slots[usize::from(slot)]
            .interrupt
            .iter_mut()
            .find(|pipe| pipe.dci == dci)
            .expect("configured pipe");
        pipe.halted = true;
        let trb = pipe.base + u64::from(pipe.deq) * 16;
        let mut event = Self::transfer_event(trb, slot, dci, 0, st.event_pcs[0]);
        event[2] = u32::from(code) << 24;
        Self::post_event(&mut st, 0, event);
    }

    /// A delayed/stale completion fixture; no DMA or dequeue is performed.
    pub fn inject_completion_at(&self, slot: u8, dci: u8, trb: u64, code: u8, residual: u32) {
        let mut st = self.st.borrow_mut();
        let mut event = Self::transfer_event(trb, slot, dci, residual, st.event_pcs[0]);
        event[2] = (event[2] & 0x00ff_ffff) | (u32::from(code) << 24);
        Self::post_event(&mut st, 0, event);
    }

    /// **Set a root-hub port's connect state, latching CSC** — the mock's
    /// hot-plug.
    ///
    /// Drives exactly what real silicon presents on a plug or unplug: CCS
    /// follows the cable and CSC latches to say it moved. Both edges are the
    /// same operation because to a driver they are the same event, and a
    /// double that could only ever attach would let a detach bug through
    /// unexamined — which is how one shipped.
    ///
    /// Ports are 1-based, matching PORTSC addressing; port 0 does not
    /// exist and is refused rather than wrapped into a register that is not
    /// a PORTSC.
    pub fn set_port_connected(&self, port: u8, connected: bool) {
        let mut st = self.st.borrow_mut();
        assert!(port >= 1, "PORTSC ports are 1-based; port 0 does not exist");
        let idx = (OP + 0x400 + (port as usize - 1) * 0x10) / 4;
        assert!(
            idx < REGS_U32,
            "port {port} is past this mock's register file"
        );
        let speed = PSI_FULL << PORTSC_SPEED_SHIFT;
        st.regs[idx] = if connected {
            PORTSC_CCS | PORTSC_CSC | speed
        } else {
            // Disconnected: CCS clear, CSC latched, PED necessarily clear —
            // a port cannot stay enabled with nothing on it.
            PORTSC_CSC
        };
    }

    fn read64(regs: &[u32; REGS_U32], off: usize) -> u64 {
        (regs[off / 4] as u64) | ((regs[off / 4 + 1] as u64) << 32)
    }

    /// Walk the command ring from the mock's dequeue, posting a Command
    /// Completion per command (following the Link TRB + toggling cycle).
    fn process_commands(st: &mut State) {
        if st.swallow {
            return;
        }
        let base = Self::read64(&st.regs, OP + CRCR) & !0x3f;
        // Bounded by the ring size so a malformed ring can't spin forever.
        for _ in 0..RING_TRBS {
            let trb_ptr = base + st.cmd_deq as u64 * 16;
            // SAFETY: the mock's IOVAs are host pointers into leaked ring
            // pages; `trb_ptr` is a 16-byte TRB slot inside one.
            let trb = unsafe {
                let p = trb_ptr as *const u32;
                [
                    p.read_volatile(),
                    p.add(1).read_volatile(),
                    p.add(2).read_volatile(),
                    p.add(3).read_volatile(),
                ]
            };
            let cycle = trb[3] & 1 != 0;
            if cycle != st.cmd_ccs {
                break; // caught up to the producer
            }
            let ttype = ((trb[3] >> 10) & 0x3f) as u8;
            if ttype == TYPE_LINK {
                // Follow back to segment start; toggle CCS if TC set.
                st.cmd_deq = 0;
                if trb[3] & (1 << 1) != 0 {
                    st.cmd_ccs = !st.cmd_ccs;
                }
                continue;
            }
            let (code, slot) = Self::handle_command(st, ttype, &trb);
            let ev = Self::completion_event(trb_ptr, code, slot, st.event_pcs[0]);
            Self::post_event(st, 0, ev);
            st.cmd_deq += 1;
        }
    }

    /// The mock's per-command response.  Address Device reads the input
    /// context the driver built (its slot's root-hub port picks the device,
    /// its EP0 dequeue pointer arms the control ring), establishing the slot.
    fn handle_command(st: &mut State, ttype: u8, trb: &[u32; 4]) -> (u8, u8) {
        if let Some((command, code)) = st.command_failure {
            if ttype == command {
                return (code, (trb[3] >> 24) as u8);
            }
        }
        match ttype {
            TYPE_ENABLE_SLOT => {
                st.next_slot += 1;
                (CC_SUCCESS, st.next_slot)
            }
            TYPE_NOOP_CMD => (CC_SUCCESS, 0),
            TYPE_DISABLE_SLOT => {
                let slot = (trb[3] >> 24) as u8;
                st.slots[usize::from(slot)] = SlotDevice::EMPTY;
                (CC_SUCCESS, slot)
            }
            TYPE_ADDRESS_DEVICE => {
                let slot_id = ((trb[3] >> 24) & 0xff) as u8;
                let ictx = (trb[0] as u64) | ((trb[1] as u64) << 32);
                // Input context: InputControl @0, Slot @dword cw, EP0 @dword
                // 2·cw (cw = context stride, 8 or 16 by CSZ).  Read the slot's
                // root port (slot dword 1) + EP0 dequeue ptr (ep dwords 2/3).
                // SAFETY: `ictx` is the host pointer to the leaked context page.
                let cw = ctx_dwords(&st.regs);
                let (root_port, ep0_base, ep0_mps) = unsafe {
                    let p = ictx as *const u32;
                    let slot_d1 = p.add(cw + 1).read_volatile();
                    let root_port = ((slot_d1 >> 16) & 0xff) as u8;
                    let ep0_d2 = p.add(2 * cw + 2).read_volatile();
                    let ep0_d3 = p.add(2 * cw + 3).read_volatile();
                    let ep0_mps = (p.add(2 * cw + 1).read_volatile() >> 16) as u16;
                    let base = ((ep0_d2 & !0xf) as u64) | ((ep0_d3 as u64) << 32);
                    (root_port, base, ep0_mps)
                };
                let kind = match root_port {
                    1 => DeviceKind::Keyboard,
                    2 => DeviceKind::Mouse,
                    3 => DeviceKind::Ecm,
                    _ => DeviceKind::None,
                };
                if (slot_id as usize) < st.slots.len() {
                    st.slots[slot_id as usize] = SlotDevice {
                        kind,
                        port: root_port,
                        ep0_base,
                        ep0_mps,
                        ..SlotDevice::EMPTY
                    };
                }
                (CC_SUCCESS, slot_id)
            }
            TYPE_CONFIGURE_ENDPOINT => {
                let slot_id = ((trb[3] >> 24) & 0xff) as u8;
                let ictx = (trb[0] as u64) | ((trb[1] as u64) << 32);
                // Input context: InputControl @0 (add-flags in dword 1), then
                // Slot @cw, endpoints @(1+dci)·cw.  **Every** add flag above
                // the slot bit is honored, not just the highest — a CDC data
                // interface brings its bulk IN and bulk OUT up in one command,
                // and a mock that saw only one of them would let a
                // half-configured driver pass.
                // SAFETY: `ictx` is the host pointer to the leaked context page.
                let cw = ctx_dwords(&st.regs);
                unsafe {
                    let p = ictx as *const u32;
                    let add = p.add(1).read_volatile();
                    for dci in 2..32u8 {
                        if add & (1 << dci) == 0 {
                            continue;
                        }
                        let ep_dw = (1 + dci as usize) * cw;
                        let d1 = p.add(ep_dw + 1).read_volatile();
                        let d2 = p.add(ep_dw + 2).read_volatile();
                        let d3 = p.add(ep_dw + 3).read_volatile();
                        let base = ((d2 & !0xf) as u64) | ((d3 as u64) << 32);
                        // EP Type (xHCI 6.2.3, endpoint context dword 1 bits
                        // 5:3): 2 = Bulk OUT, 6 = Bulk IN, 7 = Interrupt IN.
                        let ep_type = (d1 >> 3) & 0x7;
                        if (slot_id as usize) >= st.slots.len() {
                            continue;
                        }
                        let sd = &mut st.slots[slot_id as usize];
                        match ep_type {
                            2 => {
                                sd.bout_base = base;
                                sd.bout_dci = dci;
                                sd.bout_deq = 0;
                                sd.bout_ccs = true;
                            }
                            6 => {
                                sd.bin_base = base;
                                sd.bin_dci = dci;
                                sd.bin_deq = 0;
                                sd.bin_ccs = true;
                            }
                            7 => {
                                let pipe = sd
                                    .interrupt
                                    .iter_mut()
                                    .find(|pipe| pipe.dci == 0)
                                    .expect("bounded configured interrupt set");
                                *pipe = InterruptPipe {
                                    base,
                                    dci,
                                    ..InterruptPipe::EMPTY
                                };
                            }
                            _ => {}
                        }
                    }
                }
                (CC_SUCCESS, slot_id)
            }
            TYPE_EVALUATE_CONTEXT => {
                let slot = (trb[3] >> 24) as u8;
                let context = u64::from(trb[0]) | (u64::from(trb[1]) << 32);
                let cw = ctx_dwords(&st.regs);
                // SAFETY: input context is the same live DMA page supplied
                // at Address Device. Only A1 is legal for this MPS update.
                let (drop_flags, add_flags, mps) = unsafe {
                    let p = context as *const u32;
                    (
                        p.read_volatile(),
                        p.add(1).read_volatile(),
                        (p.add(2 * cw + 1).read_volatile() >> 16) as u16,
                    )
                };
                if drop_flags != 0 || add_flags != 2 {
                    return (17, slot);
                }
                st.slots[usize::from(slot)].ep0_mps = mps;
                st.ep0_evaluations += 1;
                (CC_SUCCESS, slot)
            }
            TYPE_RESET_ENDPOINT | TYPE_SET_TR_DEQUEUE => {
                let slot = (trb[3] >> 24) as u8;
                let dci = ((trb[3] >> 16) & 31) as u8;
                let device = &mut st.slots[usize::from(slot)];
                if ttype == TYPE_RESET_ENDPOINT {
                    if dci == 1 {
                        device.ep0_halted = false;
                    } else if let Some(pipe) =
                        device.interrupt.iter_mut().find(|pipe| pipe.dci == dci)
                    {
                        pipe.halted = false;
                    } else {
                        return (19, slot);
                    }
                } else {
                    let pointer = u64::from(trb[0] & !0xf) | (u64::from(trb[1]) << 32);
                    let ccs = trb[0] & 1 != 0;
                    let base = if dci == 1 {
                        device.ep0_base
                    } else {
                        device
                            .interrupt
                            .iter()
                            .find(|pipe| pipe.dci == dci)
                            .map_or(0, |pipe| pipe.base)
                    };
                    if base == 0 || pointer < base || pointer - base >= u64::from(RING_TRBS) * 16 {
                        return (17, slot);
                    }
                    let deq = ((pointer - base) / 16) as u16;
                    if dci == 1 {
                        device.ep0_deq = deq;
                        device.ep0_ccs = ccs;
                        if device.ep0_needs_dequeue {
                            st.ep0_recoveries += 1;
                        }
                        device.ep0_needs_dequeue = false;
                    } else if let Some(pipe) =
                        device.interrupt.iter_mut().find(|pipe| pipe.dci == dci)
                    {
                        pipe.deq = deq;
                        pipe.ccs = ccs;
                    }
                }
                (CC_SUCCESS, slot)
            }
            _ => (CC_SUCCESS, 0),
        }
    }

    /// **Drain the bulk OUT ring**: every Normal TRB the driver has published
    /// is a frame to take off the wire.  Each is captured into `tx` (and, in
    /// loopback, queued straight onto `rx`) and answered with its own
    /// Transfer Event, so the driver's per-transfer correlation is exercised
    /// with several completions outstanding rather than one at a time.
    fn drain_bulk_out(st: &mut State, slot_id: u8) {
        if st.swallow || slot_id == 0 || slot_id as usize >= st.slots.len() {
            return;
        }
        let sd = st.slots[slot_id as usize];
        if sd.bout_base == 0 {
            return;
        }
        let mut deq = sd.bout_deq;
        let mut ccs = sd.bout_ccs;
        for _ in 0..RING_TRBS {
            let trb_ptr = sd.bout_base + deq as u64 * 16;
            // SAFETY: host pointer into the leaked bulk-OUT ring page.
            let trb = unsafe {
                let p = trb_ptr as *const u32;
                [
                    p.read_volatile(),
                    p.add(1).read_volatile(),
                    p.add(2).read_volatile(),
                    p.add(3).read_volatile(),
                ]
            };
            if (trb[3] & 1 != 0) != ccs {
                break; // caught up to the producer
            }
            let ttype = ((trb[3] >> 10) & 0x3f) as u8;
            if ttype == TYPE_LINK {
                deq = 0;
                if trb[3] & (1 << 1) != 0 {
                    ccs = !ccs;
                }
                continue;
            }
            if ttype == TYPE_NORMAL {
                let buf = (trb[0] as u64) | ((trb[1] as u64) << 32);
                let len = ((trb[2] & 0x1_ffff) as usize).min(MOCK_FRAME_MAX);
                let target = ((trb[2] >> 22) & 0x3ff) as u16;
                let mut frame = alloc::vec![0u8; len];
                if buf != 0 && len > 0 {
                    // SAFETY: `buf` is the host pointer to a leaked
                    // device-visible page the driver named; `len` is the
                    // transfer length it published.
                    unsafe {
                        core::ptr::copy_nonoverlapping(buf as *const u8, frame.as_mut_ptr(), len)
                    };
                }
                if st.loopback {
                    st.rx.push_back(frame.clone());
                }
                st.tx.push(frame);
                let cyc = st.event_pcs.get(target as usize).copied().unwrap_or(true);
                // A transmit moves the whole buffer: no residual.
                let ev = Self::transfer_event(trb_ptr, slot_id, sd.bout_dci, 0, cyc);
                Self::post_event(st, target, ev);
            }
            deq += 1;
        }
        st.slots[slot_id as usize].bout_deq = deq;
        st.slots[slot_id as usize].bout_ccs = ccs;
        // A looped-back frame needs the receive side walked too, or it would
        // sit queued until the driver happened to ring the IN doorbell again.
        if st.loopback {
            Self::deliver_bulk_in(st, slot_id);
        }
    }

    /// **Fill posted bulk IN buffers** from the queued receive frames, one
    /// Normal TRB per frame, each answered with a Transfer Event whose
    /// residual reports the unused tail of the buffer — the short-packet
    /// arm a receive path must handle, since frames are almost never exactly
    /// the posted length.
    fn deliver_bulk_in(st: &mut State, slot_id: u8) {
        if st.swallow || slot_id == 0 || slot_id as usize >= st.slots.len() {
            return;
        }
        let sd = st.slots[slot_id as usize];
        if sd.bin_base == 0 {
            return;
        }
        let mut deq = sd.bin_deq;
        let mut ccs = sd.bin_ccs;
        for _ in 0..RING_TRBS {
            if st.rx.is_empty() {
                break;
            }
            let trb_ptr = sd.bin_base + deq as u64 * 16;
            // SAFETY: host pointer into the leaked bulk-IN ring page.
            let trb = unsafe {
                let p = trb_ptr as *const u32;
                [
                    p.read_volatile(),
                    p.add(1).read_volatile(),
                    p.add(2).read_volatile(),
                    p.add(3).read_volatile(),
                ]
            };
            if (trb[3] & 1 != 0) != ccs {
                break; // no posted buffer — the frame stays queued
            }
            let ttype = ((trb[3] >> 10) & 0x3f) as u8;
            if ttype == TYPE_LINK {
                deq = 0;
                if trb[3] & (1 << 1) != 0 {
                    ccs = !ccs;
                }
                continue;
            }
            if ttype == TYPE_NORMAL {
                let buf = (trb[0] as u64) | ((trb[1] as u64) << 32);
                let cap = (trb[2] & 0x1_ffff) as usize;
                let target = ((trb[2] >> 22) & 0x3ff) as u16;
                let frame = st.rx.pop_front().expect("non-empty checked above");
                // Bound the copy by the wire's own maximum as well as the
                // TRB's published length — the transmit drain above clamps the
                // same way.  `cap` is 17 device-supplied bits, so a driver that
                // published one larger than the buffer it actually allocated
                // would otherwise have this double walk off the end of that
                // page: a driver bug turned into UB inside the test process,
                // which is the one thing a hardware double must never do.  The
                // residual below is still computed against the UNCLAMPED `cap`,
                // so short-packet reporting is unchanged.
                let n = frame.len().min(cap).min(MOCK_FRAME_MAX);
                if buf != 0 && n > 0 {
                    // SAFETY: `buf` is the host pointer to the leaked receive
                    // page the driver posted, and the pool hands out whole
                    // pages; `n ≤ MOCK_FRAME_MAX`, well inside one.
                    unsafe { core::ptr::copy_nonoverlapping(frame.as_ptr(), buf as *mut u8, n) };
                }
                let cyc = st.event_pcs.get(target as usize).copied().unwrap_or(true);
                let ev = Self::transfer_event(
                    trb_ptr,
                    slot_id,
                    sd.bin_dci,
                    cap.saturating_sub(n) as u32,
                    cyc,
                );
                Self::post_event(st, target, ev);
            }
            deq += 1;
        }
        st.slots[slot_id as usize].bin_deq = deq;
        st.slots[slot_id as usize].bin_ccs = ccs;
    }

    /// Deliver a pending report to the slot's armed interrupt endpoint: walk
    /// its ring for a Normal TRB, copy the report into the TRB's buffer, and
    /// post a Transfer Event.  A no-op if nothing is pending or the endpoint
    /// is not yet armed (the report stays pending for a later arm).
    fn deliver_report(st: &mut State, slot_id: u8, dci: u8) {
        if slot_id == 0 || slot_id as usize >= st.slots.len() {
            return;
        }
        let Some(index) = st.slots[slot_id as usize]
            .interrupt
            .iter()
            .position(|pipe| pipe.dci == dci)
        else {
            return;
        };
        let pipe = st.slots[slot_id as usize].interrupt[index];
        let Some((report, rlen)) = pipe.pending else {
            return;
        };
        if pipe.base == 0 || pipe.halted {
            return;
        }
        let mut deq = pipe.deq;
        let mut ccs = pipe.ccs;
        for _ in 0..RING_TRBS {
            let trb_ptr = pipe.base + deq as u64 * 16;
            // SAFETY: host pointer into the leaked interrupt-ring page.
            let trb = unsafe {
                let p = trb_ptr as *const u32;
                [
                    p.read_volatile(),
                    p.add(1).read_volatile(),
                    p.add(2).read_volatile(),
                    p.add(3).read_volatile(),
                ]
            };
            if (trb[3] & 1 != 0) != ccs {
                break; // no armed transfer yet — keep the report pending
            }
            let ttype = ((trb[3] >> 10) & 0x3f) as u8;
            if ttype == TYPE_LINK {
                deq = 0;
                if trb[3] & (1 << 1) != 0 {
                    ccs = !ccs;
                }
                continue;
            }
            if ttype == TYPE_NORMAL {
                let buf = (trb[0] as u64) | ((trb[1] as u64) << 32);
                let cap = (trb[2] & 0x1_ffff) as usize;
                // Interrupter Target (Normal TRB dword 2, bits 22:31) — the
                // interrupter this transfer's completion is routed to.
                let target = ((trb[2] >> 22) & 0x3ff) as u16;
                let n = rlen.min(cap);
                // SAFETY: `buf` is the host pointer to the leaked report page.
                unsafe {
                    core::ptr::copy_nonoverlapping(report.as_ptr(), buf as *mut u8, n);
                }
                let cyc = st.event_pcs.get(target as usize).copied().unwrap_or(true);
                let ev = Self::transfer_event(
                    trb_ptr,
                    slot_id,
                    dci,
                    (cap.saturating_sub(n)) as u32,
                    cyc,
                );
                Self::post_event(st, target, ev);
                deq += 1;
                st.slots[slot_id as usize].interrupt[index].pending = None;
                break;
            }
            deq += 1;
        }
        st.slots[slot_id as usize].interrupt[index].deq = deq;
        st.slots[slot_id as usize].interrupt[index].ccs = ccs;
    }

    /// A boot device's descriptor bytes for a `GET_DESCRIPTOR` (device or
    /// configuration); empty for anything else.
    fn descriptor(kind: DeviceKind, desc_type: u8, index: u8, bad_mac: bool) -> &'static [u8] {
        match (kind, desc_type) {
            (DeviceKind::Keyboard, 1) => &KBD_DEVICE_DESC,
            (DeviceKind::Keyboard, 2) => &KBD_CONFIG_DESC,
            (DeviceKind::Mouse, 1) => &MOUSE_DEVICE_DESC,
            (DeviceKind::Mouse, 2) => &MOUSE_CONFIG_DESC,
            (DeviceKind::Ecm, 1) => &ECM_DEVICE_DESC,
            // `GET_DESCRIPTOR(CONFIGURATION, n)` indexes the configuration
            // ARRAY; `bConfigurationValue` is a field inside the descriptor
            // and is deliberately not the index.  Index 0 is RNDIS (value 2).
            (DeviceKind::Ecm, 2) if index == 0 => &ECM_RNDIS_CONFIG_DESC,
            (DeviceKind::Ecm, 2) if index == 1 => &ECM_CONFIG_DESC,
            // Only the MAC string is modelled; any other index returns
            // nothing, so a driver reading the wrong index sees a refusal
            // rather than a plausible-looking string.
            (DeviceKind::Ecm, 3) if index == ECM_MAC_STRING_INDEX => {
                if bad_mac {
                    &ECM_MAC_STRING_BAD
                } else {
                    &ECM_MAC_STRING
                }
            }
            _ => &[],
        }
    }

    /// Walk a slot's EP0 ring for one control transfer (Setup + optional Data
    /// + Status), fulfilling a `GET_DESCRIPTOR` into the Data buffer and
    /// posting a Transfer Event on the Status stage.
    fn process_ep0(st: &mut State, slot_id: u8) {
        if st.swallow || st.hold_ep0 || slot_id == 0 || slot_id as usize >= st.slots.len() {
            return;
        }
        let sd = st.slots[slot_id as usize];
        if sd.kind == DeviceKind::None || sd.ep0_base == 0 || sd.ep0_halted || sd.ep0_needs_dequeue
        {
            return;
        }
        let bad_mac = st.bad_mac;
        let mut deq = sd.ep0_deq;
        let mut ccs = sd.ep0_ccs;

        let mut setup = [0u32; 2]; // dword0/1 of the Setup TRB (the SETUP packet)
        let mut data_ptr = 0u64;
        let mut data_len = 0u32;
        let mut data_trb = 0u64;
        let mut data_isp = false;

        for _ in 0..RING_TRBS {
            let trb_ptr = sd.ep0_base + deq as u64 * 16;
            // SAFETY: host pointer into the leaked EP0 ring page.
            let trb = unsafe {
                let p = trb_ptr as *const u32;
                [
                    p.read_volatile(),
                    p.add(1).read_volatile(),
                    p.add(2).read_volatile(),
                    p.add(3).read_volatile(),
                ]
            };
            if (trb[3] & 1 != 0) != ccs {
                break; // caught up to the producer
            }
            let ttype = ((trb[3] >> 10) & 0x3f) as u8;
            match ttype {
                TYPE_LINK => {
                    deq = 0;
                    if trb[3] & (1 << 1) != 0 {
                        ccs = !ccs;
                    }
                    continue;
                }
                TYPE_SETUP => {
                    setup = [trb[0], trb[1]];
                }
                TYPE_DATA => {
                    data_ptr = (trb[0] as u64) | ((trb[1] as u64) << 32);
                    data_len = trb[2] & 0x1_ffff;
                    data_trb = trb_ptr;
                    data_isp = trb[3] & (1 << 2) != 0;
                }
                TYPE_STATUS => {
                    // The TD ends here.  Fulfill a GET_DESCRIPTOR and post the
                    // transfer event against this Status TRB.
                    let b_request = (setup[0] >> 8) & 0xff;
                    let w_value = setup[0] >> 16;
                    let w_length = setup[1] >> 16;
                    let mut residual = w_length.saturating_sub(data_len);
                    // Record any class request so a test can assert the
                    // device was really told something, not just that the
                    // driver's call returned Ok.  bmRequestType type field
                    // (bits 6:5) == 1 is a class request.
                    let bm_request_type = (setup[0] & 0xff) as u8;
                    let w_index = (setup[1] & 0xffff) as u16;
                    let mut code = CC_SUCCESS;
                    if (bm_request_type >> 5) & 0x3 == 1 {
                        st.class_requests
                            .push((b_request as u8, w_value as u16, w_index));
                        if bm_request_type & 0x80 == 0 && data_ptr != 0 {
                            let n = (data_len as usize).min(w_length as usize);
                            // SAFETY: `data_ptr` is the host pointer to the
                            // device's leaked data page; `n ≤ wLength ≤ page`.
                            let bytes =
                                unsafe { core::slice::from_raw_parts(data_ptr as *const u8, n) };
                            st.class_payloads.push((w_index, bytes.to_vec()));
                        }
                        if let Some((port, request, interface, error)) = st.hid_request_error {
                            if port == sd.port && request == b_request as u8 && interface == w_index
                            {
                                code = error;
                            }
                        }
                    }
                    if b_request == 6 && data_ptr != 0 {
                        // GET_DESCRIPTOR: wValue hi = descriptor type, lo = index.
                        let desc_type = (w_value >> 8) as u8;
                        let src = if sd.port == 1 && desc_type == 2 && st.hid_config.is_some() {
                            st.hid_config.expect("checked fixture")
                        } else {
                            Self::descriptor(sd.kind, desc_type, w_value as u8, bad_mac)
                        };
                        let mut n = (data_len as usize).min(src.len());
                        if desc_type == 1 && w_length == 8 {
                            n = n.min(st.device_prefix_len);
                        }
                        // SAFETY: `data_ptr` is the host pointer to the leaked
                        // data page (≥ one page); `n ≤ data_len ≤ page`.
                        unsafe {
                            core::ptr::copy_nonoverlapping(src.as_ptr(), data_ptr as *mut u8, n);
                            if desc_type == 1 && sd.kind != DeviceKind::Ecm && n >= 8 {
                                (data_ptr as *mut u8).add(7).write(st.hid_mps0);
                            }
                        }
                        // A longer read cannot succeed with the initial
                        // eight-byte guess on a full-speed MPS64 device.
                        if sd.kind != DeviceKind::Ecm
                            && w_length > 8
                            && sd.ep0_mps != u16::from(st.hid_mps0)
                        {
                            code = 4;
                        }
                        residual = w_length.saturating_sub(n as u32);
                    }
                    if code != CC_SUCCESS {
                        let failed_trb = if data_trb != 0 { data_trb } else { trb_ptr };
                        let mut ev =
                            Self::transfer_event(failed_trb, slot_id, 1, residual, st.event_pcs[0]);
                        ev[2] = (ev[2] & 0x00ff_ffff) | (u32::from(code) << 24);
                        Self::post_event(st, 0, ev);
                        if code == 6 {
                            st.slots[usize::from(slot_id)].ep0_halted = true;
                            st.slots[usize::from(slot_id)].ep0_needs_dequeue = true;
                            // The failed TD has not advanced to the next Setup.
                            // Reset without Set TR Dequeue must not appear green.
                            deq = ((failed_trb - sd.ep0_base) / 16) as u16;
                            break;
                        }
                    } else {
                        // A Control Data Stage and Status Stage are separate
                        // TDs. ISP makes a short Data Stage report its residue;
                        // Status still reports Success with zero residue. A
                        // mock that puts Data residue on Status hides a real
                        // descriptor-length bug (xHCI1.2b §4.11.2.2).
                        if data_trb != 0 && residual != 0 && data_isp {
                            let mut short = Self::transfer_event(
                                data_trb,
                                slot_id,
                                1,
                                residual,
                                st.event_pcs[0],
                            );
                            short[2] = (short[2] & 0x00ff_ffff) | (13 << 24);
                            Self::post_event(st, 0, short);
                        }
                        let status = Self::transfer_event(trb_ptr, slot_id, 1, 0, st.event_pcs[0]);
                        Self::post_event(st, 0, status);
                    }
                    deq += 1;
                    break;
                }
                _ => {}
            }
            deq += 1;
        }
        st.slots[slot_id as usize].ep0_deq = deq;
        st.slots[slot_id as usize].ep0_ccs = ccs;
    }

    /// Build a Transfer Event TRB (xHCI 6.4.2.1): `endpoint_id` in [3](16:20),
    /// `slot_id` in [3](24:31), Success code, residual length.
    fn transfer_event(
        trb_ptr: u64,
        slot_id: u8,
        endpoint_id: u8,
        residual: u32,
        cycle: bool,
    ) -> [u32; 4] {
        [
            trb_ptr as u32,
            (trb_ptr >> 32) as u32,
            (residual & 0xff_ffff) | ((CC_SUCCESS as u32) << 24),
            (cycle as u32)
                | (TYPE_TRANSFER_EVENT << 10)
                | ((endpoint_id as u32) << 16)
                | ((slot_id as u32) << 24),
        ]
    }

    /// Build a Command Completion Event TRB (xHCI 6.4.2.2).
    fn completion_event(cmd_ptr: u64, code: u8, slot: u8, cycle: bool) -> [u32; 4] {
        [
            cmd_ptr as u32,
            (cmd_ptr >> 32) as u32,
            (code as u32) << 24,
            (cycle as u32) | (TYPE_CMD_COMPLETION << 10) | ((slot as u32) << 24),
        ]
    }

    /// Write an event into **interrupter `intr`'s** event ring at that
    /// interrupter's enqueue, advancing (wrap → toggle its producer cycle).
    /// Commands + EP0 land on interrupter 0; an interrupt transfer lands on
    /// the interrupter its TRB targeted — the per-core routing.
    fn post_event(st: &mut State, intr: u16, ev: [u32; 4]) {
        let i = intr as usize;
        if i >= MOCK_INTRS {
            return;
        }
        let erstba_off = RT + 0x20 + i * INTR_STRIDE + ERSTBA;
        let erstba = Self::read64(&st.regs, erstba_off);
        // ERST[0].ring_base (first two dwords of the ERST).
        // SAFETY: `erstba` is the host pointer to the leaked ERST page.
        let ev_base = unsafe {
            let e = erstba as *const u32;
            (e.read_volatile() as u64) | ((e.add(1).read_volatile() as u64) << 32)
        };
        let slot_ptr = ev_base + st.event_enq[i] as u64 * 16;
        // SAFETY: `slot_ptr` is a TRB slot inside the leaked event-ring page.
        unsafe {
            let p = slot_ptr as *mut u32;
            // Body first, cycle-carrying control dword last (the discipline a
            // real controller relies on; harmless here but modeled honestly).
            p.write_volatile(ev[0]);
            p.add(1).write_volatile(ev[1]);
            p.add(2).write_volatile(ev[2]);
            p.add(3).write_volatile(ev[3]);
        }
        st.event_enq[i] += 1;
        if st.event_enq[i] == RING_TRBS {
            st.event_enq[i] = 0;
            st.event_pcs[i] = !st.event_pcs[i];
        }
        // A posted event asserts both the controller summary and the selected
        // interrupter's pending bit, as real xHCI does.
        st.regs[(OP + USBSTS) / 4] |= USBSTS_EINT;
        let iman = RT + 0x20 + i * INTR_STRIDE + IMAN;
        st.regs[iman / 4] |= IMAN_IP;
        st.ack_step[i] = 0;
    }
}

impl Default for MockXhciDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Regs for MockXhciDevice {
    fn read32(&self, off: usize) -> u32 {
        self.st.borrow().regs[off / 4]
    }

    fn write32(&self, off: usize, val: u32) {
        let mut st = self.st.borrow_mut();
        let old = st.regs[off / 4];
        st.regs[off / 4] = val;

        if st.dbc_ctrl != 0 && off == st.dbc_ctrl {
            // Debug Capability DCCTRL: DCR is read-only and follows DCE down
            // (unless the engine is modelled stuck); the rest stores.
            const DBC_DCR: u32 = 1;
            const DBC_DCE: u32 = 1 << 31;
            let dcr = if val & DBC_DCE == 0 && !st.dbc_sticky {
                0
            } else {
                old & DBC_DCR
            };
            st.regs[off / 4] = (val & !DBC_DCR) | dcr;
            return;
        }
        if off == OP + USBSTS {
            // USBSTS interrupt/status bits are RW1C. Record the first step for
            // every interrupter currently pending.
            st.regs[off / 4] = old & !(val & USBSTS_EINT);
            if val & USBSTS_EINT != 0 {
                for i in 0..MOCK_INTRS {
                    let iman = RT + 0x20 + i * INTR_STRIDE + IMAN;
                    if st.regs[iman / 4] & IMAN_IP != 0 {
                        st.ack_step[i] = 1;
                    }
                }
            }
        } else if let Some(i) = interrupter_register(off, IMAN) {
            // IP is RW1C; IE is ordinary RW. Reserved bits stay untouched.
            if old & IMAN_IP != 0 && val & IMAN_IP != 0 {
                if st.ack_step[i] != 1 {
                    st.bad_ack_order = true;
                }
                st.ack_step[i] = 2;
            }
            st.regs[off / 4] = (old & !(val & IMAN_IP) & !IMAN_IE) | (val & IMAN_IE);
        } else if let Some(i) = interrupter_register(off, ERDP) {
            if val & ERDP_EHB as u32 != 0 {
                if st.ack_step[i] != 2 {
                    st.bad_ack_order = true;
                } else {
                    st.ack_count[i] += 1;
                }
                st.ack_step[i] = 0;
            }
            // EHB is RW1C and reads back clear; keep only the dequeue address.
            st.regs[off / 4] = val & !(ERDP_EHB as u32);
        } else if off == OP + USBCMD {
            if val & USBCMD_HCRST != 0 {
                // Reset: self-clear HCRST, return to halted+ready, reset rings.
                st.regs[(OP + USBCMD) / 4] &= !USBCMD_HCRST;
                st.regs[(OP + USBSTS) / 4] = USBSTS_HCH;
                st.cmd_deq = 0;
                st.cmd_ccs = true;
                st.event_enq = [0; MOCK_INTRS];
                st.event_pcs = [true; MOCK_INTRS];
                st.next_slot = 0;
                st.slots = [SlotDevice::EMPTY; 33];
                st.ack_step = [0; MOCK_INTRS];
                st.ack_count = [0; MOCK_INTRS];
                st.bad_ack_order = false;
            } else if val & USBCMD_RS != 0 {
                st.regs[(OP + USBSTS) / 4] &= !USBSTS_HCH; // running
            } else {
                st.regs[(OP + USBSTS) / 4] |= USBSTS_HCH; // halted
            }
        } else if off == OP + CONFIG {
            // MaxSlotsEn stored; no side effect.
        } else if off == OP + CRCR && (val as u64) & CRCR_CA != 0 {
            // Command Abort: the ring is not running (we complete inline), so
            // clear CRR immediately so the abort handshake terminates.
            st.regs[(OP + CRCR) / 4] &= !(CRCR_CRR as u32);
        } else if is_portsc(off) {
            // PORTSC is RW1CS: recompute from the *old* value (RO/level bits —
            // CCS, speed, PED — are preserved; the wholesale store above is
            // wrong for them).  A write with PR completes a reset (enable +
            // latch Port Reset Change, self-clear PR); otherwise any change
            // bit the driver wrote 1 to is cleared (RW1C).
            let mut p = old;
            if val & PORTSC_PR != 0 {
                p &= !PORTSC_PR;
                p |= PORTSC_PED | PORTSC_PRC;
            } else {
                p &= !(val & PORTSC_CHANGES);
            }
            st.regs[off / 4] = p;
        } else if off >= DB && off < DB + 33 * 4 {
            // Doorbell array: index 0 = command ring, n = slot n's endpoints.
            let n = ((off - DB) / 4) as u8;
            if n == 0 {
                Self::process_commands(&mut st);
            } else if val == 1 {
                // EP0 (DCI 1): the control path.
                Self::process_ep0(&mut st, n);
            } else {
                // An endpoint doorbell: `val` is the DCI being rung.  Route by
                // which endpoint that DCI is, so one slot can hold an
                // interrupt endpoint and a bulk pair at once.
                let dci = val as u8;
                let sd = st
                    .slots
                    .get(n as usize)
                    .copied()
                    .unwrap_or(SlotDevice::EMPTY);
                if sd.bout_dci != 0 && dci == sd.bout_dci {
                    Self::drain_bulk_out(&mut st, n);
                } else if sd.bin_dci != 0 && dci == sd.bin_dci {
                    Self::deliver_bulk_in(&mut st, n);
                } else {
                    Self::deliver_report(&mut st, n, dci);
                }
            }
        }
    }
}

impl Regs64 for MockXhciDevice {
    fn read64_once(&self, off: usize) -> u64 {
        self.read64(off)
    }

    fn write64_once(&self, off: usize, value: u64) {
        // The mock has no bus-width distinction; preserve its register side
        // effects while satisfying the stronger one-transaction seam.
        self.write32(off, value as u32);
        self.write32(off + 4, (value >> 32) as u32);
    }
}

/// The mock's context stride (dwords) from HCCPARAMS1.CSZ — 16 for 64-byte
/// contexts, else 8.  Keeps the mock's input-context reads in lockstep with
/// the driver's CSZ-scaled serialization.
fn ctx_dwords(regs: &[u32; REGS_U32]) -> usize {
    if regs[0x10 / 4] & (1 << 2) != 0 {
        16
    } else {
        8
    }
}

/// Whether `off` addresses a PORTSC register (op + 0x400 + n·0x10, n < 4).
fn is_portsc(off: usize) -> bool {
    let base = OP + 0x400;
    off >= base && off < base + 4 * 0x10 && (off - base) % 0x10 == 0
}

/// Return the interrupter index when `off` names register `reg` in one of the
/// modeled runtime interrupter sets.
fn interrupter_register(off: usize, reg: usize) -> Option<usize> {
    let base = RT + 0x20 + reg;
    if off < base {
        return None;
    }
    let delta = off - base;
    (delta % INTR_STRIDE == 0)
        .then_some(delta / INTR_STRIDE)
        .filter(|i| *i < MOCK_INTRS)
}

// ── The mock DMA pool: leaked page-aligned host memory, iova == ptr ──────

/// A DMA buffer over leaked, page-aligned process memory.
#[derive(Clone, Copy)]
pub struct MockBuf {
    ptr: *mut u8,
    len: usize,
}

impl Dma for MockBuf {
    fn cpu(&self) -> *mut u8 {
        self.ptr
    }
    fn iova(&self) -> IovaAddr {
        IovaAddr(self.ptr as u64)
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// A pool handing out page-aligned zeroed buffers whose IOVA is their CPU
/// pointer (so the mock controller reads them directly).  Leaked — the
/// frames ride the whole test (the seam contract: buffers outlive handles).
pub struct MockPool;

impl DmaPool for MockPool {
    type Buf = MockBuf;

    fn alloc(&self, bytes: usize) -> Option<MockBuf> {
        let size = bytes.max(1).next_multiple_of(0x1000);
        let layout = alloc::alloc::Layout::from_size_align(size, 0x1000).ok()?;
        // SAFETY: non-zero page-multiple size; zeroed; intentionally leaked.
        let ptr = unsafe { alloc::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        Some(MockBuf { ptr, len: bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trb::Trb;
    use crate::{Controller, KeyboardReport, MouseReport};
    use xhci::ring::trb::command;

    fn spin() {}

    #[test]
    fn bring_up_reaches_running() {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        // Geometry decoded from the mock's CAP registers.
        assert_eq!(ctrl.caps().max_slots, 32);
        assert_eq!(ctrl.caps().max_ports, 4);
        assert_eq!(ctrl.caps().max_scratchpads, 2);
        assert!(!ctrl.caps().csz64, "32-byte contexts");
        // Running: USBSTS.HCH clear.
        assert_eq!(
            dev.read32(OP + USBSTS) & USBSTS_HCH,
            0,
            "controller running"
        );
    }

    #[test]
    fn prepare_defers_every_dma_pointer_until_post_activation_start() {
        use core::cell::Cell;

        /// The host's capability boundary in miniature: any DMA-pointer
        /// register written before `visible` is a fetch-before-map bug. This
        /// catches the QEMU behavior where an ERSTBA write resets the event
        /// ring and immediately reads the ERST while RS is still clear.
        struct DmaGate {
            inner: MockXhciDevice,
            visible: Cell<bool>,
            premature: Cell<bool>,
            split_publications: Cell<u8>,
            qword_publications: Cell<u8>,
        }

        impl DmaGate {
            fn is_dma_publication(off: usize) -> bool {
                matches!(
                    off,
                    o if o == OP + DCBAAP
                        || o == OP + DCBAAP + 4
                        || o == OP + CRCR
                        || o == OP + CRCR + 4
                        || o == RT + 0x20 + ERDP
                        || o == RT + 0x20 + ERDP + 4
                        || o == RT + 0x20 + ERSTBA
                        || o == RT + 0x20 + ERSTBA + 4
                )
            }
        }

        impl Regs for DmaGate {
            fn read32(&self, off: usize) -> u32 {
                self.inner.read32(off)
            }

            fn write32(&self, off: usize, val: u32) {
                if Self::is_dma_publication(off) {
                    if !self.visible.get() {
                        self.premature.set(true);
                    }
                    self.split_publications
                        .set(self.split_publications.get().saturating_add(1));
                }
                self.inner.write32(off, val);
            }
        }

        impl Regs64 for DmaGate {
            fn read64_once(&self, off: usize) -> u64 {
                self.inner.read64_once(off)
            }

            fn write64_once(&self, off: usize, value: u64) {
                if Self::is_dma_publication(off) {
                    if !self.visible.get() {
                        self.premature.set(true);
                    }
                    self.qword_publications
                        .set(self.qword_publications.get().saturating_add(1));
                }
                self.inner.write64_once(off, value);
            }
        }

        let dev = DmaGate {
            inner: MockXhciDevice::new(),
            visible: Cell::new(false),
            premature: Cell::new(false),
            split_publications: Cell::new(0),
            qword_publications: Cell::new(0),
        };
        let pool = MockPool;
        let prepared =
            Controller::<DmaGate, MockPool>::prepare(&dev, &pool, || None, spin).expect("prepare");
        let caps = *prepared.caps();

        assert_ne!(dev.read32(caps.op(USBSTS)) & USBSTS_HCH, 0);
        assert_eq!(dev.read32(caps.op(USBCMD)) & USBCMD_RS, 0);
        assert!(!dev.premature.get(), "prepare published a DMA pointer");
        assert_eq!(dev.read64(caps.op(DCBAAP)), 0, "DCBAA stays private");
        assert_eq!(dev.read64(caps.op(CRCR)), 0, "command ring stays private");
        assert_eq!(dev.read32(caps.intr(0, ERSTSZ)), 0);
        assert_eq!(
            dev.read64(caps.intr(0, ERDP)),
            0,
            "event ring stays private"
        );
        assert_eq!(dev.read64(caps.intr(0, ERSTBA)), 0, "ERST stays private");
        assert_eq!(
            dev.read32(caps.intr(0, IMAN)) & IMAN_IE,
            0,
            "interrupt delivery stays masked before start"
        );

        // This is the host's successful `activate_device_window` transition.
        dev.visible.set(true);
        let _controller = prepared.start(&dev, || None, spin).expect("start");
        assert!(!dev.premature.get());
        assert_ne!(dev.read64(caps.op(DCBAAP)), 0, "DCBAA published");
        assert_ne!(dev.read64(caps.op(CRCR)), 0, "command ring published");
        assert_eq!(dev.read32(caps.intr(0, ERSTSZ)), 1);
        assert_ne!(dev.read64(caps.intr(0, ERDP)), 0, "event ring published");
        assert_ne!(dev.read64(caps.intr(0, ERSTBA)), 0, "ERST published");
        assert_eq!(dev.read32(caps.op(USBSTS)) & USBSTS_HCH, 0);
        assert_eq!(
            dev.split_publications.get(),
            8,
            "the default publishes four pointer registers as low/high Dwords"
        );
        assert_eq!(
            dev.qword_publications.get(),
            0,
            "AC64 alone must not select native Qword MMIO"
        );

        let native = DmaGate {
            inner: MockXhciDevice::new(),
            visible: Cell::new(false),
            premature: Cell::new(false),
            split_publications: Cell::new(0),
            qword_publications: Cell::new(0),
        };
        let prepared = Controller::<DmaGate, MockPool>::prepare_with_pointer_access(
            &native,
            &pool,
            crate::PointerRegisterAccess::NativeQword,
            || None,
            spin,
        )
        .expect("native prepare");
        native.visible.set(true);
        let _controller = prepared
            .start(&native, || None, spin)
            .expect("native start");
        assert_eq!(native.split_publications.get(), 0);
        assert_eq!(
            native.qword_publications.get(),
            4,
            "the explicit quirk publishes each pointer in one Qword cycle"
        );
    }

    #[test]
    fn native_qword_is_an_exact_pci_quirk_not_an_amd_or_ac64_default() {
        use crate::controller::{PointerRegisterAccess, pointer_access_for_pci};

        for device in [0x1118, 0x111c, 0x111e, 0x1128] {
            assert_eq!(
                pointer_access_for_pci(0x1022, device),
                PointerRegisterAccess::NativeQword
            );
        }
        assert_eq!(
            pointer_access_for_pci(0x1022, 0xffff),
            PointerRegisterAccess::SplitDwords,
            "an unrelated AMD xHC does not inherit the measured quirk"
        );
        assert_eq!(
            pointer_access_for_pci(0x8086, 0x1118),
            PointerRegisterAccess::SplitDwords,
            "the device number alone is not a quirk identity"
        );
    }

    #[test]
    fn explicit_pointer_policy_preserves_the_host_clock_budget() {
        extern crate std;
        use crate::PointerRegisterAccess;
        use core::cell::Cell;

        std::thread_local! {
            static NOW: Cell<u64> = const { Cell::new(0) };
        }
        fn clock() -> Option<u64> {
            Some(NOW.get())
        }
        for policy in [
            PointerRegisterAccess::SplitDwords,
            PointerRegisterAccess::NativeQword,
        ] {
            NOW.set(0);
            let dev = MockXhciDevice::new();
            const LEGACY: usize = 0x200;
            {
                let mut state = dev.st.borrow_mut();
                state.regs[0x10 / 4] |= ((LEGACY / 4) as u32) << 16;
                state.regs[LEGACY / 4] = 1 | (1 << 16); // BIOS never releases ownership.
            }
            let mut pauses = 0;
            let result = Controller::<MockXhciDevice, MockPool>::prepare_with_pointer_access(
                &dev,
                &MockPool,
                policy,
                clock,
                || {
                    pauses += 1;
                    assert!(pauses <= 4, "the policy constructor dropped its clock");
                    NOW.set(NOW.get() + 1_000_000_000);
                },
            );
            assert!(matches!(result, Err(crate::Error::NotReady)));
            assert!(pauses > 0);
            assert_eq!(dev.read32(OP + USBCMD), 0, "no reset before ownership");
            assert_eq!(dev.read64(OP + DCBAAP), 0, "no DMA publication");
        }
    }

    #[test]
    fn bios_handoff_timeout_refuses_before_operational_writes() {
        const LEGACY: usize = 0x200;
        const HC_BIOS_OWNED: u32 = 1 << 16;

        let dev = MockXhciDevice::new();
        {
            let mut st = dev.st.borrow_mut();
            // HCCPARAMS1.xECP is a dword offset. The legacy capability has no
            // successor and deliberately never releases BIOS ownership.
            st.regs[0x10 / 4] |= ((LEGACY / 4) as u32) << 16;
            st.regs[LEGACY / 4] = 1 | HC_BIOS_OWNED;
        }
        let pool = MockPool;
        let result = Controller::<MockXhciDevice, MockPool>::prepare(&dev, &pool, || None, spin);
        assert!(matches!(result, Err(crate::Error::NotReady)));
        assert_eq!(
            dev.read32(OP + USBCMD),
            0,
            "ownership refusal precedes halt/reset writes"
        );
        assert_eq!(dev.read64(OP + DCBAAP), 0, "no DMA pointer was armed");
    }

    /// A Debug Capability firmware left running survives HCRST with
    /// firmware's DMA pointers; the handoff must disable it before any
    /// operational write and say so — and must report, not refuse, an
    /// engine that will not stop.  Chain: legacy (already released) → DbC.
    #[test]
    fn debug_capability_left_running_is_disabled_at_handoff() {
        use crate::controller::{DbcHandoff, Handoff};
        const LEGACY: usize = 0x200;
        const DBC: usize = 0x240;
        const DCCTRL: usize = 0x20;
        const DCE: u32 = 1 << 31;
        const DCR: u32 = 1;
        const RO_FIELDS: u32 = 0x3 << 24; // device address, read-only

        let running = |sticky: bool| {
            let dev = MockXhciDevice::new();
            {
                let mut st = dev.st.borrow_mut();
                st.regs[0x10 / 4] |= ((LEGACY / 4) as u32) << 16;
                // Legacy cap: BIOS ownership already clear; next → DbC.
                st.regs[LEGACY / 4] = 1 | ((((DBC - LEGACY) / 4) as u32) << 8);
                st.regs[DBC / 4] = 10;
                st.regs[(DBC + DCCTRL) / 4] = DCE | DCR | RO_FIELDS;
                st.dbc_ctrl = DBC + DCCTRL;
                st.dbc_sticky = sticky;
            }
            dev
        };

        let dev = running(false);
        let prepared =
            Controller::<MockXhciDevice, MockPool>::prepare(&dev, &MockPool, || None, spin)
                .expect("a running DbC does not refuse bring-up");
        assert_eq!(
            prepared.handoff(),
            Handoff {
                legacy_released: true,
                dbc: DbcHandoff::Disabled
            }
        );
        let ctrl = dev.read32(DBC + DCCTRL);
        assert_eq!(ctrl & (DCE | DCR), 0, "DCE cleared, engine stopped");
        assert_eq!(ctrl & RO_FIELDS, RO_FIELDS, "read-only fields untouched");
        assert_ne!(
            dev.read32(OP + CONFIG),
            0,
            "bring-up proceeded past the handoff"
        );

        let dev = running(true);
        let prepared =
            Controller::<MockXhciDevice, MockPool>::prepare(&dev, &MockPool, || None, spin)
                .expect("a DbC that will not stop is reported, not refused");
        assert_eq!(prepared.handoff().dbc, DbcHandoff::Stuck);
        assert_eq!(dev.read32(DBC + DCCTRL) & DCE, 0, "DCE was still cleared");

        // Present but idle: untouched.  Absent (no xECP): the default.
        let dev = running(false);
        dev.st.borrow_mut().regs[(DBC + DCCTRL) / 4] = DCR | RO_FIELDS; // DCE clear
        let prepared =
            Controller::<MockXhciDevice, MockPool>::prepare(&dev, &MockPool, || None, spin)
                .expect("prepares");
        assert_eq!(prepared.handoff().dbc, DbcHandoff::Idle);
        assert_eq!(
            dev.read32(DBC + DCCTRL),
            DCR | RO_FIELDS,
            "idle DbC is not written"
        );
        let dev = MockXhciDevice::new();
        let prepared =
            Controller::<MockXhciDevice, MockPool>::prepare(&dev, &MockPool, || None, spin)
                .expect("prepares");
        assert_eq!(prepared.handoff(), Handoff::default());
    }

    /// [`crate::controller::bring_up_dma_pages`] and `bring_up`'s actual
    /// draws are one fact stated twice — this pins them together (the
    /// kernel founder sizes the `role::DMA_POOL` endowment from the
    /// formula, so a drifted formula is a real machine's boot wedge).
    #[test]
    fn dma_page_formula_matches_bring_up_draws() {
        use core::cell::Cell;

        use crate::controller::bring_up_dma_pages;

        struct CountingPool(MockPool, Cell<usize>);
        impl DmaPool for CountingPool {
            type Buf = MockBuf;
            fn alloc(&self, bytes: usize) -> Option<MockBuf> {
                // `bring_up`'s draws are all single pages; a multi-page draw
                // would silently skew the page count, so refuse it loudly.
                assert!(bytes <= 0x1000, "bring_up draws single pages");
                self.1.set(self.1.get() + 1);
                self.0.alloc(bytes)
            }
        }

        let dev = MockXhciDevice::new();
        let pool = CountingPool(MockPool, Cell::new(0));
        let ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        let n = ctrl.caps().max_scratchpads;
        assert_eq!(n, 2, "the mock advertises 2 scratchpads");
        assert_eq!(
            pool.1.get(),
            bring_up_dma_pages(n),
            "formula must equal the controller's actual draw count"
        );

        // The formula's shape at the edges the mock can't exhibit: the
        // scratchpad-less QEMU controller, a real Intel part, and the
        // one-page-array ceiling (clamped, matching `setup_scratchpad`'s
        // refusal bound).
        assert_eq!(bring_up_dma_pages(0), 4, "no scratchpads ⇒ no array");
        assert_eq!(bring_up_dma_pages(32), 4 + 1 + 32);
        assert_eq!(bring_up_dma_pages(512), 4 + 1 + 512);
        assert_eq!(
            bring_up_dma_pages(1023),
            bring_up_dma_pages(512),
            "sizing clamps where the driver refuses"
        );
    }

    #[test]
    fn noop_command_round_trips() {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        let noop: Trb = command::Noop::new().into();
        let cc = ctrl
            .command(&dev, noop, || None, spin)
            .expect("noop completes");
        assert_eq!(
            cc.completion_code(),
            Ok(xhci::ring::trb::event::CompletionCode::Success)
        );
        assert!(
            dev.interrupt_acknowledged(0),
            "command batch clears EINT, IP, and EHB in order"
        );
    }

    #[test]
    fn quiesce_halts_resets_and_masks_the_controller() {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");

        ctrl.quiesce(&dev, || None, spin).expect("quiesce");

        assert_ne!(dev.read32(OP + USBSTS) & USBSTS_HCH, 0, "halted");
        assert_eq!(
            dev.read32(OP + USBCMD) & (USBCMD_RS | USBCMD_INTE | USBCMD_HCRST),
            0,
            "run, interrupt-enable, and reset are all clear"
        );
        assert_eq!(
            dev.read32(RT + 0x20 + IMAN) & IMAN_IE,
            0,
            "interrupter zero is masked before reset"
        );
    }

    #[test]
    fn interrupt_probe_separates_submit_from_correlated_completion() {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");

        let probe = ctrl.arm_interrupt_probe(&dev);
        assert!(
            !dev.interrupt_acknowledged(0),
            "submission posts an interrupt event but does not consume it"
        );
        assert!(
            matches!(
                ctrl.poll_interrupt_probe(&dev, probe),
                crate::ProbePoll::Complete
            ),
            "one poll pass finds the exact No-op completion"
        );
        assert!(
            dev.interrupt_acknowledged(0),
            "completion publishes ERDP and clears the interrupt in order"
        );
    }

    /// The **delivery-dead machine**, which is the whole reason the probe
    /// stopped spinning: a controller that never completes the command
    /// leaves every poll pass dry, and each one hands the affine witness
    /// straight back instead of burning a budget.  The caller owns the
    /// waiting (park on its route until a deadline), so a dry pass here
    /// must be cheap and repeatable — and the witness must survive it,
    /// since abandoning is the only lawful way to end the probe.
    #[test]
    fn interrupt_probe_poll_hands_the_witness_back_when_delivery_is_dead() {
        let dev = MockXhciDevice::new_swallowing();
        let pool = MockPool;
        let mut ctrl =
            Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up still works");

        let mut held = ctrl.arm_interrupt_probe(&dev);
        for _ in 0..3 {
            held = match ctrl.poll_interrupt_probe(&dev, held) {
                crate::ProbePoll::Pending(p) => p,
                crate::ProbePoll::Complete => panic!("a swallowed command cannot complete"),
                crate::ProbePoll::Failed(code) => panic!("unexpected completion code {code}"),
            };
        }
        // The deadline leg: abandon consumes the witness and aborts the
        // ring, so the stalled command cannot hold the next one behind it.
        ctrl.abandon_interrupt_probe(&dev, held, || None, spin);
        assert_eq!(
            dev.read32(OP + CRCR) & (CRCR_CRR as u32),
            0,
            "abandoning the probe stops the command ring"
        );
    }

    #[test]
    fn enable_slot_returns_a_slot_id() {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        let es: Trb = command::EnableSlot::new().into();
        let cc = ctrl
            .command(&dev, es, || None, spin)
            .expect("enable slot completes");
        assert_eq!(cc.slot_id(), 1, "first slot id");
    }

    #[test]
    fn several_commands_correlate_by_trb_pointer() {
        // Three commands in a row must each match their own completion — the
        // xHCI correlation (command TRB device address), not FIFO luck.
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        for expect_slot in 1..=3u8 {
            let es: Trb = command::EnableSlot::new().into();
            let cc = ctrl.command(&dev, es, || None, spin).expect("completes");
            assert_eq!(cc.slot_id(), expect_slot);
        }
    }

    #[test]
    fn command_times_out_and_aborts_when_swallowed() {
        let dev = MockXhciDevice::new_swallowing();
        let pool = MockPool;
        let mut ctrl =
            Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up still works");
        let noop: Trb = command::Noop::new().into();
        let r = ctrl.command(&dev, noop, || None, spin);
        assert_eq!(
            r.err(),
            Some(crate::Error::Timeout),
            "no completion → timeout+abort"
        );
        // The abort cleared CRR.
        assert_eq!(
            dev.read32(OP + CRCR) & (CRCR_CRR as u32),
            0,
            "command ring stopped"
        );
    }

    // ── Enumeration (X1b): the mock kbd (port 1) + mouse (port 2) ─────────

    #[test]
    fn ports_report_connected_then_enable_on_reset() {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        // Both device ports show connected but not-yet-enabled.
        let p1 = ctrl.port(&dev, 1);
        assert!(p1.connected());
        assert!(!p1.enabled(), "USB2 port not enabled before reset");
        assert_eq!(p1.state(), crate::PortState::Connected);
        // Reset enables it and latches Full speed.
        let speed = ctrl.reset_port(&dev, 1, || None, spin).expect("reset");
        assert_eq!(speed, crate::Speed::Full);
        assert!(ctrl.port(&dev, 1).enabled());
        // Port 3 is empty.
        assert!(!ctrl.port(&dev, 3).connected());
    }

    /// End to end: reset → address → read device + config descriptors →
    /// SET_CONFIGURATION → Configure Endpoint, for a given port.
    fn enumerate(port_num: u8) -> (crate::DeviceDescriptor, u8, bool) {
        use crate::descriptor::{ConfigDescriptor, ConfigItem, ConfigWalk, DeviceDescriptor};
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");

        ctrl.reset_port(&dev, port_num, || None, spin)
            .expect("reset");
        let port = ctrl.port(&dev, port_num);
        let mut usb = ctrl
            .address_device(&dev, &pool, &port, || None, spin)
            .expect("address");

        // Device descriptor.
        let n = ctrl
            .get_descriptor(&dev, &mut usb, 1, 0, 18, || None, spin)
            .expect("device desc");
        let dd = DeviceDescriptor::parse(usb.read_data(n)).expect("parse device");

        // Config descriptor: 9-byte header, then the full wTotalLength.
        ctrl.get_descriptor(&dev, &mut usb, 2, 0, 9, || None, spin)
            .expect("config hdr");
        let hdr = ConfigDescriptor::parse(usb.read_data(9)).expect("parse config");
        let total = hdr.total_length;
        let n = ctrl
            .get_descriptor(&dev, &mut usb, 2, 0, total, || None, spin)
            .expect("config full");
        let cfg = usb.read_data(n);

        // Find the HID boot interface + its interrupt IN endpoint.
        let mut protocol = 0u8;
        let mut ep = None;
        for item in ConfigWalk::new(cfg) {
            match item {
                ConfigItem::Interface(i) if i.class == 3 && i.subclass == 1 => {
                    protocol = i.protocol
                }
                ConfigItem::Endpoint(e) if e.is_in() && e.is_interrupt() => ep = Some(e),
                _ => {}
            }
        }
        let ep = ep.expect("interrupt IN endpoint");

        ctrl.set_configuration(&dev, &mut usb, hdr.value, || None, spin)
            .expect("set config");
        ctrl.configure_endpoint(&dev, &pool, &mut usb, &ep, || None, spin)
            .expect("configure endpoint");

        (dd, protocol, usb.int_ep().is_some())
    }

    #[test]
    fn keyboard_enumerates_as_hid_boot_keyboard() {
        let (dd, protocol, configured) = enumerate(1);
        assert_eq!(dd.vendor, 0x0627);
        assert_eq!(dd.product, 0x0001);
        assert_eq!(dd.max_packet_size0, 8);
        assert_eq!(protocol, 1, "HID boot protocol 1 = keyboard");
        assert!(configured, "interrupt IN endpoint configured");
    }

    #[test]
    fn mouse_enumerates_as_hid_boot_mouse() {
        let (dd, protocol, configured) = enumerate(2);
        assert_eq!(dd.product, 0x0002);
        assert_eq!(protocol, 2, "HID boot protocol 2 = mouse");
        assert!(configured);
    }

    #[test]
    fn enumerates_on_a_64byte_context_controller() {
        // CSZ=1: the driver builds Input<16> at a 16-dword stride and the mock
        // reads it at the same stride — enumeration + Configure Endpoint still
        // work (the real-Intel-metal path).
        use crate::descriptor::{ConfigDescriptor, ConfigItem, ConfigWalk, DeviceDescriptor};
        let dev = MockXhciDevice::new_csz64();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        assert!(ctrl.caps().csz64, "64-byte contexts");
        ctrl.reset_port(&dev, 1, || None, spin).expect("reset");
        let port = ctrl.port(&dev, 1);
        let mut usb = ctrl
            .address_device(&dev, &pool, &port, || None, spin)
            .expect("address");

        let n = ctrl
            .get_descriptor(&dev, &mut usb, 1, 0, 18, || None, spin)
            .expect("device desc");
        let dd = DeviceDescriptor::parse(usb.read_data(n)).expect("parse");
        assert_eq!(dd.vendor, 0x0627);
        assert_eq!(dd.product, 0x0001, "keyboard on port 1");

        ctrl.get_descriptor(&dev, &mut usb, 2, 0, 9, || None, spin)
            .unwrap();
        let hdr = ConfigDescriptor::parse(usb.read_data(9)).unwrap();
        ctrl.get_descriptor(&dev, &mut usb, 2, 0, hdr.total_length, || None, spin)
            .unwrap();
        let mut ep = None;
        for item in ConfigWalk::new(usb.read_data(hdr.total_length as usize)) {
            if let ConfigItem::Endpoint(e) = item {
                if e.is_in() && e.is_interrupt() {
                    ep = Some(e);
                }
            }
        }
        ctrl.set_configuration(&dev, &mut usb, hdr.value, || None, spin)
            .unwrap();
        ctrl.configure_endpoint(&dev, &pool, &mut usb, &ep.unwrap(), || None, spin)
            .expect("configure endpoint (64-byte)");
        assert!(usb.int_ep().is_some(), "interrupt endpoint configured");
    }

    // ── HID boot protocol (X1c): report path over the interrupt endpoint ──

    /// Enumerate + configure the device on `port_num`, select boot protocol,
    /// and return the live controller + device ready to arm.
    fn hid_ready(
        port_num: u8,
    ) -> (
        MockXhciDevice,
        crate::Controller<MockXhciDevice, MockPool>,
        crate::Device<MockPool>,
    ) {
        use crate::descriptor::{ConfigDescriptor, ConfigItem, ConfigWalk};
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        ctrl.reset_port(&dev, port_num, || None, spin)
            .expect("reset");
        let port = ctrl.port(&dev, port_num);
        let mut usb = ctrl
            .address_device(&dev, &pool, &port, || None, spin)
            .expect("address");

        ctrl.get_descriptor(&dev, &mut usb, 2, 0, 9, || None, spin)
            .unwrap();
        let hdr = ConfigDescriptor::parse(usb.read_data(9)).unwrap();
        ctrl.get_descriptor(&dev, &mut usb, 2, 0, hdr.total_length, || None, spin)
            .unwrap();
        let mut ep = None;
        for item in ConfigWalk::new(usb.read_data(hdr.total_length as usize)) {
            if let ConfigItem::Endpoint(e) = item {
                if e.is_in() && e.is_interrupt() {
                    ep = Some(e);
                }
            }
        }
        let ep = ep.expect("interrupt IN endpoint");
        ctrl.set_configuration(&dev, &mut usb, hdr.value, || None, spin)
            .unwrap();
        ctrl.configure_endpoint(&dev, &pool, &mut usb, &ep, || None, spin)
            .unwrap();
        ctrl.set_protocol(&dev, &mut usb, true, 0, || None, spin)
            .unwrap();
        ctrl.set_idle(&dev, &mut usb, 0, 0, || None, spin).unwrap();
        (dev, ctrl, usb)
    }

    #[test]
    fn keyboard_keypress_arrives_and_decodes() {
        let (dev, mut ctrl, mut usb) = hid_ready(1);
        // Arm, inject 'a' (usage 0x04) with Left-Shift held, poll, decode.
        ctrl.arm_interrupt(&dev, &mut usb, 0).unwrap();
        dev.inject_report(usb.slot_id, &[0x02, 0x00, 0x04, 0, 0, 0, 0, 0]);
        let len = ctrl.poll_interrupt(&dev, &usb).expect("report arrived");
        assert!(
            dev.interrupt_acknowledged(0),
            "report batch clears EINT, IP, and EHB in order"
        );
        let report = KeyboardReport::parse(usb.read_report(len)).unwrap();
        assert_eq!(report.modifiers, 0x02, "Left-Shift");
        let pressed: alloc::vec::Vec<u8> = report.pressed().collect();
        assert_eq!(pressed, alloc::vec![0x04], "'a' pressed");
        // Nothing further pending → the next poll is empty.
        ctrl.arm_interrupt(&dev, &mut usb, 0).unwrap();
        assert!(ctrl.poll_interrupt(&dev, &usb).is_none());
    }

    #[test]
    fn mouse_movement_arrives_and_decodes() {
        let (dev, mut ctrl, mut usb) = hid_ready(2);
        // Inject before arming (the report waits) — left button, +5 X, -3 Y.
        dev.inject_report(usb.slot_id, &[0x01, 5, (-3i8) as u8]);
        ctrl.arm_interrupt(&dev, &mut usb, 0).unwrap();
        let len = ctrl.poll_interrupt(&dev, &usb).expect("report arrived");
        let report = MouseReport::parse(usb.read_report(len)).unwrap();
        assert_eq!(report.buttons, 0x01);
        assert_eq!(report.dx, 5);
        assert_eq!(report.dy, -3);
    }

    #[test]
    fn interrupt_report_routes_to_a_secondary_interrupter() {
        // Multi-interrupter groundwork: arm the HID endpoint targeting
        // interrupter 1, and the report lands on interrupter 1's private event
        // ring — not the controller's interrupter 0.  (The per-core path.)
        let (dev, mut ctrl, mut usb) = hid_ready(1);
        let pool = MockPool;
        let mut intr1 = ctrl
            .create_interrupter(&dev, &pool, 1)
            .expect("interrupter 1");
        assert_eq!(intr1.index(), 1);
        let dci = usb.int_ep().unwrap().dci;

        ctrl.arm_interrupt(&dev, &mut usb, 1).unwrap();
        dev.inject_report(usb.slot_id, &[0x00, 0x00, 0x04, 0, 0, 0, 0, 0]);

        // Interrupter 0 (the controller's own) is empty; interrupter 1 has it.
        assert!(
            ctrl.poll_interrupt(&dev, &usb).is_none(),
            "completion routed away from interrupter 0"
        );
        let (slot, ep, residual) = intr1.poll(&dev).expect("report on interrupter 1");
        assert!(
            dev.interrupt_acknowledged(1),
            "secondary interrupter uses the full ack sequence"
        );
        assert_eq!(slot, usb.slot_id);
        assert_eq!(ep, dci);
        let report = KeyboardReport::parse(usb.read_report(8 - residual as usize)).unwrap();
        let pressed: alloc::vec::Vec<u8> = report.pressed().collect();
        assert_eq!(pressed, alloc::vec![0x04]);

        // A secondary interrupter can't be interrupter 0 or ≥ MaxIntrs.
        assert!(ctrl.create_interrupter(&dev, &pool, 0).is_err());
        assert!(ctrl.create_interrupter(&dev, &pool, 8).is_err());
    }

    // ── CDC-ECM: multi-endpoint configuration and the bulk data plane ──

    /// A [`MockPool`] that counts draws, so a test can assert a refusal path
    /// allocated **nothing** — in a bump allocator that never frees, a draw
    /// on a failed path is a permanent leak.
    struct TallyPool(MockPool, core::cell::Cell<usize>);

    impl TallyPool {
        fn new() -> Self {
            Self(MockPool, core::cell::Cell::new(0))
        }
        fn draws(&self) -> usize {
            self.1.get()
        }
    }

    impl DmaPool for TallyPool {
        type Buf = MockBuf;
        fn alloc(&self, bytes: usize) -> Option<MockBuf> {
            self.1.set(self.1.get() + 1);
            self.0.alloc(bytes)
        }
    }

    use crate::InterruptEvent;
    use crate::descriptor::{ConfigDescriptor, ConfigItem, ConfigWalk};
    use crate::device::EndpointKind;

    /// Bring the mock's ECM device up to a configured bulk pair, returning
    /// the controller and the addressed device.  This is the enumeration a
    /// networking class driver performs, done here against the generic xHCI
    /// surface so the surface itself is what gets proved.
    fn ecm_device<P: DmaPool>(
        dev: &MockXhciDevice,
        pool: &P,
    ) -> (Controller<MockXhciDevice, P>, crate::device::Device<P>) {
        let mut ctrl = Controller::bring_up(dev, pool, || None, spin).expect("bring-up");
        let port = ctrl.port(dev, 3);
        let mut usb = ctrl
            .address_device(dev, pool, &port, || None, spin)
            .expect("address the ECM device");

        // Select the configuration by DESCRIPTOR, walking the array by index.
        // `bConfigurationValue` is a field inside the descriptor, not the
        // index — and on this device (as on QEMU's) they are crossed over, so
        // taking index 0, or assuming value 1 lives at index 0, gets RNDIS.
        let n = ctrl
            .get_descriptor(dev, &mut usb, 1, 0, 18, || None, spin)
            .expect("device descriptor");
        let dd =
            crate::descriptor::DeviceDescriptor::parse(usb.read_data(n)).expect("parse device");
        assert!(
            dd.num_configurations >= 2,
            "the trap needs both configurations"
        );

        let mut chosen = None;
        for index in 0..dd.num_configurations {
            ctrl.get_descriptor(dev, &mut usb, 2, index, 9, || None, spin)
                .expect("config header");
            let hdr = ConfigDescriptor::parse(usb.read_data(9)).expect("parse config");
            let n = ctrl
                .get_descriptor(dev, &mut usb, 2, index, hdr.total_length, || None, spin)
                .expect("full config");
            let is_ecm = ConfigWalk::new(usb.read_data(n)).any(|item| {
                matches!(item, ConfigItem::Interface(i) if i.class == 0x02 && i.subclass == 0x06)
            });
            if is_ecm {
                chosen = Some((index, hdr));
                break;
            }
        }
        let (index, hdr) = chosen.expect("a CDC-ECM configuration");
        assert_eq!(index, 1, "ECM is the second configuration descriptor");
        assert_eq!(hdr.value, ECM_CONFIG_VALUE, "…but its value is 1");

        let n = ctrl
            .get_descriptor(dev, &mut usb, 2, index, hdr.total_length, || None, spin)
            .expect("full config");

        // The bulk pair lives on the data interface's altsetting 1.
        let mut current: Option<crate::descriptor::InterfaceDescriptor> = None;
        let mut bulk: alloc::vec::Vec<crate::descriptor::EndpointDescriptor> =
            alloc::vec::Vec::new();
        for item in ConfigWalk::new(usb.read_data(n)) {
            match item {
                ConfigItem::Interface(i) => current = Some(i),
                ConfigItem::Endpoint(e) if e.is_bulk() => {
                    if current.is_some_and(|i| i.alternate == 1) {
                        bulk.push(e);
                    }
                }
                _ => {}
            }
        }
        assert_eq!(bulk.len(), 2, "one bulk pair on altsetting 1");

        ctrl.set_configuration(dev, &mut usb, hdr.value, || None, spin)
            .expect("set configuration");
        ctrl.set_interface(dev, &mut usb, 1, 1, || None, spin)
            .expect("select the data altsetting");
        ctrl.configure_endpoints(dev, pool, &mut usb, &bulk, || None, spin)
            .expect("configure the bulk pair");
        (ctrl, usb)
    }

    /// A CDC data interface's bulk IN and OUT come up in **one** Configure
    /// Endpoint command.  Both must land, and the slot context's Context
    /// Entries must name the higher DCI — a per-endpoint command would
    /// publish an intermediate slot context that understates the set.
    #[test]
    fn configures_a_bulk_pair_in_one_command() {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (_ctrl, usb) = ecm_device(&dev, &pool);

        let bin = usb
            .endpoint_of(EndpointKind::BulkIn)
            .expect("bulk IN configured");
        let bout = usb
            .endpoint_of(EndpointKind::BulkOut)
            .expect("bulk OUT configured");
        assert_eq!(bin.dci, 5, "EP2 IN → DCI 2·2+1");
        assert_eq!(bout.dci, 4, "EP2 OUT → DCI 2·2+0");
        assert_eq!(bin.mps, 64, "QEMU's usb-net bulk packet size");
        assert_eq!(bout.mps, 64);
        // Bulk endpoints hold no driver-owned buffer: they transfer out of
        // memory the caller names.
        assert!(bin.data.is_none() && bout.data.is_none());
        // The device is not a HID device, so nothing claims the interrupt
        // accessor even though the ECM config declares a notification EP —
        // it was simply not part of this Configure Endpoint batch.
        assert!(usb.int_ep().is_none());
    }

    /// The whole transmit path: submit several frames, reap their Transfer
    /// Events, and correlate each back to the cookie it was submitted with.
    /// Cookies must come back **in submission order and unchanged** — this is
    /// what lets a caller above own the meaning of a transfer.
    #[test]
    fn bulk_transmit_correlates_every_completion_to_its_cookie() {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);
        let out = usb.dci_of(EndpointKind::BulkOut).expect("bulk OUT");

        let buf = pool.alloc(0x1000).expect("frame buffer");
        let frames: [&[u8]; 3] = [b"first frame", b"second", b"third frame here"];
        for (i, f) in frames.iter().enumerate() {
            // Stage the bytes where the controller will read them.
            // SAFETY: the pool hands out a whole leaked page; each frame is
            // far shorter, and the writes are disjoint by offset.
            unsafe { core::ptr::copy_nonoverlapping(f.as_ptr(), buf.cpu().add(i * 256), f.len()) };
            ctrl.bulk_submit(
                &dev,
                &mut usb,
                out,
                IovaAddr(buf.iova().0 + (i * 256) as u64),
                f.len() as u16,
                0xC0DE_0000 + i as u64,
                0,
            )
            .expect("submit");
        }
        assert_eq!(
            usb.endpoint_of(EndpointKind::BulkOut)
                .unwrap()
                .outstanding(),
            3
        );

        let mut events = [InterruptEvent::default(); 8];
        let n = ctrl.drain_interrupts(&dev, &mut events);
        assert_eq!(n, 3, "one Transfer Event per submitted frame");

        let mut cookies = alloc::vec::Vec::new();
        for ev in &events[..n] {
            let done = usb.complete_bulk(ev).expect("correlates");
            assert!(done.ok());
            cookies.push((done.cookie, done.transferred));
        }
        assert_eq!(
            cookies,
            alloc::vec![(0xC0DE_0000, 11), (0xC0DE_0001, 6), (0xC0DE_0002, 16),]
        );
        assert_eq!(
            usb.endpoint_of(EndpointKind::BulkOut)
                .unwrap()
                .outstanding(),
            0,
            "every submission retired"
        );

        // The bytes really reached the wire, in order.
        let sent = dev.transmitted();
        assert_eq!(sent.len(), 3);
        for (got, want) in sent.iter().zip(frames.iter()) {
            assert_eq!(got.as_slice(), *want);
        }
    }

    /// A frame written to a looped-back wire comes back on the receive side,
    /// short-packet residual and all.  This is the round trip a boot-time
    /// self-test asserts, with no second device involved.
    #[test]
    fn a_frame_round_trips_through_the_looped_back_wire() {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);
        dev.set_loopback(true);
        let out = usb.dci_of(EndpointKind::BulkOut).expect("bulk OUT");
        let inn = usb.dci_of(EndpointKind::BulkIn).expect("bulk IN");

        // Post a receive buffer first — a receive path that only posts after
        // a frame arrives drops the first one.
        let rx = pool.alloc(0x1000).expect("rx buffer");
        ctrl.bulk_submit(&dev, &mut usb, inn, rx.iova(), 1514, 0x5EED_0001, 0)
            .expect("post receive");

        let tx = pool.alloc(0x1000).expect("tx buffer");
        let frame = b"\xff\xff\xff\xff\xff\xff\x0a\x1b\x2c\x3d\x4e\x5f\x88\xb5payload";
        // SAFETY: a whole leaked page backs `tx`; the frame is far shorter.
        unsafe { core::ptr::copy_nonoverlapping(frame.as_ptr(), tx.cpu(), frame.len()) };
        ctrl.bulk_submit(
            &dev,
            &mut usb,
            out,
            tx.iova(),
            frame.len() as u16,
            0x5EED_0002,
            0,
        )
        .expect("transmit");

        let mut events = [InterruptEvent::default(); 8];
        let n = ctrl.drain_interrupts(&dev, &mut events);
        assert_eq!(n, 2, "the transmit completion and the receive completion");

        let mut received = None;
        let mut transmitted = false;
        for ev in &events[..n] {
            let done = usb.complete_bulk(ev).expect("correlates");
            assert!(done.ok());
            if done.cookie == 0x5EED_0002 {
                transmitted = true;
            } else {
                received = Some(done.transferred);
            }
        }
        assert!(transmitted, "the transmit retired");
        assert_eq!(
            received,
            Some(frame.len() as u32),
            "the receive reports the frame length, not the posted buffer size"
        );

        // SAFETY: the controller wrote the frame into the posted page.
        let got = unsafe { core::slice::from_raw_parts(rx.cpu() as *const u8, frame.len()) };
        assert_eq!(got, frame, "the same bytes came back off the wire");
    }

    /// A Transfer Event naming a TRB that is not the endpoint's oldest
    /// outstanding transfer is **refused**, not attributed to whatever is at
    /// the head of the queue.  Handing a caller someone else's cookie is a
    /// worse failure than dropping a completion, so this must return `None`
    /// and leave the queue untouched.
    #[test]
    fn a_completion_naming_the_wrong_trb_refuses_rather_than_misattributing() {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);
        let out = usb.dci_of(EndpointKind::BulkOut).expect("bulk OUT");

        let buf = pool.alloc(0x1000).expect("buffer");
        ctrl.bulk_submit(&dev, &mut usb, out, buf.iova(), 64, 0xAA, 0)
            .expect("submit");

        let mut events = [InterruptEvent::default(); 4];
        let n = ctrl.drain_interrupts(&dev, &mut events);
        assert_eq!(n, 1);

        // Same slot, same endpoint, wrong TRB address.
        let mut forged = events[0];
        forged.trb ^= 0x10;
        assert_eq!(usb.complete_bulk(&forged), None, "wrong TRB refuses");
        assert_eq!(
            usb.endpoint_of(EndpointKind::BulkOut)
                .unwrap()
                .outstanding(),
            1,
            "the refusal consumed nothing"
        );

        // An event for a slot that is not this device refuses too.
        let mut other_slot = events[0];
        other_slot.slot_id = other_slot.slot_id.wrapping_add(1);
        assert_eq!(usb.complete_bulk(&other_slot), None, "foreign slot refuses");

        // An event for an endpoint this device does not have refuses.
        let mut other_ep = events[0];
        other_ep.endpoint_id = 31;
        assert_eq!(
            usb.complete_bulk(&other_ep),
            None,
            "unknown endpoint refuses"
        );

        // The genuine event still correlates — none of the forgeries
        // corrupted the queue.
        let done = usb
            .complete_bulk(&events[0])
            .expect("the real one correlates");
        assert_eq!(done.cookie, 0xAA);
    }

    /// The correlation table is bounded, and submitting past that bound is a
    /// refusal — never an overwrite, which would silently orphan a caller's
    /// cookie.
    #[test]
    fn bulk_submit_refuses_past_the_inflight_bound() {
        use crate::device::MAX_INFLIGHT;

        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);
        // Nothing may complete while the bound is probed, or the queue drains
        // underneath the test.
        dev.st.borrow_mut().swallow = true;
        let inn = usb.dci_of(EndpointKind::BulkIn).expect("bulk IN");
        let buf = pool.alloc(0x1000).expect("buffer");

        for i in 0..MAX_INFLIGHT {
            ctrl.bulk_submit(&dev, &mut usb, inn, buf.iova(), 64, i as u64, 0)
                .expect("within the bound");
        }
        assert!(usb.endpoint_of(EndpointKind::BulkIn).unwrap().full());
        assert!(
            ctrl.bulk_submit(&dev, &mut usb, inn, buf.iova(), 64, 999, 0)
                .is_err(),
            "one past the bound refuses"
        );
        assert_eq!(
            usb.endpoint_of(EndpointKind::BulkIn).unwrap().outstanding() as usize,
            MAX_INFLIGHT,
            "the refusal recorded nothing"
        );
    }

    /// `bulk_submit` is for bulk endpoints.  Naming an unconfigured DCI, or a
    /// configured interrupt endpoint, refuses.
    #[test]
    fn bulk_submit_refuses_a_non_bulk_or_unconfigured_endpoint() {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);
        let buf = pool.alloc(0x1000).expect("buffer");

        assert!(
            ctrl.bulk_submit(&dev, &mut usb, 9, buf.iova(), 64, 1, 0)
                .is_err(),
            "an unconfigured DCI refuses"
        );
        assert!(
            ctrl.bulk_submit(&dev, &mut usb, 1, buf.iova(), 64, 1, 0)
                .is_err(),
            "EP0 is not submittable through the bulk path"
        );
    }

    /// A refused endpoint set must issue no command **and allocate nothing** —
    /// a pool draw on the refusal path is a leak in a bump allocator that
    /// never frees.
    #[test]
    fn configure_endpoints_refuses_bad_sets_without_allocating() {
        use crate::device::MAX_ENDPOINTS;

        let dev = MockXhciDevice::new_with_ecm();
        let pool = TallyPool::new();
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);
        let baseline = pool.draws();

        let bulk_in = crate::descriptor::EndpointDescriptor {
            address: 0x85,
            attributes: 0x02,
            max_packet_size: 64,
            interval: 0,
        };

        // A DCI that collides with a live endpoint — the bulk IN this device
        // already has.
        let collide = crate::descriptor::EndpointDescriptor {
            address: 0x82,
            ..bulk_in
        };
        assert!(
            ctrl.configure_endpoints(&dev, &pool, &mut usb, &[collide], || None, spin)
                .is_err(),
            "colliding with a live endpoint refuses"
        );

        // Two descriptors in one batch naming the same DCI.
        assert!(
            ctrl.configure_endpoints(&dev, &pool, &mut usb, &[bulk_in, bulk_in], || None, spin)
                .is_err(),
            "a batch that collides with itself refuses"
        );

        // A transfer type this driver does not configure (isochronous).
        let isoch = crate::descriptor::EndpointDescriptor {
            attributes: 0x01,
            ..bulk_in
        };
        assert!(
            ctrl.configure_endpoints(&dev, &pool, &mut usb, &[isoch], || None, spin)
                .is_err(),
            "an unsupported transfer type refuses"
        );

        // More endpoints than the device can hold.
        let many: alloc::vec::Vec<_> = (0..MAX_ENDPOINTS + 1)
            .map(|i| crate::descriptor::EndpointDescriptor {
                address: 0x85 + i as u8,
                ..bulk_in
            })
            .collect();
        assert!(
            ctrl.configure_endpoints(&dev, &pool, &mut usb, &many, || None, spin)
                .is_err(),
            "past MAX_ENDPOINTS refuses"
        );

        assert_eq!(pool.draws(), baseline, "no refusal path drew from the pool");

        // An empty set is a no-op, not an error.
        assert!(
            ctrl.configure_endpoints(&dev, &pool, &mut usb, &[], || None, spin)
                .is_ok()
        );
        assert_eq!(pool.draws(), baseline, "an empty set draws nothing either");
    }

    /// A CDC Ethernet device publishes its MAC as a **string descriptor**, so
    /// the generic surface has to be able to read one and decode UTF-16LE.
    #[test]
    fn reads_a_cdc_mac_from_its_string_descriptor() {
        use crate::descriptor::string_ascii;

        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);

        // The Ethernet Networking functional descriptor names the index — and
        // it lives in the ECM configuration, which is descriptor index 1.
        ctrl.get_descriptor(&dev, &mut usb, 2, ECM_CONFIG_INDEX, 9, || None, spin)
            .unwrap();
        let hdr = ConfigDescriptor::parse(usb.read_data(9)).unwrap();
        let n = ctrl
            .get_descriptor(
                &dev,
                &mut usb,
                2,
                ECM_CONFIG_INDEX,
                hdr.total_length,
                || None,
                spin,
            )
            .unwrap();
        let mac_index = ConfigWalk::new(usb.read_data(n))
            .find_map(|i| match i {
                ConfigItem::Class(c) if c.kind == 0x24 && c.subtype == 0x0F => c.byte(0),
                _ => None,
            })
            .expect("iMACAddress index");

        let n = ctrl
            .get_string_descriptor(
                &dev,
                &mut usb,
                mac_index,
                Controller::<MockXhciDevice, MockPool>::LANGID_EN_US,
                64,
                || None,
                spin,
            )
            .expect("string descriptor");
        let mut ascii = [0u8; 32];
        let len = string_ascii(usb.read_data(n), &mut ascii).expect("ASCII hex");
        assert_eq!(&ascii[..len], b"400102030405");

        // And it decodes to the MAC the mock's device actually has.
        let mut mac = [0u8; 6];
        for (i, byte) in mac.iter_mut().enumerate() {
            let hi = (ascii[i * 2] as char).to_digit(16).unwrap() as u8;
            let lo = (ascii[i * 2 + 1] as char).to_digit(16).unwrap() as u8;
            *byte = (hi << 4) | lo;
        }
        assert_eq!(mac, ECM_MAC);
    }

    /// A host-to-device class request really reaches the device: the SETUP
    /// packet's direction bit drives the data and status stages, and the mock
    /// records what it was told.  `SET_INTERFACE` (standard) and a CDC packet
    /// filter (class) are both on the bring-up path.
    #[test]
    fn control_requests_reach_the_device_with_the_right_direction() {
        const SET_ETHERNET_PACKET_FILTER: u8 = 0x43;
        const PACKET_TYPE_DIRECTED: u16 = 1 << 2;
        const PACKET_TYPE_BROADCAST: u16 = 1 << 1;

        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let (mut ctrl, mut usb) = ecm_device(&dev, &pool);

        let filter = PACKET_TYPE_DIRECTED | PACKET_TYPE_BROADCAST;
        ctrl.control(
            &dev,
            &mut usb,
            crate::device::Setup {
                request_type: 0x21, // host-to-device, class, interface
                request: SET_ETHERNET_PACKET_FILTER,
                value: filter,
                index: 0,
            },
            0,
            || None,
            spin,
        )
        .expect("the class request completes");

        let seen = dev.class_requests();
        assert!(
            seen.contains(&(SET_ETHERNET_PACKET_FILTER, filter, 0)),
            "the device was really told the filter: {seen:?}"
        );
    }
}
