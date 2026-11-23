//! **USB descriptor parsing** — the bytes a device returns over EP0, read
//! by field offset (USB 2.0 §9.6).  Pure and bounded: no allocation, no
//! trailing-length trust beyond the buffer the caller sized; a truncated
//! descriptor yields `None`, never a panic.
//!
//! Only what enumeration + HID boot need: device, configuration, interface,
//! endpoint.  String descriptors and class-specific (HID report) descriptors
//! are deliberately out of scope (the boot protocol needs none).

/// Descriptor type codes (`bDescriptorType`, USB 2.0 Table 9-5).
pub mod desc_type {
    /// Device descriptor.
    pub const DEVICE: u8 = 1;
    /// Configuration descriptor.
    pub const CONFIGURATION: u8 = 2;
    /// String descriptor — index 0 is the LANGID table, every other index a
    /// UTF-16LE string.  A CDC Ethernet interface publishes its MAC address
    /// as one of these, so this is not an optional convenience.
    pub const STRING: u8 = 3;
    /// Interface descriptor.
    pub const INTERFACE: u8 = 4;
    /// Endpoint descriptor.
    pub const ENDPOINT: u8 = 5;
}

/// USB device class 3 = HID.
pub const CLASS_HID: u8 = 3;
/// HID subclass 1 = boot interface.
pub const HID_SUBCLASS_BOOT: u8 = 1;
/// HID boot protocol 1 = keyboard.
pub const HID_PROTOCOL_KEYBOARD: u8 = 1;
/// HID boot protocol 2 = mouse.
pub const HID_PROTOCOL_MOUSE: u8 = 2;

#[inline]
fn u16le(b: &[u8], off: usize) -> u16 {
    (b[off] as u16) | ((b[off + 1] as u16) << 8)
}

/// The parsed **device descriptor** (USB 2.0 §9.6.1, 18 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceDescriptor {
    /// `bcdUSB` — USB spec release.
    pub usb: u16,
    /// `bDeviceClass`.
    pub class: u8,
    /// `bDeviceSubClass`.
    pub subclass: u8,
    /// `bDeviceProtocol`.
    pub protocol: u8,
    /// `bMaxPacketSize0` — EP0 max packet size.
    pub max_packet_size0: u8,
    /// `idVendor`.
    pub vendor: u16,
    /// `idProduct`.
    pub product: u16,
    /// `bNumConfigurations`.
    pub num_configurations: u8,
}

impl DeviceDescriptor {
    /// The minimum bytes a device descriptor occupies.
    pub const LEN: usize = 18;

    /// Parse from a returned buffer; `None` if too short or mistyped.
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < Self::LEN || b[1] != desc_type::DEVICE {
            return None;
        }
        Some(Self {
            usb: u16le(b, 2),
            class: b[4],
            subclass: b[5],
            protocol: b[6],
            max_packet_size0: b[7],
            vendor: u16le(b, 8),
            product: u16le(b, 10),
            num_configurations: b[17],
        })
    }
}

/// The parsed **configuration descriptor** header (USB 2.0 §9.6.3, 9 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConfigDescriptor {
    /// `wTotalLength` — bytes in the whole configuration (all nested
    /// interface/endpoint descriptors), the length the second GET_DESCRIPTOR
    /// requests.
    pub total_length: u16,
    /// `bNumInterfaces`.
    pub num_interfaces: u8,
    /// `bConfigurationValue` — the argument to SET_CONFIGURATION.
    pub value: u8,
}

impl ConfigDescriptor {
    /// Configuration descriptor header length.
    pub const LEN: usize = 9;

    /// Parse the header; `None` if too short or mistyped.
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < Self::LEN || b[1] != desc_type::CONFIGURATION {
            return None;
        }
        Some(Self {
            total_length: u16le(b, 2),
            num_interfaces: b[4],
            value: b[5],
        })
    }
}

/// An **interface descriptor** (USB 2.0 §9.6.5, 9 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterfaceDescriptor {
    /// `bInterfaceNumber`.
    pub number: u8,
    /// `bAlternateSetting`.
    pub alternate: u8,
    /// `bNumEndpoints` (excluding EP0).
    pub num_endpoints: u8,
    /// `bInterfaceClass`.
    pub class: u8,
    /// `bInterfaceSubClass`.
    pub subclass: u8,
    /// `bInterfaceProtocol`.
    pub protocol: u8,
}

impl InterfaceDescriptor {
    /// Whether this is a HID boot interface for `protocol` (kbd = 1, mouse
    /// = 2) — the match a boot-protocol driver keys on.
    pub fn is_hid_boot(&self, protocol: u8) -> bool {
        self.class == CLASS_HID && self.subclass == HID_SUBCLASS_BOOT && self.protocol == protocol
    }
}

/// An **endpoint descriptor** (USB 2.0 §9.6.6, 7 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndpointDescriptor {
    /// `bEndpointAddress` — bit 7 = direction (1 = IN), bits 3:0 = number.
    pub address: u8,
    /// `bmAttributes` — bits 1:0 = transfer type (3 = interrupt).
    pub attributes: u8,
    /// `wMaxPacketSize`.
    pub max_packet_size: u16,
    /// `bInterval` — polling interval (frames / 2^(n-1) µframes by speed).
    pub interval: u8,
}

impl EndpointDescriptor {
    /// The endpoint number (bits 3:0).
    pub fn number(&self) -> u8 {
        self.address & 0x0f
    }
    /// Whether the endpoint is IN (device-to-host).
    pub fn is_in(&self) -> bool {
        self.address & 0x80 != 0
    }
    /// Whether the endpoint is interrupt type (attributes bits 1:0 == 3).
    pub fn is_interrupt(&self) -> bool {
        self.attributes & 0x03 == 3
    }
    /// Whether the endpoint is bulk type (attributes bits 1:0 == 2) — the
    /// data pipes of a CDC networking interface, a mass-storage interface,
    /// and every other non-periodic class.
    pub fn is_bulk(&self) -> bool {
        self.attributes & 0x03 == 2
    }
    /// The **Device Context Index**: `2 × number + (IN ? 1 : 0)` — where this
    /// endpoint's context sits in the device/input context.
    pub fn dci(&self) -> u8 {
        2 * self.number() + if self.is_in() { 1 } else { 0 }
    }
}

/// Decode a **USB string descriptor** (`[bLength, 0x03, UTF-16LE …]`) into
/// ASCII bytes in `out`, returning how many were written.
///
/// USB strings are UTF-16LE.  Every string kore reads from a device is
/// machine-readable ASCII by its class's own definition (a CDC MAC address
/// is twelve hex digits), so a code unit outside ASCII is a **refusal**, not
/// a lossy substitution — decoding "0011²2334455" to a plausible-looking MAC
/// is exactly the silent corruption this returns `None` to prevent.
///
/// `None` when the buffer is not a well-formed string descriptor, when a
/// code unit is not ASCII, or when `out` is too small for the whole string:
/// refused, never truncated.
pub fn string_ascii(buf: &[u8], out: &mut [u8]) -> Option<usize> {
    // bLength counts the whole descriptor including its 2-byte header, and
    // must not claim more than the device actually returned.
    let len = *buf.first()? as usize;
    if buf.get(1).copied()? != desc_type::STRING || len < 2 || len > buf.len() {
        return None;
    }
    let body = &buf[2..len];
    // UTF-16LE: an odd body cannot be a whole number of code units.
    if body.len() % 2 != 0 {
        return None;
    }
    let chars = body.len() / 2;
    if chars > out.len() {
        return None;
    }
    for (i, unit) in body.chunks_exact(2).enumerate() {
        let c = u16::from_le_bytes([unit[0], unit[1]]);
        if c == 0 || c > 0x7f {
            return None;
        }
        out[i] = c as u8;
    }
    Some(chars)
}

/// A **class-specific (functional) descriptor** — a descriptor whose
/// `bDescriptorType` is outside the standard band, carrying a subtype byte
/// and a small payload.  CDC uses `CS_INTERFACE` (0x24) / `CS_ENDPOINT`
/// (0x25) this way, and it is how a CDC Ethernet interface publishes the
/// string index of its MAC address — so a walk that silently skips these
/// cannot bring up a network device.
///
/// The body is bounded inline ([`MAX_BODY`](Self::MAX_BODY)) rather than
/// borrowed, so [`ConfigItem`] stays `Copy` and lifetime-free.  Every CDC
/// functional descriptor fits (the Ethernet Networking one is 13 bytes).  A
/// longer descriptor is surfaced **truncated but with its true
/// [`len`](Self::len)**, so a caller reads [`truncated`](Self::truncated)
/// and refuses instead of silently parsing a short body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassDescriptor {
    /// `bDescriptorType` (CDC: 0x24 `CS_INTERFACE`, 0x25 `CS_ENDPOINT`).
    pub kind: u8,
    /// `bDescriptorSubtype` — the third byte, which every class-specific
    /// descriptor in this shape carries.
    pub subtype: u8,
    /// `bLength` as the device reported it, **before** any truncation.
    pub len: u8,
    body: [u8; Self::MAX_BODY],
}

impl ClassDescriptor {
    /// Bytes of payload retained after the 3-byte
    /// `bLength ‖ bDescriptorType ‖ bDescriptorSubtype` header.
    pub const MAX_BODY: usize = 16;

    /// The payload after the 3-byte header, clamped to what was retained.
    pub fn body(&self) -> &[u8] {
        let kept = (self.len as usize).saturating_sub(3).min(Self::MAX_BODY);
        &self.body[..kept]
    }

    /// Whether the device's descriptor was longer than [`MAX_BODY`] allows,
    /// so [`body`](Self::body) is a prefix and not the whole payload.
    pub fn truncated(&self) -> bool {
        (self.len as usize).saturating_sub(3) > Self::MAX_BODY
    }

    /// The payload byte at `offset` (relative to the end of the 3-byte
    /// header), or `None` when the descriptor is shorter than that — the
    /// bounds check a hand-transcribed class-layout offset needs.
    pub fn byte(&self, offset: usize) -> Option<u8> {
        self.body().get(offset).copied()
    }
}

/// One item yielded while walking a configuration's nested descriptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigItem {
    /// An interface descriptor.
    Interface(InterfaceDescriptor),
    /// An endpoint descriptor (belongs to the most recent interface).
    Endpoint(EndpointDescriptor),
    /// A class-specific functional descriptor belonging to the most recent
    /// interface (CDC's `CS_INTERFACE` family, a HID descriptor, an IAD).
    Class(ClassDescriptor),
}

/// A **forward walk** over a configuration descriptor's TLV stream — each
/// descriptor is `[bLength, bDescriptorType, …]`, so we hop by `bLength`,
/// surfacing the interfaces and endpoints and skipping class-specific items
/// (HID descriptors, IADs) the boot path ignores.
pub struct ConfigWalk<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ConfigWalk<'a> {
    /// Walk `buf` (the full configuration, `wTotalLength` bytes).
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl Iterator for ConfigWalk<'_> {
    type Item = ConfigItem;

    fn next(&mut self) -> Option<ConfigItem> {
        loop {
            // Need at least the 2-byte TLV header.
            if self.pos + 2 > self.buf.len() {
                return None;
            }
            let len = self.buf[self.pos] as usize;
            let ty = self.buf[self.pos + 1];
            // A zero length or an overrun terminates the walk (malformed).
            if len < 2 || self.pos + len > self.buf.len() {
                return None;
            }
            let item = &self.buf[self.pos..self.pos + len];
            self.pos += len;

            match ty {
                desc_type::INTERFACE if len >= InterfaceDescriptor::default_len() => {
                    return Some(ConfigItem::Interface(InterfaceDescriptor {
                        number: item[2],
                        alternate: item[3],
                        num_endpoints: item[4],
                        class: item[5],
                        subclass: item[6],
                        protocol: item[7],
                    }));
                }
                desc_type::ENDPOINT if len >= EndpointDescriptor::default_len() => {
                    return Some(ConfigItem::Endpoint(EndpointDescriptor {
                        address: item[2],
                        attributes: item[3],
                        max_packet_size: u16le(item, 4),
                        interval: item[6],
                    }));
                }
                // Standard descriptors this walk does not surface (the
                // configuration header itself, a device descriptor spliced
                // in): skip.  Everything else is class-specific and is
                // surfaced — CDC publishes the MAC-address string index
                // this way, so skipping would make a NIC unbringable.
                desc_type::DEVICE | desc_type::CONFIGURATION => continue,
                _ if len >= 3 => {
                    let mut body = [0u8; ClassDescriptor::MAX_BODY];
                    let payload = &item[3..];
                    let kept = payload.len().min(ClassDescriptor::MAX_BODY);
                    body[..kept].copy_from_slice(&payload[..kept]);
                    return Some(ConfigItem::Class(ClassDescriptor {
                        kind: ty,
                        subtype: item[2],
                        len: item[0],
                        body,
                    }));
                }
                // A 2-byte descriptor carries no subtype: nothing to surface.
                _ => continue,
            }
        }
    }
}

impl InterfaceDescriptor {
    const fn default_len() -> usize {
        9
    }
}
impl EndpointDescriptor {
    const fn default_len() -> usize {
        7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_device_descriptor() {
        let mut b = [0u8; 18];
        b[0] = 18;
        b[1] = desc_type::DEVICE;
        b[7] = 8; // MPS0
        b[8] = 0x27; // vendor 0x0627 (QEMU)
        b[9] = 0x06;
        b[10] = 0x01; // product 0x0001
        b[17] = 1; // one configuration
        let d = DeviceDescriptor::parse(&b).unwrap();
        assert_eq!(d.vendor, 0x0627);
        assert_eq!(d.product, 0x0001);
        assert_eq!(d.max_packet_size0, 8);
        assert_eq!(d.num_configurations, 1);
    }

    #[test]
    fn walks_a_hid_keyboard_configuration() {
        // config(9) + interface(9, class 3/1/1) + HID(9, skipped) + endpoint(7).
        let mut b = alloc::vec::Vec::new();
        b.extend_from_slice(&[9, desc_type::CONFIGURATION, 34, 0, 1, 1, 0, 0xA0, 50]);
        b.extend_from_slice(&[
            9,
            desc_type::INTERFACE,
            0,
            0,
            1,
            CLASS_HID,
            HID_SUBCLASS_BOOT,
            HID_PROTOCOL_KEYBOARD,
            0,
        ]);
        b.extend_from_slice(&[9, 0x21, 0x11, 0x01, 0, 1, 0x22, 63, 0]); // HID class descriptor
        b.extend_from_slice(&[7, desc_type::ENDPOINT, 0x81, 0x03, 8, 0, 10]); // EP1 IN, interrupt

        let cfg = ConfigDescriptor::parse(&b).unwrap();
        assert_eq!(cfg.total_length, 34);
        assert_eq!(cfg.value, 1);

        let items: alloc::vec::Vec<_> = ConfigWalk::new(&b).collect();
        // The config header is not yielded; interface, the HID class
        // descriptor, and the endpoint are.  The class arm is what a CDC
        // networking interface needs and HID simply ignores.
        assert_eq!(items.len(), 3);
        match items[0] {
            ConfigItem::Interface(i) => {
                assert!(i.is_hid_boot(HID_PROTOCOL_KEYBOARD));
                assert!(!i.is_hid_boot(HID_PROTOCOL_MOUSE));
            }
            _ => panic!("expected interface first"),
        }
        match items[1] {
            ConfigItem::Class(c) => {
                assert_eq!(c.kind, 0x21, "HID class descriptor");
                assert_eq!(c.subtype, 0x11, "bcdHID low byte lands in the subtype slot");
                assert_eq!(c.len, 9);
                assert!(!c.truncated());
                assert_eq!(c.body(), &[0x01, 0, 1, 0x22, 63, 0]);
            }
            _ => panic!("expected the class descriptor second"),
        }
        match items[2] {
            ConfigItem::Endpoint(e) => {
                assert!(e.is_in());
                assert!(e.is_interrupt());
                assert!(!e.is_bulk());
                assert_eq!(e.number(), 1);
                assert_eq!(e.dci(), 3, "EP1 IN → DCI 3");
            }
            _ => panic!("expected endpoint third"),
        }
    }

    /// The shape `kore-usbnet` keys on: a CDC Ethernet control interface
    /// carrying its functional descriptors, then a data interface whose
    /// **altsetting 1** is the one with the bulk pair (altsetting 0 has
    /// none — bringing up the wrong one yields a NIC that cannot move a
    /// byte).
    #[test]
    fn walks_a_cdc_ecm_configuration() {
        let mut b = alloc::vec::Vec::new();
        b.extend_from_slice(&[9, desc_type::CONFIGURATION, 71, 0, 2, 1, 0, 0xC0, 50]);
        // Communications control interface, subclass 6 = ECM.
        b.extend_from_slice(&[9, desc_type::INTERFACE, 0, 0, 1, 0x02, 0x06, 0x00, 0]);
        // CS_INTERFACE / Header functional descriptor.
        b.extend_from_slice(&[5, 0x24, 0x00, 0x10, 0x01]);
        // CS_INTERFACE / Ethernet Networking functional descriptor:
        // subtype 0x0F, iMACAddress = 4.
        b.extend_from_slice(&[13, 0x24, 0x0F, 4, 0, 0, 0, 0, 0xEA, 0x05, 0, 0, 0]);
        // The notification endpoint on the control interface.
        b.extend_from_slice(&[7, desc_type::ENDPOINT, 0x83, 0x03, 16, 0, 9]);
        // Data interface altsetting 0 — deliberately endpoint-less.
        b.extend_from_slice(&[9, desc_type::INTERFACE, 1, 0, 0, 0x0A, 0x00, 0x00, 0]);
        // Data interface altsetting 1 — the bulk pair.
        b.extend_from_slice(&[9, desc_type::INTERFACE, 1, 1, 2, 0x0A, 0x00, 0x00, 0]);
        b.extend_from_slice(&[7, desc_type::ENDPOINT, 0x81, 0x02, 0x00, 0x02, 0]);
        b.extend_from_slice(&[7, desc_type::ENDPOINT, 0x02, 0x02, 0x00, 0x02, 0]);

        let items: alloc::vec::Vec<_> = ConfigWalk::new(&b).collect();

        // The Ethernet Networking functional descriptor is reachable, and its
        // first payload byte is the MAC string index.
        let mac_index = items.iter().find_map(|i| match i {
            ConfigItem::Class(c) if c.kind == 0x24 && c.subtype == 0x0F => c.byte(0),
            _ => None,
        });
        assert_eq!(mac_index, Some(4), "iMACAddress string index");

        // Altsetting 1 of the data interface is the one carrying bulk.
        let mut current: Option<InterfaceDescriptor> = None;
        let mut bulk = alloc::vec::Vec::new();
        for item in &items {
            match item {
                ConfigItem::Interface(i) => current = Some(*i),
                ConfigItem::Endpoint(e) if e.is_bulk() => {
                    bulk.push((current.unwrap().number, current.unwrap().alternate, *e));
                }
                _ => {}
            }
        }
        assert_eq!(bulk.len(), 2, "one bulk pair");
        assert!(bulk.iter().all(|(n, alt, _)| *n == 1 && *alt == 1));
        assert!(bulk.iter().any(|(_, _, e)| e.is_in() && e.dci() == 3));
        assert!(bulk.iter().any(|(_, _, e)| !e.is_in() && e.dci() == 4));
        assert!(bulk.iter().all(|(_, _, e)| e.max_packet_size == 512));
    }

    /// A class descriptor longer than the inline bound is surfaced with its
    /// **true** length and a truncation flag, so a caller refuses rather
    /// than parsing a short body as if it were whole.
    #[test]
    fn an_oversized_class_descriptor_reports_truncation() {
        let mut b = alloc::vec::Vec::new();
        let long = 3 + ClassDescriptor::MAX_BODY + 4;
        b.push(long as u8);
        b.extend_from_slice(&[0x24, 0x0F]);
        b.extend(core::iter::repeat_n(0xAB, long - 3));

        let items: alloc::vec::Vec<_> = ConfigWalk::new(&b).collect();
        assert_eq!(items.len(), 1);
        match items[0] {
            ConfigItem::Class(c) => {
                assert_eq!(c.len as usize, long, "the device's own length survives");
                assert!(c.truncated());
                assert_eq!(c.body().len(), ClassDescriptor::MAX_BODY);
                assert_eq!(c.byte(ClassDescriptor::MAX_BODY), None, "past the body");
            }
            _ => panic!("expected a class descriptor"),
        }
    }

    /// A CDC MAC address arrives as a UTF-16LE string descriptor.  Anything
    /// that is not clean ASCII is refused — a lossy decode would hand back a
    /// plausible-looking but wrong MAC.
    #[test]
    fn string_descriptors_decode_ascii_and_refuse_everything_else() {
        let mut b = alloc::vec::Vec::new();
        b.extend_from_slice(&[2 + 24, desc_type::STRING]);
        for c in b"001122334455" {
            b.extend_from_slice(&[*c, 0]);
        }
        let mut out = [0u8; 32];
        let n = string_ascii(&b, &mut out).expect("clean ASCII decodes");
        assert_eq!(&out[..n], b"001122334455");

        // A non-ASCII code unit is a refusal, not a substitution.
        let mut bad = b.clone();
        bad[3] = 0x01; // 0x0130 — outside ASCII
        assert_eq!(string_ascii(&bad, &mut out), None);

        // Too small an output buffer refuses rather than truncating.
        let mut tiny = [0u8; 4];
        assert_eq!(string_ascii(&b, &mut tiny), None);

        // A wrong descriptor type, a short buffer, and an odd body all refuse.
        let mut wrong = b.clone();
        wrong[1] = desc_type::INTERFACE;
        assert_eq!(string_ascii(&wrong, &mut out), None);
        assert_eq!(string_ascii(&[2 + 24, desc_type::STRING], &mut out), None);
        assert_eq!(string_ascii(&[3, desc_type::STRING, 0x41], &mut out), None);
        assert_eq!(string_ascii(&[], &mut out), None);
    }
}
