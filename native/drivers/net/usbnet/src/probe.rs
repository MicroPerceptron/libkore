//! **The descriptor walk** — turning a configuration descriptor into a
//! bring-up plan, or a named refusal.
//!
//! Pure function over bytes: no hardware, no transfers, no state. That is
//! deliberate, because this is where a device's own claims are turned into
//! the addresses a driver will then poke, and it is the layer most worth
//! being able to test against hand-written adversarial descriptors.

use kore_xhci::descriptor::{ClassDescriptor, ConfigDescriptor, ConfigItem, ConfigWalk};
use kore_xhci::{EndpointDescriptor, InterfaceDescriptor};

use crate::cdc;

/// Everything bring-up needs from a device's descriptors, and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcmPlan {
    /// The **index** of the configuration in the device's array — what
    /// `GET_DESCRIPTOR(CONFIGURATION, n)` takes.
    pub config_index: u8,
    /// The configuration's `bConfigurationValue` — what `SET_CONFIGURATION`
    /// takes.
    ///
    /// Kept separate from the index on purpose. They are different numbers,
    /// they are routinely *not* equal, and QEMU's `usb-net` in particular
    /// puts RNDIS at index 0 with value 2 and CDC-ECM at index 1 with value
    /// 1 — so conflating them selects the wrong function entirely.
    pub config_value: u8,
    /// The Communications interface carrying the functional descriptors.
    pub control_interface: u8,
    /// The CDC Data interface carrying the bulk pair.
    pub data_interface: u8,
    /// The alternate setting of the data interface that actually has the
    /// bulk endpoints.
    pub data_altsetting: u8,
    /// String index of the twelve-hex-digit MAC address.
    pub mac_string: u8,
    /// `wMaxSegmentSize` the device advertises.
    pub max_segment: u16,
    /// The notification endpoint on the control interface.
    pub notify: EndpointDescriptor,
    /// Bulk IN — received frames.
    pub bulk_in: EndpointDescriptor,
    /// Bulk OUT — transmitted frames.
    pub bulk_out: EndpointDescriptor,
}

/// Why a configuration is not a usable CDC-ECM function.
///
/// Each variant names one specific missing thing, so a bring-up failure on a
/// machine with no serial console still says which rung it fell off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeError {
    /// The configuration descriptor's own header does not parse, or the
    /// buffer is shorter than it claims.
    Malformed,
    /// No interface with class 0x02 / subclass 0x06 — this configuration is
    /// something else (RNDIS, for instance).
    NotEcm,
    /// The control interface carries no Ethernet Networking functional
    /// descriptor.
    NoEthernetDescriptor,
    /// The Ethernet Networking functional descriptor is shorter than the
    /// class defines, or was longer than the walk retains — either way its
    /// fields cannot be read with confidence.
    BadEthernetDescriptor,
    /// No CDC Data interface alternate setting carries a bulk IN/OUT pair.
    ///
    /// The common cause is real: a data interface's altsetting 0 has no
    /// endpoints at all, and only altsetting 1 has the pair.
    NoBulkPair,
    /// The control interface has no interrupt IN endpoint for notifications.
    NoNotifyEndpoint,
}

/// Scan one configuration descriptor for the CDC-ECM shape.
///
/// `config` must be the **whole** configuration, `wTotalLength` bytes, not
/// just its nine-byte header.
pub fn plan_from_config(config: &[u8], config_index: u8) -> Result<EcmPlan, ProbeError> {
    let header = ConfigDescriptor::parse(config).ok_or(ProbeError::Malformed)?;
    if (header.total_length as usize) > config.len() {
        return Err(ProbeError::Malformed);
    }

    let mut current: Option<InterfaceDescriptor> = None;
    let mut control: Option<InterfaceDescriptor> = None;
    let mut ethernet: Option<ClassDescriptor> = None;
    let mut notify: Option<EndpointDescriptor> = None;
    // The bulk pair, together with the interface and altsetting that owns
    // them — an endpoint means nothing without knowing which setting to
    // select to make it exist.
    let mut bulk: Option<(u8, u8, EndpointDescriptor, EndpointDescriptor)> = None;
    let mut pending_in: Option<EndpointDescriptor> = None;
    let mut pending_out: Option<EndpointDescriptor> = None;
    let mut pending_iface: Option<InterfaceDescriptor> = None;

    for item in ConfigWalk::new(&config[..header.total_length as usize]) {
        match item {
            ConfigItem::Interface(iface) => {
                // A new interface descriptor closes the previous altsetting's
                // endpoint run.
                if bulk.is_none()
                    && let (Some(i), Some(o), Some(owner)) =
                        (pending_in, pending_out, pending_iface)
                {
                    bulk = Some((owner.number, owner.alternate, i, o));
                }
                pending_in = None;
                pending_out = None;
                pending_iface = Some(iface);
                current = Some(iface);
                if iface.class == cdc::CLASS_COMM && iface.subclass == cdc::SUBCLASS_ECM {
                    control = Some(iface);
                }
            }
            ConfigItem::Class(class) => {
                // Functional descriptors belong to the most recent interface;
                // only the control interface's Ethernet one is of interest.
                if class.kind == cdc::CS_INTERFACE
                    && class.subtype == cdc::subtype::ETHERNET
                    && current.is_some_and(|i| {
                        i.class == cdc::CLASS_COMM && i.subclass == cdc::SUBCLASS_ECM
                    })
                {
                    ethernet = Some(class);
                }
            }
            ConfigItem::Endpoint(ep) => {
                let owner = match current {
                    Some(i) => i,
                    // An endpoint before any interface descriptor is
                    // malformed; it belongs to nothing.
                    None => return Err(ProbeError::Malformed),
                };
                if owner.class == cdc::CLASS_COMM && ep.is_interrupt() && ep.is_in() {
                    notify.get_or_insert(ep);
                } else if owner.class == cdc::CLASS_CDC_DATA && ep.is_bulk() {
                    if ep.is_in() {
                        pending_in.get_or_insert(ep);
                    } else {
                        pending_out.get_or_insert(ep);
                    }
                }
            }
        }
    }
    // The last interface's endpoint run has no successor to close it.
    if bulk.is_none()
        && let (Some(i), Some(o), Some(owner)) = (pending_in, pending_out, pending_iface)
    {
        bulk = Some((owner.number, owner.alternate, i, o));
    }

    let control = control.ok_or(ProbeError::NotEcm)?;
    let ethernet = ethernet.ok_or(ProbeError::NoEthernetDescriptor)?;
    // A short descriptor cannot carry the fields; a truncated one means the
    // device declared more than the walk retained, so the tail is unknown.
    // Either way the values below would be read on faith.
    if ethernet.len < cdc::ethernet_fd::LEN || ethernet.truncated() {
        return Err(ProbeError::BadEthernetDescriptor);
    }
    let mac_string = ethernet
        .byte(cdc::ethernet_fd::I_MAC_ADDRESS)
        .ok_or(ProbeError::BadEthernetDescriptor)?;
    let lo = ethernet
        .byte(cdc::ethernet_fd::W_MAX_SEGMENT_SIZE)
        .ok_or(ProbeError::BadEthernetDescriptor)?;
    let hi = ethernet
        .byte(cdc::ethernet_fd::W_MAX_SEGMENT_SIZE + 1)
        .ok_or(ProbeError::BadEthernetDescriptor)?;
    let notify = notify.ok_or(ProbeError::NoNotifyEndpoint)?;
    let (data_interface, data_altsetting, bulk_in, bulk_out) =
        bulk.ok_or(ProbeError::NoBulkPair)?;

    Ok(EcmPlan {
        config_index,
        config_value: header.value,
        control_interface: control.number,
        data_interface,
        data_altsetting,
        mac_string,
        max_segment: u16::from_le_bytes([lo, hi]),
        notify,
        bulk_in,
        bulk_out,
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// The shape QEMU's `usb-net` presents, and a real dongle's: functional
    /// descriptors between the control interface and its notification
    /// endpoint, an endpoint-less data altsetting 0, and the bulk pair only
    /// on altsetting 1.
    fn ecm_config() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&[9, 2, 0, 0, 2, 1, 0, 0xC0, 50]); // total patched below
        b.extend_from_slice(&[9, 4, 0, 0, 1, 0x02, 0x06, 0x00, 0]); // control iface
        b.extend_from_slice(&[5, 0x24, 0x00, 0x10, 0x01]); // Header FD
        b.extend_from_slice(&[5, 0x24, 0x06, 0x00, 0x01]); // Union FD
        b.extend_from_slice(&[13, 0x24, 0x0F, 3, 0, 0, 0, 0, 0xEA, 0x05, 0, 0, 0]); // Ethernet FD
        b.extend_from_slice(&[7, 5, 0x81, 0x03, 0x10, 0, 9]); // notify, interrupt IN
        b.extend_from_slice(&[9, 4, 1, 0, 0, 0x0A, 0x00, 0x00, 0]); // data alt 0 — empty
        b.extend_from_slice(&[9, 4, 1, 1, 2, 0x0A, 0x00, 0x00, 0]); // data alt 1
        b.extend_from_slice(&[7, 5, 0x82, 0x02, 0x40, 0, 0]); // bulk IN
        b.extend_from_slice(&[7, 5, 0x02, 0x02, 0x40, 0, 0]); // bulk OUT
        let total = b.len() as u16;
        b[2..4].copy_from_slice(&total.to_le_bytes());
        b
    }

    #[test]
    fn reads_the_whole_plan_off_a_real_shaped_configuration() {
        let plan = plan_from_config(&ecm_config(), 1).expect("a usable ECM function");

        // The two numbers that are routinely different and routinely
        // conflated.
        assert_eq!(plan.config_index, 1, "the descriptor's position");
        assert_eq!(plan.config_value, 1, "SET_CONFIGURATION's argument");

        assert_eq!(plan.control_interface, 0);
        assert_eq!(plan.data_interface, 1);
        assert_eq!(
            plan.data_altsetting, 1,
            "the setting that actually has endpoints"
        );
        assert_eq!(plan.mac_string, 3);
        assert_eq!(plan.max_segment, 1514);

        assert!(plan.notify.is_in() && plan.notify.is_interrupt());
        assert_eq!(plan.notify.dci(), 3, "EP1 IN");
        assert!(plan.bulk_in.is_bulk() && plan.bulk_in.is_in());
        assert_eq!(plan.bulk_in.dci(), 5, "EP2 IN");
        assert!(plan.bulk_out.is_bulk() && !plan.bulk_out.is_in());
        assert_eq!(plan.bulk_out.dci(), 4, "EP2 OUT");
        assert_eq!(plan.bulk_in.max_packet_size, 64);
    }

    /// An RNDIS configuration must be **rejected**, not partially accepted.
    /// This is the trap: a device that offers both puts RNDIS first, so a
    /// driver that takes the first configuration drives the wrong function.
    #[test]
    fn an_rndis_configuration_is_not_ecm() {
        let mut b = Vec::new();
        b.extend_from_slice(&[9, 2, 0, 0, 2, 2, 0, 0xC0, 50]);
        b.extend_from_slice(&[9, 4, 0, 0, 1, 0x02, 0x02, 0xFF, 0]); // Comms / ACM / vendor
        b.extend_from_slice(&[7, 5, 0x81, 0x03, 0x10, 0, 9]);
        b.extend_from_slice(&[9, 4, 1, 0, 2, 0x0A, 0x00, 0x00, 0]);
        b.extend_from_slice(&[7, 5, 0x82, 0x02, 0x40, 0, 0]);
        b.extend_from_slice(&[7, 5, 0x02, 0x02, 0x40, 0, 0]);
        let total = b.len() as u16;
        b[2..4].copy_from_slice(&total.to_le_bytes());

        assert_eq!(plan_from_config(&b, 0), Err(ProbeError::NotEcm));
    }

    /// A data interface whose only setting has no endpoints has no data
    /// plane. Refused by name, because the failure it otherwise produces —
    /// a device that enumerates perfectly and moves no bytes — is opaque.
    #[test]
    fn a_data_interface_with_no_bulk_pair_refuses() {
        let full = ecm_config();
        // Cut the configuration short, just before altsetting 1.
        let alt1_at = full
            .windows(4)
            .position(|w| w == [9, 4, 1, 1])
            .expect("altsetting 1 is in the fixture");
        let mut b = full[..alt1_at].to_vec();
        let total = b.len() as u16;
        b[2..4].copy_from_slice(&total.to_le_bytes());

        assert_eq!(plan_from_config(&b, 0), Err(ProbeError::NoBulkPair));
    }

    /// Without the Ethernet Networking functional descriptor there is no MAC
    /// string index — and inventing an address is exactly what must not
    /// happen.
    #[test]
    fn a_missing_ethernet_functional_descriptor_refuses() {
        let full = ecm_config();
        let fd_at = full
            .windows(3)
            .position(|w| w == [13, 0x24, 0x0F])
            .expect("the Ethernet FD is in the fixture");
        let mut b = full[..fd_at].to_vec();
        b.extend_from_slice(&full[fd_at + 13..]);
        let total = b.len() as u16;
        b[2..4].copy_from_slice(&total.to_le_bytes());

        assert_eq!(
            plan_from_config(&b, 0),
            Err(ProbeError::NoEthernetDescriptor)
        );
    }

    /// A short Ethernet functional descriptor is refused rather than read
    /// with whatever bytes happen to follow it.
    #[test]
    fn a_short_ethernet_functional_descriptor_refuses() {
        let full = ecm_config();
        let fd_at = full
            .windows(3)
            .position(|w| w == [13, 0x24, 0x0F])
            .expect("the Ethernet FD is in the fixture");
        let mut b = full[..fd_at].to_vec();
        // Same subtype, but only 6 bytes — the fields past iMACAddress are
        // simply not there.
        b.extend_from_slice(&[6, 0x24, 0x0F, 3, 0, 0]);
        b.extend_from_slice(&full[fd_at + 13..]);
        let total = b.len() as u16;
        b[2..4].copy_from_slice(&total.to_le_bytes());

        assert_eq!(
            plan_from_config(&b, 0),
            Err(ProbeError::BadEthernetDescriptor)
        );
    }

    /// A control interface with no interrupt IN endpoint cannot report link
    /// state.
    #[test]
    fn a_missing_notification_endpoint_refuses() {
        let full = ecm_config();
        let notify_at = full
            .windows(3)
            .position(|w| w == [7, 5, 0x81])
            .expect("the notify endpoint is in the fixture");
        let mut b = full[..notify_at].to_vec();
        b.extend_from_slice(&full[notify_at + 7..]);
        let total = b.len() as u16;
        b[2..4].copy_from_slice(&total.to_le_bytes());

        assert_eq!(plan_from_config(&b, 0), Err(ProbeError::NoNotifyEndpoint));
    }

    /// A configuration claiming more bytes than it was given is malformed —
    /// refused before anything inside it is trusted.
    #[test]
    fn a_configuration_longer_than_its_buffer_refuses() {
        let mut b = ecm_config();
        let lie = (b.len() as u16) + 64;
        b[2..4].copy_from_slice(&lie.to_le_bytes());
        assert_eq!(plan_from_config(&b, 0), Err(ProbeError::Malformed));

        // And so is one too short to hold a header at all.
        assert_eq!(plan_from_config(&[9, 2], 0), Err(ProbeError::Malformed));
        assert_eq!(plan_from_config(&[], 0), Err(ProbeError::Malformed));
    }

    /// An endpoint descriptor before any interface belongs to nothing.
    #[test]
    fn an_orphan_endpoint_refuses() {
        let mut b = Vec::new();
        b.extend_from_slice(&[9, 2, 0, 0, 2, 1, 0, 0xC0, 50]);
        b.extend_from_slice(&[7, 5, 0x82, 0x02, 0x40, 0, 0]); // no interface yet
        let total = b.len() as u16;
        b[2..4].copy_from_slice(&total.to_le_bytes());

        assert_eq!(plan_from_config(&b, 0), Err(ProbeError::Malformed));
    }
}
