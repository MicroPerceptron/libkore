# `kore-usbnet` — CDC Ethernet (ECM) driver core

A [`NicBackend`](../../../crates/driver/src/net/backend.rs) over a USB
Ethernet adaptor, reached through the xHCI core's bulk and interrupt pipes.
It exists because neither kore metal target has a wired port and the peer
link (#387) needs an L2 frame service before it needs a radio; CDC-ECM is a
class standard, so nothing here is one chip's glue. The design of record is
`res/NETWORK_SERVICE.md`; this file is the crate's skeleton card in the
#809 template.

## Layout

| module | owns |
| --- | --- |
| `cdc` | the class's wire facts: constants, request encodings, the notification layout |
| `probe` | descriptor bytes → an `EcmPlan`, or a named refusal — a pure function over bytes, because it is where a device's own claims become addresses this driver will poke |
| `device` | the bring-up ladder and the `NicBackend` implementation over `kore_xhci` |

There is no `regs/` (a CDC device has no BAR of its own) and no `fw/`
(it loads no firmware). Both absences are the class README's deliberate
deviations from the ivpu skeleton, not omissions.

## The bring-up ladder

The crate's own rungs are the variants of `Error`, numbered by
`kore_svc_net::metal::usbnet_rung`. The row wraps them as readiness note
`BRING_UP_BASE + r` so a serial-less machine's `note: rung N` reads directly
against one decoder ring (`crates/abi/src/founding.rs`,
`net_note_meaning`). Renumber nothing; append only.

| crate rung `r` | `Error` | row note `16 + r` | meaning |
| --- | --- | --- | --- |
| 1 | `Probe(_)` | 17 | the descriptors do not describe a CDC Ethernet function |
| 2 | `Usb(_)` | 18 | a control transfer or endpoint setup was refused by the xHCI core |
| 3 | `BadMacString` | 19 | the adaptor's MAC address string is not twelve hex digits — refused, never substituted |
| 4 | `DmaAlloc` | 20 | no DMA buffer could be drawn for the bulk rings |
| 5 | `Exhausted` | 21 | a bounded USB resource was exhausted |

One refusal folds back into the row's `NO_ADAPTOR` (note 11) rather than
this band: `ProbeError::NotEcm`, because "this port carries some other
class of device" is a conclusion about the port, not a failure of the
adaptor. The port number rides the note's detail bits either way.

The row's own rungs (1–5 the claimant ladder every driver row shares,
11–13 this row's) are the row's, in `services/net/src/lib.rs`; the PCI
sibling's band starts at 32 (`e1000e/README.md`). The row tries the PCI
NIC first, so on a machine that has one this core is never asked; the
`run-boot-gate-x86-usbnet` variant boots without one to keep this path
proven.

## Testing

```text
make test-usbnet          # cargo test -p kore-usbnet — 24 host tests
```

The hardware double is `kore-xhci`'s `mock` feature, taken as a
dev-dependency: `MockXhciDevice::new_with_ecm` presents a CDC Ethernet
function with a loopback wire and a class-request log, and reproduces
QEMU's crossed configuration order (RNDIS at descriptor index 0 with
`bConfigurationValue` 2, ECM at index 1 with value 1) so that "select by
descriptor, never by index" stays tested. There is no in-crate mock: a
second emulated controller would be a twin of the first.

## The completion path

The controller's event ring is shared by every endpoint on it, and this
core drains it in exactly one place: `reap_tx`, the head of every drive
cycle by the class contract. `poll_rx` consumes what that drain landed
and touches the controller not at all, so a sweep of sixty-four polls
costs one pass over the ring; a receive that lands mid-sweep is latched by
the held route and ends the next park. One drain that finds anything is
one serviced interrupt in `counters()`, whatever mix of transfers it
carried; a transfer completed in error is an `rx_errors` count and an
empty slot, never invented bytes; `overruns` is an honest zero, because a
bulk IN pipe with no buffer posted holds the frame rather than dropping
it.

The notification pipe (interrupt IN) is armed at bring-up and re-armed on
every completion, so a `NetworkConnection` or `ConnectionSpeedChange`
notification reaches the driver for the row's whole life and is counted
as a link change. The link is **assumed connected** until a notification
says otherwise — the notification reports a *change*, a device that never
sends one (QEMU's `usb-net`, many dongles at plug-in) has a live link, and
Linux's `cdc_ether` takes the same posture — with the rate unknown until a
speed-change notification names it.

## Tier and evidence

**Validated.** The production x86 boot gate wires a second `qemu-xhci` with a
`usb-net` adaptor and requires the row's `net-frame-rt: PASS` and
`net-vif-isolation: PASS` witnesses, both produced by this core's datapath
(an ARP request out of a client-formatted slot, zero-copy, and a frame back
through the classifier). The tier moves into the descriptor row when #809
item 1.4 lands. Metal: a USB-C Ethernet adaptor on the second controller
of the x86 target; the port-to-controller caveat is
`res/NETWORK_SERVICE.md` §8.
