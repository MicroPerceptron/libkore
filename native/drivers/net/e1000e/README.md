# `kore-e1000e` — Intel 82574L (e1000e) driver core

A [`NicBackend`](../../../crates/driver/src/net/backend.rs) over a PCI
NIC: two descriptor rings in host memory, a tail doorbell per ring, a done
bit the device writes back into each descriptor, and one MSI-family route.
That is the NVMe-shaped half every network part has, wired or wireless, and
the half `res/NETWORK_SERVICE.md` §7 named as unproven for a frame workload
until a PCI backend existed. The 82574L is what QEMU's `-device e1000e`
emulates, both kore QEMU profiles already carry one on root port 0, and its
datasheet is public — so it is the part that proves the path today. It is
not a metal part; the class engine it establishes is `res/PEER_LINK.md` §4
P2, which the Wi-Fi part reuses once #388 D0 names it. This file is the
crate's skeleton card in the #809 template.

## Layout

| module | owns |
| --- | --- |
| `regs/i82574` | **generated** — the register map and field masks, from `tools/i82574.h` (the datasheet transcription, in the Linux `e1000e` spelling so the two can be diffed 1:1); never hand-edited |
| `regs` | the hand-written half: the compositions the core programs (`IVAR`, `IMS`, `EIAC`, `TCTL`, `RCTL`, `RXDCTL`/`TXDCTL`), each with the derivation its number came from, and the pool-free decode probe `controller_present` |
| `product` | the one runtime selection point: probed PCI identity → `Product`, never `cfg`, never on a datapath (#809 §G rule 7) |
| `desc` | the legacy 16-byte descriptors and what makes them `kore_driver` ring entries — the done bit as validity, phase ignored |
| `nic` | the bring-up ladder (`Prepared` → `E1000e` type-state), the ring engine, and the `NicBackend` impl `E1000eNic` |
| `mock` | `cfg(test)` only: a register-file double with a command engine that mirrors QEMU's MSI-X cause model |

There is no `fw/`: the 82574L loads no firmware (its station address and
configuration come from the NVM the controller auto-reads at reset). That
absence is this class README's deliberate deviation from the ivpu skeleton,
not an omission; the Wi-Fi sibling will have one.

Register access follows `kore-xhci`: the core stores no register handle;
every method takes `regs: &R`, because in the row the BAR lives in the
device's resources and the backend is a per-cycle view.

## Regenerating `regs/i82574.rs`

```text
python3 drivers/ivpu/tools/gen_ivpu_regs.py --strict \
  --source 'Intel 82574 GbE Controller Family Datasheet, transcribed by hand ({header}).' \
  --about '82574L register offsets and field masks over the `Regs` seam (BAR0 read32/write32); names follow the Linux e1000e headers so a reviewer can diff them 1:1 against the datasheet transcription.' \
  drivers/net/e1000e/tools/i82574.h > drivers/net/e1000e/src/regs/i82574.rs
```

The generated file's header records the same command. Change the header,
not the output. `tools/i82574.h` is transcribed from the datasheet by hand
with the section each line came from; the Linux header is a naming
reference, not a source (it is GPL-2.0, and nothing from it is copied).

## The bring-up ladder

The crate's own rungs are the variants of `Error`, numbered by
`Error::rung` in declaration order. The row wraps them as readiness note
`PCI_BRING_UP_BASE + r` so a serial-less machine's `note: rung N` reads
against one decoder ring (`crates/abi/src/founding.rs`,
`net_note_meaning`). Renumber nothing; append only.

| crate rung `r` | `Error` | row note `32 + r` | meaning |
| --- | --- | --- | --- |
| 1 | `NotPresent` | 33 | the BAR is too short or its control registers read all-ones: an unrouted or power-gated function |
| 2 | `ResetTimeout` | 34 | `CTRL.RST` never self-cleared within its 40 ms budget |
| 3 | `NvmNotLoaded` | 35 | the NVM auto-read never completed; the station address it loads cannot be trusted |
| 4 | `NoStationAddress` | 36 | `RAH(0).AV` clear, or the address is zero or multicast — refused, never substituted |
| 5 | `DmaAlloc` | 37 | no DMA buffer could be drawn for the rings or the receive pool |
| 6 | `TxFull` | 38 | serving-time only: the transmit ring is full and the engine retries next sweep |

`Prepared::prepare` runs rungs 1–5 under the row's `DmaPrepared` state and
leaves both engines disabled with the receive tail at zero; `Prepared::start`
enables them, posts every receive slot but one, and arms the interrupt
mask. A link that never comes up is **not** a rung: the start-time wait is
bounded, and a machine with no cable founds and serves with the link down.

The row's own rungs (1–5 the claimant ladder every driver row shares,
11–13 this row's, 16+ the USB adaptor band) are the row's, in
`services/net/src/lib.rs`.

## The interrupt model

One MSI-X message (`InterruptBlock::SINGLE`), every enabled cause routed to
table entry 0 through `IVAR`. `EIAC` lets the controller auto-clear the
queue causes from `ICR` on delivery and, with `IAME` clear, auto-mask them
from `IMS`, so a delivery is a race-free acknowledge; every drive
re-arms `IMS` in full. `IMS` carries the legacy causes too, because a
`PreferMsiX` claim may realize plain MSI, in which mode the controller
ignores `IVAR` and raises only those. One code path serves both
realizations; the mock mirrors QEMU's rule that `ICR` is not read-cleared
while `IMS` is non-zero.

The message rate is moderated from bring-up: `start` writes
`ITR_INTERVAL` (50 µs in 256 ns units — derived in `regs/mod.rs` from the
engine's receive batch and the posted ring, and coincidentally where
Linux's e1000e begins its dynamic ladder) into both `ITR` and `EITR[0]`,
so a burst raises one message per interval in whichever delivery mode the
claim realized, and the sweep behind it reaps the whole batch. Link-status
changes ride the same message (`LSC` is in `IMS_ARMED`); the link *state*
is read live from `STATUS` and the frame service derives the edge, so a
missed cause can never leave the service believing the wrong state.

What the moderation does is counted rather than assumed: `counters()`
reports the drives that found a cause latched (`irqs` — one per serviced
message, whatever it carried), the `LSC` causes among them, the `RXO`
overruns, and the descriptors written back with an error. The frame
service carries them on `Query`, so `status net` holds interrupts against
frames on a machine with no serial console.

## DMA

Rings are 2 × 4 KiB; the receive pool is 256 × 2 KiB (512 KiB). With the
row's two self-test windows that is 0.77 MiB of the 2 MiB `DMA_POOL@0`
endowment, unchanged. Every IOVA the row hands out is above 4 GiB, so the
high halves of the base registers and the 64-bit descriptor addresses are
load-bearing and the tests check them. Transmit is zero-copy: a descriptor
points at the client's own vif slot, published by `flush_tx` once per drive.
Receive copies once, out of the pool into the vif window, at the classifier
— the isolation boundary `res/NETWORK_SERVICE.md` fixes.

## Testing

```text
make test-e1000e          # cargo test -p kore-e1000e — 27 host tests
```

The hardware double is the in-crate `mock` (`cfg(test)`, `kore-nvme`'s
posture): a register file whose `CTRL.RST` runs the datasheet's reset
vector, whose `TDT` write walks the transmit ring and writes `DD` back,
whose `RDT` write fills posted receive slots from an injected inbox, and
whose cause model is QEMU's (`IVAR` routing, `EIAC` auto-clear and
auto-mask, `OTHER` folding only `LSC`/`RXO`). It refuses an extended
transmit descriptor (`DEXT`), so "legacy descriptors, always" is tested
rather than assumed. Fault injection: a reset that never self-clears, an
NVM that never loads, an address-valid bit that stays clear, a held
transmit write-back, an error descriptor.

## Tier and evidence

**Validated (QEMU).** The production x86 boot gate (`make run-boot-gate-x86`)
wires an `e1000e` on root port 0 (`QEMU_X86_NIC`), the row claims it ahead
of the USB adaptor (`services/net/src/metal/pci.rs`, the PCI-first rung in
`entry.rs`), and the gate requires `net-frame-rt: PASS`,
`net-vif-isolation: PASS` **and** `net-pci-frame-rt: PASS` — the third
witness derived from the round trip having ridden this core's rings and
route, never declared. `make run-boot-gate-x86-usbnet` boots with no PCI
NIC and requires that witness absent, which is what keeps the CDC-ECM
sibling proven beside this one. `make keytest-status-devices-x86` reads
the same testimony back off the framebuffer. Metal: none — neither kore
metal target has a wired port. The first metal part this class engine
serves is the Wi-Fi sibling, #388 D0.
