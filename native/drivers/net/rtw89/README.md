# `kore-rtw89` — Realtek RTL8922AE (rtw89, BE generation) driver core

The Wi-Fi 7 part in the metal x86 target (Gigatron, #388 D0:
`56:00.0 Network controller [0280]: Realtek RTL8922AE [10ec:8922] (rev 01)`,
subsystem Lenovo `17aa:4922`), brought up over PCI to **firmware alive**:
the MAC powered on, the efuse read, the pinned firmware verified,
downloaded through the firmware-command DMA ring and started, and one
command answered by the running firmware over the register mailbox. That
is #388's D1 — the go/no-go for a
native driver — and everything this crate does today. It is the first
metal PCI network part; the class engine it rides is the one
[`kore-e1000e`](../e1000e/README.md) proved on QEMU. This file is the
crate's skeleton card in the #809 template.

## What the part is

A mac80211-class SoftMAC: Linux's `rtw89` keeps authentication, association
and the 4-way handshake on the host and offloads scanning, rate adaptation
and the baseband/RF tables to firmware. The BE generation carries those
tables *inside* the firmware file (`rtw89/rtw8922a_fw.bin`, header format
≤ 4), so the chip-specific host code is a power-on sequence, an efuse map, a
download protocol and a mailbox — this crate. The frames (D2, a host-side
MLME and supplicant) and the second interface (D3) are `res/PEER_LINK.md`'s
later slices.

Registers live in **BAR2** (1 MiB); BAR0 is 256 bytes of legacy port I/O
the kernel never turns into a resource, and the Linux driver never touches.
The interrupt is one plain **MSI** message (no MSI-X on this part), realized
as the same `InterruptBlock::SINGLE` route the 82574L takes: the block's
`PreferMsiX` source admits an MSI-only function through the remapped
`MsiFamily` requirement.

## Layout

| module | owns |
| --- | --- |
| `regs/be` | **generated** — the register map and field masks, from `tools/rtw89_be.h` (the used subset of upstream `reg.h`, every line citing its upstream line); never hand-edited |
| `regs` | the hand-written half: the compositions the ladder programs, each with the derivation its number came from, the channel register tables, and the pool-free decode probe `controller_present` |
| `product` | the one runtime selection point: probed PCI identity → `Product`, never `cfg`, never on a datapath (#809 §G rule 7) |
| `desc` | the BE PCI descriptors (`tx_bd`, `rx_bd`, the firmware-command descriptor, `rxbd_info`, the receive descriptor) and the command/reply headers, as bytes |
| `fw` | the firmware: the BLAKE3-pinned trust decision (`firmware/manifest.toml`, parsed at compile time into `PINNED_SIZE`/`PINNED_DIGEST`), the multi-firmware wrapper, the version-0/1 inner header with its security section and key pool, and the download plan — `FirmwareImage::verified` is the only constructor, so an unverified blob cannot be parsed, let alone pushed |
| `efuse` | the physical dump through the efuse window, the logical parse of the RF block (country code, crystal trim, RF front-end type), and the four secure-boot bytes |
| `mac` | power on and off as **data**, DMA pre-init and download quotas; `mac::normal` reuses the operation engine for SCC quotas, flow control and DMAC/CMAC0 configuration after firmware TX retirement |
| `fwdl` | the firmware-command ring (channel 12) and the download protocol: CPU release, header packet, 2020-byte section packets, the status field |
| `mailbox` | the register mailbox and the `GET_FEATURE` → `PHY_CAP` witness |
| `nic` | the bring-up ladder (`Prepared` → `Rtw89` type-state) and the carve |
| `mock` | `cfg(test)` only: a byte-addressed register-file double with the power, efuse, DLE, download and mailbox state machines behind an IOMMU-shaped pool |

`firmware/` holds the manifest and README; the blob itself is an operator
artifact in the shared `vendor/firmware` submodule (`make fetch-firmware`).

Register access follows `kore-e1000e`: the core stores no register handle;
every method takes `regs: &R`, because in the row the BAR lives in the
device's resources and the backend is a per-cycle view. `R` is the seam's
`RegsNarrow`, not plain `Regs`: the part keeps byte- and halfword-wide
control registers (the mailbox trigger, the counter byte, the scoreboard)
whose neighbouring bytes are write-one-to-clear or self-triggering, so a
narrow access is never synthesized from a dword read-modify-write.

## Regenerating `regs/be.rs`

```text
python3 drivers/ivpu/tools/gen_ivpu_regs.py --strict \
  --source 'Realtek RTL8922AE (BE generation) register subset, copied from Linux v6.17 drivers/net/wireless/realtek/rtw89 with the upstream line cited per define ({header}).' \
  --about 'RTL8922AE BE-generation register offsets and field masks over the `Regs` seam (BAR2 read32/write32); names are verbatim from the Linux rtw89 headers so a reviewer can diff them 1:1 against the cited upstream lines.' \
  drivers/net/rtw89/tools/rtw89_be.h > drivers/net/rtw89/src/regs/be.rs
```

The generated file's header records the same command. Change the header,
not the output. `tools/rtw89_be.h` names the upstream line each define came
from; there is no public datasheet for this part, so the Linux header *is*
the register reference, cited rather than copied wholesale.

## The bring-up ladder

The crate's own rungs are the variants of `Error`, numbered by
`Error::rung` in declaration order. The row wraps them as readiness note
`WIFI_BRING_UP_BASE + r` so a serial-less machine's `note: rung N` reads
against one decoder ring (`crates/abi/src/founding.rs`,
`net_note_meaning`). Renumber nothing; append only.

| crate rung `r` | `Error` | row note `48 + r` | meaning |
| --- | --- | --- | --- |
| 1 | `NotPresent` | 49 | BAR2 is too short or the first registers read all-ones: an unrouted or power-gated function |
| 2 | `NoFirmware` | 50 | no firmware module in this boot (`make fetch-realtek-firmware`) |
| 3 | `BadFirmware` | 51 | the module is not the pinned blob (size, digest, no image for this cut) or, after the efuse, a header the loader refuses |
| 4 | `PowerOn` | 52 | a MAC power sequence (on, or off at quiesce) did not complete within its budget |
| 5 | `Efuse` | 53 | the efuse could not be read, or does not parse as this part's |
| 6 | `DmaAlloc` | 54 | no DMA buffer could be drawn, or one lies beyond the part's 36-bit reach |
| 7 | `DmaInit` | 55 | the DMA engine's pre-init did not complete: a DLE never ready, a channel busy, the BD-RAM bound never reset |
| 8 | `FwDownload` | 56 | the download did not complete: a path never ready, the ring stalled, or the WCPU rejected the image |
| 9 | `NoAck` | 57 | the running firmware never answered the first command |
| 10 | `MacInit` | 58 | normal DMAC/CMAC0 configuration timed out |
| 11 | `PhyInit` | 59 | PHY configuration or RF-stage receive/command retirement failed |
| 12 | `MacCommand` | 60 | MAC feature command was rejected or did not complete |

The original rungs retain their numbers; normal MAC setup (rung 10) runs
after the dual download and before the alive query (rung 9).
The original ladder is declared in the order it climbs, with one documented
exception: the container's *header* is parsed after the efuse, because
the secure-boot bytes decide which key the plan carries, so rung 3 can
surface after rung 5 as well as before rung 4. Every refusal after the MAC
is powered powers it off again before returning, so a re-founding starts
from a cold part.

`Prepared::prepare` runs rungs 1–7 under the row's `DmaPrepared` state:
the device may not master memory yet, so it is probed, its firmware
verified, powered, its efuse read, its rings carved and pointed at, and
its engine pre-initialized up to — not including — the transmit/receive
enable. `Prepared::start` runs rungs 8–12 under `DmaActive`: the enable,
the download over the firmware-command channel (WLAN-only for the original
`prepare`, WLAN plus BB for production's `prepare_with_baseband`), normal
DMAC/CMAC0 and BB/RF setup on the dual-image path, and the witness — the
register mailbox's `GET_FEATURE`, answered with `PHY_CAP` by the firmware
just loaded, bracketed by the efuse-state conversion exactly as
`rtw89_mac_read_phycap` does it.

The carve: 13 transmit BD rings (a page each; only channel 12 is ever
posted, the data channels stay stopped but programmed), 2 receive BD rings
of 8 descriptors with a 12 KiB buffer each, and 32 staging slots of 4 KiB
for the packets in flight — 316 KiB of the row's 2 MiB pool. Upstream
posts 256 receive buffers per ring; D1 posts eight because nothing drains
them yet.

## Status

| tier | evidence |
| --- | --- |
| **Host-proven** | the mock ladder: every rung's refusal and the happy path to firmware-alive |
| **Metal** | `net-wifi-fw-alive: PASS` in the founder's note band on Gigatron — the D1 go/no-go recorded in #388 |
| **Readable** | `status net` on the machine itself: the cut, firmware version, secure-boot selection, efuse country code / `xtal_cap` / `rfe_type`, the `PHY_CAP` answer and the download packet count, every code as the part reported it (`res/WIFI_LINK.md` §0). Read on Gigatron: cut 2, firmware 0.35.80.3, secure boot off, country **unprogrammed** (`0xffff`), 2×2 streams and antennas, 213 download packets |

No QEMU profile carries the part, so the CI boot gates can only prove the
row still founds its other hosts and that this rung refuses cleanly on a
machine without the function.

## What comes next

D2 — the frame path over the radio — is planned in
[`res/WIFI_LINK.md`](../../../res/WIFI_LINK.md): the chip-independent
802.11 link library, the credential boundary between this row and the
supplicant, the regulatory table rooted in the efuse country code this
crate already reads, and the seven witnesses that close it. The work this
crate grows there is W1–W5: the full firmware download (`include_bb =
true`), the BB/RF tables, the RF calibrations, channel and transmit power,
the data rings and their descriptors, the address/BSSID/security CAMs,
scan offload and rate adaptation — and `impl Dot11Radio`.


D2's first hardware checkpoint now uses `prepare_with_baseband` in the
production service. The same pinned file supplies the cut-selected BB-MCU0
image; complete MFW/element bounds are checked before power-on, and both image
plans precede DMA activation. WLAN and BB download acknowledgements are checked
separately. Host doubles prove ordering and BB-stall refusal, and the pinned
artifact's combined unretried packet count is 232 (213 WLAN + 19 BB).
The earlier 213-packet metal record above is D1 only. BB/RF execution and
post-init now extend this path as described below; full PHY acceptance
remains outstanding.

The operator's 2026-09-09 PR #960 `status net` capture reports the expected
232 packets on cut 2 with WLAN version 0.35.80.3 and the 2 TX / 2 RX PHY reply.
This is the dual-download metal observation; the exact build SHA and
`status devices` are not shown. See `res/WIFI_LINK.md` for the evidence limits.

The next checkpoint adds `mac::normal`: SCC packet-memory quotas and host
flow control, the scheduler/security/dispatcher, CMAC0 and its CAM reset,
and host-report configuration. It first requires retirement of every firmware
TX descriptor; firmware readiness alone cannot permit a DLE reset. The same
operation engine preserves register access widths and bounded handshakes.
No additional DMA allocation or lock is introduced. Data TX remains stopped,
on-air RX filters drop all frames, and interrupt admission stays closed.
At that checkpoint, BB `BOOT_RDY0` remained clear pending PHY table execution
and post-init.
`status net` now reports `MAC: normal mode, receive closed` only after normal
configuration and the subsequent firmware alive reply. The operator confirmed
these exact `status net` reports on Gigatron on 2026-09-09 after the
`0032baa08` checkpoint instructions. That normal-MAC witness does not
constitute `net-wifi-phy-up`; see the evidence limits in `res/WIFI_LINK.md`.

The next checkpoint adds a complete borrowed PHY table preflight before DMA
allocation, followed by BB writes/post-init, RF-v2 analog/direct programming,
and bounded RF H2C page submission/retirement. Gain metadata stays borrowed
from the immutable firmware for later per-channel application. Whole-program
validation covers unselected branches, selector ambiguity, RF path/slot
identity and the three-page notification limit. The executor reuses `Regs`,
`Waits`, the MAC operation engine and `FwcmdQueue`; its one 2,000-byte stack buffer
needs no additional DMA allocation or lock.

On Gigatron, expect `PHY setup: registers configured, calibration pending`
with the unchanged 232 download packets, normal-MAC line and WLAN reply.
The operator's subsequent 2026-09-09 screenshot confirms these reports on Gigatron.
RF command retirement proves DMA completion;
`net-wifi-phy-up` still requires a running-baseband witness. Channel setup,
calibration, remaining MAC/PHY setup and frame admission remain open.

The next checkpoint adds bounded RXQ0 C2H handling and the two BA-CAM user
reservations plus offload configuration. A successful done acknowledgement
must match the entire command identity, and its TX must also retire. A
receipt alone is insufficient. The existing eight RX slots are reused after
index/tag and complete envelope validation; no additional DMA allocation or
lock is introduced. Command/RX progress remains with the radio owner.

Expect `MAC features: firmware acknowledged` on the next Gigatron boot,
with the same 232 download packets and existing MAC/PHY lines. This new
checkpoint awaits metal. PCI data/interrupt admission, the running-baseband
witness, PHY dynamic-management/NCTL and calibration remain open. See
`res/WIFI_LINK.md` for the exact boundaries and failure codes.
