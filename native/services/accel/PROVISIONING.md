# Provisioning native accelerator memory

The native iVPU row accepts an owned memory supply before its first context
request. This extends its DMA pool and the aggregate client-buffer/program
budgets together. The initial 16 MiB buffer and 8 MiB program budgets are no
longer lifetime ceilings when a caller supplies additional capacity.

Use `MmClient::supply` to acquire backing and `AccelClient::provision_memory`
to transfer the returned `StashedLease`. The caller must hold both service
faces; this does not grant memory-allocation authority to the accelerator.

The sequence is:

1. Request enough supply for the sum of the buffer and program budget increments.
   Require `MmResponse::Granted` and its returned supply.
2. Call `provision_memory(supply, buffer_increment, program_increment)` on the
   accelerator client. The transfer carries `PROVISION_RIGHTS` (access, modify,
   derive, revoke). These are **increments**, not replacement limits.
3. Require `AccelResponse::Provisioned { bytes }`. The acknowledged supply length
   must cover both increments. Attach staging and create contexts afterwards.

The device owner selects the additional IOVA range, retains its CPU/device
window and supply ownership, activates the window, and then publishes the new
allocation region and byte budgets. An allocation must fit a single window;
separate supplies are not silently stitched into one contiguous buffer.

Provisioning closes permanently when the first context request is made, even
if that request fails or all later objects are destroyed. This preserves object
ID generations and keeps growth outside active execution. A failed DMA
publication seals the device epoch for containment. A refusal after capability
transfer may consume that supply; only a pre-transfer `ReturnCallError` can
return the unmoved capability.

The return rendezvous accepts retained capabilities of more than one species:
`Attach` requires a `LeasedRegion` leaf with ACCESS/MODIFY, while provisioning
requires a `MemLease` with `PROVISION_RIGHTS`. Each handler checks the
kernel-authenticated record before taking ownership. A leaf-only typed post
would reject a supply before this handler could run. Supply possession also
checks the lease species before requesting geometry: mapping a leaf produces
a virtual-address report, which cannot describe physical supply backing.

The additive wire forms are request opcode 19 (`a`: supply landing, `b`: buffer
increment, `c`: program increment) and reply tag 16 (`id`: actual supplied bytes).
Unused words and reply grants remain zero. Current providers without a native
provisioning implementation return `Unsupported`; older protocol decoders do
not recognize the opcode.

## Current limits and verification

The founded MM service receives its per-client outstanding supply budget from
`services/mm/boot.toml`: `boot_data = "supply_quota"` and
`supply_quota_bytes = 2147483648` (2 GiB in the current deployment). The manifest
and MM both reject zero, unaligned, or unrepresentable values; the page-aligned
maximum is below 64 GiB. Missing or partial boot policy refuses binding.
Receipt-table capacity is independent of byte quota. This is an admission
ceiling, not a physical-memory reservation: region exhaustion can still refuse
a request. The initial driver endowment remains 128 MiB.

Host tests cover codec validation, policy admission, closure after context
creation, lost-epoch sealing, region allocation/reclamation, and large MM
receipts. These are not evidence of a physical NPU transfer. Production
provisioning on hardware, automatic growth during execution and executable
peer DMA remain separate acceptance work. Native DMA owners now retain a
lease-derived fabric registration for each initial or grown window before
activation; registration alone does not establish a working transfer route.
