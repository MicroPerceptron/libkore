# WIP preservation and ownership replan — 2026-09-15

The user explicitly requested preserving and pushing all current changes as
WIP, revising the Kore/vkore issues, and adding necessary libkore issues.
This checkpoint preserves implementation as found; it is not a release,
completion claim, fresh validation run, or authorization to merge these branches.
The main implementation task was idle/interrupted, and no build or QEMU
process was observed when this checkpoint started.

## Saved work

| Repository | WIP branch | Base before this checkpoint |
| --- | --- | --- |
| Kore | `codex/vkore-994-hostlink` | `bc3a2ff58adc4dfdfe3dc6189ed7a52adadfe737` |
| vkore | `codex/vkore-40-current-kore` | `4d8b5e76863fa9aa1931ea6f98a9e7173b33971d` |
| libkore | `codex/vkore-libkore-wip` | `d948dc37efa3bce98d8da1bc2a8dad373fbcb3eb` |

Published commit IDs and revised issue dependencies are recorded in
[the existing umbrella, Kore #993](https://github.com/MicroPerceptron/kore/issues/993).
No default branch is changed. These snapshots include tracked/unignored source,
lockfile and generated payload changes; ignored build caches, ISO outputs,
model weights, credentials and local runtime/project state are not publication
inputs. Generated payloads are historical WIP outputs, not proof of the final
snapshot's source/artifact consistency. The existing local override configuration
must be reconciled before a reproducible release.

## Current evidence and remaining blocker

The previous journal reports host suites (341 passed, 3 ignored), new guest
packages (353 passed), kernel tests (1542 passed, 1 ignored), architecture
checks and user builds. The inspected boottest log contains `boot_test: PASS`.
These are earlier runs, not a clean final-snapshot full-suite pass. The broad
logs also contain failures and subsequent focused reruns; preserve that history.
The last inspected production provider gate reports `BOOT DEGRADED`, 18/20
services, and first-shell failure. The latest exec/provider settlement changes
were built, but a successful production boot and real model/workspace/UI
acceptance have not been established. The integration is NOT complete.

## Revised ownership

- Kore owns kernel/native runtime behavior, service implementations, shared
  transport contracts, device founding and OS enforcement.
- libkore owns the reusable application facade. Change native owners in Kore
  and mechanically refresh their libkore export until ownership is deliberately
  transferred; do not evolve the exported copy independently.
- Broker/agent policy and orchestration are reusable userland packages. Keep
  the current `kore-agent-broker` / `kore-agent-runner` packages in Kore for this
  migration, with a thin init integration. They are neither kernel policy nor
  a second vkore-only standard library. Consume libkore where its APIs exist;
  introduce missing generic client surfaces in libkore with one native owner.
- vkore owns host VM/artifact selection, inference/model supervision, host
  workspace adaptation, UI and conversation state. HostLink crosses the VM
  boundary; linking libkore into the host does not upgrade the guest.

Keep #993 and #994–#998 plus vkore #40 as existing tracking identities. Missing
heapless application support, reusable invocation/lifecycle APIs and coherent
native exports receive explicit libkore tracking. Audit existing APIs before
adding wrappers. Preserve all source carry-over decisions; do not discard the
original inventory or equate merged PR count with delivered behavior.

Aggregate CPU/accounting and exceptional automatic receipt recovery remain
explicit longer-term work under Kore #545/#546 (and #520/#541 as applicable).
They must not be advertised as enforced or recovered. They do not block the
initial runnable integration unless a demonstrated normal-use or safe-shutdown
failure makes a particular fix necessary. Normal startup, real tools, isolation,
accurate refusal/cancel behavior and safe shutdown remain acceptance requirements.

Resume with the revised issue ownership and measurable acceptance gates; do not
expand speculative hardening or repeat the issue-creation announcement as
implementation progress. No implementation work was performed by this checkpoint.

## libkore-specific state

This WIP makes hash collections optional for allocation-free native consumers
and adds local composition/heapless-check tooling. It has not been rebased onto
the newer alpha observed at `189ca9973305e1ea5a3e8196cb97d426f4c0c9b7` (#15).
Coordinate the separate open entry PR #16 without duplicating or assuming it
has landed. The existing committed native export still records source
`2a45f8bce5db65e7713f11c89d7eb5d2d9a04d3f`; it does not contain the Kore port WIP.
The export, distribution locks, downstream pins and renamed dependency/native
consumer checks remain release work. Do not hand-edit `native/` to hide drift.
