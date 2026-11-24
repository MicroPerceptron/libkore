# Owned accelerator staging

The current `Attach` operation receives an ACCESS/MODIFY leaf, maps through the
monotone MAP cursor and retains that mapping for the service task's lifetime.
Its attempt limit now precedes mapping, but it cannot support recyclable client
sessions. The receiver does not hold the immediate parent needed for exact
unshare. The additive owned-staging path is implemented in the native protocol and
iVPU host. Founded init receives the retained bootstrap client face and proves
three 16 MiB MM-supply/attach/detach cycles through the real channels on QEMU.

## Admission and ownership

`AttachOwned` (request opcode 20) preserves existing `Attach` byte and rights
semantics. It transfers a `MemLease` with access,
modify, derive and revoke rights over the existing retained return leg. The
service checks the authenticated species and rights before taking the owner.
An announcement is only a return-landing correlation value, never authority.

The client obtains the supply from MM and maps a derived client view before
transferring its parent. MM still owns admission policy; the accelerator gains
no MM face. The client stops touching shared staging while a request uses it.
The service owns a `StashedLease` and an exact `MappedLeaseView`, retained
together in that pair's state. Supply authority is never reconstructed from a leaf's geometry, and the parent
remains retained after mapping.

Choose owned staging on a pair that has never attempted legacy `Attach`.
Legacy attempts may leave task-lifetime mappings; that pair cannot switch to
owned staging or claim that detach closed those mappings.

One owned attachment occupies one pair's staging cell. A second attachment
refuses while either the live attachment or its cleanup owner remains. The
caller explicitly detaches before replacement. This avoids needing a second
mapping bank and ambiguous replacement success when old-parent retirement
fails. Legacy attachment must also refuse while owned staging is live or
closing; it cannot overwrite and lose the cleanup owner.

## Placement

The implementation uses `geom::TRANSIENT`, the existing exact-mapping
reservation. The configurable deployment moved from 16 MiB at `0x75000000` to
64 GiB at `0x200000000`, outside the existing low windows. This reserves
addresses, not physical backing; the monotone MAP cursor is unchanged.

Disjoint staging cells derive from the supplied window and the runtime's four
pair slots, leaving a 2 MiB prefix for future recyclable pair rings. Validate
nonzero page-aligned capacity, checked arithmetic, full containment and the
complete parent extent before mapping. Neither client bytes nor wire fields
choose a VA. Artifact, buffer, MM and resident-memory budgets continue to apply
independently; the VA reservation does not grant extra physical memory.

Only the accelerator service's owning control flow uses this partition of its
transient window. Other services retain their existing transient consumers.
The iVPU entry passes its actual founder geometry into the host before serving.
Missing or inadequate geometry refuses owned attachment. Other backend entries
must configure their own host before advertising this path to clients.

## Detach and failed admission

`DetachStaging` (request opcode 21) is a staging lifecycle operation, including
after device loss. Reply tag 17 (`Detached`) reports successful retirement.
It must remain reachable even when the backend refuses execution. The service
first removes all Rust access to the staging bytes, then exact-unshares its
mapping, then retires the parent lease. Parent retirement also revokes the
client's derived view. Success is returned only after both owners are closed.
The client cannot exact-unshare through its original parent slot after moving
that parent away. It retains its view owner until successful server detach,
then proves that view already dead before releasing its local name or reusing
its VA. The founded witness exercises this sequence.

A retryable retirement returns `Busy` and keeps the same affine owner and the pair cell reserved.
The client must stop using the staging view when it requests detach, including
when the response asks it to retry. No replacement or pair-generation reuse is
allowed while cleanup remains. Dropping a Rust wrapper or clearing an Option
is not proof of retirement.

A failed mapping may have crossed the kernel publication boundary. Retain the
parent and quarantine the cell until cleanup proves closure; do not interpret
an absent mapping result as proof that the VA is reusable. A refused initial
attachment may therefore consume the transferred parent, as provisioning can.
A pre-transfer failure still returns the unmoved capability to the client.

Staging is a CPU transfer window, distinct from submitted device buffers and
resident program storage. Current byte-moving handlers copy synchronously and
retain no staging slice in an accepted event. Preserve and verify that property
so detach need not cancel unrelated accepted work. It does not release that
client's contexts, buffers, programs, queues or events.

## Acceptance

The production gate requires `accel-owned-staging: PASS`: founded init obtains
and writes a 16 MiB MM supply, transfers the parent over the actual return post,
attaches, detaches, retires its revoked view, and repeats three times on the
same pair and client VA. It also checks idempotent detach. This proves ownership
and mapping reclamation; a device-backed upload/download and numerical capture
remain separate acceptance requirements. Host tests must cover geometry boundaries and
failure ordering, including mapping failure and retryable exact unshare or
parent retirement. Test that leaf-only authority cannot use the owned path and
that a closing pair cannot be reused or overwritten through legacy Attach.

The iVPU row now implements the runtime's `peer_closed` hook through the common
host's `close_peer`. It enumerates the existing object tables without allocating
a second ledger, cancels pending events, then retires events, queues, programs,
buffers and contexts for that arrival only. Staging and signal owners close
before the hook acknowledges reuse. Any refused step keeps its owner and the
closing cell for retry. DeviceLost keeps the complete graph behind the existing
hardware containment handoff. Lost and terminal device epochs never acknowledge
replacement peers: handing out the affine epoch is not proof that its external
owner finished containment, and this discard-only namespace requires fresh
founding. Legacy task-lifetime staging also refuses reuse.

Host regressions cover pending-graph retirement, another peer's surviving
context, stale handles, retryable buffer release and delayed device-loss
containment. The staging-free graph fixtures do not prove kernel mapping or
channel retirement; those retain their separate ownership tests above. The founded native witness now conveys real peer bundles, closes peers with
owned staging still attached, and observes two new generations in the same
physical cell. This validates runtime reuse on QEMU; live graph cleanup and
physical device-loss behavior retain their separate validation requirements.

Recyclable arrival stamps include a generation and are not array indexes. The
host now binds each session cell to the exact arrival, decodes only its physical
slot for staging/signal placement and completion counts, and retains the full
stamp in object ownership. A conflicting generation is refused before access;
closure resets the binding only after cleanup succeeds. Tests cover static and
dynamic collisions, old-generation staging denial, stale close calls and
one-shot completion counts on a recyclable slot. Native admission controls and the founded close/re-admit witness are now wired.

## Native peer controls

`AdmitPeer` (22) accepts one authenticated ring/kick/grant/return component from
founder pair zero, using the runtime PeerBundleAssembler. The complete bundle
must include the accelerator reply grant and retained return endpoint.
`PeerAccepted` (18) reports component receipt only. `TakePeerAdmission` (23)
consumes the outer runtime callback as `PeerAdmitted` (19), carrying its actual
arrival generation, or a refusal. Another bundle is held off until this result
is consumed. An unsolicited or repeated callback cannot bind a session.

`ClosePeer` (24) requests orderly closure of its own recyclable arrival and
replies `PeerClosing` (20); static founder pairs cannot self-close through this
operation. Receipt does not acknowledge reclamation: the runtime still waits
for `peer_closed`. Admission refuses after device loss. The one-time founder
window configuration allocates each ring slot an equal part of the reserved
2 MiB prefix, disjoint from every owned-staging cell. No wire value chooses the
mapping address. Native iVPU forwards admission results into the common host.

Host codec/state tests cover these controls and geometry. Founded init uses
`RingAccel` staged component transfer, retaining source lineages and resuming
only an exact outstanding component after uncertain transport. Its witness
uses fresh MM-backed rings, the required grant and return endpoints, and owned
staging left attached at close. The close receipt is not a cleanup verdict:
init retains and retries its view while admin round trips drive host cleanup.
Two observed replacements must carry new generations in the same physical
cell. `accel-peers: PASS` is required by both init readiness and the production
gate. This is real capability/mapping acceptance on QEMU, not physical NPU
execution, live-job cancellation or P2PDMA proof.
