# axcomp

`axcomp` is Axiom's reusable, caller-driven system-compositor state. It is
`no_std + alloc`, requires a caller-selected nightly `Allocator`, and contains
no window, event loop, clock, environment, thread, device provider, or Kore
handle type.

The crate owns reusable compositor mechanics, including those previously
entangled with the host `axui` example:

- a bounded prepared-scene LRU and bounded structural-decision memo whose
  provider payloads are returned for explicit release;
- precision-explicit planar F32 or biased-U8 materialization into linear
  XRGB8888 words or byte mappings, including padded strides and bounded damage;
- the standard cursor family (`FamilyCursor`, `FamilyPose`, `CursorShape`):
  Dart, Text, Grab, Grabbing, Pointer, four resize arrows, NotAllowed and a
  spinning Loading cursor,
  drawn by one
  distance-field raster at the destination density. Members share the Dart's
  pendulum through per-member lean gains and hand over through presence
  springs; the Dart's own pixels are unchanged. Text and hand geometry follow
  Lucide icons (ISC, see `LICENSE-LUCIDE`);
- a generation-checked target ring that gates reuse through `axgfx` producer,
  visibility, commit, presentation, and submission-release epochs. A display
  reconfigure quarantines old targets and returns each one explicitly to the
  final runtime once it is safe to destroy;
- a bounded `FrameQueue` that retains input owners separately from output
  targets, rejects new old-configuration commits after resize, and delegates
  lifecycle truth to that same target ring. Outstanding accepted frames must
  be explicitly drained; dropping them quarantines owners, not cancels work.

`axcomp` does not decide how a graph is placed or executed. `ax` and its
provider adapters retain that role. It also does not allocate platform
surfaces or present them; Kore or a host adapter retains that authority. The
staged packer is the truthful first handoff for CPU-mapped linear Kore
surfaces. A provider may bypass it only when `axgfx::TargetRoute::Direct`
truthfully binds the exact target allocation.

See [the two-window host experiment](../../tools/ax-surface-lab/KORE_PREP.md)
for resize during execution/presentation, ownership rules, and the proposed
Kore service boundary.
