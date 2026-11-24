# axgfx

`axgfx` is the platform-facing graphics contract between Axiom display lists,
render backends, and presentation authorities such as Kore's display service.
It is `no_std` and contains no native window, device, memory domain, allocator
implementation, or Iced type. `axui-core` owns display-list storage through a
caller-selected nightly `Allocator`; `FrameView<A>`, `RenderRequest<..., A>`,
and `GraphicsBackend<A>` retain that allocator provenance through submission.
Only the final host or Kore runtime chooses `A`; the framework provides no
default allocator.

The platform retains target allocation and presentation authority. A target is
an opaque backend-associated handle accompanied by a validated format, exact
byte range, plane layout, modifier, viewport, damage, and monotonically
increasing surface epoch. Backends expose capability limits, explicit
acceptance semantics, nonblocking progress, visibility, cancellation, and
release. Implementations allocate owned lowering or event storage through the
runtime-supplied `A`; the contract prohibits hidden staging and ownership
transfer, not allocation itself.

Logical-to-physical sizing has one portable rule: multiply the logical extent
by the frame scale factor and round positive half values upward. The resulting
integer extent must equal the submitted physical viewport. Platform adapters
therefore do not get to invent different DPI rounding at the last moment.

`SurfaceRetirement` models the important final distinction: producer
completion, cache visibility, submission release, commit, and presentation are
separate observations. Imported targets may require the submission to remain
live through presentation; linear coherent CPU targets may release it once
visible.

The crate deliberately does not define an IOVA or memory-domain identifier.
Kore creates and revokes mappings, and its adapter keeps that authority inside
its opaque target handle.
