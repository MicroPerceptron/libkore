# jsmn vendor record

- Upstream: `https://github.com/zserge/jsmn`
- Pin: commit `25647e692c7906b96ffd2b05ca54c097948e879c` (release 1.1.0)
- Vendored file: `jsmn.h`, implementation from that pin with only explanatory
  comments compacted; declarations and executable C tokens are unchanged
- License: MIT (`LICENSE` carried beside the source)
- Runtime posture: C89, libc-free, no allocation; caller supplies a fixed
  token array. `JSMN_STRICT` is enabled by the local build.
- Local code: `kore_jsmn.c` is the only adapter. No foreign struct or pointer
  escapes it.
- Confinement: this archive is linked only by the C parser adapter/worker, not
  the broker or filesystem service address space.
