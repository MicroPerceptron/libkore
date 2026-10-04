/*
 * The kore-owned Lua amalgamation. Zig supplies ISO C declarations while the
 * final image deliberately links no hosted libc. Every live external symbol
 * is satisfied by Rust compiler-builtins or by the Rust shim modules.
 */
#include "lprefix.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(KORE_LUA_ZIG_MUSL_ABI)
/*
 * Lua's error path enters the musl jmp_buf supplied by Zig's target headers,
 * then returns through the pinned assembly in vendor/musl-setjmp.  Admit the
 * Zig 0.16 toolchain family only when that header retains the exact register
 * prefix encoded by the assembly; a version string alone is not an ABI proof.
 */
#if defined(__x86_64__)
#define KORE_LUA_JMP_WORDS 8u
#elif defined(__aarch64__)
#define KORE_LUA_JMP_WORDS 22u
#elif defined(__riscv) && __riscv_xlen == 64
#define KORE_LUA_JMP_WORDS 26u
#else
#error unsupported musl setjmp target
#endif

_Static_assert(sizeof(unsigned long) == 8u,
               "musl setjmp assembly requires 64-bit words");
_Static_assert(offsetof(struct __jmp_buf_tag, __jb) == 0u,
               "musl jmp_buf register state must be its first field");
_Static_assert(sizeof(((struct __jmp_buf_tag *)0)->__jb) ==
                   KORE_LUA_JMP_WORDS * sizeof(unsigned long),
               "Zig musl jmp_buf no longer matches vendored setjmp assembly");
_Static_assert(_Alignof(jmp_buf) >= 8u,
               "musl setjmp assembly requires 8-byte jmp_buf alignment");
#endif

/*
 * Export the C API.  A native module `require`d into this task binds its
 * lua_* / luaL_* imports to these symbols when kore-late links it
 * (res/IMAGE.md §8); the engine is otherwise compiled -fvisibility=hidden
 * and luaconf.h's guarded default yields to this definition.  The linker
 * keeps the set in .dynsym with --export-dynamic-symbol=lua_* / luaL_*.
 * The exported set is the module ABI.
 */
#define LUA_API __attribute__((visibility("default"))) extern
/*
 * The VM runs on a kore_rt worker thread with a 64 KiB stack and a guard
 * page (crates/rt/src/thread.rs STACK_BYTES) — a quarter of what Lua's
 * stock cap of 200 nested C levels assumes.  LUAI_MAXCCALLS bounds every
 * recursion implemented in C (parser nesting, metamethod chains, C-function
 * re-entry); budgeting 1 KiB of C stack per level, 60 levels leave the
 * remaining ~4 KiB-per-level of headroom to the host frames beneath the
 * engine.  Exceeding it is a Lua "C stack overflow" error, never a fault.
 */
#define LUAI_MAXCCALLS 60
#define LUA_CORE
#define LUA_LIB
#include "luaconf.h"

/* A kore process has no ambient locale or stdio. */
#undef lua_getlocaledecpoint
#define lua_getlocaledecpoint() '.'
#define l_strcoll strcmp
#define lua_writestring(s, l) ((void)(s), (void)(l))
#define lua_writeline() ((void)0)
#define lua_writestringerror(s, p) ((void)(s), (void)(p))

/* Rust owns the deterministic numeric/formatting and fatal seams. */
extern double kore_lua_strtod(const char *text, char **end);
extern size_t kore_lua_format_number(char *dst, size_t cap, double value,
                                     size_t precision);
extern void kore_lua_abort(void) __attribute__((noreturn));

#undef lua_str2number
#define lua_str2number(s, p) kore_lua_strtod((s), (p))
/* Lua's own freestanding hexadecimal reader/writer avoids libc %a support. */
#undef lua_strx2number
#undef lua_number2strx

static int kore_lua_snprintf(char *dst, size_t cap, const char *format, ...);
#undef l_sprintf
#define l_sprintf(s, sz, f, i) kore_lua_snprintf((s), (sz), (f), (i))

/* `setjmp`/`longjmp` are the only non-local control transfer used by Lua. */
#define abort kore_lua_abort
/*
 * The VM seed: string hashing, table iteration order, and — through
 * `luaL_makeseed` — `math.random`'s initial state (lmathlib.c's
 * `setseed(..., luaL_makeseed(L), 0)`).  Upstream derives it from ASLR and
 * `time()`, neither of which a kore process has; the Rust host draws it
 * from the CPU instead, and falls back to the fixed constant this replaced
 * only where the machine exposes no entropy source at all.
 */
extern unsigned kore_lua_host_seed(void);
/* ASCII "KORE" — the value src/bridge.rs calls UNSEEDED and falls back to when
   the machine exposes no generator.  Kept identical here so a host smoke
   build, which links no Rust host, reproduces that same unseeded VM. */
#define KORE_LUA_UNSEEDED 0x4b4f5245u
__attribute__((weak)) unsigned kore_lua_host_seed(void) { return KORE_LUA_UNSEEDED; }
#define luai_makeseed() kore_lua_host_seed()

/* Keep internal engine names local to this one object. */
#undef LUAI_FUNC
#undef LUAI_DDEC
#undef LUAI_DDEF
#define LUAI_FUNC static
#define LUAI_DDEC(def)
#define LUAI_DDEF static

/* Core. */
#include "lzio.c"
#include "lctype.c"
#include "lopcodes.c"
#include "lmem.c"
#include "lundump.c"
#include "ldump.c"
#include "lstate.c"
#include "lgc.c"
#include "llex.c"
#include "lcode.c"
#include "lparser.c"
#include "ldebug.c"
#include "lfunc.c"
#include "lobject.c"
#include "ltm.c"
#include "lstring.c"
#include "ltable.c"
#include "ldo.c"
#include "lvm.c"
#include "lapi.c"

/*
 * Auxiliary and deliberately selected standard libraries.  Five auxiliary
 * functions are hosted-libc only (FILE*, errno, strerror, realloc/free):
 * renamed out of the luaL_* namespace so the exported-API glob does not
 * make them link roots (nothing in a kore image satisfies fopen, stdin,
 * strerror, or realloc; before the export they were unreferenced and
 * --gc-sections dropped them).  A native module therefore cannot reach
 * them, which is the honest ABI: states come from the host's lua_newstate
 * with the Rust allocator, never luaL_newstate.
 */
#define luaL_fileresult kore_lua_aux_fileresult_unavailable
#define luaL_execresult kore_lua_aux_execresult_unavailable
#define luaL_loadfilex kore_lua_aux_loadfilex_unavailable
#define luaL_newstate kore_lua_aux_newstate_unavailable
#define luaL_alloc kore_lua_aux_alloc_unavailable
#include "lauxlib.c"
#undef luaL_loadfilex

/* The host owns source loading; base cannot reach ambient FILE* or bytecode. */
static int kore_lua_loadfilex_disabled(lua_State *L, const char *filename,
                                       const char *mode) {
  (void)filename; (void)mode;
  return luaL_error(L, "host filesystem loading is unavailable");
}
#define luaL_loadfilex kore_lua_loadfilex_disabled
#include "lbaselib.c"
#undef luaL_loadfilex
#include "lcorolib.c"
#include "lmathlib.c"
#include "lstrlib.c"
#include "ltablib.c"
#include "lutf8lib.c"

/*
 * This is deliberately not a partial printf implementation. Lua core needs
 * exactly three conversions outside string.format: its configured integer
 * and number formats, plus the implementation-defined pointer spelling used
 * in diagnostics. string.format itself is removed from the exported library.
 *
 * The return value is the number of bytes actually stored. In particular, it
 * can never make a caller treat a truncated would-write length as initialized.
 */
static size_t kore_store(char *dst, size_t cap, const char *src, size_t len) {
  if (dst == NULL || cap == 0) return 0;
  size_t stored = len < cap - 1 ? len : cap - 1;
  if (stored != 0) memcpy(dst, src, stored);
  dst[stored] = '\0';
  return stored;
}

static size_t kore_unsigned(char *dst, unsigned long long value,
                            unsigned base) {
  static const char digits[] = "0123456789abcdef";
  char reverse[65];
  size_t n = 0;
  do {
    reverse[n++] = digits[value % base];
    value /= base;
  } while (value != 0);
  for (size_t i = 0; i < n; i++)
    dst[i] = reverse[n - i - 1];
  return n;
}

static int kore_lua_snprintf(char *dst, size_t cap, const char *format, ...) {
  char item[LUA_N2SBUFFSZ];
  size_t item_len = 0;
  va_list args;
  va_start(args, format);
  if (format != NULL && strcmp(format, LUA_INTEGER_FMT) == 0) {
    lua_Integer value = va_arg(args, LUAI_UACINT);
    unsigned long long magnitude = value < 0
        ? 0ULL - (unsigned long long)value : (unsigned long long)value;
    if (value < 0) item[item_len++] = '-';
    item_len += kore_unsigned(item + item_len, magnitude, 10);
  }
  else if (format != NULL && strcmp(format, LUA_NUMBER_FMT) == 0) {
    double value = va_arg(args, LUAI_UACNUMBER);
    item_len = kore_lua_format_number(item, sizeof(item), value, 15);
  }
  else if (format != NULL && strcmp(format, LUA_NUMBER_FMT_N) == 0) {
    double value = va_arg(args, LUAI_UACNUMBER);
    item_len = kore_lua_format_number(item, sizeof(item), value, 17);
  }
  else if (format != NULL && strcmp(format, "%p") == 0) {
    uintptr_t value = (uintptr_t)va_arg(args, const void *);
    item[item_len++] = '0'; item[item_len++] = 'x';
    item_len += kore_unsigned(item + item_len, value, 16);
  }
  else {
    /* An added call site must extend this audited core contract explicitly. */
    item[item_len++] = '?';
  }
  va_end(args);
  return (int)kore_store(dst, cap, item, item_len);
}

/*
 * `require`, over the executor (src/host.rs; res/DAGFS.md §4).  The VM runs
 * on a worker thread, so a synchronous C `require` cannot drive the async
 * filesystem: it hands the name to Rust, which posts a request to the
 * task's executor thread and parks this thread on its doorbell until the
 * answer is stored.  The answer is a text chunk the host owns until
 * kore_lua_host_release (source module, `<name>.lua` in the task's tree),
 * a C opener kore-late linked into this task (native module,
 * `libs/<name>.so`), or a NUL-terminated error.  Loaded modules are cached
 * in the registry's LUA_LOADED_TABLE exactly as stock `package` does, so
 * `require "string"` answers the standard library the opener registered.
 */
enum {
  KORE_LUA_MODULE_ERROR = 0,
  KORE_LUA_MODULE_SOURCE = 1,
  KORE_LUA_MODULE_NATIVE = 2
};
extern int kore_lua_host_require(const char *name, size_t len,
                                 const char **bytes, size_t *bytes_len,
                                 lua_CFunction *opener);
extern void kore_lua_host_release(void);

/*
 * Weak fallbacks: the native smoke builds (build.rs) link this engine with
 * no Rust host, and there `require` answers "no host".  On the kore target
 * src/host.rs defines both strongly and the linker takes those.
 */
__attribute__((weak)) int kore_lua_host_require(const char *name, size_t len,
                                                const char **bytes,
                                                size_t *bytes_len,
                                                lua_CFunction *opener) {
  static const char message[] = "no module host";
  (void)name; (void)len;
  *bytes = message;
  *bytes_len = sizeof(message) - 1;
  *opener = NULL;
  return KORE_LUA_MODULE_ERROR;
}
__attribute__((weak)) void kore_lua_host_release(void) {}

/* The rest of the host-call op vocabulary; src/bridge.rs holds the other half of
   every one of these numbers; an embedder's own start at 64. */
enum {
  KORE_LUA_OP_SLEEP = 16,
  KORE_LUA_OP_PROC_ARGS = 17,
  /* `kore.fs`, and so `io` and the filesystem half of `os`. */
  KORE_LUA_OP_FS_OPEN = 18,
  KORE_LUA_OP_FS_CLOSE = 19,
  KORE_LUA_OP_FS_READ = 20,
  KORE_LUA_OP_FS_WRITE = 21,
  /* 22 was a host-side seek.  The stream position lives in the file object
     below, so there is nothing for the host to seek; the number is left
     unused rather than recycled, because src/bridge.rs must agree. */
  KORE_LUA_OP_FS_SYNC = 23,
  KORE_LUA_OP_FS_RESIZE = 24,
  KORE_LUA_OP_FS_STAT = 25,
  KORE_LUA_OP_FS_REMOVE = 26,
  KORE_LUA_OP_FS_RENAME = 27,
  KORE_LUA_OP_FS_LIST = 28,
  /* `os.exit(code)` — records the code with the host, which reads it once
     the evaluation has unwound. */
  KORE_LUA_OP_PROC_EXIT = 29,
  /* A whole file by path, in one call: open, stream, close, all on thread
     0.  Not a convenience over `KORE_LUA_OP_FS_READ` — the host's streaming
     read is the only one that can hold a pinned version for the whole file,
     and a read assembled here from page-sized calls could not (see
     `kore_lua_file_fill`). */
  KORE_LUA_OP_FS_READ_FILE = 30
};
/* The error `os.exit` raises to unwind the evaluation.  `luaL_error` puts
   the chunk and line in front of it, so the host matches this as a *suffix*
   and reports the recorded code instead of an error; anything else is a
   genuine error and is reported as one.  src/vm.rs holds the other half of
   this spelling, as src/bridge.rs holds the other half of the op numbers. */
#define KORE_LUA_EXIT_SENTINEL "kore:exit"
/* The open-mode bits src/files.rs reads out of arg0. */
enum {
  KORE_LUA_O_READ = 1 << 0,
  KORE_LUA_O_WRITE = 1 << 1,
  KORE_LUA_O_CREATE = 1 << 2,
  KORE_LUA_O_TRUNCATE = 1 << 3,
  KORE_LUA_O_APPEND = 1 << 4,
  KORE_LUA_O_EXCLUSIVE = 1 << 5,
  KORE_LUA_O_DIRECTORY = 1 << 6
};
enum {
  KORE_LUA_STATUS_OK = 0,
  KORE_LUA_STATUS_REFUSED = 1,
  KORE_LUA_STATUS_UNAVAILABLE = 2
};
/*
 * Every value-returning host call: three argument words in, a status, two
 * result words and a text out.  One entry point for `kore.time`, an
 * embedder's libraries, and every later standard-library verb, because they are all the same
 * round trip over src/bridge.rs's one transport (`host_call` there).
 */
extern int kore_lua_host_call(unsigned op, unsigned long long arg0,
                              unsigned long long arg1, unsigned long long arg2,
                              const void *input, size_t input_len,
                              unsigned long long *result,
                              unsigned long long *result2, const char **text,
                              size_t *text_len);
__attribute__((weak)) int kore_lua_host_call(unsigned op,
                                             unsigned long long arg0,
                                             unsigned long long arg1,
                                             unsigned long long arg2,
                                             const void *input, size_t input_len,
                                             unsigned long long *result,
                                             unsigned long long *result2,
                                             const char **text, size_t *text_len) {
  static const char message[] = "no kore host";
  (void)op; (void)arg0; (void)arg1; (void)arg2; (void)input; (void)input_len;
  *result = 0; *result2 = 0;
  *text = message; *text_len = sizeof(message) - 1;
  return KORE_LUA_STATUS_UNAVAILABLE;
}

/*
 * Push the `nil, why` failure pair for a non-OK status; returns 2.  Every
 * kore library answers failure this way, so a script that learned one
 * module's error handling has learned them all.
 */
static int kore_lua_fail(lua_State *L, int status, const char *text, size_t text_len) {
  lua_pushnil(L);
  if (text == NULL) {
    text = "unavailable";
    text_len = sizeof "unavailable" - 1;
  }
  if (status == KORE_LUA_STATUS_REFUSED) {
    /* Concatenated rather than `lua_pushfstring(L, "refused: %s", text)`:
       the bridge's text is length-delimited and carries no terminator, so
       `%s` would run off the end of the reply buffer. */
    lua_pushliteral(L, "refused: ");
    lua_pushlstring(L, text, text_len);
    lua_concat(L, 2);
  } else {
    lua_pushlstring(L, text, text_len);
  }
  return 2;
}

/*
 * The same failure, in `kore.*`'s spelling: `nil` and a *table* carrying the
 * bare code (`"not-found"`), the reason it came back under, and a
 * `__tostring` so `print(err)` and `assert(f())` still read well.
 *
 * One failure, two spellings.  The stock `io` surface renders this same
 * status as `nil, message, errno` a few hundred lines below; a script that
 * wants to branch on *which* failure gets a word it can compare, instead of
 * matching a sentence.
 */
static int kore_lua_error_tostring(lua_State *L) {
  lua_getfield(L, 1, "code");
  return 1;
}

static int kore_lua_fail_table(lua_State *L, int status, const char *text,
                               size_t text_len) {
  lua_pushnil(L);
  lua_createtable(L, 0, 2);
  lua_pushlstring(L, text != NULL ? text : "unavailable", text_len);
  lua_setfield(L, -2, "code");
  lua_pushstring(L, status == KORE_LUA_STATUS_REFUSED ? "refused" : "unavailable");
  lua_setfield(L, -2, "kind");
  lua_createtable(L, 0, 1);
  lua_pushcfunction(L, kore_lua_error_tostring);
  lua_setfield(L, -2, "__tostring");
  lua_setmetatable(L, -2);
  return 2;
}

/*
 * [`kore_lua_fail`] for a NUL-terminated reason.
 *
 * The two host contracts differ and the difference is load-bearing:
 * `kore_lua_host_call` answers a text with an explicit length and **no
 * terminator** (it carries arbitrary bytes out of a Rust `Vec`), while the
 * synchronous leaf calls — `kore_lua_host_uptime_ns`, `..._wall_ns`,
 * `..._random` — answer a `&'static CStr`.  Only the latter may reach this
 * function; a length-carrying text must go to [`kore_lua_fail`] with the
 * length the bridge gave it, or `strlen` reads past the reply buffer.
 */
static int kore_lua_fail_cstr(lua_State *L, int status, const char *text) {
  return kore_lua_fail(L, status, text, text != NULL ? strlen(text) : 0);
}

/*
 * The shared shape for every `nil, why` library below: on a non-OK status
 * push the failure pair and return 2, else return 0 with the result words
 * and the answer text filled.  Callers add their own values after a 0.
 */
static int kore_lua_host_op(lua_State *L, unsigned op, unsigned long long arg0,
                         unsigned long long arg1, unsigned long long arg2,
                         unsigned long long *result,
                         unsigned long long *result2, const char **text,
                         size_t *text_len) {
  const char *why = NULL;
  size_t why_len = 0;
  int status = kore_lua_host_call(op, arg0, arg1, arg2, NULL, 0, result, result2,
                                  &why, &why_len);
  if (text != NULL) *text = why;
  if (text_len != NULL) *text_len = why_len;
  if (status != KORE_LUA_STATUS_OK)
    return kore_lua_fail(L, status, why, why_len);
  return 0;
}

/*
 * The byte-carrying form, and the one every filesystem verb uses.  `input`
 * is handed straight to the bridge as a borrow of the Lua string the caller
 * passed: the worker parks for the whole call, so the value stays rooted on
 * its stack and nothing is copied on the way down.
 *
 * `render` chooses the failure spelling — [`kore_lua_fail_table`] for
 * `kore.fs`, the `io` renderer for the stock surface — so one call site
 * serves both surfaces over one op.
 */
typedef int (*kore_lua_render_fail)(lua_State *, int, const char *, size_t);

static int kore_lua_host_bytes(lua_State *L, unsigned op,
                               unsigned long long arg0, unsigned long long arg1,
                               unsigned long long arg2, const void *input,
                               size_t input_len, unsigned long long *result,
                               unsigned long long *result2, const char **text,
                               size_t *text_len, kore_lua_render_fail render) {
  const char *why = NULL;
  size_t why_len = 0;
  int status = kore_lua_host_call(op, arg0, arg1, arg2, input, input_len, result,
                                  result2, &why, &why_len);
  if (text != NULL) *text = why;
  if (text_len != NULL) *text_len = why_len;
  if (status != KORE_LUA_STATUS_OK) return render(L, status, why, why_len);
  return 0;
}

/*
 * ------------------------------------------------------------ kore.time
 *
 * Seconds are floats; anything else carries its unit in the name.  A clock
 * that cannot be read answers `nil, why` — never a zero timestamp, which a
 * caller would read as "the monotonic epoch" rather than "no reading".
 */
extern int kore_lua_host_uptime_ns(unsigned long long *out, const char **text);
extern int kore_lua_host_wall_ns(unsigned long long *out, const char **text);

__attribute__((weak)) int kore_lua_host_uptime_ns(unsigned long long *out,
                                                  const char **text) {
  static const char message[] = "no kore host";
  *out = 0; *text = message;
  return KORE_LUA_STATUS_UNAVAILABLE;
}
__attribute__((weak)) int kore_lua_host_wall_ns(unsigned long long *out,
                                                const char **text) {
  static const char message[] = "no kore host";
  *out = 0; *text = message;
  return KORE_LUA_STATUS_UNAVAILABLE;
}

static int kore_lua_clock(lua_State *L, int wall, int as_seconds) {
  unsigned long long ns = 0;
  const char *why = NULL;
  int status = wall ? kore_lua_host_wall_ns(&ns, &why)
                    : kore_lua_host_uptime_ns(&ns, &why);
  if (status != KORE_LUA_STATUS_OK) return kore_lua_fail_cstr(L, status, why);
  /*
   * Seconds are a float, which is what a script wants to subtract and
   * compare.  A double holds 53 bits of mantissa, so a wall reading (~1.8e18
   * ns since the epoch) keeps roughly 256 ns of resolution and a monotonic
   * one keeps far more; `mono_ns` / `wall_ns` answer the exact integer for
   * callers that need every nanosecond.
   */
  if (as_seconds)
    lua_pushnumber(L, (lua_Number)ns / 1e9);
  else
    lua_pushinteger(L, (lua_Integer)ns);
  return 1;
}

static int kore_lua_time_now(lua_State *L) { return kore_lua_clock(L, 0, 1); }
static int kore_lua_time_mono_ns(lua_State *L) { return kore_lua_clock(L, 0, 0); }
static int kore_lua_time_wall(lua_State *L) { return kore_lua_clock(L, 1, 1); }
static int kore_lua_time_wall_ns(lua_State *L) { return kore_lua_clock(L, 1, 0); }

static int kore_lua_time_sleep(lua_State *L) {
  lua_Number seconds = luaL_checknumber(L, 1);
  luaL_argcheck(L, seconds >= 0, 1, "delay must not be negative");
  /* Saturate rather than wrap: an absurd delay is a long sleep, not a short
     one.  The bound is the u64 nanosecond range the executor's timer takes. */
  lua_Number ns = seconds * 1e9;
  unsigned long long delay = ns >= 1.8446744e19 ? 0xffffffffffffffffull
                                                : (unsigned long long)ns;
  unsigned long long r = 0, r2 = 0;
  int n = kore_lua_host_op(L, KORE_LUA_OP_SLEEP, delay, 0, 0, &r, &r2, NULL, NULL);
  if (n) return n;
  lua_pushboolean(L, 1);
  return 1;
}

static const luaL_Reg kore_lua_time_lib[] = {
  {"now", kore_lua_time_now},
  {"mono_ns", kore_lua_time_mono_ns},
  {"wall", kore_lua_time_wall},
  {"wall_ns", kore_lua_time_wall_ns},
  {"sleep", kore_lua_time_sleep},
  {NULL, NULL}
};

/*
 * ---------------------------------------------------------- kore.random
 *
 * A source, not a stream: every byte is a fresh CPU draw.  A script wanting
 * many cheap values should seed `math.random` (which this build now seeds
 * from the same source at startup) rather than loop here.
 */
extern int kore_lua_host_random(unsigned char *dst, size_t len, const char **text);
extern int kore_lua_host_random_available(void);

__attribute__((weak)) int kore_lua_host_random(unsigned char *dst, size_t len,
                                               const char **text) {
  static const char message[] = "no kore host";
  if (dst != NULL && len != 0) memset(dst, 0, len);
  *text = message;
  return KORE_LUA_STATUS_UNAVAILABLE;
}
__attribute__((weak)) int kore_lua_host_random_available(void) { return 0; }

/* Bounded by one Lua buffer's worth: a draw is a syscall-free CPU loop, and
   an unbounded request would be an unbounded one. */
#define KORE_LUA_RANDOM_MAX 4096

static int kore_lua_random_bytes(lua_State *L) {
  lua_Integer count = luaL_checkinteger(L, 1);
  luaL_argcheck(L, count >= 0 && count <= KORE_LUA_RANDOM_MAX, 1,
                "byte count is 0..4096");
  luaL_Buffer buffer;
  char *dst = luaL_buffinitsize(L, &buffer, (size_t)count);
  const char *why = NULL;
  int status = kore_lua_host_random((unsigned char *)dst, (size_t)count, &why);
  if (status != KORE_LUA_STATUS_OK) {
    luaL_pushresultsize(&buffer, 0);
    lua_pop(L, 1);
    return kore_lua_fail_cstr(L, status, why);
  }
  luaL_pushresultsize(&buffer, (size_t)count);
  return 1;
}

static int kore_lua_random_integer(lua_State *L) {
  lua_Integer low = luaL_checkinteger(L, 1);
  lua_Integer high = luaL_checkinteger(L, 2);
  luaL_argcheck(L, low <= high, 2, "empty interval");
  lua_Unsigned span = (lua_Unsigned)high - (lua_Unsigned)low;
  /*
   * The engine's own projection (lmathlib.c `project`): mask the draw down
   * to the smallest Mersenne number not smaller than the span, and redraw
   * while it lands outside.  Masking rather than a modulo fold keeps every
   * value equally likely, needs no division — so a full-width interval is
   * not a special case that could divide by zero — and makes
   * `kore.random.integer` and `math.random` agree on what uniform means.
   * Acceptance is at least one draw in two, and a dead entropy source ends
   * the loop with its refusal rather than spinning.
   */
  lua_Unsigned mask = span;
  for (int shift = 1; (mask & (mask + 1)) != 0; shift *= 2)
    mask |= (mask >> shift);
  for (;;) {
    unsigned long long draw = 0;
    const char *why = NULL;
    int status = kore_lua_host_random((unsigned char *)&draw, sizeof draw, &why);
    if (status != KORE_LUA_STATUS_OK) return kore_lua_fail_cstr(L, status, why);
    lua_Unsigned candidate = (lua_Unsigned)draw & mask;
    if (candidate <= span) {
      lua_pushinteger(L, (lua_Integer)(candidate + (lua_Unsigned)low));
      return 1;
    }
  }
}

static int kore_lua_random_available(lua_State *L) {
  lua_pushboolean(L, kore_lua_host_random_available());
  return 1;
}

static const luaL_Reg kore_lua_random_lib[] = {
  {"bytes", kore_lua_random_bytes},
  {"integer", kore_lua_random_integer},
  {"available", kore_lua_random_available},
  {NULL, NULL}
};

/*
 * ------------------------------------------------------------ kore.proc
 *
 * The launch arguments.  `args` is a lazy field rather than a call, because
 * that is how a Lua program expects to read them — the metatable below
 * fetches it once, on first touch, and caches it.
 */
static int kore_lua_proc_args(lua_State *L) {
  unsigned long long r = 0, r2 = 0;
  const char *text = NULL;
  size_t text_len = 0;
  int n = kore_lua_host_op(L, KORE_LUA_OP_PROC_ARGS, 0, 0, 0, &r, &r2, &text,
                           &text_len);
  if (n) return n;
  lua_pushlstring(L, text != NULL ? text : "", text_len);
  return 1;
}

/* `kore.proc.args` on first touch; thereafter the cached table. */
static int kore_lua_proc_index(lua_State *L) {
  /* A missing key answers nil for any key type, as an ordinary table does.
     Raising here would make `kore.proc[1]` an error where `{}[1]` is nil. */
  const char *key = lua_type(L, 2) == LUA_TSTRING ? lua_tostring(L, 2) : NULL;
  if (key == NULL || strcmp(key, "args") != 0) { lua_pushnil(L); return 1; }
  if (kore_lua_proc_args(L) != 1) { lua_pushnil(L); return 1; }
  /* Split the launch bytes on ASCII whitespace into a 1-based array, the
     shape a Lua program indexes and iterates with ipairs. */
  size_t len = 0;
  const char *bytes = lua_tolstring(L, -1, &len);
  lua_newtable(L);
  lua_Integer index = 0;
  size_t start = 0;
  for (size_t i = 0; i <= len; i++) {
    int boundary = i == len || bytes[i] == ' ' || bytes[i] == '\t' ||
                   bytes[i] == '\n' || bytes[i] == '\r';
    if (!boundary) continue;
    if (i > start) {
      lua_pushlstring(L, bytes + start, i - start);
      lua_rawseti(L, -2, ++index);
    }
    start = i + 1;
  }
  /* Cache on the module table itself, so the next read is a plain rawget
     and the bridge is crossed exactly once per process. */
  lua_pushvalue(L, -1);
  lua_setfield(L, 1, "args");
  return 1;
}

/*
 * `kore.proc` carries no functions: `args` is the lazy field above, and
 * termination is `os.exit`, which belongs on the stock surface where a
 * script already looks for it.  The empty list is what `luaL_newlib` wants
 * for a table whose only member arrives through `__index`.
 */
static const luaL_Reg kore_lua_proc_lib[] = {
  {NULL, NULL}
};

/* Narrow wrappers keep Rust independent of Lua's preprocessor-only API. */
typedef int (*kore_lua_append)(void *ud, const char *bytes, size_t len,
                               int stderr_stream);

typedef struct kore_lua_writer {
  kore_lua_append append;
  void *ud;
} kore_lua_writer;

/*
 * -------------------------------------------------------------- kore.fs
 *
 * The file object, and the one place a stream position lives.
 *
 * `io.open` and `kore.fs.open` answer the *same* userdata: two spellings
 * over one implementation is the whole point of having both surfaces, and
 * a script that got a file from one must be able to hand it to the other.
 * src/files.rs holds the view and answers positionally; the position, the
 * read-ahead buffer and the mode flags are here, because they are stream
 * ideas and the store has no streams.
 *
 * Rule 4 — one handle protocol: `:close()` is idempotent, `:closed()`
 * answers without raising, `__close` and `__gc` both retire the handle,
 * and `__tostring` prints an opaque identity (the slot and generation the
 * bridge assigned), never an address.
 */
#define KORE_LUA_FILE "kore.file"
/* One page. A line read must not be a bridge crossing per byte: the park
   and the kick dominate everything else, so reads go a page at a time and
   `f:read("l")` is a memchr over bytes already here. */
#define KORE_LUA_BUFSIZE 4096

/*
 * The host facade's streaming threshold (`FACADE_MIN_BYTES` in kore-dagfs,
 * `files::FACADE_STREAM_BYTES` on the Rust half of this bridge).
 *
 * A destination at least this large is read through the launch's storage
 * ring from one pinned version; below it the facade copies a window at a
 * time.  It is a fact about the host, not a tunable here, and the two
 * halves are pinned against each other by `kore_lua_read_geometry` below.
 */
#define KORE_LUA_FACADE_STREAM_MIN (256u * 1024u)

/*
 * A read of at least this much bypasses the page buffer and asks the host
 * for the whole remainder in one call.
 *
 * Reading a large file a page at a time presents a page-sized destination
 * on every call and so never reaches the threshold above, however large
 * the file is.  The decision therefore has to be made on this side, which
 * is why the constant lives next to the buffer it bypasses.
 *
 * Separately defined from the facade's own number, and overridable, so a
 * test can reach the bypass without a quarter-megabyte fixture — without
 * that lowering what this file believes the facade promises.
 */
#ifndef KORE_LUA_STREAM_MIN
#define KORE_LUA_STREAM_MIN KORE_LUA_FACADE_STREAM_MIN
#endif

_Static_assert(KORE_LUA_STREAM_MIN > KORE_LUA_BUFSIZE,
               "a bypass no larger than the page buffer saves no round trip");

/*
 * The sizes the read path's behaviour turns on, as this translation unit
 * actually compiled them.
 *
 * A harness asserts against these rather than against a header, because
 * the failure worth catching is exactly the one a header cannot show: an
 * edit to `KORE_LUA_BUFSIZE` or to the threshold that silently stops every
 * read from ever being large enough to stream.
 */
enum {
  KORE_LUA_GEOMETRY_BUFSIZE = 0,
  KORE_LUA_GEOMETRY_STREAM_MIN = 1,
  KORE_LUA_GEOMETRY_FACADE_STREAM_MIN = 2
};

unsigned long long kore_lua_read_geometry(int which) {
  switch (which) {
    case KORE_LUA_GEOMETRY_BUFSIZE: return KORE_LUA_BUFSIZE;
    case KORE_LUA_GEOMETRY_STREAM_MIN: return KORE_LUA_STREAM_MIN;
    case KORE_LUA_GEOMETRY_FACADE_STREAM_MIN: return KORE_LUA_FACADE_STREAM_MIN;
    default: return 0;
  }
}

enum {
  KORE_LUA_FILE_STORE = 0,  /* a file in the store */
  KORE_LUA_FILE_OUT = 1,    /* io.stdout */
  KORE_LUA_FILE_ERR = 2,    /* io.stderr */
  KORE_LUA_FILE_IN = 3      /* io.stdin */
};

typedef struct {
  unsigned long long handle;
  /* The stream position: where the next unbuffered read or write lands. */
  unsigned long long position;
  int kind;
  int closed;
  int readable;
  int writable;
  /* Read-ahead.  `buf[0]` is file offset `base`; `taken` bytes of the
     `have` valid ones have been handed to the script, so the script's
     logical position is `base + taken` whenever the buffer is live. */
  unsigned long long base;
  size_t have;
  size_t taken;
  int eof;
  /* The length `open` answered, kept as a *hint* for sizing a whole-file
     read and nothing else.  It goes stale the moment anyone writes or
     resizes, here or elsewhere, so no read trusts it: it only decides
     whether to try one large ask first, and every read still loops until
     the host answers zero. */
  unsigned long long length;
  /* `io.tmpfile`: the file unlinks itself when it closes, so `path` is the
     name to remove.  Nothing else records a path — an ordinary file is
     reached only through its handle, and remembering where it came from
     would be a second, staler authority over the same object. */
  int temporary;
  char path[48];
  char buf[KORE_LUA_BUFSIZE];
} kore_lua_file;

/* The registry key the host writer is stashed under, so `io.write` reaches
   the same sink `print` does rather than opening a second one. */
static const char kore_lua_writer_key = 0;

static kore_lua_writer *kore_lua_sink(lua_State *L) {
  kore_lua_writer *writer;
  lua_rawgetp(L, LUA_REGISTRYINDEX, &kore_lua_writer_key);
  writer = (kore_lua_writer *)lua_touserdata(L, -1);
  lua_pop(L, 1);
  return writer;
}

/*
 * The stock `io` surface's failure spelling: `nil, message, errno`.
 *
 * The errno is derived from the same kebab-case code `kore.fs` hands back
 * as a table, so the two surfaces cannot disagree about what happened —
 * they only disagree about how to say it.  A code with no POSIX analogue
 * answers `EIO`, which is the honest "something below failed".
 */
static int kore_lua_errno(const char *code, size_t len) {
  static const struct { const char *code; int number; } table[] = {
    {"not-found", 2},          /* ENOENT */
    {"already-exists", 17},    /* EEXIST */
    {"not-directory", 20},     /* ENOTDIR */
    {"is-directory", 21},      /* EISDIR */
    {"invalid-name", 22},      /* EINVAL */
    {"invalid-handle", 9},     /* EBADF */
    {"access-denied", 13},     /* EACCES */
    {"access-lost", 9},        /* EBADF */
    {"not-empty", 39},         /* ENOTEMPTY */
    {"out-of-storage", 28},    /* ENOSPC */
    {"out-of-handles", 24},    /* EMFILE */
    {"range", 34},             /* ERANGE */
    {"unsupported", 38},       /* ENOSYS */
    {"busy", 16},              /* EBUSY */
    {"cycle", 40},             /* ELOOP */
    {NULL, 0}
  };
  for (int i = 0; table[i].code != NULL; i++)
    if (strlen(table[i].code) == len && memcmp(table[i].code, code, len) == 0)
      return table[i].number;
  return 5;  /* EIO */
}

static int kore_lua_fail_io(lua_State *L, int status, const char *text,
                            size_t text_len) {
  (void)status;
  lua_pushnil(L);
  lua_pushlstring(L, text != NULL ? text : "unavailable", text_len);
  lua_pushinteger(L, kore_lua_errno(text != NULL ? text : "", text_len));
  return 3;
}

static kore_lua_file *kore_lua_file_check(lua_State *L, int index) {
  return (kore_lua_file *)luaL_checkudata(L, index, KORE_LUA_FILE);
}

/*
 * A closed file is a *programming* error, not an expected failure, so it
 * raises — exactly as stock Lua does, and exactly as rule 1 divides the
 * two cases.
 */
static kore_lua_file *kore_lua_file_live(lua_State *L, int index) {
  kore_lua_file *file = kore_lua_file_check(L, index);
  if (file->closed) luaL_error(L, "attempt to use a closed file");
  return file;
}

static kore_lua_file *kore_lua_file_push(lua_State *L, unsigned long long handle,
                                         int kind, int readable, int writable,
                                         unsigned long long position) {
  kore_lua_file *file =
      (kore_lua_file *)lua_newuserdatauv(L, sizeof(kore_lua_file), 0);
  file->handle = handle;
  file->position = position;
  file->kind = kind;
  file->closed = 0;
  file->readable = readable;
  file->writable = writable;
  file->base = position;
  file->have = 0;
  file->taken = 0;
  file->eof = 0;
  file->length = 0;
  file->temporary = 0;
  file->path[0] = '\0';
  luaL_setmetatable(L, KORE_LUA_FILE);
  return file;
}

/* Drop the read-ahead.  Called before anything that moves or invalidates
   the position: a seek, a write, a close. */
static void kore_lua_file_drop(kore_lua_file *file) {
  file->have = 0;
  file->taken = 0;
  file->eof = 0;
}

/*
 * Parse an `io.open` mode string into the bits src/files.rs reads.
 * Returns 0 when the spelling is not one Lua defines.
 */
static unsigned long long kore_lua_mode(const char *mode) {
  unsigned long long flags = 0;
  size_t len = strlen(mode);
  size_t i = 0;
  if (len == 0) return 0;
  switch (mode[0]) {
    case 'r': flags = KORE_LUA_O_READ; break;
    case 'w': flags = KORE_LUA_O_WRITE | KORE_LUA_O_CREATE | KORE_LUA_O_TRUNCATE; break;
    case 'a': flags = KORE_LUA_O_WRITE | KORE_LUA_O_CREATE | KORE_LUA_O_APPEND; break;
    default: return 0;
  }
  for (i = 1; i < len; i++) {
    /* `b` is accepted and ignored: kore has no text mode to differ from,
       so a binary flag would be a distinction without one. */
    if (mode[i] == 'b') continue;
    if (mode[i] == '+') {
      flags |= KORE_LUA_O_READ | KORE_LUA_O_WRITE;
      if (mode[0] == 'r') continue;       /* r+ does not create */
      flags |= KORE_LUA_O_CREATE;
      continue;
    }
    return 0;
  }
  return flags;
}

/*
 * Open, by value.  Takes the path and mode as C values rather than reading
 * fixed stack slots, because two of its three callers do not have them at
 * index 1 and 2 — `io.lines(name)` has the name at 1 and its read formats
 * after it, and `io.input(name)` has only the name.  Answers 1 with the
 * file pushed, or `render`'s failure count.
 */
static int kore_lua_open_named(lua_State *L, const char *path, size_t path_len,
                               const char *mode, int mode_index,
                               kore_lua_render_fail render) {
  unsigned long long flags = kore_lua_mode(mode);
  unsigned long long handle = 0, length = 0;
  int n;
  if (flags == 0) return luaL_argerror(L, mode_index, "invalid file mode");
  n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_OPEN, flags, 0, 0, path, path_len,
                          &handle, &length, NULL, NULL, render);
  if (n) return n;
  /* An append stream starts at the end, so `f:seek()` before the first
     write answers where those bytes will actually land. */
  {
    kore_lua_file *file =
        kore_lua_file_push(L, handle, KORE_LUA_FILE_STORE,
                           (flags & KORE_LUA_O_READ) != 0,
                           (flags & KORE_LUA_O_WRITE) != 0,
                           (flags & KORE_LUA_O_APPEND) ? length : 0);
    /* The same word `open` already answered, kept so `f:read("a")` can size
       one ask instead of discovering the end a page at a time.  A hint, not
       an authority — see the field. */
    file->length = length;
  }
  return 1;
}

/* `kore.fs.open` and `io.open`'s spelling: the path and mode are the call's
   own first two arguments; `render` picks the surface's failure shape. */
static int kore_lua_open_file(lua_State *L, kore_lua_render_fail render) {
  size_t path_len = 0;
  const char *path = luaL_checklstring(L, 1, &path_len);
  const char *mode = luaL_optstring(L, 2, "r");
  return kore_lua_open_named(L, path, path_len, mode, 2, render);
}

/*
 * `io.input`, `io.output` and `io.lines` "raise the error, instead of
 * returning an error code" (§6.8).  Upstream's wording is
 * `cannot open file 'NAME' (REASON)`, and a script that matches on it — or
 * a person reading a traceback — should see the same sentence here.
 *
 * Leaves the new file on the stack; never returns on failure.
 */
static void kore_lua_opencheck(lua_State *L, int name_index,
                               const char *mode) {
  size_t len = 0;
  const char *name = luaL_checklstring(L, name_index, &len);
  if (kore_lua_open_named(L, name, len, mode, name_index, kore_lua_fail_io) != 1) {
    /* The failure triple is on the stack: nil, message, errno. */
    const char *why = lua_tostring(L, -2);
    luaL_error(L, "cannot open file '%s' (%s)", name,
               why != NULL ? why : "unavailable");
  }
}

/*
 * Close.  Idempotent by contract (rule 4): closing twice is not an error,
 * and neither is closing a standard stream — it simply does not retire a
 * store handle, because there is none.
 */
/*
 * An `io.tmpfile` removes itself once its handle is retired.  Best effort
 * and deliberately silent: the file is already closed, the caller asked to
 * be rid of it, and a failure here has no answer a script could act on.
 */
static void kore_lua_file_unlink(kore_lua_file *file) {
  unsigned long long r = 0, r2 = 0;
  const char *why = NULL;
  size_t why_len = 0;
  if (!file->temporary || file->path[0] == '\0') return;
  (void)kore_lua_host_call(KORE_LUA_OP_FS_REMOVE, 0, 0, 0, file->path,
                           strlen(file->path), &r, &r2, &why, &why_len);
  file->temporary = 0;
  file->path[0] = '\0';
}

static int kore_lua_file_do_close(lua_State *L, kore_lua_file *file,
                                  kore_lua_render_fail render) {
  unsigned long long r = 0, r2 = 0;
  int n;
  /* "The I/O library never closes these files" (§6.8).  Upstream answers
     the refusal rather than raising, and so does this: a script that closes
     whatever `io.output()` happens to be must not die because today it is
     the terminal. */
  if (file->kind != KORE_LUA_FILE_STORE) {
    /* Exactly two values, as `io_noclose` answers: this refusal carries no
       errno because nothing was attempted below. */
    (void)render;
    lua_pushnil(L);
    lua_pushliteral(L, "cannot close standard file");
    return 2;
  }
  if (file->closed) {
    lua_pushboolean(L, 1);
    return 1;
  }
  kore_lua_file_drop(file);
  /* Marked closed before the call, not after: a refused close still
     retires the handle here, and leaving it live would let a script
     re-close a handle the host has already taken back. */
  file->closed = 1;
  n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_CLOSE, file->handle, 0, 0, NULL, 0,
                          &r, &r2, NULL, NULL, render);
  kore_lua_file_unlink(file);
  if (n) return n;
  lua_pushboolean(L, 1);
  return 1;
}

static int kore_lua_file_close(lua_State *L) {
  return kore_lua_file_do_close(L, kore_lua_file_check(L, 1), kore_lua_fail_io);
}

static int kore_lua_file_closed(lua_State *L) {
  lua_pushboolean(L, kore_lua_file_check(L, 1)->closed);
  return 1;
}

/* `__gc` and `__close` both retire the handle and neither may raise: a
   metamethod that threw here would unwind out of the collector. */
static int kore_lua_file_release(lua_State *L) {
  kore_lua_file *file = kore_lua_file_check(L, 1);
  /* A standard stream is never retired, not even by the collector: the
     host owns those, and the next chunk still writes to them. */
  if (file->kind != KORE_LUA_FILE_STORE) return 0;
  if (!file->closed) {
    unsigned long long r = 0, r2 = 0;
    const char *why = NULL;
    size_t why_len = 0;
    file->closed = 1;
    kore_lua_file_drop(file);
    (void)kore_lua_host_call(KORE_LUA_OP_FS_CLOSE, file->handle, 0, 0, NULL, 0,
                             &r, &r2, &why, &why_len);
  }
  file->closed = 1;
  kore_lua_file_unlink(file);
  return 0;
}

/* An opaque identity: the slot and generation the bridge assigned, never
   an address (rule 4). */
static int kore_lua_file_tostring(lua_State *L) {
  kore_lua_file *file = kore_lua_file_check(L, 1);
  static const char *const names[] = {"file", "stdout", "stderr", "stdin"};
  const char *name = names[file->kind & 3];
  if (file->closed)
    lua_pushfstring(L, "closed %s", name);
  else if (file->kind != KORE_LUA_FILE_STORE)
    lua_pushfstring(L, "%s", name);
  else
    lua_pushfstring(L, "file #%d.%d", (int)(file->handle & 0xffffffffu),
                    (int)(file->handle >> 32));
  return 1;
}

/*
 * Refill the read-ahead from the stream position.  Answers 1 on progress,
 * 0 at end of file, and -1 on a failure whose status and text it hands
 * back *without pushing anything*.
 *
 * Not rendering here is the whole point of the out-parameters: every
 * caller but one has a live `luaL_Buffer`, and once that buffer outgrows
 * `LUAL_BUFFERSIZE` it keeps a box userdata on the stack at a level
 * `luaL_pushresult` re-checks.  Pushing a failure triple on top of it
 * would leave the box below values it does not expect.  The caller
 * finalizes its buffer first and renders afterwards, which is safe because
 * `*why` stays alive until the *next* host call and finishing a buffer
 * makes none.
 */
static int kore_lua_file_fill(lua_State *L, kore_lua_file *file, int *status,
                              const char **why, size_t *why_len) {
  unsigned long long count = 0, len = 0;
  const char *bytes = NULL;
  size_t bytes_len = 0;
  (void)L;
  *status = KORE_LUA_STATUS_OK;
  *why = NULL;
  *why_len = 0;
  if (file->taken < file->have) return 1;       /* still buffered */
  if (file->eof) return 0;
  if (file->kind != KORE_LUA_FILE_STORE) {      /* stdin: always at end */
    file->eof = 1;
    return 0;
  }
  file->base = file->position;
  file->have = 0;
  file->taken = 0;
  *status = kore_lua_host_call(KORE_LUA_OP_FS_READ, file->handle,
                               file->position, KORE_LUA_BUFSIZE, NULL, 0,
                               &count, &len, &bytes, &bytes_len);
  if (*status != KORE_LUA_STATUS_OK) {
    *why = bytes;
    *why_len = bytes_len;
    return -1;
  }
  if (bytes == NULL) bytes_len = 0;
  if (bytes_len > KORE_LUA_BUFSIZE) bytes_len = KORE_LUA_BUFSIZE;
  if (bytes_len != 0) memcpy(file->buf, bytes, bytes_len);
  file->have = bytes_len;
  if (bytes_len == 0) { file->eof = 1; return 0; }
  return 1;
}

/* Hand out `want` bytes (or everything buffered when `want` is 0 and
   `all` is set), appending to `b`.  Answers how many were appended. */
static size_t kore_lua_file_take(kore_lua_file *file, luaL_Buffer *b,
                                 size_t want) {
  size_t ready = file->have - file->taken;
  size_t take = want < ready ? want : ready;
  luaL_addlstring(b, file->buf + file->taken, take);
  file->taken += take;
  file->position = file->base + file->taken;
  return take;
}

/*
 * A large read, from the host straight into `b`, with no page buffer in the
 * middle.
 *
 * One call for the whole `want` instead of `want`/`KORE_LUA_BUFSIZE` of
 * them.  That is the point: the host's facade streams a destination of at
 * least `KORE_LUA_STREAM_MIN` through this launch's storage ring from one
 * pinned version, and it can only do that for a destination it is handed
 * whole.  A page-at-a-time read is below the threshold on every call, so it
 * never streams however large the file is.
 *
 * The caller must have drained the buffer first: `base`, `have` and `taken`
 * describe a window this read moves past, and they are reset here.
 *
 * Answers how many bytes landed, or -1 with `*status`, `*why` and
 * `*why_len` set.  A short answer is neither an error nor necessarily the
 * end: the host bounds one reply, so the caller loops until it has what it
 * asked for or the answer is zero.  Nothing is pushed on the Lua stack
 * beyond `b`'s own growth, so a live `luaL_Buffer` stays at its level.
 */
static long long kore_lua_file_stream(kore_lua_file *file, luaL_Buffer *b,
                                      unsigned long long want, int *status,
                                      const char **why, size_t *why_len) {
  unsigned long long count = 0, len = 0;
  const char *bytes = NULL;
  size_t bytes_len = 0;
  *status = kore_lua_host_call(KORE_LUA_OP_FS_READ, file->handle,
                               file->position, want, NULL, 0, &count, &len,
                               &bytes, &bytes_len);
  if (*status != KORE_LUA_STATUS_OK) {
    *why = bytes;
    *why_len = bytes_len;
    return -1;
  }
  if (bytes == NULL) bytes_len = 0;
  /* The host answers at most what was asked; clamp rather than trust, for
     the same reason `kore_lua_file_fill` clamps to its buffer. */
  if ((unsigned long long)bytes_len > want) bytes_len = (size_t)want;
  if (bytes_len != 0) luaL_addlstring(b, bytes, bytes_len);
  file->position += bytes_len;
  file->base = file->position;
  file->have = 0;
  file->taken = 0;
  if (bytes_len == 0) file->eof = 1;
  return (long long)bytes_len;
}

/*
 * Whether the next `want` bytes should bypass the page buffer.
 *
 * Only with the buffer already drained — bytes already here are cheaper
 * than any round trip — only for a file in the store, and only when the ask
 * is one the host could stream.
 */
static int kore_lua_file_streams(const kore_lua_file *file,
                                 unsigned long long want) {
  return file->kind == KORE_LUA_FILE_STORE && !file->eof &&
         file->have == file->taken && want >= KORE_LUA_STREAM_MIN;
}

/*
 * Drop a partial read's buffer and answer the failure in `render`'s
 * spelling, as a negative count.  The buffer is finished and popped before
 * anything is pushed, so `render`'s values never land above a live box.
 */
static int kore_lua_read_failed(lua_State *L, luaL_Buffer *b, int status,
                                const char *why, size_t why_len,
                                kore_lua_render_fail render) {
  luaL_pushresult(b);
  lua_pop(L, 1);
  return -render(L, status, why, why_len);
}

/* `f:read("l")` / `"L"`: up to and including the newline. */
static int kore_lua_read_line(lua_State *L, kore_lua_file *file, int keep,
                              kore_lua_render_fail render) {
  luaL_Buffer b;
  int got = 0;
  int status = KORE_LUA_STATUS_OK;
  const char *why = NULL;
  size_t why_len = 0;
  luaL_buffinit(L, &b);
  for (;;) {
    int state = kore_lua_file_fill(L, file, &status, &why, &why_len);
    if (state < 0)
      return kore_lua_read_failed(L, &b, status, why, why_len, render);
    if (state == 0) break;
    {
      const char *from = file->buf + file->taken;
      size_t ready = file->have - file->taken;
      const char *nl = (const char *)memchr(from, '\n', ready);
      if (nl != NULL) {
        size_t upto = (size_t)(nl - from);
        luaL_addlstring(&b, from, keep ? upto + 1 : upto);
        file->taken += upto + 1;
        file->position = file->base + file->taken;
        got = 1;
        break;
      }
      luaL_addlstring(&b, from, ready);
      file->taken += ready;
      file->position = file->base + file->taken;
      got = got || ready != 0;
    }
  }
  if (!got) { luaL_pushresult(&b); lua_pop(L, 1); lua_pushnil(L); return 1; }
  luaL_pushresult(&b);
  return 1;
}

static int kore_lua_read_all(lua_State *L, kore_lua_file *file,
                             kore_lua_render_fail render) {
  luaL_Buffer b;
  int status = KORE_LUA_STATUS_OK;
  const char *why = NULL;
  size_t why_len = 0;
  luaL_buffinit(L, &b);
  for (;;) {
    /* What `open` said was left, if anything did.  A stale hint costs
       nothing either way: too high and the host answers short, too low and
       the fill below is the backstop that finds the real end. */
    unsigned long long left =
        file->length > file->position ? file->length - file->position : 0;
    if (kore_lua_file_streams(file, left)) {
      long long got =
          kore_lua_file_stream(file, &b, left, &status, &why, &why_len);
      if (got < 0)
        return kore_lua_read_failed(L, &b, status, why, why_len, render);
      if (got == 0) break;
      continue;
    }
    {
      int state = kore_lua_file_fill(L, file, &status, &why, &why_len);
      if (state < 0)
        return kore_lua_read_failed(L, &b, status, why, why_len, render);
      if (state == 0) break;
      kore_lua_file_take(file, &b, file->have - file->taken);
    }
  }
  /* "a" never fails: at end of file it answers the empty string, which is
     the one read format that does. */
  luaL_pushresult(&b);
  return 1;
}

static int kore_lua_read_count(lua_State *L, kore_lua_file *file, size_t want,
                               kore_lua_render_fail render) {
  luaL_Buffer b;
  size_t total = 0;
  int status = KORE_LUA_STATUS_OK;
  const char *why = NULL;
  size_t why_len = 0;
  if (want == 0) {
    /* Lua's zero-length read is a probe: the empty string when there is
       anything left, nil at end of file.  No buffer is live here, so the
       failure renders directly. */
    int state = kore_lua_file_fill(L, file, &status, &why, &why_len);
    if (state < 0) return -render(L, status, why, why_len);
    if (state == 0) { lua_pushnil(L); return 1; }
    lua_pushliteral(L, "");
    return 1;
  }
  luaL_buffinit(L, &b);
  while (total < want) {
    if (kore_lua_file_streams(file, want - total)) {
      long long got = kore_lua_file_stream(file, &b, want - total, &status,
                                           &why, &why_len);
      if (got < 0)
        return kore_lua_read_failed(L, &b, status, why, why_len, render);
      if (got == 0) break;
      total += (size_t)got;
      continue;
    }
    {
      int state = kore_lua_file_fill(L, file, &status, &why, &why_len);
      if (state < 0)
        return kore_lua_read_failed(L, &b, status, why, why_len, render);
      if (state == 0) break;
      total += kore_lua_file_take(file, &b, want - total);
    }
  }
  if (total == 0) { luaL_pushresult(&b); lua_pop(L, 1); lua_pushnil(L); return 1; }
  luaL_pushresult(&b);
  return 1;
}

/* `f:read("n")`: the longest prefix that is a numeral, via the engine's own
   reader so Lua and kore agree on what a number looks like. */
static int kore_lua_read_number(lua_State *L, kore_lua_file *file,
                                kore_lua_render_fail render) {
  char text[64];
  size_t len = 0;
  int seen_digit = 0;
  int status = KORE_LUA_STATUS_OK;
  const char *why = NULL;
  size_t why_len = 0;
  for (;;) {
    int state = kore_lua_file_fill(L, file, &status, &why, &why_len);
    /* The numeral is accumulated in `text`, not a `luaL_Buffer`, so this
       one renders directly. */
    if (state < 0) return -render(L, status, why, why_len);
    if (state == 0) break;
    {
      char c = file->buf[file->taken];
      if (len == 0 && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
        file->taken++; file->position = file->base + file->taken; continue;
      }
      int digit = c >= '0' && c <= '9';
      if (!(digit || c == '+' || c == '-' || c == '.' || c == 'e' ||
            c == 'E' || c == 'x' || c == 'X' || (c >= 'a' && c <= 'f') ||
            (c >= 'A' && c <= 'F'))) break;
      if (len + 1 >= sizeof text) break;
      if (digit) seen_digit = 1;
      text[len++] = c;
      file->taken++;
      file->position = file->base + file->taken;
    }
  }
  text[len] = '\0';
  if (len == 0 || !seen_digit) { lua_pushnil(L); return 1; }
  if (lua_stringtonumber(L, text) == 0) { lua_pushnil(L); return 1; }
  return 1;
}

/* One read format; answers how many values it pushed, or a negative count
   meaning "a failure of `-n` values is already on the stack".  A format
   that merely found nothing is *not* a failure: it answers 1 with nil on
   top, which is what end of file looks like to a script. */
static int kore_lua_read_one(lua_State *L, kore_lua_file *file, int index,
                             kore_lua_render_fail render) {
  if (lua_type(L, index) == LUA_TNUMBER) {
    lua_Integer want = luaL_checkinteger(L, index);
    luaL_argcheck(L, want >= 0, index, "byte count must not be negative");
    return kore_lua_read_count(L, file, (size_t)want, render);
  }
  {
    const char *format = luaL_checkstring(L, index);
    if (*format == '*') format++;   /* the 5.2 spelling, still accepted */
    switch (*format) {
      case 'l': return kore_lua_read_line(L, file, 0, render);
      case 'L': return kore_lua_read_line(L, file, 1, render);
      case 'a': return kore_lua_read_all(L, file, render);
      case 'n': return kore_lua_read_number(L, file, render);
      default:
        return luaL_argerror(L, index, "invalid format");
    }
  }
}

static int kore_lua_file_read_with(lua_State *L, kore_lua_file *file, int first,
                                   int last, kore_lua_render_fail render) {
  int top = last;
  int pushed = 0;
  if (!file->readable) {
    return render(L, KORE_LUA_STATUS_REFUSED, "access-denied",
                  sizeof "access-denied" - 1);
  }
  if (first > top) {
    int n = kore_lua_read_line(L, file, 0, render);
    return n < 0 ? -n : n;
  }
  for (int index = first; index <= top; index++) {
    int n = kore_lua_read_one(L, file, index, render);
    if (n < 0) {
      /* The read failed.  Stock `g_read` answers `luaL_fileresult` alone
         in this case, so drop whatever the earlier formats managed to
         read: the failure is the whole answer, not a tail on a partial
         one.  The failure's `-n` values are on top; each `lua_remove`
         takes the value directly beneath them. */
      int failed = -n;
      for (int drop = 0; drop < pushed; drop++) lua_remove(L, -failed - 1);
      return failed;
    }
    pushed += n;
    /* Stop at the first format that answered nil: the rest cannot succeed
       when this one hit end of file, and Lua stops too. */
    if (lua_isnil(L, -1)) break;
  }
  return pushed;
}

static int kore_lua_file_read(lua_State *L) {
  return kore_lua_file_read_with(L, kore_lua_file_live(L, 1), 2, lua_gettop(L),
                                 kore_lua_fail_io);
}

/*
 * Write.  A standard stream goes to the host sink `print` uses; a store
 * file crosses the bridge at the stream position (or, for an append file,
 * wherever src/files.rs finds the end now).
 */
static int kore_lua_file_write_with(lua_State *L, kore_lua_file *file, int first,
                                    int last, int self_index,
                                    kore_lua_render_fail render) {
  int top = last;
  if (!file->writable) {
    return render(L, KORE_LUA_STATUS_REFUSED, "access-denied",
                  sizeof "access-denied" - 1);
  }
  for (int index = first; index <= top; index++) {
    size_t len = 0;
    const char *bytes;
    /* A number is written as a number, not via tostring: that is the one
       coercion Lua's `file:write` defines, and everything else is an
       argument error rather than a silent `tostring`. */
    if (lua_type(L, index) == LUA_TNUMBER) {
      lua_pushvalue(L, index);
      bytes = lua_tolstring(L, -1, &len);
    } else {
      bytes = luaL_checklstring(L, index, &len);
      lua_pushvalue(L, index);
    }
    if (file->kind != KORE_LUA_FILE_STORE) {
      kore_lua_writer *writer = kore_lua_sink(L);
      if (writer != NULL && writer->append != NULL &&
          !writer->append(writer->ud, bytes, len,
                          file->kind == KORE_LUA_FILE_ERR))
        return luaL_error(L, "host output limit exceeded");
      lua_pop(L, 1);
      continue;
    }
    {
      unsigned long long written = 0, end = 0;
      int n;
      kore_lua_file_drop(file);
      n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_WRITE, file->handle,
                              file->position, 0, bytes, len, &written, &end,
                              NULL, NULL, render);
      if (n) return n;
      file->position = end;
      file->base = end;
      lua_pop(L, 1);
      if (written != (unsigned long long)len) {
        /* A short write is not silently successful. */
        return render(L, KORE_LUA_STATUS_REFUSED, "out-of-storage",
                      sizeof "out-of-storage" - 1);
      }
    }
  }
  /* Lua answers the file itself, so writes chain. */
  lua_pushvalue(L, self_index);
  return 1;
}

static int kore_lua_file_write(lua_State *L) {
  return kore_lua_file_write_with(L, kore_lua_file_live(L, 1), 2, lua_gettop(L),
                                  1, kore_lua_fail_io);
}

static int kore_lua_file_seek(lua_State *L) {
  kore_lua_file *file = kore_lua_file_live(L, 1);
  static const char *const whences[] = {"set", "cur", "end", NULL};
  int whence = luaL_checkoption(L, 2, "cur", whences);
  lua_Integer delta = (lua_Integer)luaL_optinteger(L, 3, 0);
  long long base = 0;
  if (file->kind != KORE_LUA_FILE_STORE) {
    /* A stream has no position to move; `seek()` on one answers its
       refusal rather than pretending a move happened. */
    return kore_lua_fail_io(L, KORE_LUA_STATUS_REFUSED, "unsupported",
                            sizeof "unsupported" - 1);
  }
  switch (whence) {
    case 0: base = 0; break;
    case 1: base = (long long)file->position; break;
    default: {
      unsigned long long length = 0, kind = 0;
      int n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_STAT, file->handle, 0, 0,
                                  NULL, 0, &length, &kind, NULL, NULL,
                                  kore_lua_fail_io);
      if (n) return n;
      base = (long long)length;
      break;
    }
  }
  if (base + (long long)delta < 0)
    return kore_lua_fail_io(L, KORE_LUA_STATUS_REFUSED, "range",
                            sizeof "range" - 1);
  kore_lua_file_drop(file);
  file->position = (unsigned long long)(base + (long long)delta);
  file->base = file->position;
  lua_pushinteger(L, (lua_Integer)file->position);
  return 1;
}

static int kore_lua_file_flush(lua_State *L) {
  kore_lua_file *file = kore_lua_file_live(L, 1);
  unsigned long long r = 0, r2 = 0;
  /* Writes are already acknowledged by the store before `f:write` returns,
     so this is a durability barrier, not a buffer drain — and on a
     standard stream there is nothing below to bar. */
  if (file->kind == KORE_LUA_FILE_STORE) {
    int n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_SYNC, file->handle, 0, 0,
                                NULL, 0, &r, &r2, NULL, NULL, kore_lua_fail_io);
    if (n) return n;
  }
  /* `true`, not the file: `f_flush` answers `luaL_fileresult`, and only
     `f:write` is special-cased to hand the file back for chaining. */
  lua_pushboolean(L, 1);
  return 1;
}

/*
 * `setvbuf` is accepted and has no effect: this build's reads are already
 * buffered a page at a time and its writes are acknowledged individually,
 * so there is no mode for it to select.  Answering `true` rather than
 * raising is the conforming choice — the manual lets an implementation
 * ignore the hint, and a script that sets a mode it does not depend on
 * should not fail.
 */
static int kore_lua_file_setvbuf(lua_State *L) {
  static const char *const modes[] = {"no", "full", "line", NULL};
  kore_lua_file_live(L, 1);
  (void)luaL_checkoption(L, 2, NULL, modes);
  lua_pushboolean(L, 1);
  return 1;
}

/* The `f:lines(...)` iterator: the file and its formats as upvalues, so
   the closure holds no state of its own beyond what the file already is. */
static int kore_lua_lines_step(lua_State *L) {
  kore_lua_file *file =
      (kore_lua_file *)lua_touserdata(L, lua_upvalueindex(1));
  int formats = (int)lua_tointeger(L, lua_upvalueindex(2));
  int close_at_end = (int)lua_tointeger(L, lua_upvalueindex(3));
  int pushed;
  if (file == NULL || file->closed)
    return luaL_error(L, "file is already closed");
  for (int i = 0; i < formats; i++) lua_pushvalue(L, lua_upvalueindex(4 + i));
  if (formats == 0) {
    pushed = kore_lua_read_line(L, file, 0, kore_lua_fail_io);
    if (pushed < 0) pushed = -pushed;
  } else {
    pushed = kore_lua_file_read_with(L, file, lua_gettop(L) - formats + 1,
                                     lua_gettop(L), kore_lua_fail_io);
  }
  /*
   * The *first* result decides, not the last: with several formats the
   * tail can be nil while the head read fine, and `io_readline` looks at
   * the head.  A falsy head with a message beside it is a read failure,
   * which stock Lua raises rather than ending the loop on — a `for` that
   * simply stopped would make an I/O error look like end of file.
   */
  if (!lua_toboolean(L, -pushed)) {
    if (pushed > 1) {
      const char *why = lua_tostring(L, -pushed + 1);
      return luaL_error(L, "%s", why != NULL ? why : "read failed");
    }
    if (close_at_end) {
      int top = lua_gettop(L);
      (void)kore_lua_file_do_close(L, file, kore_lua_fail_io);
      lua_settop(L, top);
    }
  }
  return pushed;
}

static int kore_lua_lines_push(lua_State *L, int file_index, int first,
                               int close_at_end) {
  int top = lua_gettop(L);
  int formats = top - first + 1;
  if (formats < 0) formats = 0;
  if (formats > 250) return luaL_error(L, "too many read formats");
  lua_pushvalue(L, file_index);
  lua_pushinteger(L, formats);
  lua_pushinteger(L, close_at_end);
  for (int i = 0; i < formats; i++) lua_pushvalue(L, first + i);
  lua_pushcclosure(L, kore_lua_lines_step, 3 + formats);
  return 1;
}

static int kore_lua_file_lines(lua_State *L) {
  kore_lua_file_live(L, 1);
  return kore_lua_lines_push(L, 1, 2, 0);
}

/* The file object's methods.  `kore.fs` and `io` share every one of them:
   the object is the same, so its protocol is too. */
static const luaL_Reg kore_lua_file_methods[] = {
  {"read", kore_lua_file_read},
  {"write", kore_lua_file_write},
  {"lines", kore_lua_file_lines},
  {"seek", kore_lua_file_seek},
  {"flush", kore_lua_file_flush},
  {"setvbuf", kore_lua_file_setvbuf},
  {"close", kore_lua_file_close},
  {"closed", kore_lua_file_closed},
  {NULL, NULL}
};

static void kore_lua_file_metatable(lua_State *L) {
  if (luaL_newmetatable(L, KORE_LUA_FILE)) {
    luaL_newlib(L, kore_lua_file_methods);
    lua_setfield(L, -2, "__index");
    lua_pushcfunction(L, kore_lua_file_release);
    lua_setfield(L, -2, "__gc");
    /* Deterministic close is primary; `__gc` is the backstop (rule 4). */
    lua_pushcfunction(L, kore_lua_file_release);
    lua_setfield(L, -2, "__close");
    lua_pushcfunction(L, kore_lua_file_tostring);
    lua_setfield(L, -2, "__tostring");
    lua_pushstring(L, "FILE*");
    lua_setfield(L, -2, "__name");
    /* A metatable a script could rewrite would let it forge a file. */
    lua_pushboolean(L, 0);
    lua_setfield(L, -2, "__metatable");
  }
  lua_pop(L, 1);
}

/*
 * `kore.fs`: the same object, the kebab-code failure table, and the
 * positional verbs the stream surface is built out of.
 */
static int kore_lua_fs_open(lua_State *L) {
  return kore_lua_open_file(L, kore_lua_fail_table);
}

/* One shape for every path verb that answers `true` or a failure. */
static int kore_lua_fs_path_op(lua_State *L, unsigned op) {
  size_t len = 0;
  const char *path = luaL_checklstring(L, 1, &len);
  unsigned long long r = 0, r2 = 0;
  int n = kore_lua_host_bytes(L, op, 0, 0, 0, path, len, &r, &r2, NULL, NULL,
                              kore_lua_fail_table);
  if (n) return n;
  lua_pushboolean(L, 1);
  return 1;
}

static int kore_lua_fs_remove(lua_State *L) {
  return kore_lua_fs_path_op(L, KORE_LUA_OP_FS_REMOVE);
}

static int kore_lua_fs_mkdir(lua_State *L) {
  size_t len = 0;
  const char *path = luaL_checklstring(L, 1, &len);
  unsigned long long r = 0, r2 = 0;
  int n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_OPEN, KORE_LUA_O_DIRECTORY, 0, 0,
                              path, len, &r, &r2, NULL, NULL, kore_lua_fail_table);
  if (n) return n;
  lua_pushboolean(L, 1);
  return 1;
}

/* Two paths in one window, NUL-separated: a name may not contain a NUL, so
   the split cannot be ambiguous. */
static int kore_lua_fs_rename(lua_State *L) {
  size_t from_len = 0, to_len = 0;
  const char *from = luaL_checklstring(L, 1, &from_len);
  const char *to = luaL_checklstring(L, 2, &to_len);
  unsigned long long r = 0, r2 = 0;
  luaL_Buffer b;
  char *joined;
  int n;
  luaL_buffinit(L, &b);
  joined = luaL_prepbuffsize(&b, from_len + to_len + 1);
  memcpy(joined, from, from_len);
  joined[from_len] = '\0';
  memcpy(joined + from_len + 1, to, to_len);
  n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_RENAME, 0, 0, 0, joined,
                          from_len + to_len + 1, &r, &r2, NULL, NULL,
                          kore_lua_fail_table);
  if (n) return n;
  lua_pushboolean(L, 1);
  return 1;
}

/*
 * `kore.fs.resize(file, bytes)` — the one verb that shortens or extends a
 * file without rewriting it.
 *
 * On `kore.fs` and not on the file object: `io`'s seven methods are what
 * the manual lists and a stock `file:` gains nothing here, while truncation
 * is exactly the kind of thing the kore surface exists to expose.  A
 * shortened file drags the stream position back with it — the next write
 * must not land past a hole the caller never asked for.
 */
static int kore_lua_fs_resize(lua_State *L) {
  kore_lua_file *file = kore_lua_file_live(L, 1);
  lua_Integer length = luaL_checkinteger(L, 2);
  unsigned long long r = 0, r2 = 0;
  int n;
  luaL_argcheck(L, length >= 0, 2, "length must not be negative");
  if (file->kind != KORE_LUA_FILE_STORE)
    return kore_lua_fail_table(L, KORE_LUA_STATUS_REFUSED, "unsupported",
                               sizeof "unsupported" - 1);
  kore_lua_file_drop(file);
  n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_RESIZE, file->handle,
                          (unsigned long long)length, 0, NULL, 0, &r, &r2,
                          NULL, NULL, kore_lua_fail_table);
  if (n) return n;
  if (file->position > (unsigned long long)length)
    file->position = (unsigned long long)length;
  file->base = file->position;
  lua_pushboolean(L, 1);
  return 1;
}

static int kore_lua_fs_stat(lua_State *L) {
  size_t len = 0;
  const char *path = luaL_checklstring(L, 1, &len);
  unsigned long long length = 0, kind = 0;
  int n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_STAT, 0, 0, 0, path, len,
                              &length, &kind, NULL, NULL, kore_lua_fail_table);
  if (n) return n;
  lua_createtable(L, 0, 3);
  /* Rule 5: a size carries its unit in the name. */
  lua_pushinteger(L, (lua_Integer)length);
  lua_setfield(L, -2, "size_bytes");
  lua_pushstring(L, kind ? "directory" : "file");
  lua_setfield(L, -2, "kind");
  lua_pushboolean(L, kind != 0);
  lua_setfield(L, -2, "directory");
  return 1;
}

/* `exists` is a question, not a failure, so it answers a boolean and never
   a `nil, err` pair a caller would have to unpack to learn "no". */
static int kore_lua_fs_exists(lua_State *L) {
  size_t len = 0;
  const char *path = luaL_checklstring(L, 1, &len);
  unsigned long long length = 0, kind = 0;
  const char *why = NULL;
  size_t why_len = 0;
  int status = kore_lua_host_call(KORE_LUA_OP_FS_STAT, 0, 0, 0, path, len,
                                  &length, &kind, &why, &why_len);
  lua_pushboolean(L, status == KORE_LUA_STATUS_OK);
  return 1;
}

static int kore_lua_fs_list(lua_State *L) {
  size_t len = 0;
  const char *path = luaL_checklstring(L, 1, &len);
  unsigned long long count = 0, bytes_len = 0;
  const char *bytes = NULL;
  size_t text_len = 0;
  int n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_LIST, 0, 0, 0, path, len, &count,
                              &bytes_len, &bytes, &text_len, kore_lua_fail_table);
  if (n) return n;
  lua_createtable(L, (int)count, 0);
  {
    lua_Integer index = 0;
    size_t start = 0;
    for (size_t i = 0; i < text_len; i++) {
      if (bytes[i] != '\0') continue;
      lua_pushlstring(L, bytes + start, i - start);
      lua_rawseti(L, -2, ++index);
      start = i + 1;
    }
  }
  return 1;
}

/*
 * Read a whole file without a handle — the shape a script reaches for far
 * more often than open/read/close.
 *
 * One host call, not open + read + close: the host opens, streams and
 * closes on its own thread, which is the only arrangement that can hold a
 * single pinned version for the whole file.  Assembled here from
 * `KORE_LUA_OP_FS_READ` calls it could not — each call pins and releases
 * its own version, so a concurrent writer could leave this answer a mix of
 * revisions, and no call would be large enough to stream besides.
 *
 * Bounded by the host's whole-file cap (`MAX_FILE_BYTES` in src/files.rs),
 * which refuses `range` rather than growing without one.
 */
static int kore_lua_fs_read_file(lua_State *L) {
  size_t path_len = 0;
  const char *path = luaL_checklstring(L, 1, &path_len);
  unsigned long long count = 0, len = 0;
  const char *bytes = NULL;
  size_t bytes_len = 0;
  int status = kore_lua_host_call(KORE_LUA_OP_FS_READ_FILE, 0, 0, 0, path,
                                  path_len, &count, &len, &bytes, &bytes_len);
  if (status != KORE_LUA_STATUS_OK)
    return kore_lua_fail_table(L, status, bytes, bytes_len);
  /* The reply is length-delimited and not NUL-terminated; an empty file
     answers an empty string, never nil, because it read successfully. */
  lua_pushlstring(L, bytes != NULL ? bytes : "", bytes != NULL ? bytes_len : 0);
  return 1;
}

static int kore_lua_fs_write_file(lua_State *L) {
  size_t path_len = 0, bytes_len = 0;
  const char *path = luaL_checklstring(L, 1, &path_len);
  const char *bytes = luaL_checklstring(L, 2, &bytes_len);
  unsigned long long handle = 0, length = 0, written = 0, end = 0;
  int n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_OPEN,
                              KORE_LUA_O_WRITE | KORE_LUA_O_CREATE |
                                  KORE_LUA_O_TRUNCATE,
                              0, 0, path, path_len, &handle, &length, NULL,
                              NULL, kore_lua_fail_table);
  if (n) return n;
  n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_WRITE, handle, 0, 0, bytes,
                          bytes_len, &written, &end, NULL, NULL,
                          kore_lua_fail_table);
  {
    unsigned long long r = 0, r2 = 0;
    const char *why = NULL;
    size_t why_len = 0;
    (void)kore_lua_host_call(KORE_LUA_OP_FS_CLOSE, handle, 0, 0, NULL, 0, &r,
                             &r2, &why, &why_len);
  }
  if (n) return n;
  lua_pushinteger(L, (lua_Integer)written);
  return 1;
}

static const luaL_Reg kore_lua_fs_lib[] = {
  {"open", kore_lua_fs_open},
  {"read_file", kore_lua_fs_read_file},
  {"write_file", kore_lua_fs_write_file},
  {"remove", kore_lua_fs_remove},
  {"rename", kore_lua_fs_rename},
  {"mkdir", kore_lua_fs_mkdir},
  {"resize", kore_lua_fs_resize},
  {"stat", kore_lua_fs_stat},
  {"exists", kore_lua_fs_exists},
  {"list", kore_lua_fs_list},
  {NULL, NULL}
};

/*
 * ----------------------------------------------------------------- io
 *
 * Lua's own surface, exactly as the manual specifies it, over the objects
 * above.  `io.open` and `kore.fs.open` answer the same userdata; only the
 * failure spelling differs (`nil, message, errno` here, a table there).
 *
 * The default input and output are kept in the registry rather than in the
 * `io` table, so a script that overwrites `io.stdout` changes what its own
 * name refers to and not where `io.write` goes — which is what the manual
 * means by `io.output()` being the default, and `io.stdout` being a file.
 */
static const char kore_lua_input_key = 0;
static const char kore_lua_output_key = 0;

static kore_lua_file *kore_lua_default(lua_State *L, const void *key) {
  kore_lua_file *file;
  lua_rawgetp(L, LUA_REGISTRYINDEX, key);
  file = (kore_lua_file *)luaL_checkudata(L, -1, KORE_LUA_FILE);
  return file;   /* left on the stack: the caller pops or reuses it */
}

/* `io.input()` / `io.output()`: answer the default, or set it from a file
   or a name and answer that. */
static int kore_lua_io_default(lua_State *L, const void *key, const char *mode) {
  if (!lua_isnoneornil(L, 1)) {
    if (lua_type(L, 1) == LUA_TSTRING) {
      lua_settop(L, 1);
      kore_lua_opencheck(L, 1, mode);
    } else {
      (void)kore_lua_file_check(L, 1);
      lua_pushvalue(L, 1);
    }
    lua_pushvalue(L, -1);
    lua_rawsetp(L, LUA_REGISTRYINDEX, key);
    return 1;
  }
  (void)kore_lua_default(L, key);
  return 1;
}

static int kore_lua_io_input(lua_State *L) {
  return kore_lua_io_default(L, &kore_lua_input_key, "r");
}

static int kore_lua_io_output(lua_State *L) {
  return kore_lua_io_default(L, &kore_lua_output_key, "w");
}

static int kore_lua_io_open(lua_State *L) {
  return kore_lua_open_file(L, kore_lua_fail_io);
}

static int kore_lua_io_read(lua_State *L) {
  int last = lua_gettop(L);
  kore_lua_file *file = kore_lua_default(L, &kore_lua_input_key);
  lua_pop(L, 1);   /* the registry still anchors it */
  if (file->closed) return luaL_error(L, "attempt to use a closed file");
  return kore_lua_file_read_with(L, file, 1, last, kore_lua_fail_io);
}

static int kore_lua_io_write(lua_State *L) {
  /* The arguments end where they ended before the default was pushed; the
     file itself is not one of them. */
  int last = lua_gettop(L);
  kore_lua_file *file = kore_lua_default(L, &kore_lua_output_key);
  int self_index = lua_gettop(L);
  if (file->closed) return luaL_error(L, "attempt to use a closed file");
  return kore_lua_file_write_with(L, file, 1, last, self_index,
                                  kore_lua_fail_io);
}

static int kore_lua_io_close(lua_State *L) {
  if (lua_isnoneornil(L, 1)) {
    kore_lua_file *file = kore_lua_default(L, &kore_lua_output_key);
    return kore_lua_file_do_close(L, file, kore_lua_fail_io);
  }
  return kore_lua_file_do_close(L, kore_lua_file_check(L, 1), kore_lua_fail_io);
}

static int kore_lua_io_flush(lua_State *L) {
  /* Drop any arguments first, so the default output lands at index 1 where
     `f:flush` looks for it.  Replacing slot 1 with a value that IS slot 1
     would pop it and leave nothing. */
  lua_settop(L, 0);
  kore_lua_default(L, &kore_lua_output_key);
  return kore_lua_file_flush(L);
}

/* `io.type`: "file", "closed file", or nil for anything that is not one.
   It must not raise on a foreign value — that is what it is for. */
static int kore_lua_io_type(lua_State *L) {
  luaL_checkany(L, 1);
  if (lua_getmetatable(L, 1) == 0) { lua_pushnil(L); return 1; }
  luaL_getmetatable(L, KORE_LUA_FILE);
  if (!lua_rawequal(L, -1, -2)) { lua_pushnil(L); return 1; }
  lua_pop(L, 2);
  lua_pushstring(L, ((kore_lua_file *)lua_touserdata(L, 1))->closed
                        ? "closed file" : "file");
  return 1;
}

/*
 * `io.lines()` iterates the default input and leaves it open. `io.lines(name)`
 * opens its own, and answers **four** values — the iterator, a nil state, a
 * nil control, and the file itself as the generic `for`'s to-be-closed
 * variable. That fourth value is what closes the file when the loop leaves
 * early, by `break` or by an error, and not only at end of file; the engine
 * is 5.5, so `<close>` is there to honour it.
 */
static int kore_lua_io_lines(lua_State *L) {
  int toclose;
  /* Guarantee a slot 1 to replace, exactly as `io_lines` does: `io.lines()`
     with no argument at all would otherwise replace the stack top with
     itself and leave nothing behind. */
  if (lua_isnone(L, 1)) lua_pushnil(L);
  if (lua_isnil(L, 1)) {
    kore_lua_default(L, &kore_lua_input_key);
    lua_replace(L, 1);
    toclose = 0;
  } else {
    kore_lua_opencheck(L, 1, "r");
    lua_replace(L, 1);
    toclose = 1;
  }
  kore_lua_lines_push(L, 1, 2, toclose);
  if (toclose) {
    lua_pushnil(L);       /* state */
    lua_pushnil(L);       /* control */
    lua_pushvalue(L, 1);  /* the to-be-closed file */
    return 4;
  }
  return 1;
}

/*
 * Draw an unused temporary name into `dst` (at least 32 bytes) and create
 * the file exclusively, so the name answered is one nothing else can
 * already hold.  Answers the open handle, or 0 with `why` / `why_len` set.
 *
 * The length is an out-parameter and not a `strlen` at the call site: a
 * reason coming back from `kore_lua_host_call` is length-delimited with no
 * terminator, and only the entropy draw's reason is a C string.  Both are
 * normalized to a length here so neither caller has to know which it got.
 *
 * `os.tmpname` and `io.tmpfile` are the same act — a name nobody else has —
 * differing only in what they hand back, so they share this.
 */
static unsigned long long kore_lua_tempfile(char *dst, size_t *len_out,
                                            unsigned long long flags,
                                            const char **why,
                                            size_t *why_len_out) {
  static const char digits[] = "0123456789abcdef";
  *why = NULL;
  *why_len_out = 0;
  for (int attempt = 0; attempt < 16; attempt++) {
    unsigned char draw[8];
    unsigned long long handle = 0, length = 0;
    size_t why_len = 0;
    size_t len = 0;
    int status;
    if (kore_lua_host_random(draw, sizeof draw, why) != KORE_LUA_STATUS_OK) {
      /* A leaf call's reason is a `&'static CStr`. */
      *why_len_out = *why != NULL ? strlen(*why) : 0;
      return 0;
    }
    memcpy(dst, "/tmp/lua_", 9);
    len = 9;
    for (size_t i = 0; i < sizeof draw; i++) {
      dst[len++] = digits[draw[i] >> 4];
      dst[len++] = digits[draw[i] & 15];
    }
    dst[len] = '\0';
    status = kore_lua_host_call(KORE_LUA_OP_FS_OPEN,
                                flags | KORE_LUA_O_CREATE |
                                    KORE_LUA_O_EXCLUSIVE,
                                0, 0, dst, len, &handle, &length, why,
                                &why_len);
    if (status == KORE_LUA_STATUS_OK) {
      if (len_out != NULL) *len_out = len;
      return handle + 1;   /* biased so 0 means failure */
    }
    /* Only a collision is worth another draw; anything else is the answer. */
    if (!(*why != NULL && why_len == sizeof "already-exists" - 1 &&
          memcmp(*why, "already-exists", why_len) == 0)) {
      *why_len_out = why_len;
      return 0;
    }
  }
  *why = "no unused temporary name was found";
  *why_len_out = sizeof "no unused temporary name was found" - 1;
  return 0;
}

/*
 * `io.tmpfile`: a file opened in update mode that removes itself when its
 * handle is retired.  The manual says "automatically removed when the
 * program ends"; this removes it at close or collection instead, which is
 * strictly sooner and needs no exit hook — and the VM's teardown retires
 * every handle, so the end of the program is covered by the same path.
 */
static int kore_lua_io_tmpfile(lua_State *L) {
  char path[48];
  size_t len = 0, why_len = 0;
  const char *why = NULL;
  unsigned long long biased = kore_lua_tempfile(
      path, &len, KORE_LUA_O_READ | KORE_LUA_O_WRITE, &why, &why_len);
  if (biased == 0)
    return kore_lua_fail_io(L, KORE_LUA_STATUS_UNAVAILABLE,
                            why != NULL ? why : "unavailable",
                            why != NULL ? why_len : sizeof "unavailable" - 1);
  {
    kore_lua_file *file =
        kore_lua_file_push(L, biased - 1, KORE_LUA_FILE_STORE, 1, 1, 0);
    file->temporary = 1;
    memcpy(file->path, path, len + 1);
  }
  return 1;
}

static const luaL_Reg kore_lua_io_lib[] = {
  {"tmpfile", kore_lua_io_tmpfile},
  {"open", kore_lua_io_open},
  {"close", kore_lua_io_close},
  {"read", kore_lua_io_read},
  {"write", kore_lua_io_write},
  {"lines", kore_lua_io_lines},
  {"input", kore_lua_io_input},
  {"output", kore_lua_io_output},
  {"flush", kore_lua_io_flush},
  {"type", kore_lua_io_type},
  {NULL, NULL}
};

/*
 * ----------------------------------------------------------------- os
 *
 * The calendar is ours.  There is no libc `gmtime` here and no timezone
 * database, so `os.date` computes civil time from the wall clock directly
 * — Howard Hinnant's `civil_from_days`, which is exact for every day the
 * proleptic Gregorian calendar defines and needs no table.
 *
 * kore keeps one clock and it reads UTC.  `os.date("%c")` and
 * `os.date("!%c")` therefore answer the same thing, `os.date` reports
 * `isdst = false`, and `os.time` treats a table as UTC.  That is a real
 * limitation, stated rather than papered over: a local time that silently
 * meant UTC would be worse than one that says so.
 */
typedef struct {
  int year;    /* the full year, e.g. 2026 */
  int month;   /* 1..12 */
  int day;     /* 1..31 */
  int hour;
  int minute;
  int second;
  int weekday; /* 0 = Sunday */
  int yearday; /* 1..366 */
} kore_lua_civil;

/* Days since 1970-01-01 → year/month/day.  Shifts the era to start in
   March so the leap day lands at the end of a cycle and no month table is
   needed. */
static void kore_lua_civil_from_days(long long z, kore_lua_civil *out) {
  long long era, doe, yoe, y, doy, mp, d, m;
  z += 719468;
  era = (z >= 0 ? z : z - 146096) / 146097;
  doe = z - era * 146097;                                  /* [0, 146096] */
  yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y = yoe + era * 400;
  doy = doe - (365 * yoe + yoe / 4 - yoe / 100);           /* [0, 365] */
  mp = (5 * doy + 2) / 153;                                /* [0, 11] */
  d = doy - (153 * mp + 2) / 5 + 1;                        /* [1, 31] */
  m = mp < 10 ? mp + 3 : mp - 9;                           /* [1, 12] */
  out->year = (int)(y + (m <= 2));
  out->month = (int)m;
  out->day = (int)d;
}

/* The exact inverse, so `os.time(os.date("*t", t)) == t`. */
static long long kore_lua_days_from_civil(int year, int month, int day) {
  long long y = year, era, yoe, doy, doe;
  y -= month <= 2;
  era = (y >= 0 ? y : y - 399) / 400;
  yoe = y - era * 400;                                     /* [0, 399] */
  doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}

/* Floor division: a pre-1970 timestamp must round toward the earlier day,
   not toward zero, or the hour comes out negative. */
static long long kore_lua_floor_div(long long a, long long b) {
  long long q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
  return q;
}

static void kore_lua_break_down(long long seconds, kore_lua_civil *out) {
  long long days = kore_lua_floor_div(seconds, 86400);
  long long rest = seconds - days * 86400;
  kore_lua_civil_from_days(days, out);
  out->hour = (int)(rest / 3600);
  out->minute = (int)((rest % 3600) / 60);
  out->second = (int)(rest % 60);
  /* 1970-01-01 was a Thursday (4). */
  out->weekday = (int)(((days % 7) + 11) % 7);
  out->yearday =
      (int)(days - kore_lua_days_from_civil(out->year, 1, 1) + 1);
}

/*
 * The instants this calendar answers for.
 *
 * `civil_from_days` is exact for every day the proleptic Gregorian
 * calendar defines, but `kore_lua_civil` holds an `int` year and `%Y`
 * spells four digits, so the range is stated rather than left to overflow
 * into a truncated year.  Computed from the calendar itself rather than
 * tabulated, so the bound and the conversion cannot disagree: years
 * 0000-01-01 through 9999-12-31, which is also the range `os.time` accepts
 * a table for.
 */
#define KORE_LUA_YEAR_MIN 0
#define KORE_LUA_YEAR_MAX 9999

static int kore_lua_time_representable(long long seconds) {
  long long low = kore_lua_days_from_civil(KORE_LUA_YEAR_MIN, 1, 1) * 86400;
  long long high =
      kore_lua_days_from_civil(KORE_LUA_YEAR_MAX, 12, 31) * 86400 + 86399;
  return seconds >= low && seconds <= high;
}

/*
 * Every `os.time` field is bounded before any arithmetic, so a
 * normalization cannot overflow on its way to the check above.  Generous
 * by design — far more than date arithmetic by field addition needs — and
 * it is `kore_lua_time_representable` that decides the answer.
 */
#define KORE_LUA_DATE_FIELD_MAX 1000000000ll

/* The wall clock, as whole seconds since the epoch; `ok` says whether the
   machine could answer at all. */
static long long kore_lua_wall_seconds(int *ok) {
  unsigned long long ns = 0;
  const char *why = NULL;
  int status = kore_lua_host_wall_ns(&ns, &why);
  *ok = status == KORE_LUA_STATUS_OK;
  return (long long)(ns / 1000000000ull);
}

static void kore_lua_pad2(char *dst, int value) {
  dst[0] = (char)('0' + (value / 10) % 10);
  dst[1] = (char)('0' + value % 10);
}

/* `%Y` and friends, written out: there is no `strftime` below us, and a
   partial one that silently ignored an unknown specifier would be the
   `string.format` mistake again.  An unsupported specifier is an argument
   error, so a script learns at the call rather than from wrong output. */
static int kore_lua_strftime(lua_State *L, const char *format,
                             const kore_lua_civil *t) {
  static const char *const wdays[] = {"Sunday", "Monday", "Tuesday",
                                      "Wednesday", "Thursday", "Friday",
                                      "Saturday"};
  static const char *const months[] = {"January", "February", "March", "April",
                                       "May", "June", "July", "August",
                                       "September", "October", "November",
                                       "December"};
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  for (const char *p = format; *p != '\0'; p++) {
    char pad[4];
    if (*p != '%') { luaL_addchar(&b, *p); continue; }
    p++;
    switch (*p) {
      case '\0': return luaL_error(L, "invalid conversion specifier '%%'");
      case '%': luaL_addchar(&b, '%'); break;
      case 'a': luaL_addlstring(&b, wdays[t->weekday], 3); break;
      case 'A': luaL_addstring(&b, wdays[t->weekday]); break;
      case 'b': case 'h': luaL_addlstring(&b, months[t->month - 1], 3); break;
      case 'B': luaL_addstring(&b, months[t->month - 1]); break;
      case 'd': kore_lua_pad2(pad, t->day); luaL_addlstring(&b, pad, 2); break;
      case 'e':
        pad[0] = t->day < 10 ? ' ' : (char)('0' + t->day / 10);
        pad[1] = (char)('0' + t->day % 10);
        luaL_addlstring(&b, pad, 2);
        break;
      case 'H': kore_lua_pad2(pad, t->hour); luaL_addlstring(&b, pad, 2); break;
      case 'I': {
        int hour = t->hour % 12;
        kore_lua_pad2(pad, hour == 0 ? 12 : hour);
        luaL_addlstring(&b, pad, 2);
        break;
      }
      case 'j': {
        char day[3];
        day[0] = (char)('0' + (t->yearday / 100) % 10);
        kore_lua_pad2(day + 1, t->yearday % 100);
        luaL_addlstring(&b, day, 3);
        break;
      }
      case 'm': kore_lua_pad2(pad, t->month); luaL_addlstring(&b, pad, 2); break;
      case 'M': kore_lua_pad2(pad, t->minute); luaL_addlstring(&b, pad, 2); break;
      case 'n': luaL_addchar(&b, '\n'); break;
      case 'p': luaL_addlstring(&b, t->hour < 12 ? "AM" : "PM", 2); break;
      case 'S': kore_lua_pad2(pad, t->second); luaL_addlstring(&b, pad, 2); break;
      case 't': luaL_addchar(&b, '\t'); break;
      case 'w': luaL_addchar(&b, (char)('0' + t->weekday)); break;
      case 'y': kore_lua_pad2(pad, ((t->year % 100) + 100) % 100);
                luaL_addlstring(&b, pad, 2); break;
      case 'Y': {
        char year[16];
        int len = 0;
        int value = t->year;
        int negative = value < 0;
        unsigned magnitude = (unsigned)(negative ? -value : value);
        char digits[12];
        int count = 0;
        do { digits[count++] = (char)('0' + magnitude % 10); magnitude /= 10; }
        while (magnitude != 0);
        if (negative) year[len++] = '-';
        while (count > 0) year[len++] = digits[--count];
        luaL_addlstring(&b, year, (size_t)len);
        break;
      }
      /* The one timezone kore has. */
      case 'Z': luaL_addlstring(&b, "UTC", 3); break;
      case 'z': luaL_addlstring(&b, "+0000", 5); break;
      case 'D': case 'x': {
        char date[8];
        kore_lua_pad2(date, t->month); date[2] = '/';
        kore_lua_pad2(date + 3, t->day); date[5] = '/';
        kore_lua_pad2(date + 6, ((t->year % 100) + 100) % 100);
        luaL_addlstring(&b, date, 8);
        break;
      }
      case 'F': {
        luaL_addstring(&b, "");
        {
          char date[10];
          int y = t->year;
          date[0] = (char)('0' + (y / 1000) % 10);
          date[1] = (char)('0' + (y / 100) % 10);
          kore_lua_pad2(date + 2, y % 100);
          date[4] = '-';
          kore_lua_pad2(date + 5, t->month);
          date[7] = '-';
          kore_lua_pad2(date + 8, t->day);
          luaL_addlstring(&b, date, 10);
        }
        break;
      }
      case 'R': case 'T': case 'X': {
        char clock[8];
        kore_lua_pad2(clock, t->hour); clock[2] = ':';
        kore_lua_pad2(clock + 3, t->minute);
        if (*p == 'R') { luaL_addlstring(&b, clock, 5); break; }
        clock[5] = ':';
        kore_lua_pad2(clock + 6, t->second);
        luaL_addlstring(&b, clock, 8);
        break;
      }
      case 'c': {
        char line[24];
        int len = 0;
        memcpy(line + len, wdays[t->weekday], 3); len += 3;
        line[len++] = ' ';
        memcpy(line + len, months[t->month - 1], 3); len += 3;
        line[len++] = ' ';
        line[len] = t->day < 10 ? ' ' : (char)('0' + t->day / 10);
        line[len + 1] = (char)('0' + t->day % 10);
        len += 2;
        line[len++] = ' ';
        kore_lua_pad2(line + len, t->hour); len += 2;
        line[len++] = ':';
        kore_lua_pad2(line + len, t->minute); len += 2;
        line[len++] = ':';
        kore_lua_pad2(line + len, t->second); len += 2;
        line[len++] = ' ';
        {
          int y = t->year;
          line[len++] = (char)('0' + (y / 1000) % 10);
          line[len++] = (char)('0' + (y / 100) % 10);
          kore_lua_pad2(line + len, y % 100); len += 2;
        }
        luaL_addlstring(&b, line, (size_t)len);
        break;
      }
      default:
        return luaL_error(L, "invalid conversion specifier '%%%c'", *p);
    }
  }
  luaL_pushresult(&b);
  return 1;
}

/*
 * A missing required field names itself, as `getfield` does — `checkinteger`
 * alone would blame "field ?" and leave the caller hunting.
 */
static lua_Integer kore_lua_datefield(lua_State *L, const char *name,
                                      lua_Integer fallback, int required) {
  lua_Integer value;
  if (lua_getfield(L, 1, name) == LUA_TNIL) {
    lua_pop(L, 1);
    if (required)
      return luaL_error(L, "field '%s' missing in date table", name);
    return fallback;
  }
  value = luaL_checkinteger(L, -1);
  lua_pop(L, 1);
  return value;
}

static void kore_lua_setfield(lua_State *L, const char *name, lua_Integer v) {
  lua_pushinteger(L, v);
  lua_setfield(L, 1, name);
}

static int kore_lua_os_time(lua_State *L) {
  long long when;
  if (lua_isnoneornil(L, 1)) {
    int ok = 0;
    when = kore_lua_wall_seconds(&ok);
    /* A clock that cannot be read answers `nil, why` — never 0, which a
       caller would read as 1970 rather than as "no reading". */
    if (!ok) return kore_lua_fail_cstr(L, KORE_LUA_STATUS_UNAVAILABLE,
                                       "the wall clock is not set");
    if (!kore_lua_time_representable(when))
      return luaL_error(
          L, "time result cannot be represented in this installation");
    lua_pushinteger(L, (lua_Integer)when);
    return 1;
  }
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_settop(L, 1);   /* the table is the only thing `setfield` may write to */
  {
    /* Out-of-range fields normalize, exactly as `mktime` does: `month = 13`
       is January of the next year, which is what makes date arithmetic by
       field addition work. */
    lua_Integer year = kore_lua_datefield(L, "year", 0, 1);
    lua_Integer month = kore_lua_datefield(L, "month", 0, 1);
    lua_Integer day = kore_lua_datefield(L, "day", 0, 1);
    lua_Integer hour = kore_lua_datefield(L, "hour", 12, 0);
    lua_Integer minute = kore_lua_datefield(L, "min", 0, 0);
    lua_Integer second = kore_lua_datefield(L, "sec", 0, 0);
    long long carry;
    long long days;
    kore_lua_civil t;
    /*
     * Bound the fields before any arithmetic, then bound the result.
     *
     * Two checks, not three: the field bound is what makes `(int)year`
     * exact — a year normalized from fields this size stays under 1.1e9
     * and an `int` holds it — and what keeps `days * 86400 + ...` from
     * overflowing, since the widest reachable value is about 3.4e16.
     * With overflow and truncation both excluded, `time_representable` is
     * the one thing that decides whether there is an answer.
     */
    if (day < -KORE_LUA_DATE_FIELD_MAX || day > KORE_LUA_DATE_FIELD_MAX ||
        hour < -KORE_LUA_DATE_FIELD_MAX || hour > KORE_LUA_DATE_FIELD_MAX ||
        minute < -KORE_LUA_DATE_FIELD_MAX ||
        minute > KORE_LUA_DATE_FIELD_MAX ||
        second < -KORE_LUA_DATE_FIELD_MAX ||
        second > KORE_LUA_DATE_FIELD_MAX ||
        month < -KORE_LUA_DATE_FIELD_MAX || month > KORE_LUA_DATE_FIELD_MAX ||
        year < -KORE_LUA_DATE_FIELD_MAX || year > KORE_LUA_DATE_FIELD_MAX)
      return luaL_error(
          L, "time result cannot be represented in this installation");
    carry = kore_lua_floor_div((long long)month - 1, 12);
    year += carry;
    month = (long long)month - 1 - carry * 12 + 1;
    days = kore_lua_days_from_civil((int)year, (int)month, 1) + (day - 1);
    when = days * 86400 + hour * 3600 + minute * 60 + second;
    if (!kore_lua_time_representable(when))
      return luaL_error(
          L, "time result cannot be represented in this installation");
    /* `mktime` normalizes the caller's table in place and fills in the
       fields it derived; `setallfields` is what a script reads back to
       learn the weekday it never supplied. */
    kore_lua_break_down(when, &t);
    kore_lua_setfield(L, "year", t.year);
    kore_lua_setfield(L, "month", t.month);
    kore_lua_setfield(L, "day", t.day);
    kore_lua_setfield(L, "hour", t.hour);
    kore_lua_setfield(L, "min", t.minute);
    kore_lua_setfield(L, "sec", t.second);
    kore_lua_setfield(L, "wday", t.weekday + 1);
    kore_lua_setfield(L, "yday", t.yearday);
    lua_pushboolean(L, 0);
    lua_setfield(L, 1, "isdst");
    lua_pushinteger(L, (lua_Integer)when);
    return 1;
  }
}

static int kore_lua_os_date(lua_State *L) {
  const char *format = luaL_optstring(L, 1, "%c");
  long long when;
  kore_lua_civil t;
  if (lua_isnoneornil(L, 2)) {
    int ok = 0;
    when = kore_lua_wall_seconds(&ok);
    if (!ok) return kore_lua_fail_cstr(L, KORE_LUA_STATUS_UNAVAILABLE,
                                       "the wall clock is not set");
  } else {
    when = (long long)luaL_checkinteger(L, 2);
  }
  /* kore keeps one clock and it reads UTC, so the `!` prefix selects the
     same calendar it would without it.  Accepted rather than rejected: a
     portable script writes `!` and must not fail here for doing so. */
  if (*format == '!') format++;
  if (!kore_lua_time_representable(when))
    return luaL_error(L,
                      "date result cannot be represented in this installation");
  kore_lua_break_down(when, &t);
  if (strcmp(format, "*t") == 0 || strcmp(format, "!*t") == 0) {
    lua_createtable(L, 0, 9);
    lua_pushinteger(L, t.year);    lua_setfield(L, -2, "year");
    lua_pushinteger(L, t.month);   lua_setfield(L, -2, "month");
    lua_pushinteger(L, t.day);     lua_setfield(L, -2, "day");
    lua_pushinteger(L, t.hour);    lua_setfield(L, -2, "hour");
    lua_pushinteger(L, t.minute);  lua_setfield(L, -2, "min");
    lua_pushinteger(L, t.second);  lua_setfield(L, -2, "sec");
    lua_pushinteger(L, t.weekday + 1); lua_setfield(L, -2, "wday");
    lua_pushinteger(L, t.yearday); lua_setfield(L, -2, "yday");
    /* kore has no timezone database, so there is no daylight rule to be
       inside of.  `false`, not `nil`: the field is answered, not missing. */
    lua_pushboolean(L, 0);         lua_setfield(L, -2, "isdst");
    return 1;
  }
  return kore_lua_strftime(L, format, &t);
}

static int kore_lua_os_difftime(lua_State *L) {
  lua_Number later = luaL_checknumber(L, 1);
  lua_Number earlier = luaL_optnumber(L, 2, 0);
  lua_pushnumber(L, later - earlier);
  return 1;
}

/*
 * `os.clock`: seconds of processor time.  kore gives a task a monotonic
 * counter, not a per-process CPU accounting, so this is time since the VM
 * started — the approximation the manual allows, and an exact one for the
 * single-threaded script that is the only thing measuring it.
 */
static int kore_lua_os_clock(lua_State *L) {
  unsigned long long ns = 0;
  const char *why = NULL;
  int status = kore_lua_host_uptime_ns(&ns, &why);
  if (status != KORE_LUA_STATUS_OK) return kore_lua_fail_cstr(L, status, why);
  {
    unsigned long long started =
        (unsigned long long)lua_tointeger(L, lua_upvalueindex(1));
    lua_pushnumber(L, (lua_Number)(ns - started) / 1e9);
  }
  return 1;
}

static int kore_lua_os_remove(lua_State *L) {
  size_t len = 0;
  const char *path = luaL_checklstring(L, 1, &len);
  unsigned long long r = 0, r2 = 0;
  int n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_REMOVE, 0, 0, 0, path, len, &r,
                              &r2, NULL, NULL, kore_lua_fail_io);
  if (n) return n;
  lua_pushboolean(L, 1);
  return 1;
}

static int kore_lua_os_rename(lua_State *L) {
  /* The same joined window `kore.fs.rename` builds, in `io`'s spelling. */
  size_t from_len = 0, to_len = 0;
  const char *from = luaL_checklstring(L, 1, &from_len);
  const char *to = luaL_checklstring(L, 2, &to_len);
  unsigned long long r = 0, r2 = 0;
  luaL_Buffer b;
  char *joined;
  int n;
  luaL_buffinit(L, &b);
  joined = luaL_prepbuffsize(&b, from_len + to_len + 1);
  memcpy(joined, from, from_len);
  joined[from_len] = '\0';
  memcpy(joined + from_len + 1, to, to_len);
  n = kore_lua_host_bytes(L, KORE_LUA_OP_FS_RENAME, 0, 0, 0, joined,
                          from_len + to_len + 1, &r, &r2, NULL, NULL,
                          kore_lua_fail_io);
  if (n) return n;
  lua_pushboolean(L, 1);
  return 1;
}

/*
 * `os.tmpname`.  Upstream's answers a name and leaves the caller to race
 * for it; this one *creates* the file exclusively and then answers its
 * name, so the name it returns is one nothing else can already hold.  That
 * is a deliberate strengthening, stated here because a caller that expects
 * the file to be absent would otherwise be surprised: the manual requires
 * the caller to remove the file, which this satisfies either way.
 */
static int kore_lua_os_tmpname(lua_State *L) {
  char path[48];
  size_t len = 0, why_len = 0;
  const char *why = NULL;
  unsigned long long biased =
      kore_lua_tempfile(path, &len, KORE_LUA_O_WRITE, &why, &why_len);
  if (biased == 0)
    return kore_lua_fail(L, KORE_LUA_STATUS_UNAVAILABLE,
                         why != NULL ? why : "unavailable",
                         why != NULL ? why_len : sizeof "unavailable" - 1);
  {
    /* The name is what was asked for, not the handle: close it and leave
       the file in place, which is the POSIX behaviour the manual describes
       ("this function also creates a file with that name, to avoid
       security risks"). The caller opens and removes it. */
    unsigned long long r = 0, r2 = 0;
    const char *ignored = NULL;
    size_t ignored_len = 0;
    (void)kore_lua_host_call(KORE_LUA_OP_FS_CLOSE, biased - 1, 0, 0, NULL, 0,
                             &r, &r2, &ignored, &ignored_len);
  }
  lua_pushlstring(L, path, len);
  return 1;
}

/*
 * `os.getenv` answers nil for every name, because a kore task has no
 * environment: authority arrives as capabilities in its launch, not as
 * inherited strings.  Present and conforming rather than absent — an unset
 * variable is exactly what `nil` means, and a script that probes for one
 * gets the right answer instead of a missing-function error.
 */
static int kore_lua_os_getenv(lua_State *L) {
  (void)luaL_checkstring(L, 1);
  lua_pushnil(L);
  return 1;
}

/* One locale, named as C names it. */
static int kore_lua_os_setlocale(lua_State *L) {
  const char *locale = luaL_optstring(L, 1, NULL);
  static const char *const categories[] = {"all", "collate", "ctype",
                                           "monetary", "numeric", "time", NULL};
  (void)luaL_checkoption(L, 2, "all", categories);
  if (locale == NULL || strcmp(locale, "C") == 0 || *locale == '\0')
    lua_pushliteral(L, "C");
  else
    lua_pushnil(L);   /* no other locale can be set */
  return 1;
}

/*
 * `os.exit`.  The task cannot simply leave: the evaluation has C frames
 * below it, `<close>` variables above it, and a REPL that still owes its
 * caller the output collected so far.  So this records the code with the
 * host and then raises a sentinel, which unwinds through `lua_pcall`
 * running every `__close` on the way — and the host reads the record when
 * the evaluation returns.  A verb that longjmp'd past the handlers would
 * be a member that does not do what its name says.
 */
static int kore_lua_os_exit(lua_State *L) {
  unsigned long long code = 0, r2 = 0;
  const char *why = NULL;
  size_t why_len = 0;
  int requested = 0;
  if (lua_isboolean(L, 1))
    requested = lua_toboolean(L, 1) ? 0 : 1;
  else
    requested = (int)luaL_optinteger(L, 1, 0);
  /* `close` is the second argument upstream; here the VM is always closed
     by the host after the evaluation, so it is accepted and ignored. */
  (void)lua_toboolean(L, 2);
  (void)kore_lua_host_call(KORE_LUA_OP_PROC_EXIT, (unsigned long long)requested,
                           0, 0, NULL, 0, &code, &r2, &why, &why_len);
  return luaL_error(L, KORE_LUA_EXIT_SENTINEL);
}

static const luaL_Reg kore_lua_os_lib[] = {
  {"time", kore_lua_os_time},
  {"date", kore_lua_os_date},
  {"difftime", kore_lua_os_difftime},
  {"remove", kore_lua_os_remove},
  {"rename", kore_lua_os_rename},
  {"tmpname", kore_lua_os_tmpname},
  {"getenv", kore_lua_os_getenv},
  {"setlocale", kore_lua_os_setlocale},
  {"exit", kore_lua_os_exit},
  {NULL, NULL}
};

/*
 * ----------------------------------------------------------------- kore
 *
 * One table, provisioned from what this process actually holds.  It is not
 * a lookup service for authority: every submodule below is present because
 * the image's launch profile conveyed the face it needs, and a submodule
 * whose authority is absent is absent too, so `if kore.fs then` is the
 * idiom rather than a call that always fails.
 */
static int kore_lua_has(lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  lua_getfield(L, lua_upvalueindex(1), name);
  lua_pushboolean(L, !lua_isnil(L, -1));
  return 1;
}

/*
 * `caps()` answers the names in registration order, taken from the list
 * built at open time rather than from a table walk: a walk follows hash
 * order, which now varies per boot because the VM's seed does, and a caps
 * line that reorders itself between boots is not something a gate can pin.
 * Building it at registration also means an embedder's libraries list
 * themselves — there is no second place to remember to update.
 */
static int kore_lua_caps(lua_State *L) {
  lua_newtable(L);
  lua_Integer count = (lua_Integer)lua_rawlen(L, lua_upvalueindex(1));
  for (lua_Integer i = 1; i <= count; i++) {
    lua_rawgeti(L, lua_upvalueindex(1), i);
    lua_rawseti(L, -2, i);
  }
  return 1;
}

/*
 * The embedder's libraries, handed down from Rust (`Vm::new`): a name and
 * an opener per entry, registered beneath `kore` after this crate's own.
 * Ops they post start at 64 (`OP_EMBEDDER_BASE`), which is what keeps one
 * transport serving both halves.
 *
 * They reach `luaopen_kore` as closure upvalues rather than through a file
 * static: the VM is per-thread, and a static would be a second, untyped
 * channel for something the constructor is already passing.
 */
typedef struct {
  const char *name;
  lua_CFunction opener;
} kore_lua_library;

/* Register one module and record its name for `caps()`; the caps array is
   at the top of the stack, the kore table just below it. */
static void kore_lua_module(lua_State *L, const char *name, lua_Integer *index) {
  lua_setfield(L, -3, name);
  lua_pushstring(L, name);
  lua_rawseti(L, -2, ++(*index));
}

static int luaopen_kore(lua_State *L) {
  const kore_lua_library *libraries =
      (const kore_lua_library *)lua_touserdata(L, lua_upvalueindex(1));
  size_t count = (size_t)lua_tointeger(L, lua_upvalueindex(2));
  lua_newtable(L);           /* the kore table */
  lua_newtable(L);           /* the caps array, in registration order */
  lua_Integer listed = 0;

  luaL_newlib(L, kore_lua_time_lib);
  kore_lua_module(L, "time", &listed);

  luaL_newlib(L, kore_lua_random_lib);
  kore_lua_module(L, "random", &listed);

  luaL_newlib(L, kore_lua_fs_lib);
  kore_lua_module(L, "fs", &listed);

  luaL_newlib(L, kore_lua_proc_lib);
  lua_newtable(L);
  lua_pushcfunction(L, kore_lua_proc_index);
  lua_setfield(L, -2, "__index");
  lua_setmetatable(L, -2);
  kore_lua_module(L, "proc", &listed);

  for (size_t i = 0; libraries != NULL && i < count; i++) {
    const kore_lua_library *library = &libraries[i];
    if (library->name == NULL || library->opener == NULL) continue;
    lua_pushcfunction(L, library->opener);
    lua_call(L, 0, 1);
    kore_lua_module(L, library->name, &listed);
  }

  /* `caps` closes over the array; `has` over the table it describes. */
  lua_pushcclosure(L, kore_lua_caps, 1);
  lua_setfield(L, -2, "caps");
  lua_pushvalue(L, -1);
  lua_pushcclosure(L, kore_lua_has, 1);
  lua_setfield(L, -2, "has");

  return 1;
}

static int kore_lua_require(lua_State *L) {
  size_t len = 0;
  const char *name = luaL_checklstring(L, 1, &len);
  const char *bytes = NULL;
  size_t bytes_len = 0;
  lua_CFunction opener = NULL;
  int kind;
  lua_settop(L, 1);
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);  /* 2: LOADED */
  lua_getfield(L, 2, name);
  if (lua_toboolean(L, -1)) return 1;  /* already loaded */
  lua_pop(L, 1);
  kind = kore_lua_host_require(name, len, &bytes, &bytes_len, &opener);
  if (kind == KORE_LUA_MODULE_SOURCE) {
    const char *chunkname = lua_pushfstring(L, "=%s", name);
    int status = luaL_loadbufferx(L, bytes, bytes_len, chunkname, "t");
    kore_lua_host_release();
    if (status != LUA_OK) return lua_error(L);
    lua_remove(L, -2);  /* the chunk name */
  }
  else if (kind == KORE_LUA_MODULE_NATIVE && opener != NULL) {
    /* Nothing is borrowed for a native answer — the opener is an address
       in this task's JIT arena — but the release is unconditional so the
       host has exactly one rule: every answer is released once. */
    kore_lua_host_release();
    lua_pushcfunction(L, opener);
  }
  else {
    /* `%s`, not a length: unlike `kore_lua_host_call`'s reply, a
       `require` failure is a `&'static CStr` from src/bridge.rs's
       `describe`, so it is terminated and outlives the release. */
    const char *why = bytes != NULL ? bytes : "no route";
    kore_lua_host_release();
    return luaL_error(L, "module '%s' not found: %s", name, why);
  }
  lua_pushstring(L, name);
  lua_call(L, 1, 1);  /* loader(name) */
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_pushboolean(L, 1);
  }
  lua_pushvalue(L, -1);
  lua_setfield(L, 2, name);  /* LOADED[name] = result */
  return 1;
}

/*
 * Conversion remains wholly below lua_pcall's C recovery frame. A hostile
 * __tostring may throw, but it never jumps across a Rust callback frame: the
 * append callback receives already-materialized bytes and never re-enters Lua.
 */
static int kore_lua_print(lua_State *L) {
  kore_lua_writer *writer = (kore_lua_writer *)lua_touserdata(
      L, lua_upvalueindex(1));
  int top = lua_gettop(L);
  if (writer == NULL || writer->append == NULL) return 0;
  for (int index = 1; index <= top; index++) {
    size_t len = 0;
    const char *text;
    if (index != 1 && !writer->append(writer->ud, "\t", 1, 0))
      return luaL_error(L, "host output limit exceeded");
    text = luaL_tolstring(L, index, &len);
    if (!writer->append(writer->ud, text, len, 0))
      return luaL_error(L, "host output limit exceeded");
    lua_pop(L, 1);
  }
  if (!writer->append(writer->ud, "\n", 1, 0))
    return luaL_error(L, "host output limit exceeded");
  return 0;
}

/*
 * Every library opener can allocate and therefore throw. Keep the complete
 * setup sequence below a C recovery frame so constructor OOM is reported as
 * a null state instead of longjmping through the Rust caller.
 */
/*
 * The stock `io` table, its three standard streams, and the defaults
 * `io.read` / `io.write` act on.
 *
 * `io.stdin` is a real file object that is always at end of file: a REPL's
 * input is the lines the host is already feeding the evaluator, so there
 * is nothing left for a chunk to read from underneath it.  Present and
 * answering nil is the conforming shape — that is exactly what a stream at
 * end of file does — where an absent `io.stdin` would break every script
 * that merely mentions it.
 */
static int luaopen_kore_io(lua_State *L) {
  luaL_newlib(L, kore_lua_io_lib);

  kore_lua_file_push(L, 0, KORE_LUA_FILE_IN, 1, 0, 0);
  lua_pushvalue(L, -1);
  lua_rawsetp(L, LUA_REGISTRYINDEX, &kore_lua_input_key);
  lua_setfield(L, -2, "stdin");

  kore_lua_file_push(L, 0, KORE_LUA_FILE_OUT, 0, 1, 0);
  lua_pushvalue(L, -1);
  lua_rawsetp(L, LUA_REGISTRYINDEX, &kore_lua_output_key);
  lua_setfield(L, -2, "stdout");

  kore_lua_file_push(L, 0, KORE_LUA_FILE_ERR, 0, 1, 0);
  lua_setfield(L, -2, "stderr");
  return 1;
}

/*
 * The stock `os` table.
 *
 * `os.clock` closes over the moment the VM started, because it reports
 * time *used by the program* and the machine's counter does not start
 * there.  `os.execute` and `io.popen` are absent from this crate: both
 * need to launch a process, which is a service face the embedding program
 * holds and this crate deliberately does not (see `Extension`).
 */
static int luaopen_kore_os(lua_State *L) {
  unsigned long long started = 0;
  const char *why = NULL;
  luaL_newlib(L, kore_lua_os_lib);
  if (kore_lua_host_uptime_ns(&started, &why) != KORE_LUA_STATUS_OK) started = 0;
  lua_pushinteger(L, (lua_Integer)started);
  lua_pushcclosure(L, kore_lua_os_clock, 1);
  lua_setfield(L, -2, "clock");
  return 1;
}

static int kore_lua_open_libraries(lua_State *L) {
  luaL_requiref(L, LUA_GNAME, luaopen_base, 1);
  /* Omit inert stdio/warning APIs and the unverified binary-chunk loader. */
  static const char *const omitted_base[] = {
    "dofile", "load", "loadfile", "warn", NULL
  };
  for (const char *const *name = omitted_base; *name != NULL; name++) {
    lua_pushnil(L); lua_setfield(L, -2, *name);
  }
  /* The one loader the host offers: over the store, through the executor. */
  lua_pushcfunction(L, kore_lua_require);
  lua_setfield(L, -2, "require");
  lua_pop(L, 1);
  luaL_requiref(L, LUA_COLIBNAME, luaopen_coroutine, 1); lua_pop(L, 1);
  luaL_requiref(L, LUA_TABLIBNAME, luaopen_table, 1); lua_pop(L, 1);
  luaL_requiref(L, LUA_STRLIBNAME, luaopen_string, 1);
  /* No partial printf and no producer for unsupported binary chunks. */
  lua_pushnil(L); lua_setfield(L, -2, "format");
  lua_pushnil(L); lua_setfield(L, -2, "dump");
  lua_pop(L, 1);
  luaL_requiref(L, LUA_MATHLIBNAME, luaopen_math, 1); lua_pop(L, 1);
  luaL_requiref(L, LUA_UTF8LIBNAME, luaopen_utf8, 1); lua_pop(L, 1);
  /* The file object's metatable, before anything that can create one. */
  kore_lua_file_metatable(L);
  luaL_requiref(L, LUA_IOLIBNAME, luaopen_kore_io, 1); lua_pop(L, 1);
  luaL_requiref(L, LUA_OSLIBNAME, luaopen_kore_os, 1); lua_pop(L, 1);
  /* kore's own standard library, provisioned from this process's faces.
     One table rather than a scatter of globals: `require "kore"` answers
     it, and an embedder's own libraries sit beside them.  Registered by
     hand rather than through luaL_requiref, which has no way to hand the
     opener the embedder's table; the two steps below are exactly what
     requiref does once the opener has run. */
  lua_pushlightuserdata(L, lua_touserdata(L, 1));
  lua_pushvalue(L, 2);
  lua_pushcclosure(L, luaopen_kore, 2);
  lua_call(L, 0, 1);
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
  lua_pushvalue(L, -2);
  lua_setfield(L, -2, "kore");
  lua_pop(L, 1);
  lua_setglobal(L, "kore");
  return 0;
}

lua_State *kore_lua_new(lua_Alloc alloc, void *ud, unsigned seed,
                        const kore_lua_library *libraries, size_t count) {
  lua_State *L = lua_newstate(alloc, ud, seed);
  if (L == NULL) return NULL;
  lua_pushcfunction(L, kore_lua_open_libraries);
  lua_pushlightuserdata(L, (void *)(uintptr_t)libraries);
  lua_pushinteger(L, (lua_Integer)count);
  if (lua_pcall(L, 2, 0, 0) != LUA_OK) {
    lua_close(L);
    return NULL;
  }
  return L;
}

static int kore_lua_install_print_inner(lua_State *L) {
  const kore_lua_writer *config = (const kore_lua_writer *)lua_touserdata(L, 1);
  if (config == NULL || config->append == NULL)
    return luaL_error(L, "invalid host output callback");
  kore_lua_writer *writer = (kore_lua_writer *)lua_newuserdatauv(
      L, sizeof(kore_lua_writer), 0);
  *writer = *config;
  /* The same sink, reachable by name: `io.write` and `io.stderr:write` go
     where `print` goes, rather than each opening a channel of its own. */
  lua_pushvalue(L, -1);
  lua_rawsetp(L, LUA_REGISTRYINDEX, &kore_lua_writer_key);
  lua_pushcclosure(L, kore_lua_print, 1);
  lua_setglobal(L, "print");
  return 0;
}

/*
 * The light-userdata argument is borrowed only for this synchronous pcall;
 * the inner function copies it into Lua-owned storage before returning.
 */
int kore_lua_install_print(lua_State *L, kore_lua_append append, void *ud) {
  kore_lua_writer config = { append, ud };
  lua_pushcfunction(L, kore_lua_install_print_inner);
  lua_pushlightuserdata(L, &config);
  return lua_pcall(L, 1, 0, 0);
}

int kore_lua_load(lua_State *L, const char *bytes, size_t len,
                  const char *name) {
  return luaL_loadbufferx(L, bytes, len, name, "t");
}

int kore_lua_call(lua_State *L, int arguments, int results) {
  return lua_pcall(L, arguments, results, 0);
}

static int kore_lua_print_results_inner(lua_State *L) {
  int results = lua_gettop(L);
  lua_getglobal(L, "print");
  lua_insert(L, 1);
  lua_call(L, results, 0);
  return 0;
}

int kore_lua_print_results(lua_State *L) {
  int results = lua_gettop(L);
  if (results == 0) return LUA_OK;
  /*
   * Both the global lookup (including a hostile _G.__index) and the call are
   * below this C recovery frame. No Lua longjmp can cross the Rust caller.
   */
  lua_pushcfunction(L, kore_lua_print_results_inner);
  lua_insert(L, 1);
  return lua_pcall(L, results, 0, 0);
}

int kore_lua_top(lua_State *L) { return lua_gettop(L); }
void kore_lua_set_top(lua_State *L, int top) { lua_settop(L, top); }
int kore_lua_type(lua_State *L, int index) { return lua_type(L, index); }
const char *kore_lua_error_string(lua_State *L, size_t *len) {
  static const char fallback[] = "Lua error";
  if (lua_type(L, -1) == LUA_TSTRING)
    return lua_tolstring(L, -1, len);
  *len = sizeof(fallback) - 1;
  return fallback;
}
void kore_lua_close(lua_State *L) { lua_close(L); }
