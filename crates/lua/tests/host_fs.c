/*
 * An in-memory store behind the real host transport.
 *
 * `src/kore_lua.c` declares every host entry point weak so a build with no
 * Rust host still links; with only those weak stubs, though, every verb
 * that touches a file answers "no kore host" and the whole `io` / `os` /
 * `kore.fs` surface is covered only as far as "the member exists".  That
 * is exactly the half where the interesting mistakes live: stack
 * discipline around `luaL_Buffer`, handle lifetime, the two failure
 * spellings, append and truncate, the temporary-file draw.
 *
 * So this file defines those symbols *strongly* and puts a small store
 * behind them.  It is not a filesystem: it is the smallest thing that can
 * answer the twelve filesystem ops faithfully enough that the C above it is
 * exercised for real.  Faithfully means, specifically:
 *
 *   - The reply text is **length-delimited and not NUL-terminated**, and
 *     the byte after it is poisoned.  That is the bridge's actual contract
 *     (`src/bridge.rs` answers a Rust `Vec<u8>`), and anything above that
 *     reaches for `strlen` on it reads poison and fails a test rather than
 *     passing by luck on a host where the next byte happened to be zero.
 *   - Refusals carry the kebab-case vocabulary `src/files.rs` derives from
 *     `FsError`, so the errno table and the `kore.fs` error table are
 *     tested against the words they will really see.
 *   - A handle is an index and a generation, encoded as `src/files.rs`
 *     encodes it, so a use after close is `invalid-handle` here too.
 *   - Positional reads are short by design: the store answers at most one
 *     page whatever was asked for, which is what makes both the read-ahead
 *     loop in `kore_lua.c` and its large-read bypass run more than once.
 *     The whole-file op is the one exception, because its whole contract is
 *     that there is no second call.
 *
 * Included by tests/host_smoke.c rather than compiled separately, so both
 * the libc-numeric and target-numeric builds of that harness get it.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Must agree with src/kore_lua.c's enum and src/bridge.rs's constants. */
enum {
  HOST_FS_OP_SLEEP = 16,
  HOST_FS_OP_PROC_ARGS = 17,
  HOST_FS_OP_OPEN = 18,
  HOST_FS_OP_CLOSE = 19,
  HOST_FS_OP_READ = 20,
  HOST_FS_OP_WRITE = 21,
  HOST_FS_OP_SYNC = 23,
  HOST_FS_OP_RESIZE = 24,
  HOST_FS_OP_STAT = 25,
  HOST_FS_OP_REMOVE = 26,
  HOST_FS_OP_RENAME = 27,
  HOST_FS_OP_LIST = 28,
  HOST_FS_OP_PROC_EXIT = 29,
  HOST_FS_OP_READ_FILE = 30,
  HOST_FS_OP_SAVE = 31,
  HOST_FS_OP_ENTRIES = 32,
  HOST_FS_OP_RIGHTS = 33
};

enum {
  HOST_FS_O_READ = 1 << 0,
  HOST_FS_O_WRITE = 1 << 1,
  HOST_FS_O_CREATE = 1 << 2,
  HOST_FS_O_TRUNCATE = 1 << 3,
  HOST_FS_O_APPEND = 1 << 4,
  HOST_FS_O_EXCLUSIVE = 1 << 5,
  HOST_FS_O_DIRECTORY = 1 << 6
};

enum { HOST_FS_OK = 0, HOST_FS_REFUSED = 1, HOST_FS_UNAVAILABLE = 2 };

/* One page, so `f:read("a")` over a larger file refills more than once.
   This bounds one *positional* read and nothing else: a short answer is
   the store's whole point, and it is what makes both the page-buffer loop
   and the large-read bypass in `kore_lua.c` run more than once. */
#define HOST_FS_READ_MAX 4096
#define HOST_FS_MAX_OBJECTS 32
#define HOST_FS_MAX_HANDLES 16
#define HOST_FS_PATH_MAX 64
/* Past `KORE_LUA_STREAM_MIN`, so a fixture can be large enough for the C
   side to take its bypass at the real threshold rather than a lowered one.
   Object storage is allocated on first use, so an empty store costs
   nothing for the capacity it does not hold. */
#define HOST_FS_BYTES_MAX (512 * 1024)

struct host_fs_object {
  char path[HOST_FS_PATH_MAX];
  unsigned char *bytes;
  size_t len;
  int directory;
  int used;
  /* The graph facts a stat answers.  Modelled rather than invented: a
     revision that moves on every commit, a stable identity that is not the
     slot, and a subtype carried verbatim — which is the behaviour worth
     testing, because an unknown subtype must survive the round trip. */
  unsigned long long revision;
  unsigned long long stable;
  unsigned char subtype;
};

struct host_fs_handle {
  int object;
  unsigned generation;
  int readable;
  int writable;
  int append;
  int used;
};

static struct {
  struct host_fs_object objects[HOST_FS_MAX_OBJECTS];
  struct host_fs_handle handles[HOST_FS_MAX_HANDLES];
  unsigned next_generation;
  unsigned next_stable;
  /* The reply window.  One buffer, overwritten by each call, exactly as
     the bridge's `cell.text` is: C copies it into a Lua string before it
     makes another call.  The poison byte is the point — see the header. */
  unsigned char reply[HOST_FS_BYTES_MAX + 1];
  size_t reply_len;
  /* What the harness inspects afterwards. */
  int exit_requested;
  int exit_code;
  unsigned long long slept_ns;
  /* A deterministic draw, so a temporary name is reproducible. */
  unsigned long long entropy;
  int entropy_available;
  const char *args;
  /* Fault injection.  `kore.time.sleep` is the shortest path to
     `kore_lua_fail`'s REFUSED arm, which is the only place a refusal's
     text is *concatenated* rather than pushed whole — and so the only
     place a `%s` over the length-delimited reply would run into poison. */
  int refuse_sleep;
  /* Refuse every read past the first `reads_allowed`, once armed.  This is
     the only way to reach a failure that lands *after* a partial read: by
     then `luaL_Buffer` has grown onto the stack as a box, and a failure
     pushed above it would leave the box at a level `luaL_pushresult` no
     longer expects. */
  int reads_armed;
  int reads_allowed;
  /* The largest destination a positional read ever asked for, before this
     store clamps it to one page.  The clamp is why it has to be recorded
     here: what the facade decides on is the *ask*, and after the clamp
     nothing downstream can still see how big it was. */
  unsigned long long max_read_want;
  /* Acknowledge at most this much per positional write, once armed: the
     only way to reach a *short* write, which the store otherwise never
     produces because it writes everything or refuses.  Zero acknowledges
     nothing at all without refusing — the one case the C side has no
     reason to report and must spell itself. */
  int writes_armed;
  int write_cap;
  /* Refuse every close.  The slot is still freed, so the harness's
     leak accounting stays meaningful; what is under test is the *caller's*
     stack discipline, because `kore_lua_file_do_close` pushes one value on
     success and two on a refusal. */
  int refuse_close;
} host_fs;

/* ---------------------------------------------------- little-endian out */

static void host_fs_put_le64(unsigned char *at, unsigned long long value) {
  for (int i = 0; i < 8; i++) at[i] = (unsigned char)(value >> (8 * i));
}

static void host_fs_put_le32(unsigned char *at, unsigned long value) {
  for (int i = 0; i < 4; i++) at[i] = (unsigned char)(value >> (8 * i));
}

/* ------------------------------------------------------------- replies */

/*
 * Publish `len` bytes as the reply and poison the byte after them.
 *
 * The poison is load-bearing: the bridge's text is a Rust `Vec<u8>` with
 * no terminator, so any caller that reaches for `strlen` must produce a
 * visibly wrong answer here rather than a lucky one.
 */
static void host_fs_reply_bounded(const void *bytes, size_t len, size_t cap,
                                  const char **text, size_t *text_len) {
  if (len > cap) len = cap;
  if (len != 0 && bytes != NULL) memcpy(host_fs.reply, bytes, len);
  host_fs.reply[len] = 0xff;
  host_fs.reply_len = len;
  if (text != NULL) *text = (const char *)host_fs.reply;
  if (text_len != NULL) *text_len = len;
}

/* One page at most: the short answer every positional read gets. */
static void host_fs_reply(const void *bytes, size_t len,
                          const char **text, size_t *text_len) {
  host_fs_reply_bounded(bytes, len, HOST_FS_READ_MAX, text, text_len);
}

/* A whole file in one reply — only `HOST_FS_OP_READ_FILE` answers this
   way, because only that op promises the caller there is no "again". */
static void host_fs_reply_whole(const void *bytes, size_t len,
                                const char **text, size_t *text_len) {
  host_fs_reply_bounded(bytes, len, HOST_FS_BYTES_MAX, text, text_len);
}

static int host_fs_refuse(const char *code, const char **text,
                          size_t *text_len) {
  host_fs_reply(code, strlen(code), text, text_len);
  return HOST_FS_REFUSED;
}

/* ------------------------------------------------------------ the store */

static int host_fs_find(const char *path, size_t len) {
  if (len >= HOST_FS_PATH_MAX) return -1;
  for (int i = 0; i < HOST_FS_MAX_OBJECTS; i++)
    if (host_fs.objects[i].used && strlen(host_fs.objects[i].path) == len &&
        memcmp(host_fs.objects[i].path, path, len) == 0)
      return i;
  return -1;
}

/* The parent of `path`, as src/files.rs `split` computes it: "/" for a
   top-level name, "." for a bare one. */
static void host_fs_parent(const char *path, size_t len, char *out,
                           size_t *out_len) {
  while (len > 1 && path[len - 1] == '/') len--;
  size_t cut = len;
  while (cut > 0 && path[cut - 1] != '/') cut--;
  if (cut == 0) { memcpy(out, ".", 1); *out_len = 1; return; }
  if (cut == 1) { memcpy(out, "/", 1); *out_len = 1; return; }
  memcpy(out, path, cut - 1);
  *out_len = cut - 1;
}

static int host_fs_parent_ok(const char *path, size_t len) {
  char parent[HOST_FS_PATH_MAX];
  size_t parent_len = 0;
  host_fs_parent(path, len, parent, &parent_len);
  if (parent_len == 1 && parent[0] == '.') return 1;   /* the task's cwd */
  int at = host_fs_find(parent, parent_len);
  return at >= 0 && host_fs.objects[at].directory;
}

/*
 * An object's bytes, allocated on first use.
 *
 * `HOST_FS_BYTES_MAX` is half a megabyte so a fixture can cross the C
 * side's streaming threshold; holding that inline in every slot would be
 * sixteen megabytes of store for a harness whose files are mostly a line
 * long. `NULL` is an allocation failure, which the caller answers as
 * `out-of-storage` — the same word the real store would use.
 */
static unsigned char *host_fs_bytes(struct host_fs_object *object) {
  if (object->bytes == NULL) object->bytes = (unsigned char *)calloc(1, HOST_FS_BYTES_MAX);
  return object->bytes;
}

static int host_fs_make(const char *path, size_t len, int directory) {
  if (len == 0 || len >= HOST_FS_PATH_MAX) return -1;
  for (int i = 0; i < HOST_FS_MAX_OBJECTS; i++) {
    if (host_fs.objects[i].used) continue;
    memcpy(host_fs.objects[i].path, path, len);
    host_fs.objects[i].path[len] = '\0';
    host_fs.objects[i].len = 0;
    host_fs.objects[i].directory = directory;
    /* Not the slot: a stable id must not be reconstructible from position,
       or a test could pass while the real one was being read wrong. */
    host_fs.objects[i].stable = 0x51ab1e00ull + (unsigned long long)(++host_fs.next_stable);
    host_fs.objects[i].revision = 1;
    host_fs.objects[i].subtype = 0;
    host_fs.objects[i].used = 1;
    return i;
  }
  return -1;
}

/* A handle is `(generation << 32) | index`, as src/files.rs encodes it. */
static unsigned long long host_fs_encode(int index, unsigned generation) {
  return ((unsigned long long)generation << 32) | (unsigned long long)index;
}

static struct host_fs_handle *host_fs_resolve(unsigned long long handle) {
  unsigned index = (unsigned)(handle & 0xffffffffu);
  unsigned generation = (unsigned)(handle >> 32);
  if (index >= HOST_FS_MAX_HANDLES) return NULL;
  struct host_fs_handle *slot = &host_fs.handles[index];
  if (!slot->used || slot->generation != generation) return NULL;
  return slot;
}

/* ---------------------------------------------------------- the harness */

void host_fs_reset(const char *args) {
  /* Before the wipe, or the pointers go with it: object storage is
     allocated on first use and the struct is cleared wholesale. */
  for (int i = 0; i < HOST_FS_MAX_OBJECTS; i++) {
    free(host_fs.objects[i].bytes);
    host_fs.objects[i].bytes = NULL;
  }
  memset(&host_fs, 0, sizeof host_fs);
  host_fs.entropy = 0x243f6a8885a308d3ull;   /* any fixed, well-mixed word */
  host_fs.entropy_available = 1;
  host_fs.args = args != NULL ? args : "";
  /* The two directories a task starts with. */
  host_fs_make("/", 1, 1);
  host_fs_make("/tmp", 4, 1);
}

void host_fs_set_entropy(int available) {
  host_fs.entropy_available = available;
}

void host_fs_set_refuse_sleep(int refuse) { host_fs.refuse_sleep = refuse; }

void host_fs_set_refuse_close(int refuse) { host_fs.refuse_close = refuse; }

/* Let `allowed` more reads through, then refuse; a negative count disarms. */
void host_fs_fail_reads_after(int allowed) {
  host_fs.reads_armed = allowed >= 0;
  host_fs.reads_allowed = allowed;
}

int host_fs_exit_requested(void) { return host_fs.exit_requested; }
int host_fs_exit_code(void) { return host_fs.exit_code; }
unsigned long long host_fs_slept_ns(void) { return host_fs.slept_ns; }

/* Every handle the store still believes is open — what a leak looks like. */
int host_fs_open_handles(void) {
  int live = 0;
  for (int i = 0; i < HOST_FS_MAX_HANDLES; i++)
    if (host_fs.handles[i].used) live++;
  return live;
}

/* The largest destination a positional read asked for since the last
   reset — what proves a read was big enough for the facade to stream. */
unsigned long long host_fs_max_read_want(void) { return host_fs.max_read_want; }

void host_fs_reset_max_read_want(void) { host_fs.max_read_want = 0; }

/* Acknowledge at most `cap` bytes per positional write; a negative `cap`
   disarms.  Zero acknowledges nothing, without refusing. */
void host_fs_short_writes(int cap) {
  host_fs.writes_armed = cap >= 0;
  host_fs.write_cap = cap > 0 ? cap : 0;
}

/* Stamp an object's subtype, so a test can reach one this build names and
   one it deliberately does not. */
int host_fs_set_subtype(const char *path, int subtype) {
  int at = host_fs_find(path, strlen(path));
  if (at < 0) return 0;
  host_fs.objects[at].subtype = (unsigned char)subtype;
  return 1;
}

int host_fs_exists(const char *path) {
  return host_fs_find(path, strlen(path)) >= 0;
}

/* Seed a file directly, so a test can read one it did not have to write. */
int host_fs_put(const char *path, const char *bytes, size_t len) {
  size_t path_len = strlen(path);
  int at = host_fs_find(path, path_len);
  if (at < 0) at = host_fs_make(path, path_len, 0);
  if (at < 0 || len > HOST_FS_BYTES_MAX) return 0;
  unsigned char *store = host_fs_bytes(&host_fs.objects[at]);
  if (store == NULL) return 0;
  if (len != 0) memcpy(store, bytes, len);
  host_fs.objects[at].len = len;
  return 1;
}

/* ------------------------------------------------------- the entry points */

int kore_lua_host_random(unsigned char *dst, size_t len, const char **text) {
  if (!host_fs.entropy_available) {
    if (dst != NULL && len != 0) memset(dst, 0, len);
    *text = "this machine exposes no entropy source";
    return HOST_FS_UNAVAILABLE;
  }
  for (size_t i = 0; i < len; i++) {
    /* splitmix64, so a temporary name is deterministic across runs. */
    host_fs.entropy += 0x9e3779b97f4a7c15ull;
    unsigned long long z = host_fs.entropy;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    dst[i] = (unsigned char)((z ^ (z >> 31)) & 0xff);
  }
  *text = NULL;
  return HOST_FS_OK;
}

int kore_lua_host_random_available(void) { return host_fs.entropy_available; }

int kore_lua_host_uptime_ns(unsigned long long *out, const char **text) {
  static unsigned long long tick;
  tick += 1000000ull;   /* a monotonic counter, one millisecond a read */
  *out = tick;
  *text = NULL;
  return HOST_FS_OK;
}

int kore_lua_host_wall_ns(unsigned long long *out, const char **text) {
  *out = 1700000000ull * 1000000000ull;   /* 2023-11-14T22:13:20Z */
  *text = NULL;
  return HOST_FS_OK;
}

static int host_fs_do_open(unsigned long long flags, const char *path,
                           size_t path_len, unsigned long long *result,
                           unsigned long long *result2, const char **text,
                           size_t *text_len) {
  if (path_len == 0 || path_len >= HOST_FS_PATH_MAX)
    return host_fs_refuse("invalid-name", text, text_len);
  int at = host_fs_find(path, path_len);
  if (flags & HOST_FS_O_DIRECTORY) {
    if (at >= 0) return host_fs_refuse("already-exists", text, text_len);
    if (!host_fs_parent_ok(path, path_len))
      return host_fs_refuse("not-found", text, text_len);
    if (host_fs_make(path, path_len, 1) < 0)
      return host_fs_refuse("out-of-storage", text, text_len);
    *result = 0;
    *result2 = 0;
    return HOST_FS_OK;
  }
  if (at >= 0 && (flags & HOST_FS_O_EXCLUSIVE))
    return host_fs_refuse("already-exists", text, text_len);
  if (at < 0) {
    if (!(flags & HOST_FS_O_CREATE))
      return host_fs_refuse("not-found", text, text_len);
    if (!host_fs_parent_ok(path, path_len))
      return host_fs_refuse("not-found", text, text_len);
    at = host_fs_make(path, path_len, 0);
    if (at < 0) return host_fs_refuse("out-of-storage", text, text_len);
  }
  if (host_fs.objects[at].directory)
    return host_fs_refuse("is-directory", text, text_len);
  if (flags & HOST_FS_O_TRUNCATE) host_fs.objects[at].len = 0;
  for (int i = 0; i < HOST_FS_MAX_HANDLES; i++) {
    if (host_fs.handles[i].used) continue;
    host_fs.handles[i].object = at;
    host_fs.handles[i].generation = host_fs.next_generation++;
    host_fs.handles[i].readable = (flags & HOST_FS_O_READ) != 0;
    host_fs.handles[i].writable = (flags & HOST_FS_O_WRITE) != 0;
    host_fs.handles[i].append = (flags & HOST_FS_O_APPEND) != 0;
    host_fs.handles[i].used = 1;
    *result = host_fs_encode(i, host_fs.handles[i].generation);
    *result2 = (flags & HOST_FS_O_TRUNCATE)
                   ? 0 : (unsigned long long)host_fs.objects[at].len;
    return HOST_FS_OK;
  }
  return host_fs_refuse("out-of-handles", text, text_len);
}

int kore_lua_host_call(unsigned op, unsigned long long arg0,
                       unsigned long long arg1, unsigned long long arg2,
                       const void *input, size_t input_len,
                       unsigned long long *result, unsigned long long *result2,
                       const char **text, size_t *text_len) {
  const char *path = (const char *)input;
  *result = 0;
  *result2 = 0;
  host_fs_reply(NULL, 0, text, text_len);

  switch (op) {
    case HOST_FS_OP_SLEEP:
      if (host_fs.refuse_sleep)
        return host_fs_refuse("the timer is busy", text, text_len);
      host_fs.slept_ns += arg0;
      return HOST_FS_OK;

    case HOST_FS_OP_PROC_ARGS:
      host_fs_reply(host_fs.args, strlen(host_fs.args), text, text_len);
      return HOST_FS_OK;

    case HOST_FS_OP_PROC_EXIT:
      host_fs.exit_requested = 1;
      host_fs.exit_code = (int)arg0;
      return HOST_FS_OK;

    case HOST_FS_OP_OPEN:
      return host_fs_do_open(arg0, path, input_len, result, result2, text,
                             text_len);

    case HOST_FS_OP_CLOSE: {
      struct host_fs_handle *slot = host_fs_resolve(arg0);
      if (slot == NULL) return host_fs_refuse("invalid-handle", text, text_len);
      slot->used = 0;
      if (host_fs.refuse_close)
        return host_fs_refuse("access-lost", text, text_len);
      return HOST_FS_OK;
    }

    case HOST_FS_OP_READ: {
      struct host_fs_handle *slot = host_fs_resolve(arg0);
      if (slot == NULL) return host_fs_refuse("invalid-handle", text, text_len);
      if (!slot->readable)
        return host_fs_refuse("access-denied", text, text_len);
      if (host_fs.reads_armed) {
        if (host_fs.reads_allowed == 0)
          return host_fs_refuse("access-lost", text, text_len);
        host_fs.reads_allowed--;
      }
      struct host_fs_object *object = &host_fs.objects[slot->object];
      unsigned long long offset = arg1;
      if (arg2 > host_fs.max_read_want) host_fs.max_read_want = arg2;
      size_t want = (size_t)(arg2 < HOST_FS_READ_MAX ? arg2 : HOST_FS_READ_MAX);
      if (offset >= object->len) { *result = 0; return HOST_FS_OK; }
      size_t available = object->len - (size_t)offset;
      if (want > available) want = available;
      unsigned char *store = host_fs_bytes(object);
      if (store == NULL) return host_fs_refuse("out-of-storage", text, text_len);
      host_fs_reply(store + offset, want, text, text_len);
      *result = want;
      *result2 = want;
      return HOST_FS_OK;
    }

    /*
     * A whole file by path, with no handle anywhere.  The real host opens,
     * streams and closes on its own thread under one pinned version; what
     * matters here is only that it answers the whole file in one reply and
     * leaves no handle behind, which is what the harness's leak accounting
     * then proves.
     *
     * It counts as one read against `reads_armed`: the real host performs
     * the reads itself, so fault injection has to be able to reach them
     * here too, and the verb answers once either way.
     */
    case HOST_FS_OP_READ_FILE: {
      if (host_fs.reads_armed) {
        if (host_fs.reads_allowed == 0)
          return host_fs_refuse("access-lost", text, text_len);
        host_fs.reads_allowed--;
      }
      if (input_len == 0 || input_len >= HOST_FS_PATH_MAX)
        return host_fs_refuse("invalid-name", text, text_len);
      int at = host_fs_find(path, input_len);
      if (at < 0) return host_fs_refuse("not-found", text, text_len);
      if (host_fs.objects[at].directory)
        return host_fs_refuse("is-directory", text, text_len);
      struct host_fs_object *object = &host_fs.objects[at];
      if (object->len > HOST_FS_BYTES_MAX)
        return host_fs_refuse("range", text, text_len);
      host_fs_reply_whole(object->len != 0 ? host_fs_bytes(object) : NULL,
                          object->len, text, text_len);
      *result = (unsigned long long)object->len;
      *result2 = (unsigned long long)object->len;
      return HOST_FS_OK;
    }

    /*
     * Replace an open file's whole contents.  The real host does this in
     * one held-commit transaction; what the harness can check is that it
     * is one call, that it replaces rather than overwrites in place, and
     * that the short-write cap does *not* apply to it — a save has no
     * prefix to answer, it either replaced the file or did not.
     */
    case HOST_FS_OP_SAVE: {
      struct host_fs_handle *slot = host_fs_resolve(arg0);
      if (slot == NULL) return host_fs_refuse("invalid-handle", text, text_len);
      if (!slot->writable)
        return host_fs_refuse("access-denied", text, text_len);
      if (input_len > HOST_FS_BYTES_MAX)
        return host_fs_refuse("out-of-storage", text, text_len);
      struct host_fs_object *object = &host_fs.objects[slot->object];
      unsigned char *store = host_fs_bytes(object);
      if (store == NULL) return host_fs_refuse("out-of-storage", text, text_len);
      if (input_len != 0) memcpy(store, input, input_len);
      object->len = input_len;
      object->revision++;
      *result = (unsigned long long)input_len;
      *result2 = (unsigned long long)input_len;
      return HOST_FS_OK;
    }

    case HOST_FS_OP_WRITE: {
      struct host_fs_handle *slot = host_fs_resolve(arg0);
      if (slot == NULL) return host_fs_refuse("invalid-handle", text, text_len);
      if (!slot->writable)
        return host_fs_refuse("access-denied", text, text_len);
      struct host_fs_object *object = &host_fs.objects[slot->object];
      /* An append handle writes at the end as it is *now*, which is what
         src/files.rs does — it re-stats rather than trusting the offset. */
      size_t at = slot->append ? object->len : (size_t)arg1;
      if (at > HOST_FS_BYTES_MAX || input_len > HOST_FS_BYTES_MAX - at)
        return host_fs_refuse("out-of-storage", text, text_len);
      unsigned char *store = host_fs_bytes(object);
      if (store == NULL) return host_fs_refuse("out-of-storage", text, text_len);
      /* A short acknowledgement, once armed: the store took a prefix and
         owes the rest, which is a fact and not a refusal. */
      size_t taken = input_len;
      if (host_fs.writes_armed && (size_t)host_fs.write_cap < taken)
        taken = (size_t)host_fs.write_cap;
      if (at > object->len)
        memset(store + object->len, 0, at - object->len);
      if (taken != 0) memcpy(store + at, input, taken);
      if (at + taken > object->len) object->len = at + taken;
      if (taken != 0) object->revision++;
      *result = taken;
      *result2 = at + taken;
      return HOST_FS_OK;
    }

    case HOST_FS_OP_SYNC: {
      struct host_fs_handle *slot = host_fs_resolve(arg0);
      if (slot == NULL) return host_fs_refuse("invalid-handle", text, text_len);
      return HOST_FS_OK;
    }

    case HOST_FS_OP_RESIZE: {
      struct host_fs_handle *slot = host_fs_resolve(arg0);
      if (slot == NULL) return host_fs_refuse("invalid-handle", text, text_len);
      if (!slot->writable)
        return host_fs_refuse("access-denied", text, text_len);
      struct host_fs_object *object = &host_fs.objects[slot->object];
      size_t length = (size_t)arg1;
      if (length > HOST_FS_BYTES_MAX)
        return host_fs_refuse("out-of-storage", text, text_len);
      unsigned char *store = host_fs_bytes(object);
      if (store == NULL) return host_fs_refuse("out-of-storage", text, text_len);
      if (length > object->len)
        memset(store + object->len, 0, length - object->len);
      object->len = length;
      object->revision++;
      *result = length;
      return HOST_FS_OK;
    }

    case HOST_FS_OP_STAT: {
      struct host_fs_object *object;
      if (input_len == 0) {
        struct host_fs_handle *slot = host_fs_resolve(arg0);
        if (slot == NULL)
          return host_fs_refuse("invalid-handle", text, text_len);
        object = &host_fs.objects[slot->object];
      } else {
        int at = host_fs_find(path, input_len);
        if (at < 0) return host_fs_refuse("not-found", text, text_len);
        object = &host_fs.objects[at];
      }
      /* The fixed record src/files.rs encodes, little-endian; the C half
         checks its length before reading a field. */
      unsigned char record[34];
      host_fs_put_le64(record, object->len);
      host_fs_put_le64(record + 8, object->revision);
      host_fs_put_le64(record + 16, object->stable);
      host_fs_put_le32(record + 24, 1);            /* one name per object here */
      host_fs_put_le32(record + 28, 0);            /* the harness grants nothing */
      record[32] = (unsigned char)(object->directory != 0);
      record[33] = object->subtype;
      host_fs_reply(record, sizeof record, text, text_len);
      *result = object->len;
      *result2 = sizeof record;
      return HOST_FS_OK;
    }

    /* A directory's edges, each with its target's class and subtype. */
    case HOST_FS_OP_ENTRIES: {
      int at = host_fs_find(path, input_len);
      if (at < 0) return host_fs_refuse("not-found", text, text_len);
      if (!host_fs.objects[at].directory)
        return host_fs_refuse("not-directory", text, text_len);
      unsigned char records[HOST_FS_READ_MAX];
      size_t len = 0;
      unsigned long long count = 0;
      for (int i = 0; i < HOST_FS_MAX_OBJECTS; i++) {
        if (!host_fs.objects[i].used || i == at) continue;
        char parent[HOST_FS_PATH_MAX];
        size_t parent_len = 0;
        size_t whole = strlen(host_fs.objects[i].path);
        host_fs_parent(host_fs.objects[i].path, whole, parent, &parent_len);
        if (parent_len != input_len || memcmp(parent, path, input_len) != 0)
          continue;
        size_t cut = whole;
        while (cut > 0 && host_fs.objects[i].path[cut - 1] != '/') cut--;
        size_t leaf = whole - cut;
        if (len + leaf + 3 > sizeof records)
          return host_fs_refuse("range", text, text_len);
        records[len++] = (unsigned char)(host_fs.objects[i].directory != 0);
        records[len++] = host_fs.objects[i].subtype;
        memcpy(records + len, host_fs.objects[i].path + cut, leaf);
        len += leaf;
        records[len++] = '\0';
        count++;
      }
      host_fs_reply(records, len, text, text_len);
      *result = count;
      *result2 = len;
      return HOST_FS_OK;
    }

    /*
     * The rights the task holds at a path, in the store's own `rwns`
     * spelling.  The harness grants `rwn` on everything it holds and
     * nothing on what it does not: enough for the C half's derivation of
     * booleans from letters to be exercised, including a letter it must
     * report as absent.
     */
    case HOST_FS_OP_RIGHTS: {
      int at = host_fs_find(path, input_len);
      if (at < 0) return host_fs_refuse("not-found", text, text_len);
      host_fs_reply("rwn", 3, text, text_len);
      *result = 7;
      *result2 = 3;
      return HOST_FS_OK;
    }

    case HOST_FS_OP_REMOVE: {
      /* The root has no parent to mutate; src/files.rs refuses it before
         the store ever sees it, and so does this. */
      if (input_len == 0 || (input_len == 1 && path[0] == '/'))
        return host_fs_refuse("access-denied", text, text_len);
      int at = host_fs_find(path, input_len);
      if (at < 0) return host_fs_refuse("not-found", text, text_len);
      if (host_fs.objects[at].directory) {
        for (int i = 0; i < HOST_FS_MAX_OBJECTS; i++) {
          if (!host_fs.objects[i].used || i == at) continue;
          char parent[HOST_FS_PATH_MAX];
          size_t parent_len = 0;
          host_fs_parent(host_fs.objects[i].path,
                         strlen(host_fs.objects[i].path), parent, &parent_len);
          if (parent_len == input_len && memcmp(parent, path, input_len) == 0)
            return host_fs_refuse("not-empty", text, text_len);
        }
      }
      host_fs.objects[at].used = 0;
      return HOST_FS_OK;
    }

    case HOST_FS_OP_RENAME: {
      /* Two paths in one window, NUL-separated. */
      size_t cut = 0;
      while (cut < input_len && path[cut] != '\0') cut++;
      if (cut == input_len)
        return host_fs_refuse("invalid-name", text, text_len);
      const char *to = path + cut + 1;
      size_t to_len = input_len - cut - 1;
      int at = host_fs_find(path, cut);
      if (at < 0) return host_fs_refuse("not-found", text, text_len);
      if (to_len == 0 || to_len >= HOST_FS_PATH_MAX)
        return host_fs_refuse("invalid-name", text, text_len);
      if (!host_fs_parent_ok(to, to_len))
        return host_fs_refuse("not-found", text, text_len);
      int existing = host_fs_find(to, to_len);
      if (existing >= 0) host_fs.objects[existing].used = 0;
      memcpy(host_fs.objects[at].path, to, to_len);
      host_fs.objects[at].path[to_len] = '\0';
      return HOST_FS_OK;
    }

    case HOST_FS_OP_LIST: {
      int at = host_fs_find(path, input_len);
      if (at < 0) return host_fs_refuse("not-found", text, text_len);
      if (!host_fs.objects[at].directory)
        return host_fs_refuse("not-directory", text, text_len);
      unsigned char names[HOST_FS_READ_MAX];
      size_t len = 0;
      unsigned long long count = 0;
      for (int i = 0; i < HOST_FS_MAX_OBJECTS; i++) {
        if (!host_fs.objects[i].used || i == at) continue;
        char parent[HOST_FS_PATH_MAX];
        size_t parent_len = 0;
        size_t whole = strlen(host_fs.objects[i].path);
        host_fs_parent(host_fs.objects[i].path, whole, parent, &parent_len);
        if (parent_len != input_len || memcmp(parent, path, input_len) != 0)
          continue;
        size_t cut = whole;
        while (cut > 0 && host_fs.objects[i].path[cut - 1] != '/') cut--;
        size_t leaf = whole - cut;
        if (len + leaf + 1 > sizeof names)
          return host_fs_refuse("range", text, text_len);
        memcpy(names + len, host_fs.objects[i].path + cut, leaf);
        len += leaf;
        names[len++] = '\0';
        count++;
      }
      host_fs_reply(names, len, text, text_len);
      *result = count;
      *result2 = len;
      return HOST_FS_OK;
    }

    default:
      host_fs_reply("unknown host call", sizeof "unknown host call" - 1, text,
                    text_len);
      return HOST_FS_UNAVAILABLE;
  }
  (void)arg2;
}
