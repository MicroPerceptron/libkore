/* Host-only smoke test for the pinned C engine; the metal ABI is Rust-owned. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lua.h"
#include "lauxlib.h"

/* The in-memory store behind the real host transport.  Included rather
   than compiled separately so both this harness's builds — libc numeric
   and target numeric — get the same store. */
#include "host_fs.c"

/* The engine's real arity.  This prototype MUST match src/kore_lua.c: a
   short one leaves the library array and its count as whatever the call's
   argument registers happened to hold, which `luaopen_kore` then walks. */
typedef struct { const char *name; lua_CFunction opener; } kore_lua_library;
extern lua_State *kore_lua_new(lua_Alloc, void *, unsigned,
                               const kore_lua_library *, size_t);
typedef int (*kore_lua_append)(void *, const char *, size_t, int);
extern int kore_lua_install_print(lua_State *, kore_lua_append, void *);
extern int kore_lua_load(lua_State *, const char *, size_t, const char *);
extern int kore_lua_call(lua_State *, int, int);
extern int kore_lua_print_results(lua_State *);
extern void kore_lua_set_top(lua_State *, int);
extern void kore_lua_close(lua_State *);
/* The sizes the read path turns on, as src/kore_lua.c compiled them. */
extern unsigned long long kore_lua_read_geometry(int);
enum {
  GEOMETRY_BUFSIZE = 0,
  GEOMETRY_STREAM_MIN = 1,
  GEOMETRY_FACADE_STREAM_MIN = 2
};

#if !defined(KORE_LUA_USE_TARGET_NUMERIC)
double kore_lua_strtod(const char *text, char **end) { return strtod(text, end); }

size_t kore_lua_format_number(char *dst, size_t cap, double value,
                              size_t precision) {
  /* Match the target shim's canonical spelling even when the host FPU/libc
     preserves a negative NaN sign (as x86_64 glibc does for 0.0 / 0.0). */
  if (isnan(value)) value = fabs(value);
  int written = snprintf(dst, cap, "%.*g", (int)precision, value);
  if (written < 0 || cap == 0) return 0;
  return (size_t)written < cap ? (size_t)written : cap - 1;
}
#endif

void kore_lua_abort(void) { abort(); }

static void *host_alloc(void *ud, void *ptr, size_t old_size, size_t new_size) {
  (void)ud; (void)old_size;
  if (new_size == 0) { free(ptr); return NULL; }
  return realloc(ptr, new_size);
}

struct fault_alloc {
  size_t calls;
  size_t refuse_at;
};

static void *fault_allocator(void *ud, void *ptr, size_t old_size,
                             size_t new_size) {
  struct fault_alloc *fault = ud;
  (void)old_size;
  if (new_size == 0) { free(ptr); return NULL; }
  fault->calls++;
  if (fault->refuse_at != 0 && fault->calls >= fault->refuse_at)
    return NULL;
  return realloc(ptr, new_size);
}

struct capture { char bytes[128]; size_t len; };

static int capture_append(void *ud, const char *bytes, size_t len, int err) {
  struct capture *capture = ud;
  (void)err;
  if (len > sizeof(capture->bytes) - capture->len) return 0;
  memcpy(capture->bytes + capture->len, bytes, len);
  capture->len += len;
  return 1;
}

static int run(lua_State *L, const char *source, int print_results) {
  if (kore_lua_load(L, source, strlen(source), "=host-smoke") != LUA_OK)
    return 0;
  if (kore_lua_call(L, 0, LUA_MULTRET) != LUA_OK)
    return 0;
  return !print_results || kore_lua_print_results(L) == LUA_OK;
}

static int protected_setup_smoke(void) {
  struct fault_alloc baseline = {0, 0};
  lua_State *L = kore_lua_new(fault_allocator, &baseline, 0x4b4f5245u, NULL, 0);
  if (L == NULL) return 1;
  size_t constructor_calls = baseline.calls;
  kore_lua_close(L);

  /* The final constructor allocation occurs inside library opening. */
  struct fault_alloc open_oom = {0, constructor_calls};
  L = kore_lua_new(fault_allocator, &open_oom, 0x4b4f5245u, NULL, 0);
  if (L != NULL) { kore_lua_close(L); return 2; }
  if (open_oom.calls < open_oom.refuse_at) return 3;

  struct fault_alloc install_oom = {0, 0};
  L = kore_lua_new(fault_allocator, &install_oom, 0x4b4f5245u, NULL, 0);
  if (L == NULL) return 4;
  install_oom.refuse_at = install_oom.calls + 1;
  struct capture capture = {{0}, 0};
  int status = kore_lua_install_print(L, capture_append, &capture);
  if (status == LUA_OK) { kore_lua_close(L); return 5; }
  if (install_oom.calls < install_oom.refuse_at) {
    kore_lua_close(L); return 6;
  }
  kore_lua_close(L);

  struct fault_alloc lookup_oom = {0, 0};
  L = kore_lua_new(fault_allocator, &lookup_oom, 0x4b4f5245u, NULL, 0);
  if (L == NULL) return 7;
  struct capture lookup_capture = {{0}, 0};
  if (kore_lua_install_print(L, capture_append, &lookup_capture) != LUA_OK) {
    kore_lua_close(L); return 8;
  }
  if (!run(L,
           "print=nil; setmetatable(_G,{__index=function() return {} end}); "
           "return 6*7",
           0)) {
    kore_lua_close(L); return 9;
  }
  lookup_oom.refuse_at = lookup_oom.calls + 1;
  if (kore_lua_print_results(L) == LUA_OK) {
    kore_lua_close(L); return 10;
  }
  if (lookup_oom.calls < lookup_oom.refuse_at) {
    kore_lua_close(L); return 11;
  }
  kore_lua_close(L);
  return 0;
}

/*
 * The fault hooks, reachable from Lua as `kore.test.*`.
 *
 * Registered through the same `libraries` array an embedder uses, so the
 * suite also proves that path: a name and an opener handed to
 * `kore_lua_new` land beneath `kore` and list themselves in `kore.caps()`.
 */
static int test_fail_reads_after(lua_State *L) {
  host_fs_fail_reads_after((int)luaL_checkinteger(L, 1));
  return 0;
}

static int test_fail_closes(lua_State *L) {
  host_fs_set_refuse_close(lua_toboolean(L, 1));
  return 0;
}

/* The largest destination a positional read asked the store for — the one
   thing a host build can observe about whether a read could have streamed,
   since there is no device under it to watch. */
static int test_max_read_want(lua_State *L) {
  lua_pushinteger(L, (lua_Integer)host_fs_max_read_want());
  return 1;
}

static int test_reset_max_read_want(lua_State *L) {
  (void)L;
  host_fs_reset_max_read_want();
  return 0;
}

static int test_open_library(lua_State *L) {
  static const luaL_Reg entries[] = {
    {"fail_reads_after", test_fail_reads_after},
    {"fail_closes", test_fail_closes},
    {"max_read_want", test_max_read_want},
    {"reset_max_read_want", test_reset_max_read_want},
    {NULL, NULL}
  };
  luaL_newlib(L, entries);
  return 1;
}

static const kore_lua_library test_libraries[] = {
  {"test", test_open_library}
};

/*
 * Run a chunk and report the Lua error on failure, so a broken assertion
 * names itself instead of arriving as a numeric exit code.
 */
static int run_reporting(lua_State *L, const char *source, const char *what) {
  if (kore_lua_load(L, source, strlen(source), "=host-smoke") != LUA_OK ||
      kore_lua_call(L, 0, LUA_MULTRET) != LUA_OK) {
    const char *why = lua_tostring(L, -1);
    fprintf(stderr, "%s: %s\n", what, why != NULL ? why : "(no message)");
    return 0;
  }
  return 1;
}

/*
 * The filesystem surface, over the in-memory store in tests/host_fs.c.
 *
 * Everything here needs a host to answer, which is why it is a separate
 * pass: the selftest below runs with whatever the weak stubs do, and this
 * one runs the same engine against a store that actually holds bytes.
 * Between them they cover `io`, `os` and `kore.fs` end to end.
 */
static int filesystem_smoke(void) {
  static const char suite[] =
      /* ---------------------------------------------- open, write, read */
      "local f = assert(io.open('/hello.txt','w'))\n"
      "assert(io.type(f)=='file')\n"
      "assert(f:write('hello ')==f)\n"
      "assert(f:write('world\\n')==f)\n"
      "assert(f:close()==true)\n"
      "assert(io.type(f)=='closed file')\n"
      /* A second close is idempotent, never an error (rule 4). */
      "assert(f:close()==true)\n"
      /* A closed file raises on use: a programming error, not a failure. */
      "assert(not pcall(f.read, f, 'a'))\n"

      "local r = assert(io.open('/hello.txt','r'))\n"
      "assert(r:read('a')=='hello world\\n')\n"
      "assert(r:seek('set')==0)\n"
      "assert(r:read('l')=='hello world')\n"
      "assert(r:read('l')==nil)\n"
      "assert(r:seek('set')==0)\n"
      "assert(r:read('L')=='hello world\\n')\n"
      "assert(r:seek('set',6)==6)\n"
      "assert(r:read(5)=='world')\n"
      "assert(r:seek('cur')==11)\n"
      "assert(r:seek('end')==12)\n"
      "assert(r:seek('set',0)==0)\n"
      "assert(r:read(0)=='')\n"
      "assert(r:seek('end')==12 and r:read(0)==nil)\n"
      /* A negative resulting position is a refusal, not a wrap. */
      "local ok,why,err = r:seek('set',-1)\n"
      "assert(ok==nil and why=='range' and err==34)\n"
      "assert(r:close()==true)\n"

      /* ------------------------------------------------ the `a` append mode */
      "local a = assert(io.open('/hello.txt','a'))\n"
      /* An append stream starts at the end, so a seek before the first
         write reports where those bytes will actually land. */
      "assert(a:seek('cur')==12)\n"
      "assert(a:write('again\\n'))\n"
      "assert(a:close())\n"
      "assert(kore.fs.read_file('/hello.txt')=='hello world\\nagain\\n')\n"

      /* ----------------------------------------------------- update modes */
      "local u = assert(io.open('/update.txt','w+'))\n"
      "assert(u:write('abcdef'))\n"
      "assert(u:seek('set',2)==2)\n"
      "assert(u:read(2)=='cd')\n"
      "assert(u:seek('set',0)==0)\n"
      "assert(u:write('ZY'))\n"
      "assert(u:seek('set',0)==0 and u:read('a')=='ZYcdef')\n"
      "assert(u:close())\n"
      /* `r+` reads and writes but does not create. */
      "assert(io.open('/absent.txt','r+')==nil)\n"
      "local p = assert(io.open('/update.txt','r+'))\n"
      "assert(p:read(2)=='ZY')\n"
      "assert(p:close())\n"
      /* A mode Lua does not define is an argument error, not a silent open. */
      "assert(not pcall(io.open,'/update.txt','q'))\n"
      "assert(not pcall(io.open,'/update.txt',''))\n"
      /* `b` is accepted and means nothing here. */
      "assert(io.open('/update.txt','rb')):close()\n"

      /* ------------------------------------------------- the two spellings */
      /* One failure, two renderings, both derived from the same FsError. */
      "local h,why,err = io.open('/nowhere.txt','r')\n"
      "assert(h==nil and why=='not-found' and err==2)\n"
      "local g,e = kore.fs.open('/nowhere.txt','r')\n"
      "assert(g==nil and e.code=='not-found' and e.kind=='refused')\n"
      "assert(tostring(e)=='not-found')\n"
      /* The reply the bridge hands back is length-delimited and carries no
         terminator; the store poisons the byte after it.  An exact match
         here is what proves nothing above reached for `strlen`. */
      "assert(#why==9)\n"

      /* ------------------------------------------------- read-only handles */
      "local ro = assert(io.open('/hello.txt','r'))\n"
      "local ok2,why2,err2 = ro:write('x')\n"
      "assert(ok2==nil and why2=='access-denied' and err2==13)\n"
      "assert(ro:close())\n"
      "local wo = assert(io.open('/writeonly.txt','w'))\n"
      "local ok3,why3 = wo:read('a')\n"
      "assert(ok3==nil and why3=='access-denied')\n"
      "assert(wo:close())\n"

      /* ------------------------------------------------- multi-page reads */
      /* Larger than the store's single-reply cap, so the read-ahead in
         kore_lua.c must loop rather than assume one refill is enough. */
      "local big = assert(io.open('/big.txt','w'))\n"
      "local chunk = string.rep('0123456789', 100)\n"
      "for _ = 1, 10 do assert(big:write(chunk)) end\n"
      "assert(big:close())\n"
      "local rb = assert(io.open('/big.txt','r'))\n"
      "local whole = rb:read('a')\n"
      "assert(#whole==10000, #whole)\n"
      "assert(whole==string.rep(chunk,10))\n"
      "assert(rb:seek('set')==0)\n"
      "assert(#rb:read(6000)==6000)\n"
      "assert(rb:close())\n"

      /* ------------------------------------------------------ read formats */
      "assert(kore.fs.write_file('/nums.txt','12 -3.5 0x10\\nrest\\n')>0)\n"
      "local n = assert(io.open('/nums.txt','r'))\n"
      "assert(n:read('n')==12)\n"
      "assert(n:read('n')==-3.5)\n"
      "assert(n:read('n')==16)\n"
      "assert(n:read('l')=='')\n"
      "assert(n:read('l')=='rest')\n"
      "assert(n:close())\n"
      /* Several formats in one call, and the 5.2 star spelling. */
      "local m = assert(io.open('/hello.txt','r'))\n"
      "local one,two = m:read(5,'*l')\n"
      "assert(one=='hello' and two==' world')\n"
      "assert(m:close())\n"

      /* ------------------------------------------------------------ lines */
      "assert(kore.fs.write_file('/lines.txt','a\\nb\\nc\\n')>0)\n"
      "local seen = {}\n"
      "for line in io.lines('/lines.txt') do seen[#seen+1] = line end\n"
      "assert(#seen==3 and seen[1]=='a' and seen[3]=='c')\n"
      /* io.lines(name) answers four values, the last to-be-closed. */
      "local n1,n2,n3,n4 = io.lines('/lines.txt')\n"
      "assert(type(n1)=='function' and n2==nil and n3==nil)\n"
      "assert(io.type(n4)=='file' and n4:close())\n"
      "local it,_,_,tbc = io.lines('/lines.txt')\n"
      "assert(io.type(tbc)=='file' and it()=='a')\n"
      "assert(tbc:close())\n"
      /* f:lines leaves the file open, and closing it stops the iterator. */
      "local lf = assert(io.open('/lines.txt','r'))\n"
      "local step = lf:lines('L')\n"
      "assert(step()=='a\\n')\n"
      "assert(lf:close())\n"
      "assert(not pcall(step))\n"

      /* ------------------------------------------------ default input/output */
      "local infile = io.input('/lines.txt')\n"
      "assert(io.type(infile)=='file')\n"
      "assert(io.read('l')=='a')\n"
      "assert(io.read('l')=='b')\n"
      "io.input(io.stdin)\n"
      "assert(io.read('l')==nil)\n"
      "assert(infile:close())\n"

      /* -------------------------------------------------------- kore.fs */
      "assert(kore.fs.exists('/hello.txt'))\n"
      "assert(not kore.fs.exists('/nowhere.txt'))\n"
      "local st = assert(kore.fs.stat('/hello.txt'))\n"
      "assert(st.size_bytes==18 and st.kind=='file' and st.directory==false)\n"
      "assert(kore.fs.mkdir('/dir'))\n"
      "local dst = assert(kore.fs.stat('/dir'))\n"
      "assert(dst.kind=='directory' and dst.directory==true)\n"
      /* A directory that exists is `already-exists`, not a silent success. */
      "local md,mderr = kore.fs.mkdir('/dir')\n"
      "assert(md==nil and mderr.code=='already-exists')\n"
      "assert(kore.fs.write_file('/dir/one','1')==1)\n"
      "assert(kore.fs.write_file('/dir/two','22')==2)\n"
      "local names = assert(kore.fs.list('/dir'))\n"
      "table.sort(names)\n"
      "assert(#names==2 and names[1]=='one' and names[2]=='two')\n"
      /* A non-empty directory is not removable. */
      "local rm,rmerr = kore.fs.remove('/dir')\n"
      "assert(rm==nil and rmerr.code=='not-empty')\n"
      "assert(kore.fs.remove('/dir/one'))\n"
      "assert(kore.fs.rename('/dir/two','/dir/three'))\n"
      "assert(kore.fs.read_file('/dir/three')=='22')\n"
      "assert(kore.fs.remove('/dir/three'))\n"
      "assert(kore.fs.remove('/dir'))\n"
      "assert(not kore.fs.exists('/dir'))\n"
      /* `read_file` on a missing path answers the kore spelling. */
      "local rf,rferr = kore.fs.read_file('/nowhere.txt')\n"
      "assert(rf==nil and rferr.code=='not-found')\n"
      /* The root has no parent to mutate. */
      "local root,rooterr = kore.fs.remove('/')\n"
      "assert(root==nil and rooterr.code=='access-denied')\n"

      /* ------------------------------------------------------ kore.fs.resize */
      "local rz = assert(kore.fs.open('/resize.txt','w+'))\n"
      "assert(rz:write('abcdefgh'))\n"
      "assert(kore.fs.resize(rz,3))\n"
      /* A truncation drags the position back with it. */
      "assert(rz:seek('cur')==3)\n"
      "assert(rz:seek('set')==0 and rz:read('a')=='abc')\n"
      "assert(kore.fs.resize(rz,5))\n"
      "assert(rz:seek('end')==5)\n"
      "assert(not pcall(kore.fs.resize, rz, -1))\n"
      "assert(rz:close())\n"

      /* ----------------------------------------------------------- os verbs */
      "assert(kore.fs.write_file('/gone.txt','x')==1)\n"
      "assert(os.remove('/gone.txt')==true)\n"
      "local orm,ormwhy,ormerr = os.remove('/gone.txt')\n"
      "assert(orm==nil and ormwhy=='not-found' and ormerr==2)\n"
      "assert(kore.fs.write_file('/from.txt','y')==1)\n"
      "assert(os.rename('/from.txt','/to.txt')==true)\n"
      "assert(kore.fs.read_file('/to.txt')=='y')\n"
      "assert(os.remove('/to.txt'))\n"

      /* --------------------------------------------------------- temporaries */
      /* os.tmpname creates its file, so the name it answers is one nothing
         else can already hold. */
      "local name = assert(os.tmpname())\n"
      "assert(name:sub(1,9)=='/tmp/lua_')\n"
      "assert(kore.fs.exists(name))\n"
      "local name2 = assert(os.tmpname())\n"
      "assert(name2~=name)\n"
      "assert(os.remove(name) and os.remove(name2))\n"
      /* io.tmpfile removes its own file when the handle retires. */
      "local tmp = assert(io.tmpfile())\n"
      "assert(tmp:write('scratch'))\n"
      "assert(tmp:seek('set')==0 and tmp:read('a')=='scratch')\n"
      "assert(tmp:close())\n"

      /* ------------------------------------------------------- kore.proc */
      "assert(type(kore.proc.args)=='table')\n"
      "assert(kore.proc.args[1]=='one' and kore.proc.args[2]=='two')\n"
      "assert(#kore.proc.args==2)\n"
      /* A missing key answers nil, as an ordinary table does. */
      "assert(kore.proc.nothing==nil and kore.proc[1]==nil)\n"

      /* ------------------------------------------------------- kore.time */
      "assert(kore.time.sleep(0.001)==true)\n"
      "assert(type(kore.time.now())=='number')\n"
      "assert(math.type(kore.time.mono_ns())=='integer')\n"
      "assert(kore.time.wall()==1700000000.0)\n"
      "assert(not pcall(kore.time.sleep,-1))\n"
      /* os.time with no argument reads the same wall clock. */
      "assert(os.time()==1700000000)\n"
      "assert(os.date('!%Y-%m-%dT%H:%M:%SZ')=='2023-11-14T22:13:20Z')\n"

      /* ----------------------------------------------------- kore.random */
      "assert(kore.random.available()==true)\n"
      "assert(#kore.random.bytes(32)==32)\n"
      "assert(#kore.random.bytes(0)==0)\n"
      "assert(not pcall(kore.random.bytes,-1))\n"
      "assert(not pcall(kore.random.bytes,4097))\n"
      "local lo = kore.random.integer(5,5)\n"
      "assert(lo==5)\n"
      "for _ = 1, 50 do\n"
      "  local v = kore.random.integer(-3,3)\n"
      "  assert(v>=-3 and v<=3, v)\n"
      "end\n"
      "assert(not pcall(kore.random.integer,3,-3))\n"

      /* --------------------------------------------------------- kore.caps */
      "local caps = kore.caps()\n"
      "assert(caps[1]=='time' and caps[2]=='random' and caps[3]=='fs')\n"
      "assert(caps[4]=='proc' and caps[5]=='test')\n"
      "assert(#caps==5 and type(kore.test)=='table')\n"
      "assert(kore.has('fs')==true and kore.has('nothing')==false)\n"

      /* ------------------------------------------------ the `__gc` backstop */
      /* Deterministic close is primary, but a script that simply drops a
         file must still give the store its handle back (rule 4).  The
         open-handle count the harness checks next is what proves it. */
      "do local abandoned = assert(io.open('/hello.txt','r'))\n"
      "   assert(abandoned:read(1)=='h') end\n"
      "collectgarbage('collect')\n"
      "collectgarbage('collect')\n"
      "return true\n";

  /*
   * A refused call renders through `kore_lua_fail`'s concatenating arm.
   * The store poisons the byte past its reply, so an exact match is what
   * proves the refusal was not spelled with `%s` over a buffer that has
   * no terminator.
   */
  static const char refusal_suite[] =
      "local ok,why = kore.time.sleep(0.5)\n"
      "assert(ok==nil, 'the sleep should have been refused')\n"
      "assert(why=='refused: the timer is busy', why)\n"
      "return true\n";

  /*
   * A read that fails partway answers the failure alone, as stock `g_read`
   * does — not a partial result with a failure stapled to the end.
   */
  static const char read_failure_suite[] =
      /* A failure on the very first refill: the whole answer is the
         failure triple, in `io`'s spelling. */
      "local f = assert(io.open('/hello.txt','r'))\n"
      "kore.test.fail_reads_after(0)\n"
      "local a,b,c,d = f:read('l','l')\n"
      "assert(a==nil, 'a failed read answers nil first')\n"
      "assert(b=='access-lost', tostring(b))\n"
      "assert(c==9, tostring(c))\n"
      "assert(d==nil, 'the failure is the whole answer')\n"
      "kore.test.fail_reads_after(-1)\n"
      "assert(f:close())\n"
      /* A failure *after* a partial read, which is the interesting one:
         6000 bytes needs two refills, so by the time the third is refused
         the `luaL_Buffer` has moved onto the stack as a box.  The failure
         must replace the partial result rather than trail it, and must not
         be pushed above the box. */
      "local big = assert(io.open('/big.txt','r'))\n"
      "kore.test.fail_reads_after(2)\n"
      "local e,g,h,i = big:read(6000,'l')\n"
      "assert(e==nil, 'the failure replaces the partial read')\n"
      "assert(g=='access-lost', tostring(g))\n"
      "assert(h==9, tostring(h))\n"
      "assert(i==nil, 'nothing trails the failure')\n"
      "kore.test.fail_reads_after(-1)\n"
      "assert(big:close())\n"
      /* The same shape through `read('a')`, whose buffer is the largest. */
      "local all = assert(io.open('/big.txt','r'))\n"
      "kore.test.fail_reads_after(1)\n"
      "local j,k = all:read('a')\n"
      "assert(j==nil and k=='access-lost', tostring(k))\n"
      "kore.test.fail_reads_after(-1)\n"
      "assert(all:close())\n"
      /* A file `kore.fs.open` answered is the *same* object, so its methods
         render the same way: only the two surfaces' own verbs differ, and
         `file:read` is one of `io`'s seven.  `kore.fs.open`'s failure is a
         table; `f:read`'s is a message and an errno, whichever surface
         opened the file. */
      "local kf = assert(kore.fs.open('/big.txt','r'))\n"
      "kore.test.fail_reads_after(0)\n"
      "local l,m,n2 = kf:read('a')\n"
      "assert(l==nil and m=='access-lost' and n2==9, tostring(m))\n"
      "kore.test.fail_reads_after(-1)\n"
      "assert(kf:close())\n"
      /* A read failure inside `io.lines` raises, as `io_readline` does: a
         `for` loop that merely stopped would read as end of file. */
      "local lines = io.lines('/big.txt')\n"
      "kore.test.fail_reads_after(0)\n"
      "local raised = select(2, pcall(lines))\n"
      "kore.test.fail_reads_after(-1)\n"
      "assert(type(raised)=='string' and raised:find('access%-lost'), \n"
      "       tostring(raised))\n"

      /* ------------------------------------------- a close that also fails */
      /* `kore.fs.read_file` is one host call, so it leaves no handle on
         this side to close: a store refusing every close cannot reach its
         answer at all.  That is what moving it off open/read/close bought
         — the stack discipline a fixed-count pop used to need here is now
         absent rather than merely correct. */
      "kore.test.fail_closes(true)\n"
      "local body, bodyerr = kore.fs.read_file('/hello.txt')\n"
      "assert(body=='hello world\\nagain\\n', tostring(body))\n"
      "assert(bodyerr==nil, 'a close is not the caller\\'s answer')\n"
      /* The read is still the caller's answer when it is the read that
         failed: the host performs it, so fault injection reaches it. */
      "kore.test.fail_reads_after(0)\n"
      "local none, noneerr = kore.fs.read_file('/hello.txt')\n"
      "assert(none==nil, tostring(none))\n"
      "assert(type(noneerr)=='table' and noneerr.code=='access-lost',\n"
      "       tostring(noneerr))\n"
      "kore.test.fail_reads_after(-1)\n"
      /* A refused close on the file object answers the refusal itself, and
         the handle is retired either way: a re-close is still idempotent. */
      "local rc = assert(io.open('/hello.txt','r'))\n"
      "local cok, cwhy, cerr = rc:close()\n"
      "assert(cok==nil and cwhy=='access-lost' and cerr==9, tostring(cwhy))\n"
      "assert(rc:close()==true)\n"
      "kore.test.fail_closes(false)\n"

      /* ------------------------------------ a temporary draw that cannot land */
      /* With no `/tmp` to create in, the exclusive open answers `not-found`
         and both verbs render that reason exactly.  Exact is the point: the
         store poisons the byte after its reply, so a `strlen` over it would
         answer a longer, garbled string. */
      "assert(kore.fs.remove('/tmp'))\n"
      "local tn, tnwhy = os.tmpname()\n"
      "assert(tn==nil and tnwhy=='not-found', tostring(tnwhy))\n"
      "local tf, tfwhy, tferr = io.tmpfile()\n"
      "assert(tf==nil and tfwhy=='not-found' and tferr==2, tostring(tfwhy))\n"
      "assert(kore.fs.mkdir('/tmp'))\n"
      "return true\n";

  /* `os.exit` records its code and unwinds; it is not an error. */
  static const char exit_suite[] =
      "local closed = false\n"
      "do\n"
      "  local guard <close> = setmetatable({}, {__close=function() closed=true end})\n"
      "  os.exit(7)\n"
      "end\n"
      "error('os.exit did not unwind')\n";

  host_fs_reset("one two");
  struct capture capture = {{0}, 0};
  lua_State *L = kore_lua_new(host_alloc, NULL, 0x4b4f5245u, test_libraries,
                              sizeof test_libraries / sizeof test_libraries[0]);
  if (L == NULL) return 1;
  if (kore_lua_install_print(L, capture_append, &capture) != LUA_OK) {
    kore_lua_close(L); return 2;
  }
  if (!run_reporting(L, suite, "filesystem suite")) { kore_lua_close(L); return 3; }
  kore_lua_set_top(L, 0);

  /* Every handle the suite opened was closed: a leak here means a verb
     kept one the script can no longer name. */
  if (host_fs_open_handles() != 0) { kore_lua_close(L); return 4; }
  /* io.tmpfile's file went with its handle, and os.tmpname's did not. */
  if (host_fs_exists("/tmp/lua_")) { kore_lua_close(L); return 5; }

  host_fs_set_refuse_sleep(1);
  if (!run_reporting(L, refusal_suite, "refusal rendering")) {
    kore_lua_close(L); return 6;
  }
  host_fs_set_refuse_sleep(0);
  kore_lua_set_top(L, 0);

  if (!run_reporting(L, read_failure_suite, "read failure")) {
    kore_lua_close(L); return 7;
  }
  kore_lua_set_top(L, 0);
  kore_lua_close(L);

  /* A fresh VM: the exit suite ends its own evaluation. */
  host_fs_reset("");
  struct capture exit_capture = {{0}, 0};
  L = kore_lua_new(host_alloc, NULL, 0x4b4f5245u, NULL, 0);
  if (L == NULL) return 8;
  if (kore_lua_install_print(L, capture_append, &exit_capture) != LUA_OK) {
    kore_lua_close(L); return 9;
  }
  if (kore_lua_load(L, exit_suite, strlen(exit_suite), "=exit") != LUA_OK) {
    kore_lua_close(L); return 10;
  }
  /* The sentinel reaches the host as an error — that is how it ran the
     `<close>` handler on its way out.  src/vm.rs is what recognizes it;
     here the contract under test is that the code was recorded and the
     raise carries the sentinel the host matches on. */
  if (kore_lua_call(L, 0, LUA_MULTRET) == LUA_OK) { kore_lua_close(L); return 11; }
  {
    const char *why = lua_tostring(L, -1);
    if (why == NULL || strstr(why, "kore:exit") == NULL) {
      kore_lua_close(L); return 12;
    }
  }
  if (!host_fs_exit_requested() || host_fs_exit_code() != 7) {
    kore_lua_close(L); return 13;
  }
  kore_lua_close(L);
  return 0;
}


/* The fixture, in bytes.  Past any threshold this test will accept, so the
   large-read bypass is genuinely taken; and not a multiple of the store's
   page, so the last answer is short. */
#define STREAM_FIXTURE_BYTES 300000
#define STREAM_FIXTURE_UNIT "0123456789abcdef"

/*
 * Large reads, the whole-file verb, and the sizes both turn on.
 *
 * The thing under test is a *decision*, not a transfer: the host facade
 * streams a read through this launch's storage ring only when it is handed
 * a destination of at least `KORE_LUA_FACADE_STREAM_MIN`, so a surface that
 * only ever asks for a page can never stream however large the file is.
 * Nothing here can observe a ring — there is no device under a host build —
 * but everything here can observe the ask, and the ask is what was wrong.
 *
 * The store still answers one page at a time, which is the point: the
 * bypass must assemble a short answer correctly and hand back to the page
 * buffer once the remainder is small, and both halves run in every read
 * below.
 */
static int streaming_smoke(void) {
  /* The geometry first, because every assertion after it is only
     meaningful if the engine still believes these numbers. */
  if (kore_lua_read_geometry(GEOMETRY_BUFSIZE) != 4096) return 1;
  if (kore_lua_read_geometry(GEOMETRY_FACADE_STREAM_MIN) != 256u * 1024u) return 2;
  {
    unsigned long long ask = kore_lua_read_geometry(GEOMETRY_STREAM_MIN);
    /* A read must be at least the facade's threshold to stream at all, and
       must still fit the fixture or this test stops covering the bypass. */
    if (ask < kore_lua_read_geometry(GEOMETRY_BUFSIZE)) return 3;
    if (ask > STREAM_FIXTURE_BYTES) return 4;
  }

  static const char suite[] =
      "local unit = '" STREAM_FIXTURE_UNIT "'\n"
      "local want = string.rep(unit, 300000 // #unit)\n"
      "assert(#want == 300000, 'fixture length')\n"

      /* The whole-file verb: one host call, the whole file, no handle. */
      "assert(kore.fs.read_file('/big.bin') == want)\n"

      /* `f:read('a')` over the same file.  The ask starts at the length
         `open` answered and shrinks as the file is consumed, so this runs
         the bypass first and the page buffer for the remainder. */
      "local f = assert(io.open('/big.bin','r'))\n"
      "assert(f:read('a') == want)\n"
      "assert(f:read('a') == '')\n"

      /* An explicit count, from the start and then from an offset that is
         neither page nor threshold aligned. */
      "assert(f:seek('set') == 0)\n"
      "assert(f:read(300000) == want)\n"
      "assert(f:seek('set', 10) == 10)\n"
      "assert(f:read(299990) == want:sub(11))\n"

      /* A large ask past the end answers what there is, and is not an
         error: short is short, only zero is end of file. */
      "assert(f:seek('set') == 0)\n"
      "assert(f:read(400000) == want)\n"
      "assert(f:seek('end') == 300000)\n"
      "assert(f:read(300000) == nil)\n"

      /* Small reads still take the page buffer, and still agree. */
      "assert(f:seek('set') == 0)\n"
      "assert(f:read(100) == want:sub(1, 100))\n"
      "assert(f:read('l') == want:sub(101))\n"
      "assert(f:close())\n"

      /* The whole-file verb's refusals are `kore.fs`'s table spelling, not
         `io`'s message-and-errno one. */
      "local v, e = kore.fs.read_file('/nowhere.bin')\n"
      "assert(v == nil and e.code == 'not-found' and e.kind == 'refused')\n"
      "local d, de = kore.fs.read_file('/tmp')\n"
      "assert(d == nil and de.code == 'is-directory')\n"
      /* An empty file reads successfully as an empty string, never nil. */
      "assert(kore.fs.write_file('/empty.bin', '') == 0)\n"
      "assert(kore.fs.read_file('/empty.bin') == '')\n"

      /* Small reads must still be small: the bypass is for the reads that
         can use it, not a blanket change of how every file is read. */
      "assert(kore.test.max_read_want() >= 262144, 'a read reached the facade threshold')\n"
      "local s = assert(io.open('/hello.txt','w'))\n"
      "assert(s:write('short'))\n"
      "assert(s:close())\n"
      "kore.test.reset_max_read_want()\n"
      "local t = assert(io.open('/hello.txt','r'))\n"
      "assert(t:read('a') == 'short')\n"
      "assert(t:close())\n"
      "assert(kore.test.max_read_want() <= 4096, 'a small file still reads a page at a time')\n";

  host_fs_reset("");
  {
    /* Seeded directly: the point is a file larger than the threshold, not
       the writing of one. */
    static char fixture[STREAM_FIXTURE_BYTES];
    size_t unit = sizeof STREAM_FIXTURE_UNIT - 1;
    for (size_t i = 0; i < sizeof fixture; i++) fixture[i] = STREAM_FIXTURE_UNIT[i % unit];
    if (!host_fs_put("/big.bin", fixture, sizeof fixture)) return 5;
  }

  struct capture capture = {{0}, 0};
  lua_State *L = kore_lua_new(host_alloc, NULL, 0x4b4f5245u, test_libraries,
                              sizeof test_libraries / sizeof test_libraries[0]);
  if (L == NULL) return 6;
  if (kore_lua_install_print(L, capture_append, &capture) != LUA_OK) {
    kore_lua_close(L); return 7;
  }
  if (!run_reporting(L, suite, "streaming suite")) { kore_lua_close(L); return 8; }
  kore_lua_set_top(L, 0);
  /* The whole-file verb holds no handle of its own, and every file the
     suite opened was closed. */
  if (host_fs_open_handles() != 0) { kore_lua_close(L); return 9; }
  kore_lua_close(L);
  return 0;
}

#if defined(KORE_LUA_USE_TARGET_NUMERIC)
int kore_lua_host_smoke(void) {
#else
int main(void) {
#endif
  static const char selftest[] =
      "local co=coroutine.create(function() coroutine.yield(6) end)\n"
      "local ok,value=coroutine.resume(co)\n"
      "assert(ok and value==6)\n"
      "assert(string.upper('lua')=='LUA')\n"
      "assert(table.concat({4,2})=='42')\n"
      "assert(math.floor(42.9)==42)\n"
      "assert(tostring(42.5)=='42.5')\n"
      "assert(string.format==nil)\n"
      "assert(string.dump==nil)\n"
      "assert(load==nil and loadfile==nil and dofile==nil)\n"
      "assert(warn==nil)\n"
      "assert(tostring(1e100)=='1e+100')\n"
      "assert(tostring(1e308)=='1e+308')\n"
      "assert(tostring(5e-324)=='4.94065645841247e-324')\n"
      "assert(tostring(-0.0)=='-0.0')\n"
      "assert(tostring(1e15)=='1e+15')\n"
      "assert(tostring(1234567890123456.0)=='1234567890123456.0')\n"
      "assert(tostring(math.huge)=='inf')\n"
      "assert(tostring(-math.huge)=='-inf')\n"
      "assert(tostring(0.0/0.0)=='nan')\n"
      "assert(utf8.len('\\u{3bb}')==1)\n"
      "local bad=setmetatable({},{__tostring=function() error('bad') end})\n"
      "assert(not pcall(print,bad))\n"
      "assert(not pcall(print,string.rep('x',4000)))\n"
      /* The standard library registers, and the two surfaces are one
         implementation: io.open and kore.fs.open are the same function. */
      "assert(type(io)=='table' and type(os)=='table')\n"
      "assert(type(kore.fs)=='table')\n"
      "assert(io.type(io.stdout)=='file')\n"
      /* "The I/O library never closes these files" (manual 6.8): the close
         is refused, and the stream is still usable afterwards. */
      "local ok,why=io.stdout:close()\n"
      "assert(ok==nil and why=='cannot close standard file')\n"
      "assert(io.type(io.stdout)=='file')\n"
      /* Every member the manual lists is either present or absent for a
         stated reason; none is present-but-partial. */
      "for _,name in ipairs{'close','flush','input','lines','open','output',\n"
      "                     'read','tmpfile','type','write'} do\n"
      "  assert(type(io[name])=='function', 'io.'..name)\n"
      "end\n"
      "for _,name in ipairs{'clock','date','difftime','exit','getenv',\n"
      "                     'remove','rename','setlocale','time',\n"
      "                     'tmpname'} do\n"
      "  assert(type(os[name])=='function', 'os.'..name)\n"
      "end\n"
      "local h=io.open('/nowhere','r')\n"
      "assert(h==nil)\n"
      "for _,name in ipairs{'close','flush','lines','read','seek',\n"
      "                     'setvbuf','write'} do\n"
      "  assert(type(io.stdout[name])=='function', 'file:'..name)\n"
      "end\n"
      /* luaL_fileresult's exact shapes: a standard close answers two values
         and no errno; flush answers true; write answers the file so writes
         chain.  Writes here are empty so the capture assertions below still
         see only what print produced. */
      "assert(select('#', io.stdout:close())==2)\n"
      "local ok,why,err=io.stdout:close()\n"
      "assert(ok==nil and why=='cannot close standard file' and err==nil)\n"
      "assert(io.stdout:flush()==true and io.flush()==true)\n"
      "assert(io.stdout:write('')==io.stdout)\n"
      "assert(io.write('')==io.stdout)\n"
      "assert(not pcall(io.write,{}) and not pcall(io.write,true))\n"
      "assert(io.stdout:setvbuf('full')==true)\n"
      "assert(not pcall(io.stdout.setvbuf,io.stdout,'bogus'))\n"
      "assert(io.input()==io.stdin and io.output()==io.stdout)\n"
      "assert(io.output(io.stderr)==io.stderr)\n"
      "io.output(io.stdout)\n"
      /* io.lines() over the default input answers one value and leaves it
         open; io.lines(name) answers four, the last to-be-closed. */
      "assert(select('#', io.lines())==1)\n"
      "for _ in io.lines() do error('stdin is at end of file') end\n"
      "assert(io.type(io.stdin)=='file')\n"
      /* Every read format at end of file, including the zero-length probe
         and the 5.2 star spelling. */
      "assert(io.stdin:read(0)==nil and io.stdin:read('a')=='')\n"
      "assert(io.stdin:read('l')==nil and io.stdin:read('L')==nil)\n"
      "assert(io.stdin:read('n')==nil and io.stdin:read(5)==nil)\n"
      "assert(io.stdin:read('*l')==nil)\n"
      "assert(not pcall(io.stdin.read,io.stdin,'z'))\n"
      "assert(select(2, io.stdout:seek())~=nil)\n"
      /* os.time normalizes its table in place and fills the derived
         fields, as mktime/setallfields do. */
      "local t={year=2023,month=11,day=14,hour=22,min=13,sec=20}\n"
      "assert(os.time(t)==1700000000)\n"
      "assert(t.wday==3 and t.yday==318 and t.isdst==false)\n"
      "local roll={year=2023,month=13,day=1,hour=0}\n"
      "assert(os.time(roll)==os.time{year=2024,month=1,day=1,hour=0})\n"
      "assert(roll.year==2024 and roll.month==1)\n"
      "local named=select(2, pcall(os.time,{month=1,day=1}))\n"
      "assert(named:find(\"field 'year' missing\"), named)\n"
      "for _,x in ipairs{0,1,951782400,1700000000,-86400} do\n"
      "  assert(os.time(os.date('*t',x))==x, x)\n"
      "end\n"
      /* io.input/io.output/io.lines raise rather than answer a code. */
      "local raised=select(2, pcall(io.lines,'/nowhere'))\n"
      "assert(type(raised)=='string' and raised:find('cannot open file'),\n"
      "       tostring(raised))\n"
      "assert(not pcall(io.input,'/nowhere'))\n"
      /* `io.output` opens for writing, so it fails only where it cannot
         create: a parent that is not there. */
      "assert(not pcall(io.output,'/nodir/file'))\n"
      "assert(not pcall(io.lines,'/nodir/file'))\n"
      "assert(io.type(42)==nil and io.type('x')==nil)\n"
      "assert(io.stdin:read('a')=='')\n"
      "assert(io.stdin:read('l')==nil)\n"
      /* The calendar is pure computation, so a host build checks it in
         full.  These are the dates every off-by-one shows up in: the
         epoch, a leap day, a century that is not a leap year, one that
         is, and a pre-epoch instant whose seconds must floor rather than
         truncate toward zero. */
      "assert(os.date('%Y-%m-%d',0)=='1970-01-01')\n"
      "assert(os.date('%H:%M:%S',0)=='00:00:00')\n"
      "assert(os.date('%A',0)=='Thursday')\n"
      "assert(os.date('%Y-%m-%d',951782400)=='2000-02-29')\n"
      "assert(os.date('%Y-%m-%d',68169600)=='1972-02-29')\n"
      "assert(os.date('%Y-%m-%d',4107542400)=='2100-03-01')\n"
      "assert(os.date('%Y-%m-%d %H:%M:%S',-1)=='1969-12-31 23:59:59')\n"
      "assert(os.date('%Y-%m-%d',-86400)=='1969-12-31')\n"
      "assert(os.date('%j',0)=='001')\n"
      "assert(os.date('%F',1700000000)=='2023-11-14')\n"
      "assert(os.date('%T',1700000000)=='22:13:20')\n"
      "assert(os.date('%p',0)=='AM' and os.date('%p',43200)=='PM')\n"
      "assert(os.date('%I',0)=='12' and os.date('%I',43200)=='12')\n"
      "assert(os.date('%%',0)=='%')\n"
      "assert(not pcall(os.date,'%Q',0))\n"
      /* The calendar states its range instead of truncating a huge year
         into a plausible-looking date: the broken-down year is an `int`
         and `%F` spells four digits.  The bounds are the first and last
         instants of years 0000 and 9999. */
      "assert(os.date('%F',253402300799)=='9999-12-31')\n"
      "assert(os.date('%T',253402300799)=='23:59:59')\n"
      "assert(os.date('%F',-62167219200)=='0000-01-01')\n"
      "local over = select(2, pcall(os.date,'%F',253402300800))\n"
      "assert(over:find('cannot be represented'), tostring(over))\n"
      "assert(not pcall(os.date,'%F',-62167219201))\n"
      "assert(not pcall(os.date,'*t',1e18))\n"
      "assert(not pcall(os.date,'%F',-1e18))\n"
      "assert(not pcall(os.time,{year=99999,month=1,day=1,hour=0}))\n"
      "assert(not pcall(os.time,{year=2023,month=1,day=1e15,hour=0}))\n"
      "assert(not pcall(os.time,{year=-5,month=1,day=1,hour=0}))\n"
      /* `*t` and `os.time` are exact inverses, which is what makes date
         arithmetic by field addition work. */
      "local t=os.date('*t',1700000000)\n"
      "assert(t.year==2023 and t.month==11 and t.day==14)\n"
      "assert(t.hour==22 and t.min==13 and t.sec==20)\n"
      "assert(t.wday==3 and t.yday==318 and t.isdst==false)\n"
      "assert(os.time(t)==1700000000)\n"
      "assert(os.time{year=1970,month=1,day=1,hour=0}==0)\n"
      /* An out-of-range field normalizes, as mktime does. */
      "assert(os.time{year=2023,month=13,day=1,hour=0}==\n"
      "       os.time{year=2024,month=1,day=1,hour=0})\n"
      "assert(os.time{year=2024,month=0,day=1,hour=0}==\n"
      "       os.time{year=2023,month=12,day=1,hour=0})\n"
      "assert(os.difftime(10,4)==6.0)\n"
      /* A kore task has no environment, so every name is unset. */
      "assert(os.getenv('PATH')==nil)\n"
      "assert(os.setlocale()=='C' and os.setlocale('fr_FR')==nil)\n"
      /* Absent, and absent for a stated reason: both need to launch a
         process, which is the embedding program's face, not this crate's. */
      "assert(os.execute==nil and io.popen==nil)\n"
      /* With no host linked, a store verb answers its reason rather than
         pretending; the failure carries an errno on the io surface and a
         code on the kore one. */
      "local f,why,err=io.open('/nowhere','r')\n"
      "assert(f==nil and type(why)=='string' and math.type(err)=='integer')\n"
      "local g,e=kore.fs.open('/nowhere','r')\n"
      "assert(g==nil and type(e)=='table' and type(e.code)=='string')\n"
      "assert(tostring(e)==e.code)\n"
      "print(6*7)\n";
  int setup_status = protected_setup_smoke();
  if (setup_status != 0) return 10 + setup_status;
  /* A store with nothing in it: the selftest below asserts what an absent
     path answers, and the pass after it fills one in. */
  host_fs_reset("");
  int fs_status = filesystem_smoke();
  if (fs_status != 0) return 30 + fs_status;
  int stream_status = streaming_smoke();
  if (stream_status != 0) return 50 + stream_status;
  host_fs_reset("");
  struct capture capture = {{0}, 0};
  lua_State *L = kore_lua_new(host_alloc, NULL, 0x4b4f5245u, NULL, 0);
  if (L == NULL) return 1;
  if (kore_lua_install_print(L, capture_append, &capture) != LUA_OK) {
    kore_lua_close(L); return 2;
  }
  if (!run_reporting(L, selftest, "selftest")) return 2;
  if (capture.len != 3 || memcmp(capture.bytes, "42\n", 3) != 0) return 3;

  /* Saved-script regression: a local closure adds mixed integer/float
   * arguments. This runs with both libc and the target numeric shim. */
  static const char saved_script[] =
      "print(\"hello\")\n"
      "local function add(x, y)\n return x + y\nend\n"
      "local x = 100\nlocal y = 9.9\nprint(add(x, y))\n";
  if (!run(L, saved_script, 0)) return 9;
  if (capture.len != 15 || memcmp(capture.bytes + 3, "hello\n109.9\n", 12) != 0) return 10;
  capture.len = 3;

  /* Protected errors must return through the selected setjmp seam. */
  if (kore_lua_load(L, "error('expected')", 17, "=error") != LUA_OK) return 4;
  if (kore_lua_call(L, 0, LUA_MULTRET) == LUA_OK) return 5;
  kore_lua_set_top(L, 0);
  if (!run(L, "return 6*7", 1)) return 6;
  if (capture.len != 6 || memcmp(capture.bytes + 3, "42\n", 3) != 0) return 7;
  kore_lua_set_top(L, 0);
  if (run(L,
          "print=nil; setmetatable(_G,{__index=function(_,key) "
          "if key=='print' then error('hostile print lookup') end end}); "
          "return 6*7",
          1))
    return 8;
  kore_lua_close(L);
  return 0;
}
