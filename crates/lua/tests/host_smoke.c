/* Host-only smoke test for the pinned C engine; the metal ABI is Rust-owned. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lua.h"
#include "lauxlib.h"

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
      "assert(not pcall(io.output,'/nowhere'))\n"
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
  struct capture capture = {{0}, 0};
  lua_State *L = kore_lua_new(host_alloc, NULL, 0x4b4f5245u, NULL, 0);
  if (L == NULL) return 1;
  if (kore_lua_install_print(L, capture_append, &capture) != LUA_OK) {
    kore_lua_close(L); return 2;
  }
  if (!run(L, selftest, 0)) return 2;
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
