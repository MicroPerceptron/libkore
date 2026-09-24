/* Freestanding runtime probe for the exact musl setjmp assembly linked by Lua. */
#include <setjmp.h>

typedef unsigned long usize;

static jmp_buf environment;
static jmp_buf zero_environment;
static volatile unsigned phase;

#if defined(__aarch64__)

static void put_byte(unsigned char byte) {
  volatile unsigned *const uart_data = (volatile unsigned *)0x09000000ul;
  volatile unsigned *const uart_flags = (volatile unsigned *)0x09000018ul;
  while ((*uart_flags & (1u << 5)) != 0) {}
  *uart_data = byte;
}

__attribute__((noreturn)) static void finish(int status) {
  usize reason[2] = {0x20026ul, (usize)status};
  register usize operation __asm__("x0") = 0x20;
  register usize argument __asm__("x1") = (usize)reason;
  __asm__ volatile("hlt #0xf000"
                   : "+r"(operation)
                   : "r"(argument)
                   : "memory");
  for (;;) __asm__ volatile("wfe");
}

#elif defined(__riscv)

static void put_byte(unsigned char byte) {
  volatile unsigned char *const uart_data =
      (volatile unsigned char *)0x10000000ul;
  volatile unsigned char *const uart_status =
      (volatile unsigned char *)0x10000005ul;
  while ((*uart_status & (1u << 5)) == 0) {}
  *uart_data = byte;
}

__attribute__((noreturn)) static void finish(int status) {
  volatile unsigned *const test = (volatile unsigned *)0x100000ul;
  *test = status == 0 ? 0x5555u : 0x3333u;
  for (;;) __asm__ volatile("wfi");
}

#else
#error unsupported target
#endif

static void write_text(const char *text) {
  while (*text != 0) put_byte((unsigned char)*text++);
}

__attribute__((noinline, noreturn)) static void jump_nonzero(void) {
  volatile usize scratch[64];
  for (usize index = 0; index < 64; ++index)
    scratch[index] = index ^ 0x5a5aul;
  longjmp(environment, 42);
}

__attribute__((noinline, noreturn)) static void jump_zero(void) {
  volatile usize scratch[64];
  for (usize index = 0; index < 64; ++index)
    scratch[index] = index ^ 0xa5a5ul;
  longjmp(zero_environment, 0);
}

__attribute__((noreturn)) void harness_start(void) {
  int value = setjmp(environment);
  if (value == 0) {
    phase = 1;
    jump_nonzero();
  }
  if (value != 42 || phase != 1) {
    write_text("setjmp-runtime: nonzero failed\n");
    finish(1);
  }

  value = setjmp(zero_environment);
  if (value == 0) {
    phase = 2;
    jump_zero();
  }
  if (value != 1 || phase != 2) {
    write_text("setjmp-runtime: zero failed\n");
    finish(1);
  }

  write_text("setjmp-runtime: ok\n");
  finish(0);
}
