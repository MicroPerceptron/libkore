.section .text.start,"ax",@progbits
.global _start
.type _start,@function
_start:
    /* The generic loader enters in M-mode with floating-point state off. */
    csrr t0, mstatus
    li t1, 0x6000
    or t0, t0, t1
    csrw mstatus, t0
    la sp, __stack_top
    call harness_start
1:
    wfi
    j 1b

.section .bss.stack,"aw",@nobits
.balign 16
__stack_bottom:
    .space 16384
__stack_top:
