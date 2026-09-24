.section .text.start,"ax",%progbits
.global _start
.type _start,%function
_start:
    /* QEMU's generic loader enters at EL1 with FP/SIMD disabled. */
    mrs x1, cpacr_el1
    mov x2, #0x300000
    orr x1, x1, x2
    msr cpacr_el1, x1
    isb
    ldr x0, =__stack_top
    mov sp, x0
    bl harness_start
1:
    wfe
    b 1b

.section .bss.stack,"aw",%nobits
.balign 16
__stack_bottom:
    .space 16384
__stack_top:
