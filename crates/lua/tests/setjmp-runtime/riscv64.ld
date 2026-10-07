ENTRY(_start)
SECTIONS {
    /* QEMU virt DRAM starts at 0x80000000; leave its reset payload clear. */
    . = 0x80200000;
    .text : { KEEP(*(.text.start)) *(.text .text.*) }
    .rodata : ALIGN(16) { *(.rodata .rodata.*) }
    .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) *(.got .got.*) }
    .bss : ALIGN(16) { *(.bss .bss.*) *(.sbss .sbss.*) COMMON }
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
