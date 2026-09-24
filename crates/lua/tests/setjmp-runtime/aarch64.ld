ENTRY(_start)
SECTIONS {
    /* Outside the virt DTB reservation, inside its first RAM bank. */
    . = 0x41000000;
    .text : { KEEP(*(.text.start)) *(.text .text.*) }
    .rodata : ALIGN(16) { *(.rodata .rodata.*) }
    .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) *(.got .got.*) }
    .bss : ALIGN(16) { *(.bss .bss.*) *(.sbss .sbss.*) COMMON }
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
