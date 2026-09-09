/* Generated allocation aliases. Include instead of host stdlib.h. */
#ifndef KORE_STDLIB_H
#define KORE_STDLIB_H
#include "../kore.h"
#define malloc kore_alloc_malloc
#define calloc kore_alloc_calloc
#define realloc kore_alloc_realloc
#define free kore_alloc_free
#endif
