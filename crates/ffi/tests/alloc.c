#include "kore/stdlib.h"

uint32_t c_check_alloc(void) {
    if (calloc(SIZE_MAX, 2) != NULL) return 1;
    if (malloc(SIZE_MAX) != NULL) return 2;
    free(NULL);
    unsigned char *p = calloc(17, 3);
    if (!p || (uintptr_t)p % KORE_ALLOC_MIN_ALIGNMENT) return 3;
    for (size_t i = 0; i < 51; ++i) {
        if (p[i] != 0) return 4;
        p[i] = (unsigned char)i;
    }
    if (realloc(p, SIZE_MAX) != NULL || p[50] != 50) return 5;
    unsigned char *q = realloc(p, 300);
    if (!q) { free(p); return 6; }
    for (size_t i = 0; i < 51; ++i) if (q[i] != i) return 7;
    free(q);
    p = kore_alloc_aligned(33, 4096);
    if (!p || (uintptr_t)p % 4096) return 8;
    p[32] = 73;
    q = realloc(p, 100);
    if (!q) { free(p); return 9; }
    if ((uintptr_t)q % 4096 || q[32] != 73) return 10;
    if (realloc(q, 0) != NULL) return 11;
    p = malloc(0);
    if (!p) return 12;
    free(p);
    KoreHashTable *table = kore_hash_table_create_default(4);
    if (!table) return 13;
    if (kore_hash_table_reserve(table, 100) != KORE_HASH_OK) return 14;
    kore_hash_table_destroy(table);
    return 0;
}
