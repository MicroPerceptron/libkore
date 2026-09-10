#include "kore.h"
#include <assert.h>

struct arena { _Alignas(64) unsigned char bytes[32768]; size_t used, live, remaining; };
static void *allocate(void *context, size_t size, size_t align) {
    struct arena *a = context;
    if (!a->remaining) return NULL;
    uintptr_t base = (uintptr_t)a->bytes;
    uintptr_t address = (base + a->used + align - 1) & ~(uintptr_t)(align - 1);
    size_t offset = address - base;
    if (offset > sizeof a->bytes || size > sizeof a->bytes - offset) return NULL;
    a->used = offset + size;
    ++a->live; --a->remaining;
    return a->bytes + offset;
}
static void deallocate(void *context, void *ptr, size_t size, size_t align) {
    struct arena *a = context;
    assert((uintptr_t)ptr % align == 0);
    assert((unsigned char *)ptr >= a->bytes);
    assert((unsigned char *)ptr + size <= a->bytes + sizeof a->bytes);
    assert(a->live); --a->live;
}
struct record { uint64_t key, value; };
static int32_t equal(const void *entry, const void *query, void *context) {
    (void)context;
    return ((const struct record *)entry)->key == ((const struct record *)query)->key;
}
static int32_t visit(void *entry, void *context) {
    *(uint64_t *)context += ((struct record *)entry)->value;
    return 1;
}

void c_check_hash(void) {
    struct arena a = {.remaining = 1000};
    KoreHashAllocator allocator = {&a, allocate, deallocate};
    KoreHashTable *table = kore_hash_table_create(allocator, 2);
    assert(table);
    struct record entries[64];
    void *out = NULL;
    for (uint64_t i = 0; i < 64; ++i) {
        entries[i] = (struct record){i, i + 1};
        /* Identical hashes exercise equality and growth under collisions. */
        assert(kore_hash_table_insert(table, 0, &entries[i], equal, NULL, &out) == KORE_HASH_OK);
        assert(out == NULL);
    }
    struct record replacement = {4, 100};
    assert(kore_hash_table_insert(table, 0, &replacement, equal, NULL, &out) == KORE_HASH_REPLACED);
    assert(out == &entries[4]);
    a.remaining = 0;
    assert(kore_hash_table_reserve(table, 100000) == KORE_HASH_NO_MEMORY);
    assert(kore_hash_table_len(table) == 64);
    assert(kore_hash_table_find(table, 0, &entries[4], equal, NULL, &out) == KORE_HASH_OK);
    assert(out == &replacement);
    assert(kore_hash_table_remove(table, 0, &entries[7], equal, NULL, &out) == KORE_HASH_OK);
    assert(out == &entries[7]);
    assert(kore_hash_table_find(table, 0, &entries[7], equal, NULL, &out) == KORE_HASH_MISSING);
    assert(out == NULL);
    uint64_t sum = 0;
    assert(kore_hash_table_visit(table, visit, &sum) == KORE_HASH_OK);
    assert(sum == 64 * 65 / 2 - 5 + 100 - 8);
    assert(kore_hash_table_find(NULL, 0, NULL, equal, NULL, &out) == KORE_HASH_INVALID);
    assert(out == NULL);
    assert(kore_hash_table_insert(table, 0, NULL, equal, NULL, &out) == KORE_HASH_INVALID);
    assert(kore_hash_table_find(table, 0, NULL, NULL, NULL, &out) == KORE_HASH_INVALID);
    assert(kore_hash_table_find(table, 0, NULL, equal, NULL, NULL) == KORE_HASH_INVALID);
    kore_hash_table_clear(table);
    assert(kore_hash_table_len(table) == 0);
    /* Retained capacity works even when no further allocations are possible. */
    assert(kore_hash_table_insert(table, 0, &entries[0], equal, NULL, &out) == KORE_HASH_OK);
    kore_hash_table_destroy(table);
    assert(a.live == 0);
    assert(entries[0].value == 1 && replacement.value == 100);
    /* Growth must rehash using each entry's cached caller-supplied hash. */
    a.used = 0; a.remaining = 1000;
    table = kore_hash_table_create(allocator, 0);
    assert(table);
    for (uint64_t i = 0; i < 64; ++i)
        assert(kore_hash_table_insert(table, i * UINT64_C(0x9e3779b97f4a7c15),
            &entries[i], equal, NULL, &out) == KORE_HASH_OK);
    for (uint64_t i = 0; i < 64; ++i) {
        assert(kore_hash_table_find(table, i * UINT64_C(0x9e3779b97f4a7c15),
            &entries[i], equal, NULL, &out) == KORE_HASH_OK);
        assert(out == &entries[i]);
    }
    kore_hash_table_destroy(table);
    assert(a.live == 0);
    kore_hash_table_destroy(NULL);
    kore_hash_table_clear(NULL);
    assert(kore_hash_table_len(NULL) == 0);
    /* Fail the handle allocation after successfully allocating bucket storage. */
    a.used = 0; a.remaining = 1;
    assert(kore_hash_table_create(allocator, 8) == NULL);
    assert(a.live == 0);
    a.used = 0; a.remaining = 1;
    table = kore_hash_table_create(allocator, 0);
    assert(table);
    assert(kore_hash_table_insert(table, 0, &entries[0], equal, NULL, &out) == KORE_HASH_NO_MEMORY);
    assert(out == NULL && kore_hash_table_len(table) == 0);
    kore_hash_table_destroy(table);
    assert(a.live == 0);
    allocator.allocate = NULL;
    assert(kore_hash_table_create(allocator, 0) == NULL);

}
