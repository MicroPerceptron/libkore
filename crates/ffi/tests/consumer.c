#include "kore.h"

_Static_assert(sizeof(KoreStatus) == 4, "status ABI");
_Static_assert(sizeof(KoreFsHandle) == 16, "handle ABI");
_Static_assert(offsetof(KoreFsCompletion, count) == 32, "completion ABI");
_Static_assert(sizeof(KoreFsCompletion) == 80, "completion ABI");

uint32_t c_submit_write(KoreFsContext *ctx, KoreFsHandle handle) {
    uint8_t data[] = { 17, 23, 31, 47 };
    KoreStatus first = kore_fs_write(ctx, handle, 13, data, sizeof data);
    data[0] = 99; /* Rust must have copied this before the foreign stack returns. */
    if (first != KORE_STATUS_QUEUED) return 1;
    if (kore_fs_close(ctx, handle) != KORE_STATUS_BUSY) return 2;
    return 0;
}

uint32_t c_read_completion(KoreFsContext *ctx, const KoreFsCompletion *completion) {
    uint8_t data[4] = { 0 };
    if (completion->abi_version != KORE_ABI_VERSION || completion->operation != KORE_OP_READ ||
        completion->status != KORE_STATUS_OK || completion->count != 4) return 1;
    if (kore_fs_read_data(ctx, data, 3) != KORE_STATUS_BUFFER_TOO_SMALL) return 2;
    if (kore_fs_read_data(ctx, data, 4) != KORE_STATUS_OK) return 3;
    return data[0] == 17 && data[3] == 47 ? 0 : 4;
}

_Static_assert(sizeof(KoreTaskAction) == 16, "task action ABI");
_Static_assert(sizeof(KoreTaskHandle) == 16, "task handle ABI");
_Static_assert(sizeof(KoreDispatch) == 8, "dispatch ABI");
KoreTaskAction c_task_step(void *user) {
    uint32_t *turns = user;
    *turns += 1;
    return (KoreTaskAction) { .operation = *turns == 1 ? KORE_TASK_YIELD : KORE_TASK_DONE };
}
uint32_t c_prepare_task(KoreTaskBatch *batch, uint32_t *turns, KoreTaskHandle *handle) {
    return kore_task_prepare(batch, c_task_step, turns, handle);
}
uint32_t c_check_cancelled(KoreTaskHandle handle) {
    if (kore_task_poll(handle) != KORE_TASK_STATE_CANCELLED) return 1;
    if (kore_task_release(handle) != KORE_STATUS_OK) return 2;
    return kore_task_poll(handle) == KORE_TASK_STATE_STALE ? 0 : 3;
}
uint32_t c_sleep_and_random(KoreFsContext *ctx) {
    uint64_t output = 77;
    if (kore_random_fill(NULL, 0) != KORE_RANDOM_STATUS_OK) return 1;
    if (kore_random_range(9, 2, &output) != KORE_RANDOM_STATUS_INVALID_RANGE || output != 77) return 2;
    if (kore_time_sleep_for(ctx, 1250000) != KORE_STATUS_QUEUED) return 3;
    return kore_time_yield(ctx) == KORE_STATUS_BUSY ? 0 : 4;
}

#include "kore/pthread.h"
static void *c_thread_identity(void *argument) { return argument; }
uint32_t c_check_thread_refusal(void) {
    pthread_t thread = (pthread_t)(uintptr_t)16;
    if (pthread_create(&thread, NULL, c_thread_identity, NULL) != KORE_THREAD_EAGAIN) return 1;
    if (thread != (pthread_t)(uintptr_t)16) return 2;
    if (pthread_join(NULL, NULL) != KORE_THREAD_EINVAL) return 3;
    return pthread_detach(NULL) == KORE_THREAD_EINVAL ? 0 : 4;
}
