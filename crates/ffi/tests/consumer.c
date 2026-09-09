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
