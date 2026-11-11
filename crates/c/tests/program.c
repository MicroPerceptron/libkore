/* kore/program.h as an independent C program uses it: a session whose
 * callbacks are kore.h's own KoreFsStep, run through kore_fs_run, and a
 * descriptor described through kore_fd_describe. Compiled
 * freestanding for every Kore ISA by tools/check-bindings.py. */
#include <kore/program.h>

struct reader {
    int done;
};

static void open_root(struct KoreFsContext *context, const struct KoreFsCompletion *completion,
                      void *user)
{
    struct reader *reader = user;
    if (completion->operation == KORE_OP_START) {
        (void)kore_fs_open(context, (const uint8_t *)"", 0, KORE_FS_READ);
        return;
    }
    reader->done = 1;
}

/* The entry's exact type, so a drift in either header fails the build. */
static KoreStatus (*const run)(KoreFsStep, void *, KoreFsError *) = kore_fs_run;

int program_session(void)
{
    struct reader reader = {0};
    KoreFsError error = KORE_FS_ERROR_NONE;
    KoreStatus status = run(open_root, &reader, &error);
    return status == KORE_STATUS_OK && error == KORE_FS_ERROR_NONE && reader.done;
}

/* A descriptor's description: the exact entry type, its fields and the
 * constants its fields take. */
static KoreStatus (*const describe)(int, KoreFdInfo *) = kore_fd_describe;

int program_memory(int fd)
{
    KoreFdInfo info = {0};
    if (describe(fd, &info) != KORE_STATUS_OK)
        return 0;
    return info.kind == KORE_FD_MEMORY && (info.access & (KORE_FD_READ | KORE_FD_WRITE)) != 0 &&
           info.length != 0;
}
