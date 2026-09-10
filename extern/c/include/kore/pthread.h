/* Generated lifecycle aliases; default attributes only, not a complete POSIX API. */
#ifndef KORE_PTHREAD_H
#define KORE_PTHREAD_H
#include "../kore.h"
typedef struct KoreThread *pthread_t;
typedef struct KoreThreadAttr pthread_attr_t;
#define pthread_create kore_thread_create
#define pthread_join kore_thread_join
#define pthread_detach kore_thread_detach
#define pthread_cancel kore_thread_cancel
#define pthread_key_create kore_thread_key_create
#define pthread_key_delete kore_thread_key_delete
#define pthread_setspecific kore_thread_setspecific
#define pthread_getspecific kore_thread_getspecific
#define pthread_setcancelstate kore_thread_setcancelstate
#define pthread_setcanceltype kore_thread_setcanceltype
#define pthread_testcancel kore_thread_testcancel
#define pthread_exit kore_thread_exit
typedef KoreThreadKey pthread_key_t;
#define PTHREAD_CANCEL_ENABLE KORE_THREAD_CANCEL_ENABLE
#define PTHREAD_CANCEL_DISABLE KORE_THREAD_CANCEL_DISABLE
#define PTHREAD_CANCEL_DEFERRED KORE_THREAD_CANCEL_DEFERRED
#define PTHREAD_CANCEL_ASYNCHRONOUS KORE_THREAD_CANCEL_ASYNCHRONOUS
#define PTHREAD_DESTRUCTOR_ITERATIONS KORE_THREAD_DESTRUCTOR_ITERATIONS
#define PTHREAD_KEYS_MAX KORE_THREAD_KEYS_MAX
#define PTHREAD_CANCELED (kore_thread_canceled())
#define pthread_cleanup_push(routine, argument) do { KoreThreadCleanup _kore_cleanup; kore_thread_cleanup_push(&_kore_cleanup, (routine), (argument));
#define pthread_cleanup_pop(execute) kore_thread_cleanup_pop(&_kore_cleanup, (execute)); } while (0)
#endif
