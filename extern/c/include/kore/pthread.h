/* Generated lifecycle aliases; default attributes only, not a complete POSIX API. */
#ifndef KORE_PTHREAD_H
#define KORE_PTHREAD_H
#include "../kore.h"
typedef struct KoreThread *pthread_t;
typedef struct KoreThreadAttr pthread_attr_t;
#define pthread_create kore_thread_create
#define pthread_join kore_thread_join
#define pthread_detach kore_thread_detach
#endif
