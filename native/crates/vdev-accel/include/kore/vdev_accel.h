/* <kore/vdev_accel.h>: the virtual accelerator, for a driver in C.
 *
 * The accelerator is a virtio-accel 1.0 device with no hardware behind it
 * (`/kore/device/virtual/accel`, <kore/program.h>). A driver brings it up
 * over its register block and then speaks the protocol to it: one command
 * is one chain of its virtqueue, a request the device reads and a response
 * it writes, each a header and a payload.
 *
 * It computes six functions of one FP8 (E4M3) value, elementwise over a
 * buffer. A session:
 *
 *   CREATE_CONTEXT; ALLOCATE_BUFFER twice (operands, results);
 *   WRITE_BUFFER the operands; CREATE_QUEUE;
 *   LOAD_PROGRAM naming a kernel; SUBMIT binding the operands to slot 0
 *   for reading and the results to slot 1 for writing;
 *   POLL_EVENT until the event is complete; DESTROY_EVENT; READ_BUFFER.
 *
 * A buffer bound to a submission is the event's until the event is
 * destroyed, so results are read after DESTROY_EVENT. A reset of the
 * device gives every object a new namespace: an id from before it names
 * nothing after it.
 *
 * Every word is little-endian, and no frame has padding: the structures
 * are packed, and each asserts its size and its fields' offsets.
 *
 * Generated from the protocol's own definitions by kore-vdev-accel's
 * `c_header` test. Do not edit. */
#ifndef KORE_VDEV_ACCEL_H
#define KORE_VDEV_ACCEL_H

#include <stddef.h>
#include <stdint.h>

/* The protocol, and the device's name in the register block. */
#define KORE_ACCEL_PROTOCOL_MAJOR 1
#define KORE_ACCEL_PROTOCOL_MINOR 0
#define KORE_ACCEL_DEVICE_ID 0x6c636361u /* "accl" */

/* Commands: a request's `opcode`. */
#define KORE_ACCEL_OP_GET_DEVICE_INFO 0x0001
#define KORE_ACCEL_OP_CREATE_CONTEXT 0x0100
#define KORE_ACCEL_OP_DESTROY_CONTEXT 0x0101
#define KORE_ACCEL_OP_ALLOCATE_BUFFER 0x0200
#define KORE_ACCEL_OP_FREE_BUFFER 0x0201
#define KORE_ACCEL_OP_WRITE_BUFFER 0x0202
#define KORE_ACCEL_OP_READ_BUFFER 0x0203
#define KORE_ACCEL_OP_LOAD_PROGRAM 0x0300
#define KORE_ACCEL_OP_UNLOAD_PROGRAM 0x0301
#define KORE_ACCEL_OP_CREATE_QUEUE 0x0400
#define KORE_ACCEL_OP_DESTROY_QUEUE 0x0401
#define KORE_ACCEL_OP_SUBMIT 0x0500
#define KORE_ACCEL_OP_POLL_EVENT 0x0501
#define KORE_ACCEL_OP_CANCEL_EVENT 0x0502
#define KORE_ACCEL_OP_DESTROY_EVENT 0x0503

/* A response's `status`. */
#define KORE_ACCEL_STATUS_OK 0
#define KORE_ACCEL_STATUS_UNSUPPORTED 1
#define KORE_ACCEL_STATUS_INCOMPATIBLE 2
#define KORE_ACCEL_STATUS_INVALID_ARGUMENT 3
#define KORE_ACCEL_STATUS_OUT_OF_BOUNDS 4
#define KORE_ACCEL_STATUS_BUSY 5
#define KORE_ACCEL_STATUS_OUT_OF_MEMORY 6
#define KORE_ACCEL_STATUS_RESOURCE_LIMIT 7
#define KORE_ACCEL_STATUS_DEADLINE_EXPIRED 8
#define KORE_ACCEL_STATUS_DEVICE_LOST 9
#define KORE_ACCEL_STATUS_PERMISSION_DENIED 10
#define KORE_ACCEL_STATUS_STALE_OBJECT 11
#define KORE_ACCEL_STATUS_INTERNAL_ERROR 0xffff

/* An event's `state`. */
#define KORE_ACCEL_EVENT_PENDING 0
#define KORE_ACCEL_EVENT_COMPLETE 1
#define KORE_ACCEL_EVENT_FAILED 2
#define KORE_ACCEL_EVENT_CANCELLED 3

/* A buffer's `usage` bits, its `memory_domain`, and a binding's `access`. */
#define KORE_ACCEL_BUFFER_TRANSFER_SOURCE 0x1u /* READ_BUFFER may read it */
#define KORE_ACCEL_BUFFER_TRANSFER_DESTINATION 0x2u /* WRITE_BUFFER may write it */
#define KORE_ACCEL_BUFFER_PROGRAM_INPUT 0x4u /* a program may read it */
#define KORE_ACCEL_BUFFER_PROGRAM_OUTPUT 0x8u /* a program may write it */
#define KORE_ACCEL_BUFFER_MUTABLE_STATE 0x10u
#define KORE_ACCEL_MEMORY_HOST 1
#define KORE_ACCEL_ACCESS_READ 1
#define KORE_ACCEL_ACCESS_WRITE 2
#define KORE_ACCEL_ACCESS_READ_WRITE 3

/* A program of this device is a registered kernel: LOAD_PROGRAM with this
 * format, this identity in `target[0]`, a kernel in `target[1]`, this
 * artifact as the bytes after the request, and this resident charge. */
#define KORE_ACCEL_PROGRAM_FORMAT 0x4d50534bu /* "KSPM" */
#define KORE_ACCEL_PROGRAM_IDENTITY 0x4d50534bu /* "KSPM" */
#define KORE_ACCEL_PROGRAM_ARTIFACT_BYTES 8
#define KORE_ACCEL_PROGRAM_ARTIFACT { 0x4b, 0x53, 0x50, 0x4d, 0x01, 0x00, 0x00, 0x00 } /* a uint8_t array's initializer */
#define KORE_ACCEL_PROGRAM_RESIDENT_BYTES 64

/* The kernels. Each reads slot 0 and writes slot 1, one E4M3 value a byte,
 * and is defined by a 256-entry table: rounded to nearest, ties to even;
 * saturating at +-448 (0x7e, 0xfe); a NaN operand or a result that is not
 * real is the canonical NaN 0x7f. */
#define KORE_ACCEL_KERNEL_SIN 0x0100u
#define KORE_ACCEL_KERNEL_COS 0x0101u
#define KORE_ACCEL_KERNEL_TAN 0x0102u
#define KORE_ACCEL_KERNEL_EXP 0x0103u
#define KORE_ACCEL_KERNEL_LOG 0x0104u
#define KORE_ACCEL_KERNEL_SQRT 0x0105u

/* The device configuration, at the register block's `config`. */
struct kore_accel_config {
    uint16_t protocol_major; /* 1 */
    uint16_t protocol_minor; /* 0 */
    uint16_t command_queue_count; /* 1 */
    uint16_t max_chain_descriptors; /* the longest chain, and the smallest queue */
    uint32_t max_request_bytes; /* the largest request frame, header included */
    uint32_t max_response_bytes; /* the largest response frame, header included */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_config) == 16, "config");
_Static_assert(offsetof(struct kore_accel_config, protocol_major) == 0, "config.protocol_major");
_Static_assert(offsetof(struct kore_accel_config, protocol_minor) == 2, "config.protocol_minor");
_Static_assert(offsetof(struct kore_accel_config, command_queue_count) == 4, "config.command_queue_count");
_Static_assert(offsetof(struct kore_accel_config, max_chain_descriptors) == 6, "config.max_chain_descriptors");
_Static_assert(offsetof(struct kore_accel_config, max_request_bytes) == 8, "config.max_request_bytes");
_Static_assert(offsetof(struct kore_accel_config, max_response_bytes) == 12, "config.max_response_bytes");

/* The head of every request. Its payload follows it. */
struct kore_accel_request {
    uint16_t opcode; /* KORE_ACCEL_OP_* */
    uint16_t flags; /* 0 */
    uint32_t payload_bytes; /* bytes after this header */
    uint64_t request_id; /* the driver's own; the response repeats it */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_request) == 16, "request");
_Static_assert(offsetof(struct kore_accel_request, opcode) == 0, "request.opcode");
_Static_assert(offsetof(struct kore_accel_request, flags) == 2, "request.flags");
_Static_assert(offsetof(struct kore_accel_request, payload_bytes) == 4, "request.payload_bytes");
_Static_assert(offsetof(struct kore_accel_request, request_id) == 8, "request.request_id");

/* The head of every response. Its payload follows it. */
struct kore_accel_response {
    uint16_t status; /* KORE_ACCEL_STATUS_* */
    uint16_t flags; /* 0 */
    uint32_t payload_bytes; /* bytes after this header */
    uint64_t request_id; /* the request's */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_response) == 16, "response");
_Static_assert(offsetof(struct kore_accel_response, status) == 0, "response.status");
_Static_assert(offsetof(struct kore_accel_response, flags) == 2, "response.flags");
_Static_assert(offsetof(struct kore_accel_response, payload_bytes) == 4, "response.payload_bytes");
_Static_assert(offsetof(struct kore_accel_response, request_id) == 8, "response.request_id");

/* GET_DEVICE_INFO's response payload. */
struct kore_accel_device_info {
    uint8_t uuid[16]; /* the backend's name */
    uint16_t class; /* what kind of accelerator */
    uint16_t reserved; /* 0 */
    uint32_t vendor_id;
    uint32_t device_id;
    uint64_t capabilities;
    uint32_t max_contexts;
    uint32_t max_buffers_per_context;
    uint32_t max_programs_per_context;
    uint32_t max_queues_per_context;
    uint32_t max_events_per_context;
    uint32_t max_bindings_per_submission;
    uint64_t max_buffer_bytes;
    uint64_t max_artifact_bytes;
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_device_info) == 76, "device_info");
_Static_assert(offsetof(struct kore_accel_device_info, uuid) == 0, "device_info.uuid");
_Static_assert(offsetof(struct kore_accel_device_info, class) == 16, "device_info.class");
_Static_assert(offsetof(struct kore_accel_device_info, reserved) == 18, "device_info.reserved");
_Static_assert(offsetof(struct kore_accel_device_info, vendor_id) == 20, "device_info.vendor_id");
_Static_assert(offsetof(struct kore_accel_device_info, device_id) == 24, "device_info.device_id");
_Static_assert(offsetof(struct kore_accel_device_info, capabilities) == 28, "device_info.capabilities");
_Static_assert(offsetof(struct kore_accel_device_info, max_contexts) == 36, "device_info.max_contexts");
_Static_assert(offsetof(struct kore_accel_device_info, max_buffers_per_context) == 40, "device_info.max_buffers_per_context");
_Static_assert(offsetof(struct kore_accel_device_info, max_programs_per_context) == 44, "device_info.max_programs_per_context");
_Static_assert(offsetof(struct kore_accel_device_info, max_queues_per_context) == 48, "device_info.max_queues_per_context");
_Static_assert(offsetof(struct kore_accel_device_info, max_events_per_context) == 52, "device_info.max_events_per_context");
_Static_assert(offsetof(struct kore_accel_device_info, max_bindings_per_submission) == 56, "device_info.max_bindings_per_submission");
_Static_assert(offsetof(struct kore_accel_device_info, max_buffer_bytes) == 60, "device_info.max_buffer_bytes");
_Static_assert(offsetof(struct kore_accel_device_info, max_artifact_bytes) == 68, "device_info.max_artifact_bytes");

/* CREATE_CONTEXT's payload. The response is an object: the context. */
struct kore_accel_create_context {
    uint32_t flags; /* 0 */
    uint32_t reserved; /* 0 */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_create_context) == 8, "create_context");
_Static_assert(offsetof(struct kore_accel_create_context, flags) == 0, "create_context.flags");
_Static_assert(offsetof(struct kore_accel_create_context, reserved) == 4, "create_context.reserved");

/* One object's id: what creating answers and destroying takes. */
struct kore_accel_object {
    uint64_t object_id;
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_object) == 8, "object");
_Static_assert(offsetof(struct kore_accel_object, object_id) == 0, "object.object_id");

/* ALLOCATE_BUFFER's payload. The response is an object: the buffer. */
struct kore_accel_allocate_buffer {
    uint64_t context_id;
    uint64_t bytes;
    uint64_t alignment; /* a power of two; 1 for bytes */
    uint8_t memory_domain; /* KORE_ACCEL_MEMORY_* */
    uint8_t reserved0[7]; /* 0 */
    uint32_t usage; /* KORE_ACCEL_BUFFER_* bits */
    uint32_t reserved1; /* 0 */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_allocate_buffer) == 40, "allocate_buffer");
_Static_assert(offsetof(struct kore_accel_allocate_buffer, context_id) == 0, "allocate_buffer.context_id");
_Static_assert(offsetof(struct kore_accel_allocate_buffer, bytes) == 8, "allocate_buffer.bytes");
_Static_assert(offsetof(struct kore_accel_allocate_buffer, alignment) == 16, "allocate_buffer.alignment");
_Static_assert(offsetof(struct kore_accel_allocate_buffer, memory_domain) == 24, "allocate_buffer.memory_domain");
_Static_assert(offsetof(struct kore_accel_allocate_buffer, reserved0) == 25, "allocate_buffer.reserved0");
_Static_assert(offsetof(struct kore_accel_allocate_buffer, usage) == 32, "allocate_buffer.usage");
_Static_assert(offsetof(struct kore_accel_allocate_buffer, reserved1) == 36, "allocate_buffer.reserved1");

/* WRITE_BUFFER's payload, with the bytes after it, and READ_BUFFER's, whose response is the bytes. */
struct kore_accel_transfer_buffer {
    uint64_t buffer_id;
    uint64_t offset;
    uint64_t bytes;
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_transfer_buffer) == 24, "transfer_buffer");
_Static_assert(offsetof(struct kore_accel_transfer_buffer, buffer_id) == 0, "transfer_buffer.buffer_id");
_Static_assert(offsetof(struct kore_accel_transfer_buffer, offset) == 8, "transfer_buffer.offset");
_Static_assert(offsetof(struct kore_accel_transfer_buffer, bytes) == 16, "transfer_buffer.bytes");

/* LOAD_PROGRAM's payload, with the artifact after it. The response is an object: the program. */
struct kore_accel_load_program {
    uint64_t context_id;
    uint32_t format; /* KORE_ACCEL_PROGRAM_FORMAT */
    uint32_t flags; /* 0 */
    uint32_t target[12]; /* [0] KORE_ACCEL_PROGRAM_IDENTITY, [1] a kernel, the rest 0 */
    uint64_t payload_bytes; /* the artifact's bytes */
    uint64_t resident_bytes; /* KORE_ACCEL_PROGRAM_RESIDENT_BYTES */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_load_program) == 80, "load_program");
_Static_assert(offsetof(struct kore_accel_load_program, context_id) == 0, "load_program.context_id");
_Static_assert(offsetof(struct kore_accel_load_program, format) == 8, "load_program.format");
_Static_assert(offsetof(struct kore_accel_load_program, flags) == 12, "load_program.flags");
_Static_assert(offsetof(struct kore_accel_load_program, target) == 16, "load_program.target");
_Static_assert(offsetof(struct kore_accel_load_program, payload_bytes) == 64, "load_program.payload_bytes");
_Static_assert(offsetof(struct kore_accel_load_program, resident_bytes) == 72, "load_program.resident_bytes");

/* CREATE_QUEUE's payload. The response is an object: the execution queue. */
struct kore_accel_create_queue {
    uint64_t context_id;
    uint32_t flags; /* 0 */
    uint32_t reserved; /* 0 */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_create_queue) == 16, "create_queue");
_Static_assert(offsetof(struct kore_accel_create_queue, context_id) == 0, "create_queue.context_id");
_Static_assert(offsetof(struct kore_accel_create_queue, flags) == 8, "create_queue.flags");
_Static_assert(offsetof(struct kore_accel_create_queue, reserved) == 12, "create_queue.reserved");

/* SUBMIT's payload, with `binding_count` bindings after it. */
struct kore_accel_submit {
    uint64_t queue_id;
    uint64_t program_id;
    uint32_t binding_count;
    uint32_t flags; /* 0 */
    uint64_t timeout_ns; /* from admission; 0 is none */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_submit) == 32, "submit");
_Static_assert(offsetof(struct kore_accel_submit, queue_id) == 0, "submit.queue_id");
_Static_assert(offsetof(struct kore_accel_submit, program_id) == 8, "submit.program_id");
_Static_assert(offsetof(struct kore_accel_submit, binding_count) == 16, "submit.binding_count");
_Static_assert(offsetof(struct kore_accel_submit, flags) == 20, "submit.flags");
_Static_assert(offsetof(struct kore_accel_submit, timeout_ns) == 24, "submit.timeout_ns");

/* One buffer range a submission's program reads or writes. */
struct kore_accel_binding {
    uint64_t buffer_id;
    uint64_t offset;
    uint64_t bytes;
    uint32_t slot; /* which of the program's operands */
    uint8_t access; /* KORE_ACCEL_ACCESS_* */
    uint8_t reserved[3]; /* 0 */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_binding) == 32, "binding");
_Static_assert(offsetof(struct kore_accel_binding, buffer_id) == 0, "binding.buffer_id");
_Static_assert(offsetof(struct kore_accel_binding, offset) == 8, "binding.offset");
_Static_assert(offsetof(struct kore_accel_binding, bytes) == 16, "binding.bytes");
_Static_assert(offsetof(struct kore_accel_binding, slot) == 24, "binding.slot");
_Static_assert(offsetof(struct kore_accel_binding, access) == 28, "binding.access");
_Static_assert(offsetof(struct kore_accel_binding, reserved) == 29, "binding.reserved");

/* SUBMIT's response payload: the event the submission ends as. */
struct kore_accel_submit_response {
    uint64_t event_id;
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_submit_response) == 8, "submit_response");
_Static_assert(offsetof(struct kore_accel_submit_response, event_id) == 0, "submit_response.event_id");

/* POLL_EVENT's response payload. */
struct kore_accel_event_state {
    uint16_t state; /* KORE_ACCEL_EVENT_* */
    uint16_t error; /* a KORE_ACCEL_STATUS_* when the state is FAILED */
    uint32_t reserved; /* 0 */
} __attribute__((packed));
_Static_assert(sizeof(struct kore_accel_event_state) == 8, "event_state");
_Static_assert(offsetof(struct kore_accel_event_state, state) == 0, "event_state.state");
_Static_assert(offsetof(struct kore_accel_event_state, error) == 2, "event_state.error");
_Static_assert(offsetof(struct kore_accel_event_state, reserved) == 4, "event_state.reserved");

#endif /* KORE_VDEV_ACCEL_H */
