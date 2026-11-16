//! `<kore/vdev_accel.h>`: what a C driver of the virtual accelerator needs,
//! written from the Rust that defines it.
//!
//! The header is not typed by hand. This test renders it from the
//! protocol's own types (`virtio-accel-proto`'s frames, with each field's
//! offset taken by `offset_of!`), the protocol's enumerations and this
//! device's registry, and requires the committed file to be exactly that.
//! A frame that moves, a status that is renumbered or a kernel that is added
//! fails here until the header is rendered again:
//!
//! ```text
//! KORE_BLESS=1 cargo test -p kore-vdev-accel --test c_header
//! ```
//!
//! The header also carries its own check, a `_Static_assert` for every
//! frame's size and every field's offset, so each compiler that reads it
//! proves its layout is the protocol's. A second test hands the header to
//! the host's C compiler for that proof.

use core::mem::{offset_of, size_of};
use std::fmt::Write as _;
use std::path::PathBuf;

use kore_svc_accel_spmd::{FORMAT_REGISTERED, IDENTITY_MAGIC, REGISTERED_MANIFEST};
use kore_vdev_accel::device::{DEVICE_ID, PROGRAM_RESIDENT_BYTES};
use kore_vdev_accel::fp8::Op;
use virtio_accel_core::{AccessMode, BufferUsage, MemoryDomain};
use virtio_accel_proto::{
    AllocateBufferRequest, CreateContextRequest, CreateQueueRequest, KnownEventState, KnownOpcode,
    LoadProgramRequest, ObjectPayload, PROTOCOL_MAJOR, PROTOCOL_MINOR, RequestHeader,
    ResponseHeader, StatusCode, SubmitRequest, SubmitResponse, TransferBufferRequest, WireBinding,
    WireConfig, WireDeviceInfo, WireEventState,
};

/// A field's C type.
#[derive(Clone, Copy)]
enum C {
    U8,
    U16,
    U32,
    U64,
    /// `uint8_t name[n]`.
    Bytes(usize),
    /// `uint32_t name[n]`.
    Words(usize),
}

impl C {
    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::Bytes(n) => n,
            Self::Words(n) => 4 * n,
        }
    }

    fn declare(self, name: &str) -> String {
        match self {
            Self::U8 => format!("uint8_t {name};"),
            Self::U16 => format!("uint16_t {name};"),
            Self::U32 => format!("uint32_t {name};"),
            Self::U64 => format!("uint64_t {name};"),
            Self::Bytes(n) => format!("uint8_t {name}[{n}];"),
            Self::Words(n) => format!("uint32_t {name}[{n}];"),
        }
    }
}

struct Field {
    name: &'static str,
    ty: C,
    offset: usize,
    note: &'static str,
}

struct Frame {
    name: &'static str,
    about: &'static str,
    bytes: usize,
    fields: Vec<Field>,
}

/// One frame: its C name, what it is, and each field with its C type. The
/// offsets and the size are the Rust type's.
macro_rules! frame {
    ($ty:ty as $name:literal, $about:literal, { $($field:ident: $c:expr, $note:literal;)* }) => {
        Frame {
            name: $name,
            about: $about,
            bytes: size_of::<$ty>(),
            fields: vec![$(Field {
                name: stringify!($field),
                ty: $c,
                offset: offset_of!($ty, $field),
                note: $note,
            }),*],
        }
    };
}

fn frames() -> Vec<Frame> {
    vec![
        frame!(WireConfig as "config",
            "The device configuration, at the register block's `config`.", {
            protocol_major: C::U16, "1";
            protocol_minor: C::U16, "0";
            command_queue_count: C::U16, "1";
            max_chain_descriptors: C::U16, "the longest chain, and the smallest queue";
            max_request_bytes: C::U32, "the largest request frame, header included";
            max_response_bytes: C::U32, "the largest response frame, header included";
        }),
        frame!(RequestHeader as "request",
            "The head of every request. Its payload follows it.", {
            opcode: C::U16, "KORE_ACCEL_OP_*";
            flags: C::U16, "0";
            payload_bytes: C::U32, "bytes after this header";
            request_id: C::U64, "the driver's own; the response repeats it";
        }),
        frame!(ResponseHeader as "response",
            "The head of every response. Its payload follows it.", {
            status: C::U16, "KORE_ACCEL_STATUS_*";
            flags: C::U16, "0";
            payload_bytes: C::U32, "bytes after this header";
            request_id: C::U64, "the request's";
        }),
        frame!(WireDeviceInfo as "device_info",
            "GET_DEVICE_INFO's response payload.", {
            uuid: C::Bytes(16), "the backend's name";
            class: C::U16, "what kind of accelerator";
            reserved: C::U16, "0";
            vendor_id: C::U32, "";
            device_id: C::U32, "";
            capabilities: C::U64, "";
            max_contexts: C::U32, "";
            max_buffers_per_context: C::U32, "";
            max_programs_per_context: C::U32, "";
            max_queues_per_context: C::U32, "";
            max_events_per_context: C::U32, "";
            max_bindings_per_submission: C::U32, "";
            max_buffer_bytes: C::U64, "";
            max_artifact_bytes: C::U64, "";
        }),
        frame!(CreateContextRequest as "create_context",
            "CREATE_CONTEXT's payload. The response is an object: the context.", {
            flags: C::U32, "0";
            reserved: C::U32, "0";
        }),
        frame!(ObjectPayload as "object",
            "One object's id: what creating answers and destroying takes.", {
            object_id: C::U64, "";
        }),
        frame!(AllocateBufferRequest as "allocate_buffer",
            "ALLOCATE_BUFFER's payload. The response is an object: the buffer.", {
            context_id: C::U64, "";
            bytes: C::U64, "";
            alignment: C::U64, "a power of two; 1 for bytes";
            memory_domain: C::U8, "KORE_ACCEL_MEMORY_*";
            reserved0: C::Bytes(7), "0";
            usage: C::U32, "KORE_ACCEL_BUFFER_* bits";
            reserved1: C::U32, "0";
        }),
        frame!(TransferBufferRequest as "transfer_buffer",
            "WRITE_BUFFER's payload, with the bytes after it, and READ_BUFFER's, whose response is the bytes.", {
            buffer_id: C::U64, "";
            offset: C::U64, "";
            bytes: C::U64, "";
        }),
        frame!(LoadProgramRequest as "load_program",
            "LOAD_PROGRAM's payload, with the artifact after it. The response is an object: the program.", {
            context_id: C::U64, "";
            format: C::U32, "KORE_ACCEL_PROGRAM_FORMAT";
            flags: C::U32, "0";
            target: C::Words(12), "[0] KORE_ACCEL_PROGRAM_IDENTITY, [1] a kernel, the rest 0";
            payload_bytes: C::U64, "the artifact's bytes";
            resident_bytes: C::U64, "KORE_ACCEL_PROGRAM_RESIDENT_BYTES";
        }),
        frame!(CreateQueueRequest as "create_queue",
            "CREATE_QUEUE's payload. The response is an object: the execution queue.", {
            context_id: C::U64, "";
            flags: C::U32, "0";
            reserved: C::U32, "0";
        }),
        frame!(SubmitRequest as "submit",
            "SUBMIT's payload, with `binding_count` bindings after it.", {
            queue_id: C::U64, "";
            program_id: C::U64, "";
            binding_count: C::U32, "";
            flags: C::U32, "0";
            timeout_ns: C::U64, "from admission; 0 is none";
        }),
        frame!(WireBinding as "binding",
            "One buffer range a submission's program reads or writes.", {
            buffer_id: C::U64, "";
            offset: C::U64, "";
            bytes: C::U64, "";
            slot: C::U32, "which of the program's operands";
            access: C::U8, "KORE_ACCEL_ACCESS_*";
            reserved: C::Bytes(3), "0";
        }),
        frame!(SubmitResponse as "submit_response",
            "SUBMIT's response payload: the event the submission ends as.", {
            event_id: C::U64, "";
        }),
        frame!(WireEventState as "event_state",
            "POLL_EVENT's response payload.", {
            state: C::U16, "KORE_ACCEL_EVENT_*";
            error: C::U16, "a KORE_ACCEL_STATUS_* when the state is FAILED";
            reserved: C::U32, "0";
        }),
    ]
}

fn define(out: &mut String, name: &str, value: impl core::fmt::Display, note: &str) {
    let define = format!("#define KORE_ACCEL_{name} {value}");
    if note.is_empty() {
        writeln!(out, "{define}").unwrap();
    } else {
        writeln!(out, "{define} /* {note} */").unwrap();
    }
}

/// A four-character code as a C unsigned literal, with its spelling.
fn fourcc(code: u32) -> (String, String) {
    let spelled: String = code.to_le_bytes().iter().map(|&b| b as char).collect();
    (format!("0x{code:08x}u"), format!("\"{spelled}\""))
}

const BANNER: &str = r#"/* <kore/vdev_accel.h>: the virtual accelerator, for a driver in C.
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

"#;

fn render() -> String {
    let mut out = String::from(BANNER);

    out.push_str("/* The protocol, and the device's name in the register block. */\n");
    define(&mut out, "PROTOCOL_MAJOR", PROTOCOL_MAJOR, "");
    define(&mut out, "PROTOCOL_MINOR", PROTOCOL_MINOR, "");
    let (id, spelled) = fourcc(DEVICE_ID);
    define(&mut out, "DEVICE_ID", id, &spelled);

    out.push_str("\n/* Commands: a request's `opcode`. */\n");
    for (name, opcode) in [
        ("GET_DEVICE_INFO", KnownOpcode::GetDeviceInfo),
        ("CREATE_CONTEXT", KnownOpcode::CreateContext),
        ("DESTROY_CONTEXT", KnownOpcode::DestroyContext),
        ("ALLOCATE_BUFFER", KnownOpcode::AllocateBuffer),
        ("FREE_BUFFER", KnownOpcode::FreeBuffer),
        ("WRITE_BUFFER", KnownOpcode::WriteBuffer),
        ("READ_BUFFER", KnownOpcode::ReadBuffer),
        ("LOAD_PROGRAM", KnownOpcode::LoadProgram),
        ("UNLOAD_PROGRAM", KnownOpcode::UnloadProgram),
        ("CREATE_QUEUE", KnownOpcode::CreateQueue),
        ("DESTROY_QUEUE", KnownOpcode::DestroyQueue),
        ("SUBMIT", KnownOpcode::Submit),
        ("POLL_EVENT", KnownOpcode::PollEvent),
        ("CANCEL_EVENT", KnownOpcode::CancelEvent),
        ("DESTROY_EVENT", KnownOpcode::DestroyEvent),
    ] {
        let value = format!("0x{:04x}", opcode as u16);
        define(&mut out, &format!("OP_{name}"), value, "");
    }

    out.push_str("\n/* A response's `status`. */\n");
    for (name, status) in [
        ("OK", StatusCode::OK),
        ("UNSUPPORTED", StatusCode::UNSUPPORTED),
        ("INCOMPATIBLE", StatusCode::INCOMPATIBLE),
        ("INVALID_ARGUMENT", StatusCode::INVALID_ARGUMENT),
        ("OUT_OF_BOUNDS", StatusCode::OUT_OF_BOUNDS),
        ("BUSY", StatusCode::BUSY),
        ("OUT_OF_MEMORY", StatusCode::OUT_OF_MEMORY),
        ("RESOURCE_LIMIT", StatusCode::RESOURCE_LIMIT),
        ("DEADLINE_EXPIRED", StatusCode::DEADLINE_EXPIRED),
        ("DEVICE_LOST", StatusCode::DEVICE_LOST),
        ("PERMISSION_DENIED", StatusCode::PERMISSION_DENIED),
        ("STALE_OBJECT", StatusCode::STALE_OBJECT),
    ] {
        define(&mut out, &format!("STATUS_{name}"), status.0, "");
    }
    let internal = format!("0x{:04x}", StatusCode::INTERNAL_ERROR.0);
    define(&mut out, "STATUS_INTERNAL_ERROR", internal, "");

    out.push_str("\n/* An event's `state`. */\n");
    for (name, state) in [
        ("PENDING", KnownEventState::Pending),
        ("COMPLETE", KnownEventState::Complete),
        ("FAILED", KnownEventState::Failed),
        ("CANCELLED", KnownEventState::Cancelled),
    ] {
        define(&mut out, &format!("EVENT_{name}"), state as u16, "");
    }

    out.push_str(
        "\n/* A buffer's `usage` bits, its `memory_domain`, and a binding's `access`. */\n",
    );
    for (name, usage, note) in [
        (
            "TRANSFER_SOURCE",
            BufferUsage::TRANSFER_SOURCE,
            "READ_BUFFER may read it",
        ),
        (
            "TRANSFER_DESTINATION",
            BufferUsage::TRANSFER_DESTINATION,
            "WRITE_BUFFER may write it",
        ),
        (
            "PROGRAM_INPUT",
            BufferUsage::PROGRAM_INPUT,
            "a program may read it",
        ),
        (
            "PROGRAM_OUTPUT",
            BufferUsage::PROGRAM_OUTPUT,
            "a program may write it",
        ),
        ("MUTABLE_STATE", BufferUsage::MUTABLE_STATE, ""),
    ] {
        let value = format!("0x{:x}u", usage.bits());
        define(&mut out, &format!("BUFFER_{name}"), value, note);
    }
    define(&mut out, "MEMORY_HOST", MemoryDomain::Host as u8, "");
    define(&mut out, "ACCESS_READ", AccessMode::Read as u8, "");
    define(&mut out, "ACCESS_WRITE", AccessMode::Write as u8, "");
    define(
        &mut out,
        "ACCESS_READ_WRITE",
        AccessMode::ReadWrite as u8,
        "",
    );

    out.push_str(
        "\n/* A program of this device is a registered kernel: LOAD_PROGRAM with this\n \
         * format, this identity in `target[0]`, a kernel in `target[1]`, this\n \
         * artifact as the bytes after the request, and this resident charge. */\n",
    );
    let (format, spelled) = fourcc(FORMAT_REGISTERED);
    define(&mut out, "PROGRAM_FORMAT", format, &spelled);
    let (identity, spelled) = fourcc(IDENTITY_MAGIC);
    define(&mut out, "PROGRAM_IDENTITY", identity, &spelled);
    define(
        &mut out,
        "PROGRAM_ARTIFACT_BYTES",
        REGISTERED_MANIFEST.len(),
        "",
    );
    let artifact: Vec<String> = REGISTERED_MANIFEST
        .iter()
        .map(|byte| format!("0x{byte:02x}"))
        .collect();
    let artifact = format!("{{ {} }}", artifact.join(", "));
    define(
        &mut out,
        "PROGRAM_ARTIFACT",
        artifact,
        "a uint8_t array's initializer",
    );
    define(
        &mut out,
        "PROGRAM_RESIDENT_BYTES",
        PROGRAM_RESIDENT_BYTES,
        "",
    );

    out.push_str(
        "\n/* The kernels. Each reads slot 0 and writes slot 1, one E4M3 value a byte,\n \
         * and is defined by a 256-entry table: rounded to nearest, ties to even;\n \
         * saturating at +-448 (0x7e, 0xfe); a NaN operand or a result that is not\n \
         * real is the canonical NaN 0x7f. */\n",
    );
    for op in Op::ALL {
        let name = format!("KERNEL_{}", format!("{op:?}").to_uppercase());
        define(&mut out, &name, format!("0x{:04x}u", op.kernel_id()), "");
    }

    for frame in frames() {
        writeln!(out, "\n/* {} */", frame.about).unwrap();
        writeln!(out, "struct kore_accel_{} {{", frame.name).unwrap();
        for field in &frame.fields {
            let declared = field.ty.declare(field.name);
            if field.note.is_empty() {
                writeln!(out, "    {declared}").unwrap();
            } else {
                writeln!(out, "    {declared} /* {} */", field.note).unwrap();
            }
        }
        out.push_str("} __attribute__((packed));\n");
        writeln!(
            out,
            "_Static_assert(sizeof(struct kore_accel_{}) == {}, \"{}\");",
            frame.name, frame.bytes, frame.name
        )
        .unwrap();
        for field in &frame.fields {
            writeln!(
                out,
                "_Static_assert(offsetof(struct kore_accel_{}, {}) == {}, \"{}.{}\");",
                frame.name, field.name, field.offset, frame.name, field.name
            )
            .unwrap();
        }
    }

    out.push_str("\n#endif /* KORE_VDEV_ACCEL_H */\n");
    out
}

fn committed() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include/kore/vdev_accel.h")
}

/// The table above is the protocol's frames, whole: the fields are every
/// byte of each type, in order, with nothing between them.
#[test]
fn every_frame_is_its_fields_end_to_end() {
    for frame in frames() {
        let mut at = 0;
        for field in &frame.fields {
            assert_eq!(field.offset, at, "{}.{}", frame.name, field.name);
            at += field.ty.bytes();
        }
        assert_eq!(at, frame.bytes, "{}: fields cover the frame", frame.name);
    }
}

#[test]
fn the_committed_header_is_what_the_protocol_renders() {
    let rendered = render();
    let path = committed();
    if std::env::var_os("KORE_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().expect("the header has a directory")).unwrap();
        std::fs::write(&path, &rendered).unwrap();
    }
    let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        on_disk == rendered,
        "{} is not what the protocol renders; run\n  KORE_BLESS=1 cargo test -p kore-vdev-accel --test c_header",
        path.display()
    );
}

/// The header's own assertions hold for a real C compiler, the host's: its
/// packed structures are laid out as the protocol's frames are.
#[test]
fn a_c_compiler_agrees_with_every_assertion() {
    let cc = std::env::var_os("KORE_HOST_CC").unwrap_or_else(|| "cc".into());
    let dir = std::env::temp_dir().join(format!("kore-vdev-accel-header-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("check.c");
    std::fs::write(
        &source,
        "#include <kore/vdev_accel.h>\n\
         static const uint8_t artifact[KORE_ACCEL_PROGRAM_ARTIFACT_BYTES] = KORE_ACCEL_PROGRAM_ARTIFACT;\n\
         int kore_vdev_accel_header(void) { return artifact[0] + (int)KORE_ACCEL_KERNEL_SQRT; }\n",
    )
    .unwrap();
    let output = std::process::Command::new(&cc)
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
        ])
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include"))
        .arg(&source)
        .output()
        .unwrap_or_else(|error| panic!("the host C compiler {cc:?} did not run: {error}"));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        output.status.success(),
        "the header does not compile:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
