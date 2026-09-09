//! Capability-neutral content extraction contracts.
//!
//! This crate deliberately knows no filesystem paths, namespace roots,
//! storage placement, or graph mutation authority. A consumer identifies one
//! immutable byte source, a bounded broker selects a versioned parser from
//! content signatures plus caller policy, and a worker returns only the
//! normalized records in [`schema`].
//!
//! [`Broker`] plus the frozen frames in [`worker_wire`] are the reusable
//! ordinary-caller orchestration surface. A source-owning task submits an
//! immutable identity, receives a parser selection, invokes only the
//! corresponding delegated worker capability, then validates and commits the
//! provenance-bearing result. The crate never owns executable authority or a
//! filesystem view; the boot-time init pipeline is an acceptance consumer of
//! this same API, not a privileged extraction service required by callers.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod broker;
pub mod registry;
pub mod schema;
pub mod wire;
pub mod worker_wire;

pub use broker::{Broker, BrokerJobId, BrokerRefusal, JobState};
pub use registry::{
    AccessPattern, Format, ParserDeclaration, ParserId, ParserLimits, ParserPolicy, Registry,
    Selection, VendorRecord,
};
pub use schema::{
    CacheIdentity, Confidence, Diagnostic, DiagnosticCode, EmbeddedRange, Fact, FactKind,
    FactValue, InputIdentity, NormalizedResult, ParserProvenance, Reference, ResultBuilder,
    SchemaError, SourceRange, Symbol, TextSpan, Thumbnail,
};
pub use wire::{
    DecodeError as ResultDecodeError, EncodeError as ResultEncodeError, RESULT_WIRE_MAGIC,
    RESULT_WIRE_VERSION, decode_result, encode_result,
};
pub use worker_wire::{
    BrokerDispatch, COVERAGE_ARCHIVE, COVERAGE_BROKER_VALIDATED, COVERAGE_C_JSON, COVERAGE_CSV,
    COVERAGE_DOCUMENT, COVERAGE_FONT, COVERAGE_JSON, COVERAGE_LIMIT_REJECTIONS,
    COVERAGE_PLAIN_TEXT, COVERAGE_RASTER, COVERAGE_XML, WORKER_REPORT_BYTES, WorkerReport,
    WorkerReportError, decode_input, decode_job, encode_input, encode_job,
};
