//! TLS client operations over caller-provided transport buffers.
#![cfg_attr(target_os = "none", no_std)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

use alloc::{boxed::Box, sync::Arc};
use core::{ptr::NonNull, slice, str, time::Duration};
use ring::{digest, rand::SecureRandom};
use rustls::{
    ClientConfig, RootCertStore,
    client::{Resumption, UnbufferedClientConnection},
    pki_types::{CertificateDer, ServerName, UnixTime, pem::PemObject},
    time_provider::TimeProvider,
    unbuffered::ConnectionState,
};

/// Space required in the output buffer of any TLS step.
pub const KORE_TLS_OUTPUT_BYTES: usize = 32 * 1024;
/// Maximum application write per TLS step.
pub const KORE_TLS_WRITE_BYTES: usize = 16 * 1024;
/// Maximum PEM bundle size accepted by the client constructor.
pub const KORE_TLS_ROOTS_BYTES: usize = 1024 * 1024;

/// Invalid arguments leave the connection untouched. Other step errors require freeing it.
#[repr(C)]
pub enum KoreTlsStatus {
    Ok,
    InvalidArgument,
    InvalidTrust,
    Certificate,
    Protocol,
    Entropy,
    Clock,
}

/// Poll with no plaintext, send one plaintext chunk, or send close_notify.
#[repr(u32)]
pub enum KoreTlsAction {
    Poll,
    Write,
    Shutdown,
}

#[repr(C)]
pub enum KoreTlsEvent {
    /// Send the output bytes before calling step again.
    Ciphertext,
    /// Deliver the output bytes as application data.
    Plaintext,
    /// Append more ciphertext to the unconsumed input and retry.
    NeedInput,
    /// The authenticated connection can accept application writes.
    Ready,
    /// Send the output bytes. The requested plaintext chunk was consumed.
    Wrote,
    /// Send close_notify from the output buffer.
    ShutdownSent,
    /// The peer authenticated the end of its stream with close_notify.
    PeerClosed,
}

#[repr(C)]
pub struct KoreTlsProgress {
    pub event: KoreTlsEvent,
    /// Remove these bytes from the front of the input buffer before the next step.
    pub consumed: usize,
    /// Number of bytes written to the output buffer.
    pub produced: usize,
}

/// An owned TLS session. Serialize calls that use the same session.
pub struct KoreTlsClient {
    connection: UnbufferedClientConnection,
    failed: bool,
}

/// A streaming SHA-256 context.
pub struct KoreTlsSha256(digest::Context);

#[derive(Debug)]
struct WallClock;
impl TimeProvider for WallClock {
    fn current_time(&self) -> Option<UnixTime> {
        #[cfg(target_os = "none")]
        let seconds = kore_std::runtime::clock::wall().ok()?.as_secs();
        #[cfg(not(target_os = "none"))]
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        Some(UnixTime::since_unix_epoch(Duration::from_secs(seconds)))
    }
}

#[cfg(target_os = "none")]
fn kore_entropy(bytes: &mut [u8]) -> Result<(), getrandom::Error> {
    kore_std::runtime::random::fill(bytes).map_err(|_| getrandom::Error::UNSUPPORTED)
}

#[cfg(target_os = "none")]
getrandom::register_custom_getrandom!(kore_entropy);

// The entropy hook is an implementation detail of getrandom, not C ABI.
#[cfg(target_os = "none")]
core::arch::global_asm!(".hidden __getrandom_custom");

fn status(error: rustls::Error) -> KoreTlsStatus {
    match error {
        rustls::Error::InvalidCertificate(_) => KoreTlsStatus::Certificate,
        rustls::Error::FailedToGetCurrentTime => KoreTlsStatus::Clock,
        rustls::Error::FailedToGetRandomBytes => KoreTlsStatus::Entropy,
        _ => KoreTlsStatus::Protocol,
    }
}

fn open(host: &[u8], pem: &[u8]) -> Result<KoreTlsClient, KoreTlsStatus> {
    let name = str::from_utf8(host).map_err(|_| KoreTlsStatus::InvalidArgument)?;
    let name = ServerName::try_from(name)
        .map_err(|_| KoreTlsStatus::InvalidArgument)?
        .to_owned();
    let mut roots = RootCertStore::empty();
    for certificate in CertificateDer::pem_slice_iter(pem) {
        roots
            .add(certificate.map_err(|_| KoreTlsStatus::InvalidTrust)?)
            .map_err(|_| KoreTlsStatus::InvalidTrust)?;
    }
    if roots.is_empty() {
        return Err(KoreTlsStatus::InvalidTrust);
    }
    let clock = Arc::new(WallClock);
    if clock.current_time().is_none() {
        return Err(KoreTlsStatus::Clock);
    }
    let mut config = ClientConfig::builder_with_details(
        Arc::new(rustls::crypto::ring::default_provider()),
        clock,
    )
    .with_safe_default_protocol_versions()
    .map_err(status)?
    .with_root_certificates(roots)
    .with_no_client_auth();
    config.resumption = Resumption::disabled();
    let connection = UnbufferedClientConnection::new(Arc::new(config), name).map_err(status)?;
    Ok(KoreTlsClient {
        connection,
        failed: false,
    })
}

// SAFETY: the caller vouches for the buffer, and only call-local slices are made.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    let ptr = NonNull::new(ptr.cast_mut()).unwrap_or_else(NonNull::dangling);
    unsafe { slice::from_raw_parts(ptr.as_ptr(), len) }
}

unsafe fn bytes_mut<'a>(ptr: *mut u8, len: usize) -> &'a mut [u8] {
    let ptr = NonNull::new(ptr).unwrap_or_else(NonNull::dangling);
    unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), len) }
}

fn valid_buffer(ptr: *const u8, len: usize) -> bool {
    len <= isize::MAX as usize && (len == 0 || !ptr.is_null())
}

/// Create a client with a DNS name or IP and one or more PEM trust anchors.
/// The platform wall clock supplies certificate validation time.
///
/// # Safety
/// Both input buffers must be readable. The output must point to a writable
/// pointer. The caller owns the resulting handle and must free it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_tls_client_new(
    host: *const u8,
    host_len: usize,
    roots: *const u8,
    roots_len: usize,
    client: *mut *mut KoreTlsClient,
) -> KoreTlsStatus {
    if client.is_null() {
        return KoreTlsStatus::InvalidArgument;
    }
    unsafe { client.write(core::ptr::null_mut()) };
    if !valid_buffer(host, host_len)
        || !(1..=253).contains(&host_len)
        || !valid_buffer(roots, roots_len)
        || !(1..=KORE_TLS_ROOTS_BYTES).contains(&roots_len)
    {
        return KoreTlsStatus::InvalidArgument;
    }
    match open(unsafe { bytes(host, host_len) }, unsafe {
        bytes(roots, roots_len)
    }) {
        Ok(connection) => {
            unsafe { client.write(Box::into_raw(Box::new(connection))) };
            KoreTlsStatus::Ok
        }
        Err(error) => error,
    }
}

/// Advance at most one event. Fully send every ciphertext output before
/// calling again. Retry a write after intermediate events until Wrote.
/// Report transport EOF separately from PeerClosed, which is authenticated.
///
/// # Safety
/// The handle must be live and exclusively borrowed. Input, plaintext, output
/// and progress buffers must be valid and disjoint for this call. Only empty
/// input buffers may be null. Output must hold KORE_TLS_OUTPUT_BYTES bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_tls_client_step(
    client: *mut KoreTlsClient,
    action: u32,
    input: *mut u8,
    input_len: usize,
    plaintext: *const u8,
    plaintext_len: usize,
    output: *mut u8,
    output_len: usize,
    progress: *mut KoreTlsProgress,
) -> KoreTlsStatus {
    let write = action == KoreTlsAction::Write as u32;
    let shutdown = action == KoreTlsAction::Shutdown as u32;
    if client.is_null()
        || progress.is_null()
        || !valid_buffer(input, input_len)
        || !valid_buffer(output, output_len)
        || output_len < KORE_TLS_OUTPUT_BYTES
        || !valid_buffer(plaintext, plaintext_len)
        || (write && !(1..=KORE_TLS_WRITE_BYTES).contains(&plaintext_len))
        || (!write && plaintext_len != 0)
        || (!write && !shutdown && action != KoreTlsAction::Poll as u32)
    {
        return KoreTlsStatus::InvalidArgument;
    }
    let client = unsafe { &mut *client };
    match client.step(
        unsafe { bytes_mut(input, input_len) },
        unsafe { bytes(plaintext, plaintext_len) },
        unsafe { bytes_mut(output, output_len) },
        shutdown,
    ) {
        Ok(result) => {
            unsafe { progress.write(result) };
            KoreTlsStatus::Ok
        }
        Err(error) => {
            client.failed = true;
            error
        }
    }
}

impl KoreTlsClient {
    fn step(
        &mut self,
        input: &mut [u8],
        plaintext: &[u8],
        output: &mut [u8],
        shutdown: bool,
    ) -> Result<KoreTlsProgress, KoreTlsStatus> {
        if self.failed {
            return Err(KoreTlsStatus::Protocol);
        }
        let incoming = self.connection.process_tls_records(input);
        let mut consumed = incoming.discard;
        let (event, produced) = match incoming.state.map_err(status)? {
            ConnectionState::EncodeTlsData(mut data) => (
                KoreTlsEvent::Ciphertext,
                data.encode(output).map_err(|_| KoreTlsStatus::Protocol)?,
            ),
            ConnectionState::TransmitTlsData(data) => {
                // Every previous output was transmitted before this call.
                data.done();
                (KoreTlsEvent::Ciphertext, 0)
            }
            ConnectionState::ReadTraffic(mut data) => {
                let record = data
                    .next_record()
                    .ok_or(KoreTlsStatus::Protocol)?
                    .map_err(status)?;
                consumed += record.discard;
                let destination = output
                    .get_mut(..record.payload.len())
                    .ok_or(KoreTlsStatus::Protocol)?;
                destination.copy_from_slice(record.payload);
                (KoreTlsEvent::Plaintext, destination.len())
            }
            ConnectionState::BlockedHandshake => (KoreTlsEvent::NeedInput, 0),
            ConnectionState::WriteTraffic(mut data) if shutdown => (
                KoreTlsEvent::ShutdownSent,
                data.queue_close_notify(output)
                    .map_err(|_| KoreTlsStatus::Protocol)?,
            ),
            ConnectionState::WriteTraffic(mut data) if !plaintext.is_empty() => (
                KoreTlsEvent::Wrote,
                data.encrypt(plaintext, output)
                    .map_err(|_| KoreTlsStatus::Protocol)?,
            ),
            ConnectionState::WriteTraffic(_) => (KoreTlsEvent::Ready, 0),
            ConnectionState::PeerClosed | ConnectionState::Closed => (KoreTlsEvent::PeerClosed, 0),
            _ => return Err(KoreTlsStatus::Protocol),
        };
        Ok(KoreTlsProgress {
            event,
            consumed,
            produced,
        })
    }
}

/// # Safety
/// Pass null or a still-owned handle returned by kore_tls_client_new.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_tls_client_free(client: *mut KoreTlsClient) {
    if !client.is_null() {
        drop(unsafe { Box::from_raw(client) });
    }
}

/// Fill bytes using the platform secure random source.
///
/// # Safety
/// The output buffer must be valid and exclusively writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_tls_random(output: *mut u8, len: usize) -> KoreTlsStatus {
    if !valid_buffer(output, len) {
        return KoreTlsStatus::InvalidArgument;
    }
    match ring::rand::SystemRandom::new().fill(unsafe { bytes_mut(output, len) }) {
        Ok(()) => KoreTlsStatus::Ok,
        Err(_) => KoreTlsStatus::Entropy,
    }
}

/// Allocate a streaming SHA-256 context.
#[unsafe(no_mangle)]
pub extern "C" fn kore_tls_sha256_new() -> *mut KoreTlsSha256 {
    Box::into_raw(Box::new(KoreTlsSha256(digest::Context::new(
        &digest::SHA256,
    ))))
}

/// # Safety
/// The handle must be live and exclusively borrowed. The input must be readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_tls_sha256_update(
    context: *mut KoreTlsSha256,
    data: *const u8,
    len: usize,
) -> KoreTlsStatus {
    if context.is_null() || !valid_buffer(data, len) {
        return KoreTlsStatus::InvalidArgument;
    }
    unsafe { (*context).0.update(bytes(data, len)) };
    KoreTlsStatus::Ok
}

/// # Safety
/// The handle must be live. The output must have room for 32 bytes.
/// The context remains owned by the caller and must be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_tls_sha256_digest(
    context: *const KoreTlsSha256,
    output: *mut u8,
) -> KoreTlsStatus {
    if context.is_null() || output.is_null() {
        return KoreTlsStatus::InvalidArgument;
    }
    unsafe {
        bytes_mut(output, 32).copy_from_slice((*context).0.clone().finish().as_ref());
    }
    KoreTlsStatus::Ok
}

/// # Safety
/// Pass null or a still-owned handle returned by kore_tls_sha256_new.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_tls_sha256_free(context: *mut KoreTlsSha256) {
    if !context.is_null() {
        drop(unsafe { Box::from_raw(context) });
    }
}
