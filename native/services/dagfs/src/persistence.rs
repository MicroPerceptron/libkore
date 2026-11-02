//! Versioned, checksummed durable namespace publication and recovery.
//!
//! Two one-block root records select two copy-on-write metadata arenas. A
//! commit writes the inactive arena, flushes, publishes its root record, and
//! flushes again. Recovery validates both roots and every referenced record,
//! then chooses the highest completely valid generation. Disk records carry
//! stable object IDs and extents only: session IDs, handles, grants,
//! capabilities, channels, mappings, and request state are never encoded.

use alloc::{boxed::Box, vec::Vec};

use crc::{CRC_32_ISCSI, Crc};
use kore_driver::block::DirectBlockDevice;

use crate::{
    Engine, FsError, Name, ObjectKind, ObjectSubtype, ProviderError, SessionId, StableObjectId,
    bounds::{
        MAX_BLOCK_PLACEMENTS, MAX_DERIVED_RESULTS, MAX_DIR_ENTRIES, MAX_DIRECTORIES, MAX_NAME_LEN,
        MAX_OBJECTS,
    },
    namespace::Namespace,
    object::FIRST_REVISION,
    provider::{BlockKey, BlockProvider, DurableExtent, LEGACY_DATA_START_LBA, StorageProvider},
};

const CRC: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);
const ROOT_MAGIC: &[u8; 8] = b"KOREROOT";
const IMAGE_MAGIC: &[u8; 8] = b"KOREFSM1";
const LEGACY_FORMAT_VERSION: u32 = 1;
/// The version that added the derived-result table to the image.  Named
/// because the table's presence is decided by "version 2 or later", never by
/// "equals the current version" — the latter silently drops a v2 volume's
/// index the moment the current version moves on.
const DERIVED_FORMAT_VERSION: u32 = 2;
/// The version that added the three facts the mount previously had to
/// assume: where the data arena starts, how many blocks it spans, and the
/// 32-byte era stamp identifying the build whose store this is.
const ARENA_FORMAT_VERSION: u32 = 3;
/// The version that gave the object record its subtype byte (offset 10,
/// reserved-must-be-zero before it). Named because "can this volume record
/// a subtype" is decided by "version 4 or later", never by "equals the
/// current version".
pub const SUBTYPE_FORMAT_VERSION: u32 = 4;
/// The version whose root names a content-digest sidecar ([`crate::digests`]):
/// derived data, CRC'd and keyed by revision, so mount and every lend stop
/// rehashing files whose bytes have not changed. A version-4 volume mounts
/// as it is and is upgraded by its next commit (`res/STORAGE_TRANSPORT.md`
/// §1 ruling 3); older versions keep committing at their own version.
pub const DIGEST_FORMAT_VERSION: u32 = 5;
/// The current version.
pub const FORMAT_VERSION: u32 = DIGEST_FORMAT_VERSION;
/// Metadata reservation used by versions 1 and 2.
const LEGACY_IMAGE_BYTES: usize = 16 * 1024;
/// Maximum encoded namespace image in the current format.
///
/// Two fixed reservations of this size live below the volume's recorded
/// `data_start`; commits transfer only the encoded, block-rounded prefix.
pub const MAX_IMAGE_BYTES: usize = 128 * 1024;
/// Root-record header through version 2.
const ROOT_HEADER_V2: usize = 60;
/// Root-record header at version 3: `data_start` (8) + `arena_blocks` (8) +
/// `era` (32) after v2's fields, with the trailing checksum after them.
const ROOT_HEADER_V3: usize = 108;
/// Root-record header at version 5: the version-3 fields, then the digest
/// sidecar's extent (`lba` 8, `blocks` 4, `len` 4) and its CRC (4), with the
/// trailing checksum after them. Zero `blocks` means no sidecar.
const ROOT_HEADER_V5: usize = 128;
/// Bytes of era stamp carried in a version-3 root record.
pub const ERA_BYTES: usize = 32;

/// The header length one format version declares.
///
/// The length is validated *exactly*, so it cannot be a single constant
/// across versions: a v2 volume records 60 and a v3 volume records 108, and
/// each must still be recognized by a reader that knows both.
const fn root_header_bytes(version: u32) -> Option<usize> {
    match version {
        LEGACY_FORMAT_VERSION | DERIVED_FORMAT_VERSION => Some(ROOT_HEADER_V2),
        ARENA_FORMAT_VERSION | SUBTYPE_FORMAT_VERSION => Some(ROOT_HEADER_V3),
        DIGEST_FORMAT_VERSION => Some(ROOT_HEADER_V5),
        _ => None,
    }
}
const IMAGE_HEADER: usize = 64;
const OBJECT_BYTES: usize = 48;
const EDGE_BYTES: usize = 56;
const DERIVED_BYTES: usize = 72;
const LEGACY_MAX_DURABLE_OBJECTS: usize = 64;
const LEGACY_MAX_DURABLE_EDGES: usize = 128;
const LEGACY_MAX_DIR_ENTRIES: usize = 32;
pub const MAX_DURABLE_OBJECTS: usize = MAX_OBJECTS;

/// Containment edges one image may record.
///
/// **Deliberately not `MAX_DIRECTORIES * MAX_DIR_ENTRIES`.** That product is
/// the shape a namespace *could* take if every directory filled; it is not
/// the shape one takes. A desktop tree measured for
/// `res/FILESYSTEM_SCALE.md` carries about **1.05 edges per object**, because
/// a namespace is a tree with a few extra links, while the product reserved
/// sixteen per object and so spent 87% of the image on adjacency that does
/// not exist. Paying that ratio is what held the object bound at 136.
///
/// The engine's *structural* maximum is still the product, which is larger
/// than this. A tree that exceeds this cap is refused at commit with
/// [`PersistenceError::Bounds`] rather than at `link`, because the engine
/// keeps no global edge total and adding one would put a counter on every
/// mutation. That is a real gap, and it is bounded on both sides: reaching it
/// needs twice as many edges as the object bound allows objects, against a
/// measured ratio near one. Closing it properly wants either a global
/// admission check or variable-length edge records, which is the same change
/// that lets `MAX_NAME_LEN` leave 32.
///
/// Chosen so an existing volume keeps mounting: `MAX_DIR_ENTRIES` and the
/// slot size are untouched, and the **reader** admits everything a current
/// image could record before this decoupling
/// ([`PRE_DECOUPLING_MAX_DURABLE_EDGES`]), so nothing that was recordable
/// becomes unreadable and no reformat is implied. This constant is the
/// **writer's** ceiling only. A pre-decoupling volume carrying more edges
/// than this — reachable only through hard links, at more than seven per
/// object — mounts and reads whole; its next commit refuses
/// [`PersistenceError::Bounds`] until links drop below the ceiling, which is
/// the same typed refusal any over-bound tree receives.
pub const MAX_DURABLE_EDGES: usize = 1024;

/// Containment edges a current-profile image written **before**
/// [`MAX_DURABLE_EDGES`] was decoupled may record: the `16 × 128`
/// directory × entry product the profile reserved when it was introduced.
/// Reading validates against this rather than the writer's ceiling, so a
/// volume that mounted under the old bound still mounts. A count in
/// `MAX_DURABLE_EDGES + 1 ..= PRE_DECOUPLING_MAX_DURABLE_EDGES` therefore
/// identifies an old writer, and a count above it is corruption.
///
/// A literal because neither factor is a live constant any more:
/// `MAX_DIRECTORIES` has since been derived from `MAX_OBJECTS`, and the
/// product was never a bound this format should have carried
/// (`res/FILESYSTEM_SCALE.md` §4.1).
const PRE_DECOUPLING_MAX_DURABLE_EDGES: usize = 2048;

// A reader must admit at least what the writer emits, or a freshly committed
// image would refuse to recover.
const _: () = assert!(PRE_DECOUPLING_MAX_DURABLE_EDGES >= MAX_DURABLE_EDGES);

// The serialized maxima are one contract: raising any constituent without
// raising the reservation must fail at compile time, not as a late commit.
const _: () = assert!(
    IMAGE_HEADER
        + MAX_DURABLE_OBJECTS * OBJECT_BYTES
        + MAX_DURABLE_EDGES * EDGE_BYTES
        + MAX_DERIVED_RESULTS * DERIVED_BYTES
        <= MAX_IMAGE_BYTES,
    "the current durable graph must fit its recorded image reservation"
);

#[derive(Clone, Copy)]
struct ImageLayout {
    bytes: usize,
    slot_blocks: u32,
    data_start: u64,
}

/// The wire version and capacity profile are deliberately separate. Slice 5
/// produced v3 roots before slice 7b enlarged the slots; those roots declare
/// the legacy floor and must keep using the legacy slot addresses. Fresh v3
/// formats declare the current floor and receive the raised bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageProfile {
    Legacy,
    Current,
}

fn image_layout(profile: ImageProfile, block_size: u32) -> Result<ImageLayout, PersistenceError> {
    let bytes = match profile {
        ImageProfile::Legacy => LEGACY_IMAGE_BYTES,
        ImageProfile::Current => MAX_IMAGE_BYTES,
    };
    let block = block_size as usize;
    if block == 0 || !bytes.is_multiple_of(block) {
        return Err(PersistenceError::Incompatible);
    }
    let slot_blocks = u32::try_from(bytes / block).map_err(|_| PersistenceError::Bounds)?;
    let metadata_end = 2u64
        .checked_add(
            u64::from(slot_blocks)
                .checked_mul(2)
                .ok_or(PersistenceError::Bounds)?,
        )
        .ok_or(PersistenceError::Bounds)?;
    Ok(ImageLayout {
        bytes,
        slot_blocks,
        // Versions 1 and 2 reserved 72 blocks even when their two 16 KiB
        // slots ended earlier. Current volumes extend that separation where
        // the larger slots require.
        // Current profiles always use a distinguishable floor. On 4 KiB
        // media both slot layouts otherwise end below legacy block 72, and a
        // pre-7b v3 root would not say which slot-1 address it meant.
        data_start: match profile {
            ImageProfile::Legacy => metadata_end.max(LEGACY_DATA_START_LBA),
            ImageProfile::Current => metadata_end.max(LEGACY_DATA_START_LBA + 1),
        },
    })
}

fn root_profile(
    version: u32,
    data_start: u64,
    block_size: u32,
) -> Result<ImageProfile, PersistenceError> {
    match version {
        LEGACY_FORMAT_VERSION | DERIVED_FORMAT_VERSION => Ok(ImageProfile::Legacy),
        // A pre-7b version-3 volume kept the legacy slot profile; the profile
        // is decided by where the data arena starts, never by the version.
        ARENA_FORMAT_VERSION | SUBTYPE_FORMAT_VERSION | DIGEST_FORMAT_VERSION
            if data_start == LEGACY_DATA_START_LBA =>
        {
            Ok(ImageProfile::Legacy)
        }
        ARENA_FORMAT_VERSION | SUBTYPE_FORMAT_VERSION | DIGEST_FORMAT_VERSION
            if data_start >= image_layout(ImageProfile::Current, block_size)?.data_start =>
        {
            Ok(ImageProfile::Current)
        }
        ARENA_FORMAT_VERSION | SUBTYPE_FORMAT_VERSION | DIGEST_FORMAT_VERSION => {
            Err(PersistenceError::Corrupt)
        }
        _ => Err(PersistenceError::Incompatible),
    }
}

/// The writer's `(objects, edges)` ceilings for `profile`.
fn durable_bounds(profile: ImageProfile) -> (usize, usize) {
    match profile {
        ImageProfile::Legacy => (LEGACY_MAX_DURABLE_OBJECTS, LEGACY_MAX_DURABLE_EDGES),
        ImageProfile::Current => (MAX_DURABLE_OBJECTS, MAX_DURABLE_EDGES),
    }
}

/// Containment edges a reader admits from an image of `profile`: the
/// writer's ceiling plus everything an older writer of the same profile could
/// lawfully have recorded. The legacy profile never moved, so its two
/// figures coincide.
const fn readable_durable_edges(profile: ImageProfile) -> usize {
    match profile {
        ImageProfile::Legacy => LEGACY_MAX_DURABLE_EDGES,
        ImageProfile::Current => PRE_DECOUPLING_MAX_DURABLE_EDGES,
    }
}

const fn durable_dir_entries(profile: ImageProfile) -> usize {
    match profile {
        ImageProfile::Legacy => LEGACY_MAX_DIR_ENTRIES,
        ImageProfile::Current => MAX_DIR_ENTRIES,
    }
}

/// Conservative provider floor used before either root has authenticated its
/// exact per-volume arena.
pub(crate) fn bootstrap_data_start(block_size: u32) -> Result<u64, PersistenceError> {
    Ok(image_layout(ImageProfile::Current, block_size)?.data_start)
}

/// One fallibly allocated metadata image retained by a mounted volume and
/// reused for every recovery read and commit.
struct ImageScratch(Box<[u8]>);

impl ImageScratch {
    fn new() -> Result<Self, PersistenceError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(MAX_IMAGE_BYTES)
            .map_err(|_| PersistenceError::Bounds)?;
        bytes.resize(MAX_IMAGE_BYTES, 0);
        Ok(Self(bytes.into_boxed_slice()))
    }

    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistenceError {
    Unformatted,
    Incompatible,
    Corrupt,
    Bounds,
    Interrupted(CommitStage),
    Provider(ProviderError),
    Filesystem(FsError),
}

/// What the two reserved root blocks say about the underlying medium.
///
/// This classification is deliberately weaker than recovery: recognizing the
/// kore magic does not bless a root record, it only says that recovery owns
/// the right to diagnose it.  Conversely, arbitrary nonzero bytes are
/// [`Foreign`](Self::Foreign), not a damaged kore volume.  Real boot disks
/// normally carry GPT/MBR data in these blocks and must never be formatted by
/// an automatic service bring-up path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeProbe {
    /// Both reserved root blocks are entirely zero.
    Blank,
    /// At least one root block begins with the kore filesystem magic.
    Kore,
    /// Nonzero metadata exists, but neither root names the kore format.
    Foreign,
}

impl From<ProviderError> for PersistenceError {
    fn from(value: ProviderError) -> Self {
        Self::Provider(value)
    }
}

impl From<FsError> for PersistenceError {
    fn from(value: FsError) -> Self {
        Self::Filesystem(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitStage {
    BeforeMetadata,
    AfterMetadata,
    AfterMetadataFlush,
    AfterRootWrite,
    AfterRootFlush,
}

/// The stage of [`PersistentFs::recover_witnessed`] that refused a volume, in
/// the order recovery runs. Frozen: append, never renumber — the number is
/// printed inside the startup refusal of a boot that cannot mount its
/// workspace (`durable.rs`, detail 17), and a photograph of that screen has
/// to stay decodable.
///
/// The image stages (`RootDecode` through `ImageDecode`) are per root: a
/// root that fails one falls back to the other, so one of them names the
/// newer root's failure only when the older root failed too. Every later
/// stage refuses the mount outright from whichever root reached it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RecoverStage {
    /// The medium's block is larger than a root record's scratch.
    Geometry = 1,
    /// Reading the two reserved root blocks.
    RootBlocks = 2,
    /// Neither root block names the format.
    Unformatted = 3,
    /// Both roots claim one generation.
    SameGeneration = 4,
    /// The roots disagree on the allocation arena.
    ArenaDisagrees = 5,
    /// The image scratch could not be allocated.
    Scratch = 6,
    /// A root record did not decode.
    RootDecode = 7,
    /// The root's arena bounds were refused by the provider.
    Arena = 8,
    /// The root's version names no image profile this reader has.
    Profile = 9,
    /// The image length does not fit its region.
    ImageSpan = 10,
    /// Reading the image.
    ImageRead = 11,
    /// The image failed its checksum.
    ImageCrc = 12,
    /// The image did not decode.
    ImageDecode = 13,
    /// The live or fallback extent tables overflowed.
    ExtentTable = 14,
    /// The provider refused the recovered extents.
    Extents = 15,
    /// Restoring an object into the namespace.
    Objects = 16,
    /// The derived-result table overflowed.
    DerivedTable = 17,
    /// The recovered extents and the image's objects disagree in count.
    ExtentWalk = 18,
    /// Restoring a containment edge.
    Edges = 19,
    /// The image's root object is not among its objects.
    Root = 20,
    /// The engine refused the recovered namespace.
    Engine = 21,
}

/// Why and where a recovery refused a volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecoverFailure {
    pub stage: RecoverStage,
    pub error: PersistenceError,
}

impl RecoverFailure {
    pub const fn at(stage: RecoverStage, error: PersistenceError) -> Self {
        Self { stage, error }
    }

    /// The failure as four bytes, low to high: the stage, the error's kind
    /// (1 unformatted, 2 incompatible, 3 corrupt, 4 bounds, 5 interrupted,
    /// 6 provider, 7 filesystem), then the cause — the commit stage for an
    /// interrupted commit, the provider's wire code, or the filesystem's
    /// wire status with its provider code beside it. Every code is a frozen
    /// wire vocabulary ([`kore_dagfs::wire`]), never a new one.
    pub fn witness(self) -> u32 {
        use kore_dagfs::wire::{fs_status_code, provider_code};
        let (kind, cause, beside) = match self.error {
            PersistenceError::Unformatted => (1, 0, 0),
            PersistenceError::Incompatible => (2, 0, 0),
            PersistenceError::Corrupt => (3, 0, 0),
            PersistenceError::Bounds => (4, 0, 0),
            PersistenceError::Interrupted(stage) => (5, stage as u8 + 1, 0),
            PersistenceError::Provider(error) => (6, provider_code(error), 0),
            PersistenceError::Filesystem(error) => {
                let beside = match error {
                    FsError::ProviderFailure(provider) => provider_code(provider),
                    _ => 0,
                };
                (7, fs_status_code(error), beside)
            }
        };
        u32::from_le_bytes([self.stage as u8, kind, cause, beside])
    }
}

/// Tag a fallible step of recovery with its stage.
fn at<E: Into<PersistenceError>>(stage: RecoverStage) -> impl FnOnce(E) -> RecoverFailure {
    move |error| RecoverFailure::at(stage, error.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MountStatus {
    pub generation: u64,
    pub used_fallback_root: bool,
    pub mode: MountMode,
    /// The era this volume carries, round-tripped through mount and commit.
    ///
    /// Held here rather than defaulted at each commit because a field that is
    /// re-zeroed on every write is not persisted at all: the stamp has to
    /// survive the commits between the mount that read it and the policy that
    /// will judge it. Nothing reads it for a decision yet.
    pub era: [u8; ERA_BYTES],
}

/// Whether recovered state may be published again without an explicit
/// repair/revalidation step. Falling back around corruption is readable but
/// deliberately not writable: silently overwriting the evidence would turn
/// a deterministic degraded mount into data loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountMode {
    ReadWrite,
    DegradedReadOnly,
}

#[derive(Clone, Copy)]
struct DiskObject {
    stable: StableObjectId,
    kind: ObjectKind,
    subtype: ObjectSubtype,
    len: u64,
    revision: u64,
    extent: Option<DurableExtent>,
}

#[derive(Clone, Copy)]
struct DiskEdge {
    parent: StableObjectId,
    child: StableObjectId,
    name: Name,
}

#[derive(Clone, Copy)]
struct DiskDerived {
    object: StableObjectId,
    input_version: u64,
    extent: DurableExtent,
    digest: [u8; 32],
}

/// One root's decoded metadata image.
///
/// The three tables are heap-backed and reserved to the **counts this image
/// actually records**, not to the profile's worst case. Inline they came to
/// 132,168 bytes — the edge table alone was 114,696 — and `Decoded` is built
/// on the mount path, which runs inside the service's `serve` future on the
/// founder's shared 4 MiB stack. That is the same coupling the namespace
/// arenas had, and recovery decodes each of the two roots in turn.
///
/// Reserving the recorded counts rather than the bounds also means a small
/// volume pays for a small volume: the profile maxima are already validated
/// against [`durable_bounds`] before a single record is read, so the
/// reservation cannot be driven past them by disk bytes.
struct Decoded {
    generation: u64,
    era: [u8; ERA_BYTES],
    root: StableObjectId,
    objects: Vec<DiskObject>,
    edges: Vec<DiskEdge>,
    derived: Vec<DiskDerived>,
}

/// The one canonical order of a decoded root's payload extents: each object
/// that carries one, then every derived record. `recover_extents` returns
/// the live root's keys following exactly this sequence, and the mount path
/// consumes them through one iterator — the pairing between extents and
/// their objects has this single definition, never twin hand-maintained
/// walks that must agree by luck.
fn root_payload_extents(decoded: &Decoded) -> impl Iterator<Item = DurableExtent> + '_ {
    decoded
        .objects
        .iter()
        .filter_map(|object| object.extent)
        .chain(decoded.derived.iter().map(|record| record.extent))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RootRecord {
    version: u32,
    generation: u64,
    block_size: u32,
    block_count: u64,
    slot: u8,
    root: StableObjectId,
    image_len: u32,
    image_crc: u32,
    /// First block of the data arena.  Recorded from version 3 so the floor
    /// is a property of the volume rather than a compile-time constant every
    /// reader must already agree on; versions 1 and 2 decode to
    /// [`LEGACY_DATA_START_LBA`], which is what they meant.
    data_start: u64,
    /// Blocks the data arena spans.  Versions 1 and 2 decode to everything
    /// above the floor, which is what they meant.
    arena_blocks: u64,
    /// Which build's store this is — blake3 of the image archive. All-zero
    /// means unstamped, which is what versions 1 and 2 decode to.
    era: [u8; ERA_BYTES],
    /// The content-digest sidecar this root names (version 5), if any.
    sidecar: Option<SidecarRef>,
}

/// Where a root's content-digest sidecar lives, and the CRC its bytes carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SidecarRef {
    extent: DurableExtent,
    crc: u32,
}

/// An engine whose durable root is published through its direct block
/// provider. Runtime authority is recreated for `session` on recovery.
pub struct PersistentFs<
    D: DirectBlockDevice,
    const BUFFERS: usize = 8,
    const OUTSTANDING: usize = 32,
> {
    engine: Engine<BlockProvider<D, BUFFERS, OUTSTANDING>>,
    state: PersistenceState,
}

/// The durability cursor kept beside a mounted request engine.
///
/// `version` pins legacy volumes to their legacy reservation until deliberate
/// reprovisioning; `image` is the single reusable heap allocation that keeps
/// the large current reservation off every async commit stack.
pub struct PersistenceState {
    status: MountStatus,
    version: u32,
    profile: ImageProfile,
    image: ImageScratch,
    /// The digest sidecar the live root names, and its placement — replaced,
    /// never rewritten in place, when a commit finds the memo changed.
    sidecar: Option<(BlockKey, SidecarRef)>,
}

impl PersistenceState {
    pub const fn status(&self) -> MountStatus {
        self.status
    }
}

impl<D: DirectBlockDevice, const B: usize, const O: usize> PersistentFs<D, B, O> {
    /// Classify the reserved roots without modifying the medium.
    ///
    /// The observation borrows the provider so its affine channel can then be
    /// moved into recovery, explicit blank-media formatting, or containment.
    /// Only an all-zero pair is blank. A GPT header, partition table, another
    /// filesystem, or any other non-kore data is foreign and is never an
    /// implicit invitation to overwrite the namespace.
    pub async fn probe(provider: &BlockProvider<D, B, O>) -> Result<VolumeProbe, PersistenceError> {
        let geometry = provider.geometry();
        let block = geometry.block_size as usize;
        if block == 0 || block > 4096 {
            return Err(PersistenceError::Incompatible);
        }
        let mut roots = [[0u8; 4096]; 2];
        provider.read_metadata(0, &mut roots[0][..block]).await?;
        provider.read_metadata(1, &mut roots[1][..block]).await?;
        let classify = |root: &[u8]| {
            if root.iter().all(|byte| *byte == 0) {
                VolumeProbe::Blank
            } else if root.get(..ROOT_MAGIC.len()) == Some(ROOT_MAGIC.as_slice()) {
                VolumeProbe::Kore
            } else {
                VolumeProbe::Foreign
            }
        };
        let roots = [classify(&roots[0][..block]), classify(&roots[1][..block])];
        Ok(if roots.contains(&VolumeProbe::Kore) {
            VolumeProbe::Kore
        } else if roots.iter().all(|root| *root == VolumeProbe::Blank) {
            VolumeProbe::Blank
        } else {
            VolumeProbe::Foreign
        })
    }

    /// Whether every root that recovery could select is a valid version-3
    /// root stamped with `expected`, with at least one such root present.
    ///
    /// This is the system-store admission check. Requiring agreement across
    /// both nonblank slots is stronger than comparing only the newest header:
    /// if its image later proves corrupt, ordinary recovery may fall back to
    /// the other slot. That fallback must never publish an older archive era.
    /// User volumes do not call this function and remain era-agnostic.
    pub async fn roots_match_era(
        provider: &BlockProvider<D, B, O>,
        expected: [u8; ERA_BYTES],
    ) -> Result<bool, PersistenceError> {
        let geometry = provider.geometry();
        let block = geometry.block_size as usize;
        if block == 0 || block > 4096 {
            return Err(PersistenceError::Incompatible);
        }
        let mut roots = [[0u8; 4096]; 2];
        provider.read_metadata(0, &mut roots[0][..block]).await?;
        provider.read_metadata(1, &mut roots[1][..block]).await?;
        let mut matched = false;
        for bytes in roots {
            let bytes = &bytes[..block];
            if bytes.iter().all(|byte| *byte == 0) {
                continue;
            }
            if bytes.get(..ROOT_MAGIC.len()) != Some(ROOT_MAGIC.as_slice()) {
                return Ok(false);
            }
            let Ok(root) = decode_root(bytes, geometry.block_size, geometry.block_count) else {
                return Ok(false);
            };
            if root.version != FORMAT_VERSION || root.era != expected {
                return Ok(false);
            }
            matched = true;
        }
        Ok(matched)
    }

    pub async fn format(
        provider: BlockProvider<D, B, O>,
        session: SessionId,
    ) -> Result<Self, PersistenceError> {
        Self::format_with_era(provider, session, [0u8; ERA_BYTES]).await
    }

    /// Format a fresh namespace and stamp its first durable generation with
    /// `era`. The caller owns policy: user formatting passes zero; only the
    /// already-clamped system path passes the measured archive digest.
    pub async fn format_with_era(
        provider: BlockProvider<D, B, O>,
        session: SessionId,
        era: [u8; ERA_BYTES],
    ) -> Result<Self, PersistenceError> {
        let geometry = provider.geometry();
        let layout = image_layout(ImageProfile::Current, geometry.block_size)?;
        provider.configure_arena(
            layout.data_start,
            geometry.block_count.saturating_sub(layout.data_start),
        )?;
        let engine = Engine::new(provider);
        engine.attach_session(session).await?;
        let mut volume = Self {
            engine,
            state: PersistenceState {
                status: MountStatus {
                    generation: 0,
                    used_fallback_root: false,
                    mode: MountMode::ReadWrite,
                    era,
                },
                version: FORMAT_VERSION,
                profile: ImageProfile::Current,
                image: ImageScratch::new()?,
                sidecar: None,
            },
        };
        volume.commit(session).await?;
        Ok(volume)
    }

    /// Destructively replace a rebuildable system namespace after its provider
    /// has been narrowed to the designated system extent.
    ///
    /// Both root slots are zeroed and flushed before the new generation is
    /// formatted. A cut before that flush leaves an old mismatched root and is
    /// retried by the era policy; a cut after it leaves blank media; a cut
    /// during the format leaves either blank media or one complete current-era
    /// root. No state can make recovery select a stale-era fallback.
    pub async fn reformat_with_era(
        provider: BlockProvider<D, B, O>,
        session: SessionId,
        era: [u8; ERA_BYTES],
    ) -> Result<Self, PersistenceError> {
        let block = provider.geometry().block_size as usize;
        if block == 0 || block > 4096 {
            return Err(PersistenceError::Incompatible);
        }
        let roots = [0u8; 8192];
        let bytes = block
            .checked_mul(2)
            .filter(|bytes| *bytes <= roots.len())
            .ok_or(PersistenceError::Bounds)?;
        provider.write_metadata(0, &roots[..bytes]).await?;
        provider.flush_device().await?;
        // Keep the fresh-format state machine inline. Awaiting
        // `format_with_era` here would nest two futures that each own the
        // affine provider/engine return shape; in an unoptimized build that
        // needlessly more than doubles the bounded startup frame (and does the
        // same to the founded service's root future).
        let geometry = provider.geometry();
        let layout = image_layout(ImageProfile::Current, geometry.block_size)?;
        provider.configure_arena(
            layout.data_start,
            geometry.block_count.saturating_sub(layout.data_start),
        )?;
        let engine = Engine::new(provider);
        engine.attach_session(session).await?;
        let mut volume = Self {
            engine,
            state: PersistenceState {
                status: MountStatus {
                    generation: 0,
                    used_fallback_root: false,
                    mode: MountMode::ReadWrite,
                    era,
                },
                version: FORMAT_VERSION,
                profile: ImageProfile::Current,
                image: ImageScratch::new()?,
                sidecar: None,
            },
        };
        volume.commit(session).await?;
        Ok(volume)
    }

    pub fn engine(&self) -> &Engine<BlockProvider<D, B, O>> {
        &self.engine
    }
    pub fn engine_mut(&mut self) -> &mut Engine<BlockProvider<D, B, O>> {
        &mut self.engine
    }
    pub const fn status(&self) -> MountStatus {
        self.state.status()
    }

    pub async fn commit(&mut self, session: SessionId) -> Result<u64, PersistenceError> {
        self.commit_with_cut(session, None).await
    }

    /// Deterministic power-cut injection. A cut means no durability
    /// acknowledgement; subsequent recovery may select the complete old or
    /// complete new root depending on whether publication became durable.
    pub async fn commit_with_cut(
        &mut self,
        session: SessionId,
        cut: Option<CommitStage>,
    ) -> Result<u64, PersistenceError> {
        commit_engine(&mut self.engine, &mut self.state, session, cut).await
    }

    /// Clean shutdown crosses the same durability boundary as an ordinary
    /// commit. No boot-local handle or capability is serialized; a later
    /// process still performs full root validation and recreates authority.
    pub async fn clean_shutdown(&mut self, session: SessionId) -> Result<u64, PersistenceError> {
        self.commit(session).await
    }

    pub async fn recover(
        provider: BlockProvider<D, B, O>,
        session: SessionId,
    ) -> Result<Self, PersistenceError> {
        Self::recover_witnessed(provider, session)
            .await
            .map_err(|failure| failure.error)
    }

    /// [`Self::recover`], naming the stage that refused the volume: what a
    /// boot that cannot mount its workspace reports, so a photographed
    /// failure says where recovery stopped and why
    /// ([`RecoverFailure::witness`]).
    pub async fn recover_witnessed(
        provider: BlockProvider<D, B, O>,
        session: SessionId,
    ) -> Result<Self, RecoverFailure> {
        let geometry = provider.geometry();
        let block = geometry.block_size as usize;
        if block > 4096 {
            return Err(RecoverFailure::at(RecoverStage::Geometry, PersistenceError::Incompatible));
        }
        let mut blocks = [[0u8; 4096]; 2];
        provider.read_metadata(0, &mut blocks[0][..block]).await.map_err(at(RecoverStage::RootBlocks))?;
        provider.read_metadata(1, &mut blocks[1][..block]).await.map_err(at(RecoverStage::RootBlocks))?;
        let parsed = [
            decode_root(
                &blocks[0][..block],
                geometry.block_size,
                geometry.block_count,
            ),
            decode_root(
                &blocks[1][..block],
                geometry.block_size,
                geometry.block_count,
            ),
        ];
        if parsed
            .iter()
            .all(|entry| matches!(entry, Err(PersistenceError::Unformatted)))
        {
            return Err(RecoverFailure::at(RecoverStage::Unformatted, PersistenceError::Unformatted));
        }
        if let (Ok(a), Ok(b)) = (parsed[0], parsed[1])
            && a.generation == b.generation
        {
            // Two independently published slots cannot lawfully claim the
            // same generation. Choosing by slot would silently bless an
            // ambiguous/tampered history.
            return Err(RecoverFailure::at(RecoverStage::SameGeneration, PersistenceError::Corrupt));
        }
        if let (Ok(a), Ok(b)) = (parsed[0], parsed[1])
            && (a.data_start != b.data_start || a.arena_blocks != b.arena_blocks)
        {
            // Both slots are generations of one volume and therefore one
            // allocation arena. Selecting either of two contradictory bounds
            // would make the other root's extents mean something different.
            return Err(RecoverFailure::at(RecoverStage::ArenaDisagrees, PersistenceError::Corrupt));
        }
        let mut order = [0usize, 1usize];
        let generation = |i: usize| parsed[i].as_ref().map_or(0, |root| root.generation);
        if generation(order[1]) > generation(order[0]) {
            order.swap(0, 1);
        }
        let newest_generation = generation(order[0]);
        let mut last = RecoverFailure::at(RecoverStage::RootDecode, PersistenceError::Corrupt);
        let mut image = ImageScratch::new().map_err(at(RecoverStage::Scratch))?;
        for (attempt, index) in order.into_iter().enumerate() {
            let root = match parsed[index] {
                Ok(root) => root,
                Err(error) => {
                    last = RecoverFailure::at(RecoverStage::RootDecode, error);
                    continue;
                }
            };
            if let Err(error) = provider.configure_arena(root.data_start, root.arena_blocks) {
                last = RecoverFailure::at(RecoverStage::Arena, error.into());
                continue;
            }
            let profile = root_profile(root.version, root.data_start, geometry.block_size).map_err(at(RecoverStage::Profile))?;
            let layout = image_layout(profile, geometry.block_size).map_err(at(RecoverStage::Profile))?;
            let arena_lba = image_lba(profile, root.slot, geometry.block_size).map_err(at(RecoverStage::Profile))?;
            let len = root.image_len as usize;
            let span = match image_transfer_bytes(len, geometry.block_size, layout.bytes) {
                Ok(span) => span,
                Err(error) => {
                    last = RecoverFailure::at(RecoverStage::ImageSpan, error);
                    continue;
                }
            };
            if let Err(error) = provider
                .read_metadata(arena_lba, &mut image.as_mut()[..span])
                .await
            {
                last = RecoverFailure::at(RecoverStage::ImageRead, error.into());
                continue;
            }
            if CRC.checksum(&image.as_mut()[..len]) != root.image_crc {
                last = RecoverFailure::at(RecoverStage::ImageCrc, PersistenceError::Corrupt);
                continue;
            }
            let decoded = match decode_image(
                &image.as_mut()[..len],
                root,
                geometry.block_size,
                geometry.block_count,
            ) {
                Ok(decoded) => decoded,
                Err(error) => {
                    last = RecoverFailure::at(RecoverStage::ImageDecode, error);
                    continue;
                }
            };
            // Read both recovery-selectable roots before installing any
            // allocation state, then rebuild the free index as one sorted
            // transaction. Exact current/fallback matches need one
            // reservation; fallback-only extents remain protected.
            let other = 1usize - index;
            let older = if let Ok(other_root) = parsed[other]
                && other_root.generation < decoded.generation
            {
                read_decoded(&provider, other_root, image.as_mut())
                    .await
                    .ok()
            } else {
                None
            };
            // Arena-backed like the decoded image above: these are sized by
            // the placement and derived bounds, and mount runs inside the
            // service's `serve` future on the founder's shared stack.
            let alloc = crate::service_alloc();
            let mut live_extents =
                crate::slot::BoundedVec::<DurableExtent, crate::ServiceAlloc>::try_new_in(
                    MAX_BLOCK_PLACEMENTS,
                    alloc,
                )
                .map_err(|_| RecoverFailure::at(RecoverStage::ExtentTable, PersistenceError::Bounds))?;
            for extent in root_payload_extents(&decoded) {
                live_extents
                    .push(extent)
                    .map_err(|_| RecoverFailure::at(RecoverStage::ExtentTable, PersistenceError::Bounds))?;
            }
            // The digest sidecar last, after every payload extent — the
            // order the recovered keys are consumed in below. A sidecar whose
            // extent does not lie inside the arena is derived data gone
            // wrong: it is dropped, never allowed to fail the mount.
            let sidecar = root
                .sidecar
                .filter(|sidecar| sidecar_in_arena(sidecar.extent, root, geometry.block_size));
            if let Some(sidecar) = sidecar {
                live_extents
                    .push(sidecar.extent)
                    .map_err(|_| RecoverFailure::at(RecoverStage::ExtentTable, PersistenceError::Bounds))?;
            }
            let mut fallback_extents =
                crate::slot::BoundedVec::<DurableExtent, crate::ServiceAlloc>::try_new_in(
                    MAX_BLOCK_PLACEMENTS,
                    alloc,
                )
                .map_err(|_| RecoverFailure::at(RecoverStage::ExtentTable, PersistenceError::Bounds))?;
            if let Some(older) = older.as_ref() {
                for extent in root_payload_extents(older) {
                    fallback_extents
                        .push(extent)
                        .map_err(|_| RecoverFailure::at(RecoverStage::ExtentTable, PersistenceError::Bounds))?;
                }
                // The older root's sidecar stays protected while that root
                // is still a recovery choice.
                if let Ok(other_root) = parsed[other]
                    && let Some(sidecar) = other_root.sidecar.filter(|sidecar| {
                        sidecar_in_arena(sidecar.extent, other_root, geometry.block_size)
                    })
                {
                    fallback_extents
                        .push(sidecar.extent)
                        .map_err(|_| RecoverFailure::at(RecoverStage::ExtentTable, PersistenceError::Bounds))?;
                }
            }
            let recovered_extents = provider.recover_extents(&live_extents, &fallback_extents)
                .map_err(at(RecoverStage::Extents))?;
            // Consume the keys through one iterator in the same canonical
            // order [`root_payload_extents`] defines; the exhaustion check
            // at the end proves the walks agreed.
            let mut recovered = recovered_extents.iter().copied();
            let ns = Namespace::<BlockKey>::new();
            for object in decoded.objects.iter() {
                let storage = match object.extent {
                    Some(_) => Some(recovered.next().ok_or(RecoverFailure::at(RecoverStage::ExtentWalk, PersistenceError::Corrupt))?),
                    None => None,
                };
                ns.restore_object(
                    object.stable,
                    object.kind,
                    object.subtype,
                    object.len,
                    object.revision,
                    storage,
                )
                .map_err(at(RecoverStage::Objects))?;
            }
            let mut recovered_derived = crate::slot::BoundedVec::<
                (DiskDerived, BlockKey),
                crate::ServiceAlloc,
            >::try_new_in(MAX_DERIVED_RESULTS, alloc)
            .map_err(|_| RecoverFailure::at(RecoverStage::DerivedTable, PersistenceError::Bounds))?;
            for record in decoded.derived.iter().copied() {
                let key = recovered.next().ok_or(RecoverFailure::at(RecoverStage::ExtentWalk, PersistenceError::Corrupt))?;
                recovered_derived
                    .push((record, key))
                    .map_err(|_| RecoverFailure::at(RecoverStage::DerivedTable, PersistenceError::Bounds))?;
            }
            let sidecar_key = match sidecar {
                Some(sidecar) => Some((recovered.next().ok_or(RecoverFailure::at(RecoverStage::ExtentWalk, PersistenceError::Corrupt))?, sidecar)),
                None => None,
            };
            if recovered.next().is_some() {
                return Err(RecoverFailure::at(RecoverStage::ExtentWalk, PersistenceError::Corrupt));
            }
            for edge in decoded.edges.iter() {
                ns.restore_edge(edge.parent, edge.name, edge.child)
                    .map_err(at(RecoverStage::Edges))?;
            }
            let root_id = ns
                .find_stable(decoded.root)
                .ok_or(RecoverFailure::at(RecoverStage::Root, PersistenceError::Corrupt))?;
            let engine = Engine::from_recovered(
                provider,
                ns,
                session,
                root_id,
                root.version >= SUBTYPE_FORMAT_VERSION,
            )
            .map_err(at(RecoverStage::Engine))?;
            // Load the remembered digests before revalidation needs them. A
            // sidecar that fails its CRC or shape loads nothing and is
            // marked for replacement; mount goes on either way.
            if let Some((key, sidecar)) = sidecar_key {
                let loaded = load_sidecar(&engine, key, sidecar).await;
                if !loaded {
                    engine.digests().borrow_mut().mark_dirty();
                }
            }
            for (record, storage) in recovered_derived {
                let recovered: Result<(), PersistenceError> = async {
                    let object = engine
                        .namespace()
                        .find_stable(record.object)
                        .ok_or(PersistenceError::Corrupt)?;
                    let current = engine.namespace().object(object)?;
                    let file_storage = current.storage();
                    let file_len = current.len();
                    let current_revision = current.revision();
                    drop(current);
                    if current_revision != record.input_version {
                        return Err(PersistenceError::Corrupt);
                    }

                    let mut wire = [0u8; kore_svc_extract::WORKER_REPORT_BYTES];
                    let wire_len =
                        usize::try_from(record.extent.len).map_err(|_| PersistenceError::Bounds)?;
                    let read = engine
                        .provider()
                        .read(storage, 0, &mut wire[..wire_len])
                        .await?;
                    if read != wire_len
                        || blake3::hash(&wire[..wire_len]).as_bytes() != &record.digest
                    {
                        return Err(PersistenceError::Corrupt);
                    }
                    let mut result = kore_svc_extract::decode_result(&wire[..wire_len])
                        .map_err(|_| PersistenceError::Corrupt)?;

                    // The source word is a boot-local object correlation id, not
                    // durable authority. Rebind it to the recovered generation,
                    // while proving the durable digest against the selected file
                    // payload before publishing the hot index.
                    let remembered = engine.digests().borrow().get(record.object, current_revision);
                    let mut hasher = blake3::Hasher::new();
                    let mut offset = if remembered.is_some() { file_len } else { 0u64 };
                    let mut chunk = [0u8; crate::bounds::MAX_TRANSFER];
                    while offset < file_len {
                        let want = usize::try_from((file_len - offset).min(chunk.len() as u64))
                            .map_err(|_| PersistenceError::Bounds)?;
                        let Some(file_storage) = file_storage else {
                            return Err(PersistenceError::Corrupt);
                        };
                        let count = engine
                            .provider()
                            .read(file_storage, offset, &mut chunk[..want])
                            .await?;
                        if count != want {
                            return Err(PersistenceError::Corrupt);
                        }
                        hasher.update(&chunk[..count]);
                        offset = offset
                            .checked_add(count as u64)
                            .ok_or(PersistenceError::Bounds)?;
                    }
                    let digest = match remembered {
                        Some(digest) => digest,
                        None => {
                            let digest = *hasher.finalize().as_bytes();
                            engine
                                .digests()
                                .borrow_mut()
                                .put(record.object, current_revision, digest);
                            digest
                        }
                    };
                    if result.provenance.input.len != file_len
                        || result.provenance.input.version != current_revision
                        || result.provenance.input.digest != digest
                    {
                        return Err(PersistenceError::Corrupt);
                    }
                    result.provenance.input.source = object.to_bits();
                    let parser = kore_svc_extract::registry::Registry::get_version(
                        result.provenance.parser,
                        result.provenance.parser_version,
                    )
                    .ok_or(PersistenceError::Incompatible)?;
                    kore_svc_extract::registry::Registry::validate(
                        &result,
                        result.provenance.input,
                        parser.limits.max_output,
                    )
                    .map_err(|_| PersistenceError::Corrupt)?;
                    engine.restore_derived_result(
                        object,
                        result,
                        storage,
                        u32::try_from(wire_len).map_err(|_| PersistenceError::Bounds)?,
                        record.digest,
                    )?;
                    Ok(())
                }
                .await;
                // Derived data is rebuildable, never authority. A corrupt,
                // obsolete, or no-longer-supported batch is omitted from the
                // hot index and becomes a typed IndexIncomplete observation;
                // it cannot make the authoritative namespace unmountable.
                if recovered.is_err() {
                    continue;
                }
            }
            let used_fallback_root = attempt != 0
                || decoded.generation != newest_generation
                || matches!(
                    parsed[other],
                    Err(PersistenceError::Corrupt | PersistenceError::Incompatible)
                );
            return Ok(Self {
                engine,
                state: PersistenceState {
                    status: MountStatus {
                        generation: decoded.generation,
                        used_fallback_root,
                        mode: if geometry.read_only || used_fallback_root {
                            MountMode::DegradedReadOnly
                        } else {
                            MountMode::ReadWrite
                        },
                        era: decoded.era,
                    },
                    version: root.version,
                    profile,
                    image,
                    sidecar: sidecar_key,
                },
            });
        }
        Err(last)
    }

    pub async fn restart(self, session: SessionId) -> Result<Self, PersistenceError> {
        let provider = self.engine.into_provider();
        let bootstrap = bootstrap_data_start(provider.geometry().block_size)?;
        let provider = provider.restart_with_data_start(bootstrap)?;
        Self::recover(provider, session).await
    }

    /// Split the mounted volume into the request engine and the durability
    /// cursor a founded service keeps beside it.
    pub fn into_parts(self) -> (Engine<BlockProvider<D, B, O>>, PersistenceState) {
        (self.engine, self.state)
    }
}

/// Publish one engine snapshot through the same dual-root transaction used by
/// [`PersistentFs`]. This is the founded service's commit hook: request
/// dispatch owns the engine, while this helper owns durable publication.
pub async fn commit_engine<D: DirectBlockDevice, const B: usize, const O: usize>(
    engine: &mut Engine<BlockProvider<D, B, O>>,
    state: &mut PersistenceState,
    session: SessionId,
    cut: Option<CommitStage>,
) -> Result<u64, PersistenceError> {
    if state.status.mode == MountMode::DegradedReadOnly {
        return Err(PersistenceError::Provider(ProviderError::ReadOnly));
    }
    // A version-4 volume is upgraded by this commit; older ones keep their
    // version (their reservation and records stay what they were).
    let version = if state.version == SUBTYPE_FORMAT_VERSION {
        DIGEST_FORMAT_VERSION
    } else {
        state.version
    };
    // The digest sidecar is staged into a fresh extent before the image, so
    // the root below can name it. It is derived data: a sidecar that cannot
    // be written leaves the root naming the previous one (or none), never a
    // failed commit.
    let staged = if version >= DIGEST_FORMAT_VERSION {
        stage_sidecar(engine, state).await
    } else {
        SidecarStage::Keep
    };
    let sidecar = match staged {
        SidecarStage::Keep => state.sidecar.map(|(_, sidecar)| sidecar),
        SidecarStage::Fresh { sidecar, .. } => Some(sidecar),
        SidecarStage::Drop => None,
    };
    let mut submitted = false;
    match publish_root(engine, state, session, cut, version, sidecar, &mut submitted).await {
        Ok(generation) => {
            state.version = version;
            let retired = match staged {
                SidecarStage::Keep => None,
                SidecarStage::Fresh { key, sidecar } => state.sidecar.replace((key, sidecar)),
                SidecarStage::Drop => state.sidecar.take(),
            };
            if let Some((old, _)) = retired
                && engine.provider().release(old).await.is_err()
            {
                engine.namespace().note_leaked_placement();
            }
            Ok(generation)
        }
        Err(error) => {
            // The memo's change is not durable: the next commit writes it.
            if !matches!(staged, SidecarStage::Keep) {
                engine.digests().borrow_mut().mark_dirty();
            }
            // Before the root write was submitted nothing durable can name
            // the fresh sidecar; after it, the new root may have landed, so
            // the extent stays put until a mount rebuilds the allocator.
            if let SidecarStage::Fresh { key, .. } = staged
                && !submitted
                && engine.provider().release(key).await.is_err()
            {
                engine.namespace().note_leaked_placement();
            }
            Err(error)
        }
    }
}

/// What a commit does with the digest sidecar.
#[derive(Clone, Copy)]
enum SidecarStage {
    /// The memo is unchanged (or could not be written): the root names the
    /// sidecar the previous one did.
    Keep,
    /// A fresh sidecar was written; the root names it.
    Fresh { key: BlockKey, sidecar: SidecarRef },
    /// The memo is empty: the root names none.
    Drop,
}

/// Prune the memo to live revisions and, if it changed, write it into a
/// fresh extent. Never fails the commit: every refusal is [`SidecarStage::Keep`]
/// with the memo left dirty for the next commit.
async fn stage_sidecar<D: DirectBlockDevice, const B: usize, const O: usize>(
    engine: &Engine<BlockProvider<D, B, O>>,
    state: &PersistenceState,
) -> SidecarStage {
    let encoded = {
        let ns = engine.namespace();
        let mut memo = engine.digests().borrow_mut();
        memo.prune(|stable| {
            let id = ns.find_stable(stable)?;
            ns.object(id).ok().map(|object| object.revision())
        });
        if !memo.dirty() && (state.sidecar.is_some() || memo.is_empty()) {
            return SidecarStage::Keep;
        }
        if memo.is_empty() {
            memo.mark_clean();
            return SidecarStage::Drop;
        }
        let mut encoded = Vec::new();
        if encoded.try_reserve_exact(memo.encoded_bytes()).is_err() {
            return SidecarStage::Keep;
        }
        encoded.resize(memo.encoded_bytes(), 0);
        if memo.encode(&mut encoded).is_none() {
            memo.mark_dirty();
            return SidecarStage::Keep;
        }
        encoded
    };
    let provider = engine.provider();
    let written: Result<BlockKey, ProviderError> = async {
        let key = provider
            .allocate(ObjectKind::File, encoded.len() as u64)
            .await?;
        if let Err(error) = async {
            provider.write(key, 0, &encoded).await?;
            provider.sync(key).await
        }
        .await
        {
            let _ = provider.release(key).await;
            return Err(error);
        }
        Ok(key)
    }
    .await;
    let Ok(key) = written else {
        engine.digests().borrow_mut().mark_dirty();
        return SidecarStage::Keep;
    };
    let Ok(extent) = provider.extent(key) else {
        let _ = provider.release(key).await;
        engine.digests().borrow_mut().mark_dirty();
        return SidecarStage::Keep;
    };
    SidecarStage::Fresh {
        key,
        sidecar: SidecarRef {
            extent,
            crc: CRC.checksum(&encoded),
        },
    }
}

/// The root publication itself: image, flush, root, flush, checkpoint.
/// `submitted` turns true the moment the root write may have landed.
async fn publish_root<D: DirectBlockDevice, const B: usize, const O: usize>(
    engine: &mut Engine<BlockProvider<D, B, O>>,
    state: &mut PersistenceState,
    session: SessionId,
    cut: Option<CommitStage>,
    version: u32,
    sidecar: Option<SidecarRef>,
    submitted: &mut bool,
) -> Result<u64, PersistenceError> {
    let generation = state
        .status
        .generation
        .checked_add(1)
        .ok_or(PersistenceError::Bounds)?;
    let slot = (generation & 1) as u8;
    let root_id = engine.root_object(session)?;
    let image = state.image.as_mut();
    let (image_len, root) = encode_image(
        engine,
        root_id,
        generation,
        version,
        state.profile,
        image,
    )?;
    let provider = engine.provider();
    let geometry = provider.geometry();
    let layout = image_layout(state.profile, geometry.block_size)?;
    let arena_lba = image_lba(state.profile, slot, geometry.block_size)?;
    if cut == Some(CommitStage::BeforeMetadata) {
        return Err(PersistenceError::Interrupted(CommitStage::BeforeMetadata));
    }
    // Zero the tail of the written span rather than trusting the buffer to
    // have arrived clean: the padding lands on disk, and a future scratch
    // buffer reused across commits would otherwise publish stale bytes.
    let span = image_transfer_bytes(image_len, geometry.block_size, layout.bytes)?;
    image
        .get_mut(image_len..span)
        .ok_or(PersistenceError::Bounds)?
        .fill(0);
    provider.write_metadata(arena_lba, &image[..span]).await?;
    if cut == Some(CommitStage::AfterMetadata) {
        return Err(PersistenceError::Interrupted(CommitStage::AfterMetadata));
    }
    provider.flush_device().await?;
    if cut == Some(CommitStage::AfterMetadataFlush) {
        return Err(PersistenceError::Interrupted(
            CommitStage::AfterMetadataFlush,
        ));
    }
    let mut root_block = [0u8; 4096];
    let block = geometry.block_size as usize;
    encode_root(
        RootRecord {
            version,
            generation,
            block_size: geometry.block_size,
            block_count: geometry.block_count,
            slot,
            root,
            image_len: image_len as u32,
            image_crc: CRC.checksum(&image[..image_len]),
            data_start: provider.arena().0,
            arena_blocks: provider.arena().1,
            // Preserved, not re-stamped: this commit is not the authority on
            // which build's store this is.
            era: state.status.era,
            sidecar,
        },
        &mut root_block[..block],
    )?;
    // Once the root write is submitted, a failed completion or flush can
    // still leave this root durable. Freeze every extent it may name before
    // that cut so a later request cannot overwrite a recoverable generation
    // in place.
    provider.prepare_publication();
    *submitted = true;
    provider
        .write_metadata(slot as u64, &root_block[..block])
        .await?;
    if cut == Some(CommitStage::AfterRootWrite) {
        return Err(PersistenceError::Interrupted(CommitStage::AfterRootWrite));
    }
    provider.flush_device().await?;
    if cut == Some(CommitStage::AfterRootFlush) {
        return Err(PersistenceError::Interrupted(CommitStage::AfterRootFlush));
    }
    state.status = MountStatus {
        generation,
        used_fallback_root: false,
        mode: MountMode::ReadWrite,
        era: state.status.era,
    };
    provider.checkpoint().await;
    Ok(generation)
}

/// Whether a root's sidecar extent lies inside that root's data arena and
/// is long enough to hold what it claims.
fn sidecar_in_arena(extent: DurableExtent, root: RootRecord, block_size: u32) -> bool {
    let arena_end = root.data_start.saturating_add(root.arena_blocks);
    let end = extent.lba.checked_add(u64::from(extent.blocks));
    extent.blocks != 0
        && extent.lba >= root.data_start
        && end.is_some_and(|end| end <= arena_end)
        && extent.len <= u64::from(extent.blocks) * u64::from(block_size)
        && extent.len as usize >= crate::digests::HEADER_BYTES
}

/// Read and load the sidecar the root names; `false` when it could not be
/// read, fails its CRC, or does not decode.
async fn load_sidecar<D: DirectBlockDevice, const B: usize, const O: usize>(
    engine: &Engine<BlockProvider<D, B, O>>,
    key: BlockKey,
    sidecar: SidecarRef,
) -> bool {
    let Ok(len) = usize::try_from(sidecar.extent.len) else {
        return false;
    };
    let mut bytes = Vec::new();
    if bytes.try_reserve_exact(len).is_err() {
        return false;
    }
    bytes.resize(len, 0);
    let mut read = 0;
    while read < len {
        let want = (len - read).min(engine.provider().limits().max_transfer);
        match engine
            .provider()
            .read(key, read as u64, &mut bytes[read..read + want])
            .await
        {
            Ok(n) if n == want => read += n,
            _ => return false,
        }
    }
    if CRC.checksum(&bytes) != sidecar.crc {
        return false;
    }
    engine.digests().borrow_mut().load(&bytes).is_ok()
}

async fn read_decoded<D: DirectBlockDevice, const B: usize, const O: usize>(
    provider: &BlockProvider<D, B, O>,
    root: RootRecord,
    image: &mut [u8],
) -> Result<Decoded, PersistenceError> {
    let geometry = provider.geometry();
    let profile = root_profile(root.version, root.data_start, geometry.block_size)?;
    let layout = image_layout(profile, geometry.block_size)?;
    let arena_lba = image_lba(profile, root.slot, geometry.block_size)?;
    let len = root.image_len as usize;
    let span = image_transfer_bytes(len, geometry.block_size, layout.bytes)?;
    provider
        .read_metadata(arena_lba, &mut image[..span])
        .await?;
    if CRC.checksum(&image[..len]) != root.image_crc {
        return Err(PersistenceError::Corrupt);
    }
    decode_image(
        &image[..len],
        root,
        geometry.block_size,
        geometry.block_count,
    )
}

/// Bytes of an image slot a generation actually has to move.
///
/// The slot is a fixed *reservation*; a commit only has to transfer the bytes
/// it wrote. `image_len` is already recorded in the root record and already
/// bounds the CRC, so both halves agree without inventing a second source of
/// truth. A tree that fits in one block costs one block rather than the whole
/// reservation — every `touch` and `write` on a small volume was moving 16 KiB
/// of mostly-zero metadata, and it is what makes raising the reservation
/// affordable instead of an 8x write amplification on every file operation.
fn image_transfer_bytes(
    image_len: usize,
    block_size: u32,
    image_bytes: usize,
) -> Result<usize, PersistenceError> {
    let block = block_size as usize;
    if block == 0 || !image_bytes.is_multiple_of(block) {
        return Err(PersistenceError::Incompatible);
    }
    if image_len == 0 || image_len > image_bytes {
        return Err(PersistenceError::Bounds);
    }
    // `block` divides `image_bytes` and `image_len <= image_bytes`, so this
    // cannot exceed the reservation.
    Ok(image_len.div_ceil(block) * block)
}

fn image_lba(profile: ImageProfile, slot: u8, block_size: u32) -> Result<u64, PersistenceError> {
    if slot > 1 {
        return Err(PersistenceError::Corrupt);
    }
    let layout = image_layout(profile, block_size)?;
    let start = 2u64
        .checked_add(u64::from(slot) * u64::from(layout.slot_blocks))
        .ok_or(PersistenceError::Bounds)?;
    let end = start
        .checked_add(u64::from(layout.slot_blocks))
        .ok_or(PersistenceError::Bounds)?;
    if end > layout.data_start {
        return Err(PersistenceError::Incompatible);
    }
    Ok(start)
}

fn encode_image<D: DirectBlockDevice, const B: usize, const O: usize>(
    engine: &Engine<BlockProvider<D, B, O>>,
    root_id: crate::ObjectId,
    generation: u64,
    version: u32,
    profile: ImageProfile,
    image: &mut [u8],
) -> Result<(usize, StableObjectId), PersistenceError> {
    let (max_objects, max_edges) = durable_bounds(profile);
    let graph = engine
        .namespace()
        .durable_graph(root_id, MAX_DURABLE_EDGES)?;
    if graph.objects.len() > max_objects || graph.edges.len() > max_edges {
        return Err(PersistenceError::Bounds);
    }
    let max_dir_entries = durable_dir_entries(profile);
    if graph.objects.iter().any(|object| {
        object.kind.is_dir()
            && graph
                .edges
                .iter()
                .filter(|edge| edge.parent == object.stable)
                .count()
                > max_dir_entries
    }) {
        return Err(PersistenceError::Bounds);
    }
    let root = graph.root;
    let derived = if version >= DERIVED_FORMAT_VERSION {
        engine.durable_derived_results::<MAX_DERIVED_RESULTS>()?
    } else {
        heapless::Vec::new()
    };
    image
        .get_mut(..IMAGE_HEADER)
        .ok_or(PersistenceError::Bounds)?
        .fill(0);
    put(image, 0, IMAGE_MAGIC)?;
    put_u32(image, 8, version)?;
    put_u32(image, 12, IMAGE_HEADER as u32)?;
    put_u64(image, 16, generation)?;
    put_u64(image, 24, root.get())?;
    put_u32(image, 32, graph.objects.len() as u32)?;
    put_u32(image, 36, graph.edges.len() as u32)?;
    put_u32(image, 40, OBJECT_BYTES as u32)?;
    put_u32(image, 44, EDGE_BYTES as u32)?;
    if version >= DERIVED_FORMAT_VERSION {
        put_u32(image, 48, derived.len() as u32)?;
        put_u32(image, 52, DERIVED_BYTES as u32)?;
    }
    let mut at = IMAGE_HEADER;
    for object in graph.objects {
        image
            .get_mut(at..at + OBJECT_BYTES)
            .ok_or(PersistenceError::Bounds)?
            .fill(0);
        let extent = match object.storage {
            Some(key) => Some(engine.provider().extent(key)?),
            None => None,
        };
        put_u64(image, at, object.stable.get())?;
        image[at + 8] = if object.kind.is_file() { 1 } else { 2 };
        image[at + 9] = u8::from(extent.is_some());
        // Byte 10 is the subtype from version 4; before it the byte is
        // reserved-must-be-zero, and a fact the version cannot record is
        // refused rather than dropped.
        if version >= SUBTYPE_FORMAT_VERSION {
            image[at + 10] = object.subtype.to_bits();
        } else if !object.subtype.is_plain() {
            return Err(PersistenceError::Incompatible);
        }
        put_u64(image, at + 16, object.len)?;
        put_u64(image, at + 24, object.revision)?;
        if let Some(extent) = extent {
            put_u64(image, at + 32, extent.lba)?;
            put_u32(image, at + 40, extent.blocks)?;
        }
        let checksum = CRC.checksum(&image[at..at + 44]);
        put_u32(image, at + 44, checksum)?;
        at += OBJECT_BYTES;
    }
    for edge in graph.edges {
        image
            .get_mut(at..at + EDGE_BYTES)
            .ok_or(PersistenceError::Bounds)?
            .fill(0);
        put_u64(image, at, edge.parent.get())?;
        put_u64(image, at + 8, edge.child.get())?;
        image[at + 16] = edge.name.len() as u8;
        put(image, at + 20, edge.name.as_bytes())?;
        let checksum = CRC.checksum(&image[at..at + 52]);
        put_u32(image, at + 52, checksum)?;
        at += EDGE_BYTES;
    }
    for record in derived {
        image
            .get_mut(at..at + DERIVED_BYTES)
            .ok_or(PersistenceError::Bounds)?
            .fill(0);
        let object = engine.namespace().object(record.object)?;
        if !object.kind().is_file()
            || object.revision() != record.input.version
            || object.len() != record.input.len
        {
            return Err(PersistenceError::Corrupt);
        }
        let extent = engine.provider().extent(record.key)?;
        if extent.len != u64::from(record.len) {
            return Err(PersistenceError::Corrupt);
        }
        put_u64(image, at, object.stable().get())?;
        put_u64(image, at + 8, record.input.version)?;
        put_u64(image, at + 16, extent.lba)?;
        put_u32(image, at + 24, extent.blocks)?;
        put_u32(image, at + 28, record.len)?;
        put(image, at + 32, &record.digest)?;
        let checksum = CRC.checksum(&image[at..at + 64]);
        put_u32(image, at + 64, checksum)?;
        at += DERIVED_BYTES;
    }
    Ok((at, root))
}

fn encode_root(root: RootRecord, block: &mut [u8]) -> Result<(), PersistenceError> {
    let header = root_header_bytes(root.version).ok_or(PersistenceError::Incompatible)?;
    if block.len() < header {
        return Err(PersistenceError::Incompatible);
    }
    block.fill(0);
    put(block, 0, ROOT_MAGIC)?;
    put_u32(block, 8, root.version)?;
    put_u32(block, 12, header as u32)?;
    put_u64(block, 16, root.generation)?;
    put_u32(block, 24, root.block_size)?;
    block[28] = root.slot;
    block[29] = 1; // complete root record
    put_u64(block, 32, root.block_count)?;
    put_u64(block, 40, root.root.get())?;
    put_u32(block, 48, root.image_len)?;
    put_u32(block, 52, root.image_crc)?;
    if header >= ROOT_HEADER_V3 {
        put_u64(block, 56, root.data_start)?;
        put_u64(block, 64, root.arena_blocks)?;
        put(block, 72, &root.era)?;
    }
    if header >= ROOT_HEADER_V5
        && let Some(sidecar) = root.sidecar
    {
        put_u64(block, 104, sidecar.extent.lba)?;
        put_u32(block, 112, sidecar.extent.blocks)?;
        put_u32(
            block,
            116,
            u32::try_from(sidecar.extent.len).map_err(|_| PersistenceError::Bounds)?,
        )?;
        put_u32(block, 120, sidecar.crc)?;
    }
    // The checksum is the header's last four bytes and covers everything
    // before it — one rule at every version, so a new field is added by
    // moving the boundary rather than by inventing a second layout.
    let covered = header - 4;
    let checksum = CRC.checksum(&block[..covered]);
    put_u32(block, covered, checksum)
}

fn decode_root(
    block: &[u8],
    block_size: u32,
    block_count: u64,
) -> Result<RootRecord, PersistenceError> {
    if block.iter().all(|byte| *byte == 0) {
        return Err(PersistenceError::Unformatted);
    }
    if block.get(..8) != Some(ROOT_MAGIC.as_slice()) {
        return Err(PersistenceError::Corrupt);
    }
    let version = get_u32(block, 8)?;
    // A version this build does not know, or a header length that disagrees
    // with the one that version defines, is incompatible — never guessed at.
    let Some(header) = root_header_bytes(version) else {
        return Err(PersistenceError::Incompatible);
    };
    if get_u32(block, 12)? as usize != header {
        return Err(PersistenceError::Incompatible);
    }
    if get_u32(block, 24)? != block_size || get_u64(block, 32)? != block_count {
        return Err(PersistenceError::Incompatible);
    }
    let covered = header - 4;
    if block.get(29) != Some(&1)
        || block
            .get(30..32)
            .is_none_or(|reserved| reserved.iter().any(|byte| *byte != 0))
        || block
            .get(header..)
            .is_none_or(|reserved| reserved.iter().any(|byte| *byte != 0))
        || CRC.checksum(block.get(..covered).ok_or(PersistenceError::Corrupt)?)
            != get_u32(block, covered)?
    {
        return Err(PersistenceError::Corrupt);
    }
    // Versions before 3 recorded neither the arena nor the era. They decode
    // to what they meant: the compile-time floor, everything above it, and an
    // unstamped era — not to zero, which would describe an empty volume.
    //
    // The fields are present exactly when the header is the version-3 one —
    // the rule `encode_root` writes by. Keying this on "equals the current
    // version" instead read every version-3 root as legacy once version 4
    // landed: a raised-profile v3 volume then located its image at the
    // legacy slot and its era read as unstamped.
    let (data_start, arena_blocks, era) = if header >= ROOT_HEADER_V3 {
        let mut era = [0u8; ERA_BYTES];
        era.copy_from_slice(
            block
                .get(72..72 + ERA_BYTES)
                .ok_or(PersistenceError::Corrupt)?,
        );
        (get_u64(block, 56)?, get_u64(block, 64)?, era)
    } else {
        (
            LEGACY_DATA_START_LBA,
            block_count.saturating_sub(LEGACY_DATA_START_LBA),
            [0u8; ERA_BYTES],
        )
    };
    let sidecar = if header >= ROOT_HEADER_V5 {
        let blocks = get_u32(block, 112)?;
        (blocks != 0)
            .then(|| -> Result<SidecarRef, PersistenceError> {
                Ok(SidecarRef {
                    extent: DurableExtent {
                        lba: get_u64(block, 104)?,
                        blocks,
                        len: u64::from(get_u32(block, 116)?),
                    },
                    crc: get_u32(block, 120)?,
                })
            })
            .transpose()?
    } else {
        None
    };
    let root = RootRecord {
        version,
        generation: get_u64(block, 16)?,
        block_size,
        block_count,
        slot: *block.get(28).ok_or(PersistenceError::Corrupt)?,
        root: StableObjectId::new(get_u64(block, 40)?).ok_or(PersistenceError::Corrupt)?,
        image_len: get_u32(block, 48)?,
        image_crc: get_u32(block, 52)?,
        data_start,
        arena_blocks,
        era,
        sidecar,
    };
    let profile = root_profile(version, root.data_start, block_size)?;
    let layout = image_layout(profile, block_size)?;
    if root.generation == 0 || root.slot > 1 || root.image_len as usize > layout.bytes {
        return Err(PersistenceError::Corrupt);
    }
    // An arena that starts below the metadata reservation, or runs past the
    // medium, describes a volume that cannot exist.
    let arena_end = root
        .data_start
        .checked_add(root.arena_blocks)
        .ok_or(PersistenceError::Corrupt)?;
    if root.data_start < layout.data_start || root.arena_blocks == 0 || arena_end > block_count {
        return Err(PersistenceError::Corrupt);
    }
    Ok(root)
}

fn decode_image(
    image: &[u8],
    root: RootRecord,
    block_size: u32,
    _block_count: u64,
) -> Result<Decoded, PersistenceError> {
    if image.get(..8) != Some(IMAGE_MAGIC.as_slice())
        || get_u32(image, 8)? != root.version
        || get_u32(image, 12)? as usize != IMAGE_HEADER
        || get_u64(image, 16)? != root.generation
        || get_u64(image, 24)? != root.root.get()
        || get_u32(image, 40)? as usize != OBJECT_BYTES
        || get_u32(image, 44)? as usize != EDGE_BYTES
    {
        return Err(PersistenceError::Corrupt);
    }
    let object_count = get_u32(image, 32)? as usize;
    let edge_count = get_u32(image, 36)? as usize;
    let derived_count = if root.version >= DERIVED_FORMAT_VERSION {
        if get_u32(image, 52)? as usize != DERIVED_BYTES
            || image[56..IMAGE_HEADER].iter().any(|byte| *byte != 0)
        {
            return Err(PersistenceError::Corrupt);
        }
        get_u32(image, 48)? as usize
    } else {
        if image[48..IMAGE_HEADER].iter().any(|byte| *byte != 0) {
            return Err(PersistenceError::Corrupt);
        }
        0
    };
    let profile = root_profile(root.version, root.data_start, block_size)?;
    let (max_objects, _) = durable_bounds(profile);
    // Edges are read against what any writer of this profile could have
    // recorded, not the current writer's ceiling: see
    // `PRE_DECOUPLING_MAX_DURABLE_EDGES`. The byte total is still pinned by
    // the exact-length check below, and the image itself by its slot.
    let max_edges = readable_durable_edges(profile);
    if object_count == 0
        || object_count > max_objects
        || edge_count > max_edges
        || derived_count > MAX_DERIVED_RESULTS
    {
        return Err(PersistenceError::Bounds);
    }
    let required = IMAGE_HEADER
        .checked_add(
            object_count
                .checked_mul(OBJECT_BYTES)
                .ok_or(PersistenceError::Bounds)?,
        )
        .and_then(|n| n.checked_add(edge_count.checked_mul(EDGE_BYTES)?))
        .and_then(|n| n.checked_add(derived_count.checked_mul(DERIVED_BYTES)?))
        .ok_or(PersistenceError::Bounds)?;
    if required != image.len() {
        return Err(PersistenceError::Corrupt);
    }
    // Counts are bounds-checked above, so these reservations are bounded by
    // the profile and fail closed rather than growing during the fill.
    let mut objects = Vec::new();
    objects
        .try_reserve_exact(object_count)
        .map_err(|_| PersistenceError::Bounds)?;
    let mut at = IMAGE_HEADER;
    for _ in 0..object_count {
        if CRC.checksum(&image[at..at + 44]) != get_u32(image, at + 44)? {
            return Err(PersistenceError::Corrupt);
        }
        let stable = StableObjectId::new(get_u64(image, at)?).ok_or(PersistenceError::Corrupt)?;
        if objects.iter().any(|old: &DiskObject| old.stable == stable) {
            return Err(PersistenceError::Corrupt);
        }
        let kind = match image[at + 8] {
            1 => ObjectKind::File,
            2 => ObjectKind::Directory,
            _ => return Err(PersistenceError::Corrupt),
        };
        let has_extent = match image[at + 9] {
            0 => false,
            1 => true,
            _ => return Err(PersistenceError::Corrupt),
        };
        let subtype = if root.version >= SUBTYPE_FORMAT_VERSION {
            ObjectSubtype::from_bits(image[at + 10])
        } else {
            if image[at + 10] != 0 {
                return Err(PersistenceError::Corrupt);
            }
            ObjectSubtype::PLAIN
        };
        if kind.is_dir() && !subtype.is_plain() {
            return Err(PersistenceError::Corrupt);
        }
        if image[at + 11..at + 16].iter().any(|byte| *byte != 0) {
            return Err(PersistenceError::Corrupt);
        }
        let len = get_u64(image, at + 16)?;
        let revision = get_u64(image, at + 24)?;
        if revision < FIRST_REVISION {
            return Err(PersistenceError::Corrupt);
        }
        let extent = has_extent.then(|| DurableExtent {
            lba: get_u64(image, at + 32).unwrap_or(0),
            blocks: get_u32(image, at + 40).unwrap_or(0),
            len,
        });
        if !has_extent && (get_u64(image, at + 32)? != 0 || get_u32(image, at + 40)? != 0) {
            return Err(PersistenceError::Corrupt);
        }
        if kind.is_dir() && (len != 0 || extent.is_some()) {
            return Err(PersistenceError::Corrupt);
        }
        if kind.is_file() && len != 0 && extent.is_none() {
            return Err(PersistenceError::Corrupt);
        }
        if let Some(extent) = extent {
            let expected = len
                .checked_add(block_size as u64 - 1)
                .ok_or(PersistenceError::Bounds)?
                / block_size as u64;
            let end = extent
                .lba
                .checked_add(extent.blocks as u64)
                .ok_or(PersistenceError::Bounds)?;
            let arena_end = root
                .data_start
                .checked_add(root.arena_blocks)
                .ok_or(PersistenceError::Bounds)?;
            // An empty file's placement holds no blocks and so no address:
            // the provider records it as `lba 0, blocks 0`
            // (`BlockProvider::allocate`), and `recover_extents` reserves
            // nothing for it. That one form is lawful outside the arena; any
            // other address on a zero-block extent is not something a writer
            // produces.
            let empty = extent.blocks == 0;
            if expected != extent.blocks as u64
                || (empty && extent.lba != 0)
                || (!empty && (extent.lba < root.data_start || end > arena_end))
            {
                return Err(PersistenceError::Corrupt);
            }
            if objects
                .iter()
                .filter_map(|old| old.extent)
                .any(|old| extent.lba < old.lba + old.blocks as u64 && old.lba < end)
            {
                return Err(PersistenceError::Corrupt);
            }
        }
        objects.push(DiskObject {
            stable,
            kind,
            subtype,
            len,
            revision,
            extent,
        });
        at += OBJECT_BYTES;
    }
    if objects.iter().filter(|object| object.kind.is_dir()).count() > MAX_DIRECTORIES {
        return Err(PersistenceError::Bounds);
    }
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(edge_count)
        .map_err(|_| PersistenceError::Bounds)?;
    for _ in 0..edge_count {
        if CRC.checksum(&image[at..at + 52]) != get_u32(image, at + 52)? {
            return Err(PersistenceError::Corrupt);
        }
        let parent = StableObjectId::new(get_u64(image, at)?).ok_or(PersistenceError::Corrupt)?;
        let child =
            StableObjectId::new(get_u64(image, at + 8)?).ok_or(PersistenceError::Corrupt)?;
        let len = image[at + 16] as usize;
        if len == 0
            || len > MAX_NAME_LEN
            || image[at + 17..at + 20].iter().any(|byte| *byte != 0)
            || image[at + 20 + len..at + 52].iter().any(|byte| *byte != 0)
        {
            return Err(PersistenceError::Corrupt);
        }
        let name = Name::from_bytes(&image[at + 20..at + 20 + len])
            .map_err(|_| PersistenceError::Corrupt)?;
        if !objects
            .iter()
            .any(|o| o.stable == parent && o.kind.is_dir())
            || !objects.iter().any(|o| o.stable == child)
            || edges
                .iter()
                .any(|old: &DiskEdge| old.parent == parent && old.name == name)
        {
            return Err(PersistenceError::Corrupt);
        }
        edges.push(DiskEdge {
            parent,
            child,
            name,
        });
        at += EDGE_BYTES;
    }
    let max_dir_entries = durable_dir_entries(profile);
    if objects.iter().any(|object| {
        object.kind.is_dir()
            && edges
                .iter()
                .filter(|edge| edge.parent == object.stable)
                .count()
                > max_dir_entries
    }) {
        return Err(PersistenceError::Bounds);
    }
    let mut derived = Vec::new();
    derived
        .try_reserve_exact(derived_count)
        .map_err(|_| PersistenceError::Bounds)?;
    for _ in 0..derived_count {
        if CRC.checksum(&image[at..at + 64]) != get_u32(image, at + 64)?
            || image[at + 68..at + DERIVED_BYTES]
                .iter()
                .any(|byte| *byte != 0)
        {
            return Err(PersistenceError::Corrupt);
        }
        let object = StableObjectId::new(get_u64(image, at)?).ok_or(PersistenceError::Corrupt)?;
        let input_version = get_u64(image, at + 8)?;
        let extent = DurableExtent {
            lba: get_u64(image, at + 16)?,
            blocks: get_u32(image, at + 24)?,
            len: u64::from(get_u32(image, at + 28)?),
        };
        let digest = image[at + 32..at + 64]
            .try_into()
            .map_err(|_| PersistenceError::Corrupt)?;
        let owner = objects
            .iter()
            .find(|candidate| candidate.stable == object)
            .ok_or(PersistenceError::Corrupt)?;
        let expected_blocks = extent
            .len
            .checked_add(block_size as u64 - 1)
            .ok_or(PersistenceError::Bounds)?
            / block_size as u64;
        let end = extent
            .lba
            .checked_add(u64::from(extent.blocks))
            .ok_or(PersistenceError::Bounds)?;
        let arena_end = root
            .data_start
            .checked_add(root.arena_blocks)
            .ok_or(PersistenceError::Bounds)?;
        if !owner.kind.is_file()
            || owner.revision != input_version
            || extent.len == 0
            || extent.len > kore_svc_extract::WORKER_REPORT_BYTES as u64
            || expected_blocks != u64::from(extent.blocks)
            || extent.lba < root.data_start
            || end > arena_end
            || objects
                .iter()
                .filter_map(|old| old.extent)
                .any(|old| extent.lba < old.lba + u64::from(old.blocks) && old.lba < end)
            || derived.iter().any(|old: &DiskDerived| {
                extent.lba < old.extent.lba + u64::from(old.extent.blocks) && old.extent.lba < end
            })
            || derived.iter().any(|old: &DiskDerived| old.object == object)
        {
            return Err(PersistenceError::Corrupt);
        }
        derived.push(DiskDerived {
            object,
            input_version,
            extent,
            digest,
        });
        at += DERIVED_BYTES;
    }
    validate_graph(root.root, &objects, &edges)?;
    Ok(Decoded {
        generation: root.generation,
        era: root.era,
        root: root.root,
        objects,
        edges,
        derived,
    })
}

/// A decoded image must not put the volume on the mount path's stack.
///
/// Inline, `Decoded` was **132,168 bytes** — the edge table alone 114,696 —
/// and recovery decodes each of the two roots in turn inside the service's
/// `serve` future on the founder's shared 4 MiB stack. The ceiling here is
/// loose on purpose: it pins the *property* (three handles plus scalars, no
/// table) rather than a figure, so adding a field is free and re-inlining a
/// table fails.
#[cfg(test)]
#[test]
fn a_decoded_image_holds_no_table_inline() {
    let bytes = core::mem::size_of::<Decoded>();
    assert!(
        bytes <= 512,
        "Decoded grew to {bytes} B: a volume-scaled table is inline again"
    );
}

fn validate_graph(
    root: StableObjectId,
    objects: &[DiskObject],
    edges: &[DiskEdge],
) -> Result<(), PersistenceError> {
    let Some(root_index) = objects
        .iter()
        .position(|object| object.stable == root && object.kind.is_dir())
    else {
        return Err(PersistenceError::Corrupt);
    };

    // Kahn's algorithm is both the cycle proof and the reachability proof.
    // Crucially, a directory reached from two parents has indegree two and is
    // queued only after both lawful routes are processed; it is not mistaken
    // for a cycle merely because it was seen before.
    let mut incoming = [0usize; MAX_DURABLE_OBJECTS];
    for edge in edges {
        let child = objects
            .iter()
            .position(|object| object.stable == edge.child)
            .ok_or(PersistenceError::Corrupt)?;
        incoming[child] = incoming[child]
            .checked_add(1)
            .ok_or(PersistenceError::Bounds)?;
    }
    if incoming[root_index] != 0 {
        return Err(PersistenceError::Corrupt);
    }

    let mut pending = heapless::Vec::<usize, MAX_DURABLE_OBJECTS>::new();
    pending
        .push(root_index)
        .map_err(|_| PersistenceError::Bounds)?;
    let mut processed = [false; MAX_DURABLE_OBJECTS];
    let mut count = 0usize;
    while let Some(parent_index) = pending.pop() {
        if processed[parent_index] {
            return Err(PersistenceError::Corrupt);
        }
        processed[parent_index] = true;
        count += 1;
        let parent = objects[parent_index].stable;
        for edge in edges.iter().filter(|edge| edge.parent == parent) {
            let child = objects
                .iter()
                .position(|object| object.stable == edge.child)
                .ok_or(PersistenceError::Corrupt)?;
            incoming[child] = incoming[child]
                .checked_sub(1)
                .ok_or(PersistenceError::Corrupt)?;
            if incoming[child] == 0 {
                pending.push(child).map_err(|_| PersistenceError::Bounds)?;
            }
        }
    }
    // A disconnected component never enters the root-seeded queue; a cycle
    // retains positive indegree. Both therefore fail the same closed check.
    if count != objects.len() {
        return Err(PersistenceError::Corrupt);
    }
    Ok(())
}

fn put(dst: &mut [u8], at: usize, src: &[u8]) -> Result<(), PersistenceError> {
    let end = at.checked_add(src.len()).ok_or(PersistenceError::Bounds)?;
    dst.get_mut(at..end)
        .ok_or(PersistenceError::Bounds)?
        .copy_from_slice(src);
    Ok(())
}
fn put_u32(dst: &mut [u8], at: usize, value: u32) -> Result<(), PersistenceError> {
    put(dst, at, &value.to_le_bytes())
}
fn put_u64(dst: &mut [u8], at: usize, value: u64) -> Result<(), PersistenceError> {
    put(dst, at, &value.to_le_bytes())
}
fn get_u32(src: &[u8], at: usize) -> Result<u32, PersistenceError> {
    let bytes: [u8; 4] = src
        .get(at..at + 4)
        .ok_or(PersistenceError::Corrupt)?
        .try_into()
        .map_err(|_| PersistenceError::Corrupt)?;
    Ok(u32::from_le_bytes(bytes))
}
fn get_u64(src: &[u8], at: usize) -> Result<u64, PersistenceError> {
    let bytes: [u8; 8] = src
        .get(at..at + 8)
        .ok_or(PersistenceError::Corrupt)?
        .try_into()
        .map_err(|_| PersistenceError::Corrupt)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_driver::block::{BlockChannel, BlockError, CompletionMode, memory::MemoryBlockDevice};
    use kore_svc_extract::{
        NormalizedResult, ParserProvenance, WORKER_REPORT_BYTES, encode_result,
        registry::{PLAIN_TEXT, Registry},
    };
    use kore_svc_extract_worker_rust::parse_plain_text;

    type Device = MemoryBlockDevice<327680, 8192>;
    type Volume = PersistentFs<Device, 3, 8>;

    fn provider() -> BlockProvider<Device, 3, 8> {
        let channel =
            BlockChannel::new(Device::new(512, 16).unwrap(), CompletionMode::Poll).unwrap();
        let bootstrap = bootstrap_data_start(channel.geometry().block_size).unwrap();
        BlockProvider::new_with_data_start(channel, bootstrap).unwrap()
    }

    fn name(value: &str) -> Name {
        Name::new(value).unwrap()
    }

    fn on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("persistent-fs-test".into())
            // Async durability moves the by-value provider, decoded graph,
            // and engine through one recovery future while I/O is pending.
            // Exercise that real ownership shape without relying on the host
            // test runner's much smaller default stack; the 128 KiB image
            // itself remains heap-backed and is tested separately below.
            .stack_size(32 * 1024 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    fn fixture(objects: &[DiskObject], edges: &[DiskEdge]) -> (Box<[u8]>, usize, RootRecord) {
        // Mirror the mounted volume's retained publication scratch. Returning
        // the full current reservation inline would make adversarial parser
        // tests retain several 128 KiB arrays in one stack frame.
        let mut image = ImageScratch::new().unwrap().0;
        put(&mut image, 0, IMAGE_MAGIC).unwrap();
        put_u32(&mut image, 8, FORMAT_VERSION).unwrap();
        put_u32(&mut image, 12, IMAGE_HEADER as u32).unwrap();
        put_u64(&mut image, 16, 7).unwrap();
        put_u64(&mut image, 24, 1).unwrap();
        put_u32(&mut image, 32, objects.len() as u32).unwrap();
        put_u32(&mut image, 36, edges.len() as u32).unwrap();
        put_u32(&mut image, 40, OBJECT_BYTES as u32).unwrap();
        put_u32(&mut image, 44, EDGE_BYTES as u32).unwrap();
        put_u32(&mut image, 48, 0).unwrap();
        put_u32(&mut image, 52, DERIVED_BYTES as u32).unwrap();
        let mut at = IMAGE_HEADER;
        for object in objects {
            put_u64(&mut image, at, object.stable.get()).unwrap();
            image[at + 8] = if object.kind.is_file() { 1 } else { 2 };
            image[at + 9] = u8::from(object.extent.is_some());
            image[at + 10] = object.subtype.to_bits();
            put_u64(&mut image, at + 16, object.len).unwrap();
            put_u64(&mut image, at + 24, object.revision).unwrap();
            if let Some(extent) = object.extent {
                put_u64(&mut image, at + 32, extent.lba).unwrap();
                put_u32(&mut image, at + 40, extent.blocks).unwrap();
            }
            let checksum = CRC.checksum(&image[at..at + 44]);
            put_u32(&mut image, at + 44, checksum).unwrap();
            at += OBJECT_BYTES;
        }
        for edge in edges {
            put_u64(&mut image, at, edge.parent.get()).unwrap();
            put_u64(&mut image, at + 8, edge.child.get()).unwrap();
            image[at + 16] = edge.name.len() as u8;
            put(&mut image, at + 20, edge.name.as_bytes()).unwrap();
            let checksum = CRC.checksum(&image[at..at + 52]);
            put_u32(&mut image, at + 52, checksum).unwrap();
            at += EDGE_BYTES;
        }
        let root = RootRecord {
            version: FORMAT_VERSION,
            generation: 7,
            block_size: 512,
            block_count: 256,
            slot: 1,
            root: StableObjectId::new(1).unwrap(),
            image_len: at as u32,
            image_crc: CRC.checksum(&image[..at]),
            data_start: LEGACY_DATA_START_LBA,
            arena_blocks: 256 - LEGACY_DATA_START_LBA,
            era: [0u8; ERA_BYTES],
            sidecar: None,
        };
        (image, at, root)
    }

    /// Rewrite the active fresh-v3 generation into the legacy slot profile.
    /// This models both a real v2 disk and a v3 disk produced after slice 5
    /// but before slice 7b enlarged the metadata reservation.
    fn rewrite_active_as_legacy(volume: &Volume, version: u32) {
        let provider = volume.engine().provider();
        let block_size = provider.geometry().block_size;
        let block = block_size as usize;
        let slot = (volume.status().generation & 1) as u8;
        let current_lba = image_lba(ImageProfile::Current, slot, block_size).unwrap();
        let legacy_lba = image_lba(ImageProfile::Legacy, slot, block_size).unwrap();

        let mut image = [0u8; LEGACY_IMAGE_BYTES];
        let mut head = [0u8; 4096];
        crate::block_on(provider.read_metadata(slot as u64, &mut head[..block])).unwrap();
        let current =
            decode_root(&head[..block], block_size, provider.geometry().block_count).unwrap();
        assert_eq!(
            root_profile(current.version, current.data_start, block_size),
            Ok(ImageProfile::Current),
            "the fixture must start at the raised v3 profile"
        );
        let span = image_transfer_bytes(current.image_len as usize, block_size, LEGACY_IMAGE_BYTES)
            .unwrap();
        crate::block_on(provider.read_metadata(current_lba, &mut image[..span])).unwrap();

        put_u32(&mut image, 8, version).unwrap();
        let image_len = current.image_len as usize;
        let legacy = RootRecord {
            version,
            image_crc: CRC.checksum(&image[..image_len]),
            data_start: LEGACY_DATA_START_LBA,
            arena_blocks: provider
                .geometry()
                .block_count
                .saturating_sub(LEGACY_DATA_START_LBA),
            ..current
        };
        encode_root(legacy, &mut head[..block]).unwrap();
        crate::block_on(provider.write_metadata(legacy_lba, &image[..span])).unwrap();
        crate::block_on(provider.write_metadata(slot as u64, &head[..block])).unwrap();

        // The older fallback still names the raised synthetic fixture. A real
        // legacy disk never has that mixed history, so leave one valid root
        // instead of manufacturing contradictory arena bounds.
        head[..block].fill(0);
        crate::block_on(provider.write_metadata(u64::from(slot ^ 1), &head[..block])).unwrap();
        crate::block_on(provider.flush_device()).unwrap();
    }

    #[test]
    fn probe_distinguishes_blank_kore_and_foreign_media_without_writing() {
        on_large_stack(|| {
            let blank = provider();
            assert_eq!(
                crate::block_on(Volume::probe(&blank)),
                Ok(VolumeProbe::Blank)
            );

            let foreign = provider();
            let mut gpt = [0u8; 512];
            gpt[..8].copy_from_slice(b"EFI PART");
            crate::block_on(foreign.write_metadata(0, &gpt)).unwrap();
            assert_eq!(
                crate::block_on(Volume::probe(&foreign)),
                Ok(VolumeProbe::Foreign)
            );
            let mut observed = [0u8; 512];
            crate::block_on(foreign.read_metadata(0, &mut observed)).unwrap();
            assert_eq!(observed, gpt, "the probe never rewrites foreign media");

            let volume = crate::block_on(Volume::format(provider(), SessionId::new(0))).unwrap();
            assert_eq!(
                crate::block_on(Volume::probe(volume.engine().provider())),
                Ok(VolumeProbe::Kore)
            );
        });
    }

    #[test]
    fn v3_era_survives_commits_and_cold_recovery() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let era = [0x35; ERA_BYTES];
            let mut volume =
                crate::block_on(Volume::format_with_era(provider(), session, era)).unwrap();
            assert_eq!(volume.status().era, era);
            assert!(
                crate::block_on(Volume::roots_match_era(volume.engine().provider(), era)).unwrap()
            );

            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("era"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"current",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            assert!(
                crate::block_on(Volume::roots_match_era(volume.engine().provider(), era)).unwrap()
            );

            let recovered = crate::block_on(volume.restart(SessionId::new(1))).unwrap();
            assert_eq!(recovered.status().era, era);
            assert!(
                crate::block_on(Volume::roots_match_era(recovered.engine().provider(), era,))
                    .unwrap()
            );
        });
    }

    #[test]
    fn system_admission_rejects_a_stale_fallback_era() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let era = [0x46; ERA_BYTES];
            let mut volume =
                crate::block_on(Volume::format_with_era(provider(), session, era)).unwrap();
            // Publish the other slot so recovery has two lawful candidates.
            crate::block_on(volume.commit(session)).unwrap();
            let provider = volume.engine().provider();
            let block = provider.geometry().block_size as usize;
            let mut root = [0u8; 4096];
            crate::block_on(provider.read_metadata(0, &mut root[..block])).unwrap();
            let mut decoded = decode_root(
                &root[..block],
                provider.geometry().block_size,
                provider.geometry().block_count,
            )
            .unwrap();
            decoded.era = [0x47; ERA_BYTES];
            encode_root(decoded, &mut root[..block]).unwrap();
            crate::block_on(provider.write_metadata(0, &root[..block])).unwrap();
            crate::block_on(provider.flush_device()).unwrap();

            assert!(
                !crate::block_on(Volume::roots_match_era(provider, era)).unwrap(),
                "a recovery-selectable stale era must forbid ordinary recovery"
            );
        });
    }

    #[test]
    fn reformat_replaces_only_the_namespace_and_stamps_the_new_era() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let old_era = [0x58; ERA_BYTES];
            let new_era = [0x59; ERA_BYTES];
            let mut old =
                crate::block_on(Volume::format_with_era(provider(), session, old_era)).unwrap();
            let root = old
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            crate::block_on(old.engine_mut().create(
                session,
                root.handle,
                &name("old-era-only"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(old.commit(session)).unwrap();

            let (engine, _) = old.into_parts();
            let provider = engine.into_provider();
            assert!(!crate::block_on(Volume::roots_match_era(&provider, new_era)).unwrap());
            let mut current =
                crate::block_on(Volume::reformat_with_era(provider, session, new_era)).unwrap();
            assert_eq!(current.status().era, new_era);
            assert!(
                crate::block_on(Volume::roots_match_era(
                    current.engine().provider(),
                    new_era,
                ))
                .unwrap()
            );
            let root = current
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            assert_eq!(
                current.engine().open_at(
                    session,
                    root.handle,
                    &name("old-era-only"),
                    crate::Rights::READ
                ),
                Err(crate::FsError::NotFound),
                "old system names must not survive an era rebuild"
            );
        });
    }

    /// Byte 10 of the object record is the subtype from version 4: a stamped
    /// file round-trips, a directory may not carry one, and the bytes after
    /// it stay reserved.
    #[test]
    fn a_version_4_object_record_round_trips_its_subtype() {
        let root_object = DiskObject {
            stable: StableObjectId::new(1).unwrap(),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let tool = DiskObject {
            stable: StableObjectId::new(2).unwrap(),
            kind: ObjectKind::File,
            subtype: ObjectSubtype::EXECUTABLE_IMAGE,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let edge = DiskEdge {
            parent: StableObjectId::new(1).unwrap(),
            name: Name::new("tool").unwrap(),
            child: StableObjectId::new(2).unwrap(),
        };
        let (image, len, root) = fixture(&[root_object, tool], &[edge]);
        // The fixture is at the current version, which carries subtypes.
        assert!(root.version >= SUBTYPE_FORMAT_VERSION);
        let decoded = decode_image(&image[..len], root, 512, 256).unwrap();
        assert_eq!(decoded.objects[1].subtype, ObjectSubtype::EXECUTABLE_IMAGE);
        assert_eq!(decoded.objects[0].subtype, ObjectSubtype::PLAIN);

        // An unknown subtype is data: carried, not refused.
        let (mut image, len, mut root) = fixture(&[root_object, tool], &[edge]);
        let at = IMAGE_HEADER + OBJECT_BYTES;
        image[at + 10] = 0x7f;
        let checksum = CRC.checksum(&image[at..at + 44]);
        put_u32(&mut image, at + 44, checksum).unwrap();
        root.image_crc = CRC.checksum(&image[..len]);
        let decoded = decode_image(&image[..len], root, 512, 256).unwrap();
        assert_eq!(decoded.objects[1].subtype, ObjectSubtype::from_bits(0x7f));
    }

    #[test]
    fn a_directory_with_a_subtype_is_corrupt() {
        let root_object = DiskObject {
            stable: StableObjectId::new(1).unwrap(),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::EXECUTABLE_IMAGE,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let (image, len, root) = fixture(&[root_object], &[]);
        assert!(matches!(
            decode_image(&image[..len], root, 512, 256),
            Err(PersistenceError::Corrupt)
        ));
    }

    /// Before version 4 the byte was reserved-must-be-zero, and a reader of
    /// those versions keeps saying so; an encoder asked to record a subtype
    /// at such a version refuses rather than drops it.
    #[test]
    fn a_version_3_reader_keeps_byte_ten_reserved() {
        let root_object = DiskObject {
            stable: StableObjectId::new(1).unwrap(),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let tool = DiskObject {
            stable: StableObjectId::new(2).unwrap(),
            kind: ObjectKind::File,
            subtype: ObjectSubtype::EXECUTABLE_IMAGE,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let edge = DiskEdge {
            parent: StableObjectId::new(1).unwrap(),
            name: Name::new("tool").unwrap(),
            child: StableObjectId::new(2).unwrap(),
        };
        let (mut image, len, mut root) = fixture(&[root_object, tool], &[edge]);
        put_u32(&mut image, 8, ARENA_FORMAT_VERSION).unwrap();
        root.version = ARENA_FORMAT_VERSION;
        root.image_crc = CRC.checksum(&image[..len]);
        assert!(
            matches!(
                decode_image(&image[..len], root, 512, 256),
                Err(PersistenceError::Corrupt)
            ),
            "a version-3 record with byte ten set is corrupt, not subtyped"
        );

        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            crate::block_on(volume.engine_mut().create_typed(
                session,
                root.handle,
                &name("tool"),
                ObjectKind::File,
                ObjectSubtype::EXECUTABLE_IMAGE,
            ))
            .unwrap();
            let root_id = volume.engine().root_object(session).unwrap();
            let mut scratch = ImageScratch::new().unwrap();
            assert_eq!(
                encode_image(
                    volume.engine(),
                    root_id,
                    11,
                    ARENA_FORMAT_VERSION,
                    ImageProfile::Current,
                    scratch.as_mut(),
                )
                .unwrap_err(),
                PersistenceError::Incompatible,
                "a version that cannot record the stamp refuses rather than drops it"
            );
        });
    }

    #[test]
    fn subtype_survives_commit_and_cold_recovery() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let made = crate::block_on(volume.engine_mut().create_typed(
                session,
                root.handle,
                &name("tool"),
                ObjectKind::File,
                ObjectSubtype::EXECUTABLE_IMAGE,
            ))
            .unwrap();
            assert_eq!(made.subtype, ObjectSubtype::EXECUTABLE_IMAGE);
            crate::block_on(volume.engine_mut().write_at(
                session,
                made.handle,
                0,
                b"ELF",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();

            let mut recovered = crate::block_on(volume.restart(SessionId::new(1))).unwrap();
            let session = SessionId::new(1);
            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let tool = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("tool"), crate::Rights::READ)
                .unwrap();
            assert_eq!(tool.subtype, ObjectSubtype::EXECUTABLE_IMAGE);
            assert_eq!(
                recovered
                    .engine()
                    .stat(session, tool.handle)
                    .unwrap()
                    .subtype,
                ObjectSubtype::EXECUTABLE_IMAGE
            );
            // A recovered version-4 volume keeps recording subtypes.
            crate::block_on(recovered.engine_mut().create_typed(
                session,
                root.handle,
                &name("another"),
                ObjectKind::File,
                ObjectSubtype::EXECUTABLE_IMAGE,
            ))
            .unwrap();
        });
    }

    /// A version-3 volume mounts, reads every object as plain, keeps
    /// committing at version 3, and refuses to record a subtype it could not
    /// keep.
    #[test]
    fn a_version_3_volume_mounts_plain_and_refuses_subtyped_creates() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("kept"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            rewrite_active_as_legacy(&volume, ARENA_FORMAT_VERSION);

            let mut recovered = crate::block_on(volume.restart(session)).unwrap();
            assert_eq!(recovered.status().mode, MountMode::ReadWrite);
            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let kept = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("kept"), crate::Rights::READ)
                .unwrap();
            assert_eq!(kept.subtype, ObjectSubtype::PLAIN);
            assert_eq!(
                crate::block_on(recovered.engine_mut().create_typed(
                    session,
                    root.handle,
                    &name("tool"),
                    ObjectKind::File,
                    ObjectSubtype::EXECUTABLE_IMAGE,
                ))
                .unwrap_err(),
                crate::FsError::Unsupported
            );
            crate::block_on(recovered.engine_mut().create(
                session,
                root.handle,
                &name("plain"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(recovered.commit(session)).unwrap();

            // Preserved, not re-stamped: the volume still says version 3.
            let provider = recovered.engine().provider();
            let block = provider.geometry().block_size as usize;
            let mut head = [0u8; 4096];
            let slot = (recovered.status().generation & 1) as u64;
            crate::block_on(provider.read_metadata(slot, &mut head[..block])).unwrap();
            let current = decode_root(
                &head[..block],
                provider.geometry().block_size,
                provider.geometry().block_count,
            )
            .unwrap();
            assert_eq!(current.version, ARENA_FORMAT_VERSION);
        });
    }

    #[test]
    fn legacy_v1_image_decodes_without_fabricating_derived_state() {
        let root_object = DiskObject {
            stable: StableObjectId::new(1).unwrap(),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let (mut image, len, mut root) = fixture(&[root_object], &[]);
        put_u32(&mut image, 8, LEGACY_FORMAT_VERSION).unwrap();
        image[48..IMAGE_HEADER].fill(0);
        root.version = LEGACY_FORMAT_VERSION;
        root.image_crc = CRC.checksum(&image[..len]);

        let decoded = decode_image(&image[..len], root, 512, 256).unwrap();
        assert_eq!(decoded.objects.len(), 1);
        assert!(decoded.derived.is_empty());
    }

    #[test]
    fn retained_scratch_reencodes_v1_without_stale_reserved_bytes() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root_id = volume.engine().root_object(session).unwrap();
            let mut scratch = ImageScratch::new().unwrap();
            scratch.as_mut().fill(0xa5);
            let (len, root) = encode_image(
                volume.engine(),
                root_id,
                11,
                LEGACY_FORMAT_VERSION,
                ImageProfile::Legacy,
                scratch.as_mut(),
            )
            .unwrap();
            assert!(
                scratch.as_mut()[48..IMAGE_HEADER]
                    .iter()
                    .all(|byte| *byte == 0)
            );
            let record = RootRecord {
                version: LEGACY_FORMAT_VERSION,
                generation: 11,
                block_size: 512,
                block_count: 640,
                slot: 1,
                root,
                image_len: len as u32,
                image_crc: CRC.checksum(&scratch.as_mut()[..len]),
                data_start: LEGACY_DATA_START_LBA,
                arena_blocks: 640 - LEGACY_DATA_START_LBA,
                era: [0u8; ERA_BYTES],
                sidecar: None,
            };
            assert!(
                decode_image(&scratch.as_mut()[..len], record, 512, 640)
                    .unwrap()
                    .derived
                    .is_empty()
            );
        });
    }

    #[test]
    fn legacy_profile_refuses_a_directory_beyond_its_32_entry_contract() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("base"),
                ObjectKind::File,
            ))
            .unwrap();
            for index in 0..32 {
                volume
                    .engine_mut()
                    .link(
                        session,
                        root.handle,
                        &name(&std::format!("l{index}")),
                        file.handle,
                    )
                    .unwrap();
            }
            let root_id = volume.engine().root_object(session).unwrap();
            let mut scratch = ImageScratch::new().unwrap();
            assert_eq!(
                encode_image(
                    volume.engine(),
                    root_id,
                    12,
                    DERIVED_FORMAT_VERSION,
                    ImageProfile::Legacy,
                    scratch.as_mut(),
                ),
                Err(PersistenceError::Bounds)
            );
        });
    }

    #[test]
    fn decoder_refuses_graphs_that_exceed_live_directory_bounds() {
        let dir = |stable| DiskObject {
            stable: StableObjectId::new(stable).unwrap(),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let file = DiskObject {
            stable: StableObjectId::new(2).unwrap(),
            kind: ObjectKind::File,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let current = image_layout(ImageProfile::Current, 512).unwrap();
        let make_current = |record: &mut RootRecord| {
            record.block_count = 640;
            record.data_start = current.data_start;
            record.arena_blocks = 640 - current.data_start;
        };

        let mut edges = std::vec::Vec::new();
        for index in 0..=MAX_DIR_ENTRIES {
            edges.push(DiskEdge {
                parent: StableObjectId::new(1).unwrap(),
                child: file.stable,
                name: name(&std::format!("e{index}")),
            });
        }
        let (image, len, mut record) = fixture(&[dir(1), file], &edges);
        make_current(&mut record);
        assert!(matches!(
            decode_image(&image[..len], record, 512, 640),
            Err(PersistenceError::Bounds)
        ));

        let mut directories = std::vec::Vec::new();
        let mut edges = std::vec::Vec::new();
        for stable in 1..=MAX_DIRECTORIES as u64 + 1 {
            directories.push(dir(stable));
            if stable != 1 {
                edges.push(DiskEdge {
                    parent: StableObjectId::new(1).unwrap(),
                    child: StableObjectId::new(stable).unwrap(),
                    name: name(&std::format!("d{stable}")),
                });
            }
        }
        let (image, len, mut record) = fixture(&directories, &edges);
        make_current(&mut record);
        assert!(matches!(
            decode_image(&image[..len], record, 512, 640),
            Err(PersistenceError::Bounds)
        ));
    }

    /// A current-profile image written before the edge budget was decoupled
    /// could record up to the old `16 × 128` product. Such a volume must keep
    /// mounting: the reader admits the old product, and only a count past it
    /// is refused. The writer's ceiling is a separate, lower figure.
    #[test]
    fn decoder_admits_pre_decoupling_edge_counts_but_not_more() {
        let dir = |stable: u64| DiskObject {
            stable: StableObjectId::new(stable).unwrap(),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let file = DiskObject {
            stable: StableObjectId::new(10).unwrap(),
            kind: ObjectKind::File,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let current = image_layout(ImageProfile::Current, 512).unwrap();
        let make_current = |record: &mut RootRecord| {
            record.block_count = 640;
            record.data_start = current.data_start;
            record.arena_blocks = 640 - current.data_start;
        };
        let edge = |parent: u64, child: u64, name_text: &str| DiskEdge {
            parent: StableObjectId::new(parent).unwrap(),
            child: StableObjectId::new(child).unwrap(),
            name: name(name_text),
        };

        // Root 1 holds eight directories and one link to the file; each
        // directory holds 127 more links. 8 + 1 + 8 × 127 = 1025 edges: one
        // past the writer's ceiling, well inside the old product, and every
        // directory inside `MAX_DIR_ENTRIES`.
        let mut objects = std::vec![dir(1)];
        let mut edges = std::vec::Vec::new();
        for sub in 2..=9u64 {
            objects.push(dir(sub));
            edges.push(edge(1, sub, &std::format!("d{sub}")));
            for link in 0..127 {
                edges.push(edge(sub, 10, &std::format!("l{link}")));
            }
        }
        objects.push(file);
        edges.push(edge(1, 10, "f"));
        assert_eq!(edges.len(), MAX_DURABLE_EDGES + 1);
        let (image, len, mut record) = fixture(&objects, &edges);
        make_current(&mut record);
        let decoded = decode_image(&image[..len], record, 512, 640).unwrap();
        assert_eq!(decoded.edges.len(), MAX_DURABLE_EDGES + 1);
        assert_eq!(decoded.objects.len(), objects.len());

        // One past the old product is not an old writer; it is corruption.
        let mut edges = std::vec::Vec::new();
        for index in 0..=PRE_DECOUPLING_MAX_DURABLE_EDGES {
            edges.push(edge(1, 10, &std::format!("e{index}")));
        }
        let (image, len, mut record) = fixture(&[dir(1), file], &edges);
        make_current(&mut record);
        assert!(matches!(
            decode_image(&image[..len], record, 512, 640),
            Err(PersistenceError::Bounds)
        ));
    }

    /// A version-2 image still yields its derived table.
    ///
    /// The gate used to read `root.version == FORMAT_VERSION`, so the moment
    /// the current version moved past 2 every v2 volume would have mounted
    /// with an EMPTY index while its batches sat on disk — a silent
    /// false-negative for every `CurrentOnly` search on that volume. The gate
    /// is "2 or later" for exactly this reason.
    #[test]
    fn a_version_2_image_keeps_its_derived_table() {
        let root_object = DiskObject {
            stable: StableObjectId::new(1).unwrap(),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: FIRST_REVISION,
            extent: None,
        };
        let (mut image, len, mut root) = fixture(&[root_object], &[]);
        put_u32(&mut image, 8, DERIVED_FORMAT_VERSION).unwrap();
        root.version = DERIVED_FORMAT_VERSION;
        root.image_crc = CRC.checksum(&image[..len]);

        let decoded = decode_image(&image[..len], root, 512, 256).unwrap();
        assert_eq!(decoded.objects.len(), 1);
        // The fixture publishes no batches, but the COUNT FIELD was read
        // rather than forced to zero: a v2 image whose header says otherwise
        // is no longer silently treated as unindexed.
        assert!(decoded.derived.is_empty());
    }

    /// Everything version 3 added has to survive a round trip, or the fields
    /// exist on disk without meaning anything — at EVERY version that carries
    /// the version-3 header, not only the current one.
    ///
    /// This test once encoded only `FORMAT_VERSION`, so when version 4 landed
    /// it kept passing while every real version-3 root decoded as legacy: a
    /// raised-profile v3 volume located its image at the legacy slot and lost
    /// its era. Version 3 is enumerated explicitly so the next bump cannot
    /// repeat that.
    #[test]
    fn a_version_3_root_round_trips_its_arena_and_era() {
        let era = core::array::from_fn(|index| index as u8 ^ 0xa5);
        let data_start = image_layout(ImageProfile::Current, 512).unwrap().data_start + 8;
        for version in [ARENA_FORMAT_VERSION, SUBTYPE_FORMAT_VERSION, FORMAT_VERSION] {
            let record = RootRecord {
                version,
                generation: 9,
                block_size: 512,
                block_count: 4096,
                slot: 1,
                root: StableObjectId::new(3).unwrap(),
                image_len: 64,
                image_crc: 0x1234_5678,
                data_start,
                arena_blocks: 1000,
                era,
                sidecar: None,
            };
            let mut block = [0u8; 512];
            encode_root(record, &mut block).unwrap();
            assert_eq!(
                Some(get_u32(&block, 12).unwrap() as usize),
                root_header_bytes(version)
            );

            let back = decode_root(&block, 512, 4096).unwrap();
            assert_eq!(back.version, version);
            assert_eq!(back.generation, 9);
            assert_eq!(back.data_start, data_start, "v{version} arena start");
            assert_eq!(back.arena_blocks, 1000, "v{version} arena span");
            assert_eq!(back.era, era, "v{version} era");
            assert_eq!(back.root, record.root);
            assert_eq!(back.image_crc, record.image_crc);
            // The raised profile is what the recorded arena start selects;
            // decoding v3 as legacy would have answered `Legacy` here.
            assert_eq!(
                root_profile(back.version, back.data_start, 512),
                Ok(ImageProfile::Current),
                "v{version} slot profile"
            );
        }
    }

    /// A pre-v3 root records neither arena nor era, so it must decode to what
    /// it MEANT — the compile-time floor and everything above it — not to
    /// zero, which would describe a volume with no data region at all.
    #[test]
    fn a_pre_v3_root_decodes_to_the_arena_it_meant() {
        for version in [LEGACY_FORMAT_VERSION, DERIVED_FORMAT_VERSION] {
            let record = RootRecord {
                version,
                generation: 4,
                block_size: 512,
                block_count: 4096,
                slot: 0,
                root: StableObjectId::new(1).unwrap(),
                image_len: 64,
                image_crc: 7,
                // Ignored at this version: the fields do not exist on disk.
                data_start: 999,
                arena_blocks: 999,
                era: [0xff; ERA_BYTES],
                sidecar: None,
            };
            let mut block = [0u8; 512];
            encode_root(record, &mut block).unwrap();
            assert_eq!(
                get_u32(&block, 12).unwrap() as usize,
                ROOT_HEADER_V2,
                "v{version} must keep its own header length"
            );
            // Nothing of the v3 fields was written into a v2 record.
            assert!(block[ROOT_HEADER_V2..].iter().all(|byte| *byte == 0));

            let back = decode_root(&block, 512, 4096).unwrap();
            assert_eq!(back.version, version);
            assert_eq!(back.data_start, LEGACY_DATA_START_LBA);
            assert_eq!(back.arena_blocks, 4096 - LEGACY_DATA_START_LBA);
            assert_eq!(back.era, [0u8; ERA_BYTES]);
        }
    }

    /// The header length is validated against the one its version defines, so
    /// a record claiming a version it does not carry the fields for is
    /// refused rather than read past its own data.
    #[test]
    fn a_version_and_header_length_that_disagree_are_incompatible() {
        let record = RootRecord {
            version: DERIVED_FORMAT_VERSION,
            generation: 4,
            block_size: 512,
            block_count: 4096,
            slot: 0,
            root: StableObjectId::new(1).unwrap(),
            image_len: 64,
            image_crc: 7,
            data_start: LEGACY_DATA_START_LBA,
            arena_blocks: 4096 - LEGACY_DATA_START_LBA,
            era: [0u8; ERA_BYTES],
            sidecar: None,
        };
        let mut block = [0u8; 512];
        encode_root(record, &mut block).unwrap();
        // v2 bytes relabelled as v3: the v3 reader would look for an era at
        // offset 72 that was never written.
        put_u32(&mut block, 8, FORMAT_VERSION).unwrap();
        assert_eq!(
            decode_root(&block, 512, 4096),
            Err(PersistenceError::Incompatible)
        );

        // And a version this build has never heard of is refused, not guessed.
        let mut block = [0u8; 512];
        encode_root(record, &mut block).unwrap();
        put_u32(&mut block, 8, 99).unwrap();
        assert_eq!(
            decode_root(&block, 512, 4096),
            Err(PersistenceError::Incompatible)
        );
    }

    /// An arena below the metadata reservation or past the end of the medium
    /// describes a volume that cannot exist; recording it does not make it
    /// real.
    #[test]
    fn an_impossible_v3_arena_is_refused() {
        let base = RootRecord {
            version: FORMAT_VERSION,
            generation: 4,
            block_size: 512,
            block_count: 4096,
            slot: 0,
            root: StableObjectId::new(1).unwrap(),
            image_len: 64,
            image_crc: 7,
            data_start: LEGACY_DATA_START_LBA,
            arena_blocks: 4096 - LEGACY_DATA_START_LBA,
            era: [0u8; ERA_BYTES],
            sidecar: None,
        };
        for (data_start, arena_blocks) in [
            (0u64, 16u64),                     // under the reservation
            (LEGACY_DATA_START_LBA, 4096),     // past the medium
            (LEGACY_DATA_START_LBA, u64::MAX), // overflowing
        ] {
            let mut block = [0u8; 512];
            encode_root(
                RootRecord {
                    data_start,
                    arena_blocks,
                    ..base
                },
                &mut block,
            )
            .unwrap();
            assert_eq!(
                decode_root(&block, 512, 4096),
                Err(PersistenceError::Corrupt),
                "arena [{data_start}, +{arena_blocks}) must be refused"
            );
        }
    }

    /// A commit moves only the bytes it wrote.
    ///
    /// Proven by poisoning the tail of the slot the next generation publishes
    /// into and showing the poison SURVIVES: if the commit still transferred
    /// the whole reservation, it would have overwritten it. This is what makes
    /// raising the reservation affordable — otherwise every `touch` on a small
    /// volume pays the full metadata arena.
    #[test]
    fn a_commit_transfers_only_the_image_it_wrote() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let block_size = volume.engine().provider().geometry().block_size;
            let block = block_size as usize;
            let slot_blocks = image_layout(ImageProfile::Current, block_size)
                .unwrap()
                .slot_blocks;

            // The generation about to be published, and therefore its slot.
            let next_slot = ((volume.status().generation + 1) & 1) as u8;
            let arena_lba = image_lba(ImageProfile::Current, next_slot, block_size).unwrap();

            // Poison every block of that slot except the first.
            let poison = [0xc7u8; 512];
            for offset in 1..slot_blocks as u64 {
                crate::block_on(
                    volume
                        .engine()
                        .provider()
                        .write_metadata(arena_lba + offset, &poison[..block]),
                )
                .unwrap();
            }

            crate::block_on(volume.commit(session)).unwrap();
            assert_eq!(
                (volume.status().generation & 1) as u8,
                next_slot,
                "the commit must have published into the poisoned slot"
            );

            // A fresh tree's image is one block, so every poisoned block is
            // still poisoned — the commit did not touch the reservation's tail.
            let mut back = [0u8; 512];
            for offset in 1..slot_blocks as u64 {
                crate::block_on(
                    volume
                        .engine()
                        .provider()
                        .read_metadata(arena_lba + offset, &mut back[..block]),
                )
                .unwrap();
                assert_eq!(
                    &back[..block],
                    &poison[..block],
                    "block {offset} of the slot was rewritten despite carrying no image"
                );
            }

            // And the stale tail is never read: recovery reads only the length
            // the root records, so the volume still mounts.
            let recovered = crate::block_on(volume.restart(session)).unwrap();
            assert_eq!(recovered.status().mode, MountMode::ReadWrite);
        });
    }

    /// The transfer span is a whole number of blocks, never more than the
    /// reservation, and an empty or oversized image is refused rather than
    /// silently clamped.
    #[test]
    fn the_image_transfer_span_is_block_aligned_and_bounded() {
        assert_eq!(image_transfer_bytes(1, 512, MAX_IMAGE_BYTES), Ok(512));
        assert_eq!(image_transfer_bytes(512, 512, MAX_IMAGE_BYTES), Ok(512));
        assert_eq!(image_transfer_bytes(513, 512, MAX_IMAGE_BYTES), Ok(1024));
        assert_eq!(
            image_transfer_bytes(MAX_IMAGE_BYTES, 512, MAX_IMAGE_BYTES),
            Ok(MAX_IMAGE_BYTES)
        );
        assert_eq!(
            image_transfer_bytes(MAX_IMAGE_BYTES + 1, 512, MAX_IMAGE_BYTES),
            Err(PersistenceError::Bounds)
        );
        assert_eq!(
            image_transfer_bytes(0, 512, MAX_IMAGE_BYTES),
            Err(PersistenceError::Bounds)
        );
        // A block size the reservation is not a multiple of cannot address the
        // slot at all.
        assert_eq!(
            image_transfer_bytes(64, 768, MAX_IMAGE_BYTES),
            Err(PersistenceError::Incompatible)
        );
    }

    #[test]
    fn current_and_legacy_profiles_have_unambiguous_slot_boundaries() {
        let legacy_512 = image_layout(ImageProfile::Legacy, 512).unwrap();
        let current_512 = image_layout(ImageProfile::Current, 512).unwrap();
        assert_eq!(legacy_512.data_start, LEGACY_DATA_START_LBA);
        assert_eq!(current_512.data_start, 514);
        assert_eq!(bootstrap_data_start(512), Ok(current_512.data_start));

        // Both natural metadata ends fit below block 72 on 4 KiB media, so
        // current uses block 73 as an explicit profile discriminator.
        let legacy_4k = image_layout(ImageProfile::Legacy, 4096).unwrap();
        let current_4k = image_layout(ImageProfile::Current, 4096).unwrap();
        assert_eq!(legacy_4k.data_start, LEGACY_DATA_START_LBA);
        assert_eq!(current_4k.data_start, LEGACY_DATA_START_LBA + 1);
        assert_ne!(legacy_4k.slot_blocks, current_4k.slot_blocks);
    }

    /// **A version-2 volume written by an older build still mounts, and its
    /// data comes back.**
    ///
    /// The two decode halves are unit-tested above, but the guarantee that
    /// matters is end to end: a real disk carrying a v2 root and a v2 image —
    /// which is what every volume in the field carries — recovers its whole
    /// namespace and its bytes under a build whose current version is 3. This
    /// downgrades a freshly formatted volume's on-disk records to v2 and
    /// recovers through the ordinary path, no test seam in production code.
    #[test]
    fn a_version_2_volume_on_disk_still_mounts_and_returns_its_data() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("kept"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"v2-bytes",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();

            // Rewrite the published generation as version 2, exactly as a
            // build that predates v3 would have left it.
            rewrite_active_as_legacy(&volume, DERIVED_FORMAT_VERSION);

            // The ordinary recovery path, over v2 bytes.
            let mut recovered = crate::block_on(volume.restart(session)).unwrap();
            assert_eq!(
                recovered.status().mode,
                MountMode::ReadWrite,
                "a v2 volume must mount writable, not land in a degraded posture"
            );
            assert_eq!(
                recovered.status().era,
                [0u8; ERA_BYTES],
                "an unstamped volume reads as unstamped, never as some other era"
            );

            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let kept = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("kept"), crate::Rights::ALL)
                .unwrap();
            let mut back = [0u8; 8];
            assert_eq!(
                crate::block_on(
                    recovered
                        .engine_mut()
                        .read(session, kept.handle, 0, &mut back)
                ),
                Ok(8)
            );
            assert_eq!(&back, b"v2-bytes");

            // And it can be written again without silently raising its
            // reservation. Capacity grows only through deliberate reprovision.
            crate::block_on(recovered.commit(session)).unwrap();
        });
    }

    #[test]
    fn a_pre_7b_version_3_volume_keeps_its_legacy_slot_profile() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("old-v3"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"legacy-v3",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            rewrite_active_as_legacy(&volume, FORMAT_VERSION);

            let mut recovered = crate::block_on(volume.restart(session)).unwrap();
            assert_eq!(recovered.state.profile, ImageProfile::Legacy);
            assert_eq!(
                recovered.engine().provider().arena().0,
                LEGACY_DATA_START_LBA
            );
            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("old-v3"), crate::Rights::ALL)
                .unwrap();
            let mut back = [0u8; 9];
            assert_eq!(
                crate::block_on(
                    recovered
                        .engine_mut()
                        .read(session, file.handle, 0, &mut back,)
                ),
                Ok(9)
            );
            assert_eq!(&back, b"legacy-v3");

            // A commit preserves the old slot addresses and bounds. It does
            // not silently reinterpret a pre-7b v3 volume as a raised v3 one.
            crate::block_on(recovered.commit(session)).unwrap();
            let provider = recovered.engine().provider();
            let block_size = provider.geometry().block_size;
            let block = block_size as usize;
            let slot = (recovered.status().generation & 1) as u64;
            let mut head = [0u8; 4096];
            assert_eq!(
                crate::block_on(provider.read_metadata(slot, &mut head[..block])),
                Ok(())
            );
            let root =
                decode_root(&head[..block], block_size, provider.geometry().block_count).unwrap();
            assert_eq!(root.version, FORMAT_VERSION);
            assert_eq!(root.data_start, LEGACY_DATA_START_LBA);
            assert_eq!(
                root_profile(root.version, root.data_start, block_size),
                Ok(ImageProfile::Legacy)
            );
        });
    }

    #[test]
    fn current_profile_commits_a_graph_beyond_both_legacy_totals() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let directory = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("links"),
                ObjectKind::Directory,
            ))
            .unwrap();
            for index in 0..64 {
                let entry = name(&std::format!("f{index}"));
                let file = crate::block_on(volume.engine_mut().create(
                    session,
                    root.handle,
                    &entry,
                    ObjectKind::File,
                ))
                .unwrap();
                volume
                    .engine_mut()
                    .link(session, directory.handle, &entry, file.handle)
                    .unwrap();
                volume.engine().close(session, file.handle).unwrap();
            }
            let root_id = volume.engine().root_object(session).unwrap();
            let graph = volume
                .engine()
                .namespace()
                .durable_graph(root_id, MAX_DURABLE_EDGES)
                .unwrap();
            assert!(graph.objects.len() > LEGACY_MAX_DURABLE_OBJECTS);
            assert!(graph.edges.len() > LEGACY_MAX_DURABLE_EDGES);
            crate::block_on(volume.commit(session)).unwrap();

            let mut recovered = crate::block_on(volume.restart(session)).unwrap();
            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let directory = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("links"), crate::Rights::ALL)
                .unwrap();
            assert!(
                recovered
                    .engine_mut()
                    .open_at(session, directory.handle, &name("f63"), crate::Rights::ALL,)
                    .is_ok()
            );
        });
    }

    #[test]
    fn complete_namespace_and_bytes_survive_with_fresh_authority() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let left = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("left"),
                ObjectKind::Directory,
            ))
            .unwrap();
            let right = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("right"),
                ObjectKind::Directory,
            ))
            .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                left.handle,
                &name("data"),
                ObjectKind::File,
            ))
            .unwrap();
            let revision = crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"durable-bytes",
                crate::ANY_REVISION,
            ))
            .unwrap();
            volume
                .engine_mut()
                .link(session, root.handle, &name("alias"), file.handle)
                .unwrap();
            volume
                .engine_mut()
                .rename(session, root.handle, &name("alias"), &name("renamed"))
                .unwrap();
            volume
                .engine_mut()
                .relocate(
                    session,
                    left.handle,
                    &name("data"),
                    right.handle,
                    &name("moved"),
                )
                .unwrap();
            crate::block_on(
                volume
                    .engine_mut()
                    .unlink(session, root.handle, &name("renamed")),
            )
            .unwrap();
            let moved = volume
                .engine_mut()
                .open_at(session, right.handle, &name("moved"), crate::Rights::ALL)
                .unwrap();
            let revision = crate::block_on(volume.engine_mut().resize(
                session,
                moved.handle,
                7,
                revision,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().sync(session, moved.handle)).unwrap();
            let stable = volume
                .engine()
                .stat(session, moved.handle)
                .unwrap()
                .stable_object;
            let stale_handle = moved.handle;
            crate::block_on(volume.commit(session)).unwrap();

            let recovered_session = SessionId::new(2);
            let mut volume = crate::block_on(volume.restart(recovered_session)).unwrap();
            assert_eq!(
                volume.engine().authority_edge_count(),
                1,
                "disk minted no old grants or sessions"
            );
            assert_eq!(
                volume.engine().stat(recovered_session, stale_handle),
                Err(FsError::InvalidHandle)
            );
            let root = volume
                .engine_mut()
                .open_root(recovered_session, crate::Rights::ALL)
                .unwrap();
            assert!(
                volume
                    .engine_mut()
                    .open_at(
                        recovered_session,
                        root.handle,
                        &name("left"),
                        crate::Rights::ALL
                    )
                    .is_ok()
            );
            assert_eq!(
                volume
                    .engine_mut()
                    .open_at(
                        recovered_session,
                        root.handle,
                        &name("renamed"),
                        crate::Rights::ALL
                    )
                    .unwrap_err(),
                FsError::NotFound
            );
            let right = volume
                .engine_mut()
                .open_at(
                    recovered_session,
                    root.handle,
                    &name("right"),
                    crate::Rights::ALL,
                )
                .unwrap();
            let moved = volume
                .engine_mut()
                .open_at(
                    recovered_session,
                    right.handle,
                    &name("moved"),
                    crate::Rights::ALL,
                )
                .unwrap();
            let stat = volume
                .engine()
                .stat(recovered_session, moved.handle)
                .unwrap();
            assert_eq!(stat.stable_object, stable);
            assert_eq!(stat.revision, revision);
            assert_eq!(stat.links, 1);
            let mut bytes = [0u8; 16];
            assert_eq!(
                crate::block_on(volume.engine_mut().read(
                    recovered_session,
                    moved.handle,
                    0,
                    &mut bytes
                ))
                .unwrap(),
                7
            );
            assert_eq!(&bytes[..7], b"durable");
        });
    }

    #[test]
    fn current_content_index_survives_a_cold_provider_restart() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("note"),
                ObjectKind::File,
            ))
            .unwrap();
            // A newly created empty file already has a complete application
            // batch. Mutating its authoritative revision must publish a root
            // without that now-stale derived extent, not reject the namespace
            // commit as corrupt.
            let empty_owner = crate::extraction::ExtractionOwner::new(40);
            let (empty_lease, empty_input) = crate::block_on(volume.engine_mut().lend_version(
                empty_owner,
                session,
                file.handle,
            ))
            .unwrap();
            let parser = Registry::get(PLAIN_TEXT).unwrap();
            let empty = NormalizedResult::empty(ParserProvenance::deterministic(
                empty_input,
                parser.id,
                parser.version,
            ));
            let mut empty_wire = [0u8; WORKER_REPORT_BYTES];
            let empty_used = encode_result(&empty, &mut empty_wire).unwrap();
            crate::block_on(volume.engine_mut().commit_encoded_derived_result(
                empty_owner,
                empty_lease,
                empty,
                &empty_wire[..empty_used],
                0,
            ))
            .unwrap();
            crate::block_on(
                volume
                    .engine_mut()
                    .release_lent_version(empty_owner, empty_lease),
            )
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();

            let bytes = b"durable capability search";
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                bytes,
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            assert_eq!(
                volume.engine().search(
                    session,
                    root.handle,
                    crate::SearchQuery {
                        content: Some(crate::SearchContent {
                            plane: crate::SearchPlane::FullText,
                            term: crate::SearchTerm::new("capability").unwrap(),
                        }),
                        freshness: crate::FreshnessPolicy::CurrentOnly,
                        ..crate::SearchQuery::ALL
                    },
                    None,
                ),
                Err(FsError::IndexIncomplete)
            );
            let owner = crate::extraction::ExtractionOwner::new(41);
            let (lease, input) = crate::block_on(volume.engine_mut().lend_version(
                owner,
                session,
                file.handle,
            ))
            .unwrap();
            let result = parse_plain_text(input, bytes).unwrap();
            let mut wire = [0u8; WORKER_REPORT_BYTES];
            let used = encode_result(&result, &mut wire).unwrap();
            crate::block_on(volume.engine_mut().commit_encoded_derived_result(
                owner,
                lease,
                result,
                &wire[..used],
                0,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
            crate::block_on(volume.commit(session)).unwrap();

            let recovered_session = SessionId::new(2);
            let mut volume = crate::block_on(volume.restart(recovered_session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(recovered_session, crate::Rights::ALL)
                .unwrap();
            let page = volume
                .engine()
                .search(
                    recovered_session,
                    root.handle,
                    crate::SearchQuery {
                        content: Some(crate::SearchContent {
                            plane: crate::SearchPlane::FullText,
                            term: crate::SearchTerm::new("capability").unwrap(),
                        }),
                        freshness: crate::FreshnessPolicy::CurrentOnly,
                        ..crate::SearchQuery::ALL
                    },
                    None,
                )
                .unwrap();
            assert_eq!(page.hits.len(), 1);
            assert_eq!(page.hits[0].path[0], name("note"));
        });
    }

    #[test]
    fn block_backed_file_larger_than_one_transfer_stays_index_incomplete_without_a_batch() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("large-note"),
                ObjectKind::File,
            ))
            .unwrap();
            let first = [b'x'; crate::bounds::MAX_TRANSFER];
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                &first,
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                crate::bounds::MAX_TRANSFER as u64,
                b"n",
                crate::ANY_REVISION,
            ))
            .unwrap();
            assert_eq!(
                volume.engine().stat(session, file.handle).unwrap().len,
                crate::bounds::MAX_TRANSFER as u64 + 1
            );
            assert_eq!(
                volume.engine().search(
                    session,
                    root.handle,
                    crate::SearchQuery {
                        content: Some(crate::SearchContent {
                            plane: crate::SearchPlane::FullText,
                            term: crate::SearchTerm::new("needle").unwrap(),
                        }),
                        freshness: crate::FreshnessPolicy::CurrentOnly,
                        ..crate::SearchQuery::ALL
                    },
                    None,
                ),
                Err(FsError::IndexIncomplete)
            );
        });
    }

    /// A valid batch larger than the one the store keeps per version, on the
    /// durable provider a machine mounts: before, it passed the provider's
    /// transfer bound, was written and synced to a fresh placement, and was
    /// then refused `Range` when the hot store could not pack it — failing
    /// the write it describes, and `find reindex`. Now it is refused as a
    /// batch this index cannot hold before a block is spent, and the version
    /// stays unindexed.
    #[test]
    fn an_oversized_batch_is_refused_before_a_placement_is_spent() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("notes"),
                ObjectKind::File,
            ))
            .unwrap();
            let text: std::string::String = (0..20)
                .map(|line| {
                    std::format!(
                        "minute{line} reviewed storage{line} batches{line} answer{line} quickly{line}\n"
                    )
                })
                .collect();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                text.as_bytes(),
                crate::ANY_REVISION,
            ))
            .unwrap();
            let owner = crate::extraction::ExtractionOwner::new(43);
            let (lease, input) = crate::block_on(volume.engine_mut().lend_version(
                owner,
                session,
                file.handle,
            ))
            .unwrap();
            // The parser's spans plus a token fact per distinct word: the
            // batch the plain-text producer published before it bounded it.
            let mut result = parse_plain_text(input, text.as_bytes()).unwrap();
            for word in text.split(|byte: char| !byte.is_ascii_alphanumeric()) {
                if word.is_empty() || result.facts.iter().any(|fact| fact.key.as_str() == word) {
                    continue;
                }
                let fact = kore_svc_extract::Fact {
                    kind: kore_svc_extract::FactKind::Custom,
                    key: word.try_into().unwrap(),
                    value: kore_svc_extract::FactValue::Utf8(word.try_into().unwrap()),
                    confidence: kore_svc_extract::Confidence::CERTAIN,
                    source: None,
                };
                if result.facts.push(fact).is_err() {
                    break;
                }
            }
            let mut wire = [0u8; 2 * WORKER_REPORT_BYTES];
            let used = encode_result(&result, &mut wire).unwrap();
            assert!(
                used > WORKER_REPORT_BYTES
                    && used <= volume.engine().provider().limits().max_transfer,
                "the batch must clear the provider's bound and miss the store's"
            );
            let (placements, writes) = (
                volume.engine().provider().placements(),
                volume.engine().provider().stats().writes,
            );

            assert_eq!(
                crate::block_on(volume.engine_mut().commit_encoded_derived_result(
                    owner,
                    lease,
                    result,
                    &wire[..used],
                    0,
                )),
                Err(crate::ExtractionError::Oversize)
            );
            assert_eq!(volume.engine().provider().placements(), placements);
            assert_eq!(volume.engine().provider().stats().writes, writes);
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
            assert_eq!(
                volume
                    .engine()
                    .derived_result(session, file.handle, PLAIN_TEXT),
                Ok(None)
            );
        });
    }

    #[test]
    fn corrupt_derived_batch_does_not_make_the_namespace_unmountable() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("note"),
                ObjectKind::File,
            ))
            .unwrap();
            let bytes = b"rebuildable derived search";
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                bytes,
                crate::ANY_REVISION,
            ))
            .unwrap();
            let owner = crate::extraction::ExtractionOwner::new(42);
            let (lease, input) = crate::block_on(volume.engine_mut().lend_version(
                owner,
                session,
                file.handle,
            ))
            .unwrap();
            let result = parse_plain_text(input, bytes).unwrap();
            let mut wire = [0u8; WORKER_REPORT_BYTES];
            let used = encode_result(&result, &mut wire).unwrap();
            crate::block_on(volume.engine_mut().commit_encoded_derived_result(
                owner,
                lease,
                result,
                &wire[..used],
                0,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
            crate::block_on(volume.commit(session)).unwrap();

            let durable = volume.engine().durable_derived_results::<1>().unwrap();
            let extent = volume.engine().provider().extent(durable[0].key).unwrap();
            volume
                .engine()
                .provider()
                .control(|device| device.corrupt_byte(extent.lba as usize * 512 + 3, 0x80))
                .unwrap()
                .unwrap();

            let recovered_session = SessionId::new(2);
            let mut volume = crate::block_on(volume.restart(recovered_session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(recovered_session, crate::Rights::ALL)
                .unwrap();
            assert_eq!(
                volume.engine().search(
                    recovered_session,
                    root.handle,
                    crate::SearchQuery {
                        content: Some(crate::SearchContent {
                            plane: crate::SearchPlane::FullText,
                            term: crate::SearchTerm::new("derived").unwrap(),
                        }),
                        freshness: crate::FreshnessPolicy::CurrentOnly,
                        ..crate::SearchQuery::ALL
                    },
                    None,
                ),
                Err(FsError::IndexIncomplete)
            );
            assert!(
                volume
                    .engine_mut()
                    .open_at(
                        recovered_session,
                        root.handle,
                        &name("note"),
                        crate::Rights::READ,
                    )
                    .is_ok(),
                "derived corruption cannot hide or invalidate authoritative files"
            );
        });
    }

    #[test]
    fn directory_diamond_survives_commit_and_detaches_one_route_exactly() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let left = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("left"),
                ObjectKind::Directory,
            ))
            .unwrap();
            let right = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("right"),
                ObjectKind::Directory,
            ))
            .unwrap();
            let shared = crate::block_on(volume.engine_mut().create(
                session,
                left.handle,
                &name("shared"),
                ObjectKind::Directory,
            ))
            .unwrap();
            volume
                .engine_mut()
                .link(session, right.handle, &name("shared"), shared.handle)
                .unwrap();
            crate::block_on(volume.engine_mut().create(
                session,
                shared.handle,
                &name("leaf"),
                ObjectKind::File,
            ))
            .unwrap();
            let stable = volume
                .engine()
                .stat(session, shared.handle)
                .unwrap()
                .stable_object;
            crate::block_on(volume.commit(session)).unwrap();

            let mut volume = crate::block_on(volume.restart(session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let left = volume
                .engine_mut()
                .open_at(session, root.handle, &name("left"), crate::Rights::ALL)
                .unwrap();
            let right = volume
                .engine_mut()
                .open_at(session, root.handle, &name("right"), crate::Rights::ALL)
                .unwrap();
            let through_left = volume
                .engine_mut()
                .open_at(session, left.handle, &name("shared"), crate::Rights::ALL)
                .unwrap();
            let through_right = volume
                .engine_mut()
                .open_at(session, right.handle, &name("shared"), crate::Rights::ALL)
                .unwrap();
            assert_eq!(
                volume
                    .engine()
                    .stat(session, through_left.handle)
                    .unwrap()
                    .stable_object,
                stable
            );
            assert_eq!(
                volume
                    .engine()
                    .stat(session, through_right.handle)
                    .unwrap()
                    .stable_object,
                stable
            );

            crate::block_on(
                volume
                    .engine_mut()
                    .detach_edge(session, left.handle, &name("shared")),
            )
            .unwrap();
            assert_eq!(
                volume.engine().stat(session, through_left.handle),
                Err(FsError::AccessLost)
            );
            assert!(volume.engine().stat(session, through_right.handle).is_ok());
            assert!(
                volume
                    .engine_mut()
                    .open_at(
                        session,
                        through_right.handle,
                        &name("leaf"),
                        crate::Rights::ALL,
                    )
                    .is_ok()
            );
        });
    }

    #[test]
    fn atomic_replacement_cuts_recover_old_or_complete_new() {
        on_large_stack(|| {
            for stage in [
                CommitStage::BeforeMetadata,
                CommitStage::AfterMetadata,
                CommitStage::AfterMetadataFlush,
                CommitStage::AfterRootWrite,
                CommitStage::AfterRootFlush,
            ] {
                let session = SessionId::new(0);
                let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
                let root = volume
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let file = crate::block_on(volume.engine_mut().create(
                    session,
                    root.handle,
                    &name("state"),
                    ObjectKind::File,
                ))
                .unwrap();
                let old_revision = crate::block_on(volume.engine_mut().write_at(
                    session,
                    file.handle,
                    0,
                    b"old",
                    crate::ANY_REVISION,
                ))
                .unwrap();
                crate::block_on(volume.commit(session)).unwrap();
                let mut replacement = crate::block_on(volume.engine().begin_replace(
                    session,
                    file.handle,
                    3,
                    old_revision,
                ))
                .unwrap();
                crate::block_on(volume.engine().write_replacement(
                    session,
                    &mut replacement,
                    0,
                    b"n",
                ))
                .unwrap();
                // Another session's commit while staging must still encode old bytes.
                crate::block_on(volume.commit(session)).unwrap();
                crate::block_on(volume.engine().write_replacement(
                    session,
                    &mut replacement,
                    1,
                    b"ew",
                ))
                .unwrap();
                let new_revision =
                    crate::block_on(volume.engine().commit_replace(session, &replacement)).unwrap();
                assert_eq!(
                    crate::block_on(volume.commit_with_cut(session, Some(stage))),
                    Err(PersistenceError::Interrupted(stage))
                );
                let mut recovered = crate::block_on(volume.restart(session)).unwrap();
                let root = recovered
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let file = recovered
                    .engine_mut()
                    .open_at(session, root.handle, &name("state"), crate::Rights::ALL)
                    .unwrap();
                let mut bytes = [0; 3];
                crate::block_on(
                    recovered
                        .engine_mut()
                        .read(session, file.handle, 0, &mut bytes),
                )
                .unwrap();
                let stat = recovered.engine().stat(session, file.handle).unwrap();
                let old = bytes == *b"old" && stat.revision == old_revision;
                let new = bytes == *b"new" && stat.revision == new_revision;
                assert!(old || new, "{stage:?} recovered a hybrid");
            }
        });
    }

    #[test]
    fn every_publication_cut_recovers_one_complete_root() {
        on_large_stack(|| {
            for stage in [
                CommitStage::BeforeMetadata,
                CommitStage::AfterMetadata,
                CommitStage::AfterMetadataFlush,
                CommitStage::AfterRootWrite,
                CommitStage::AfterRootFlush,
            ] {
                let session = SessionId::new(0);
                let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
                let root = volume
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let file = crate::block_on(volume.engine_mut().create(
                    session,
                    root.handle,
                    &name("state"),
                    ObjectKind::File,
                ))
                .unwrap();
                let old_revision = crate::block_on(volume.engine_mut().write_at(
                    session,
                    file.handle,
                    0,
                    b"old",
                    crate::ANY_REVISION,
                ))
                .unwrap();
                crate::block_on(volume.commit(session)).unwrap();
                let new_revision = crate::block_on(volume.engine_mut().write_at(
                    session,
                    file.handle,
                    0,
                    b"new",
                    old_revision,
                ))
                .unwrap();
                assert_eq!(
                    crate::block_on(volume.commit_with_cut(session, Some(stage))),
                    Err(PersistenceError::Interrupted(stage))
                );
                let mut recovered = crate::block_on(volume.restart(session)).unwrap();
                let root = recovered
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let file = recovered
                    .engine_mut()
                    .open_at(session, root.handle, &name("state"), crate::Rights::ALL)
                    .unwrap();
                let mut bytes = [0; 3];
                crate::block_on(
                    recovered
                        .engine_mut()
                        .read(session, file.handle, 0, &mut bytes),
                )
                .unwrap();
                let stat = recovered.engine().stat(session, file.handle).unwrap();
                let old = bytes == *b"old" && stat.revision == old_revision;
                let new = bytes == *b"new" && stat.revision == new_revision;
                assert!(old || new, "{stage:?} recovered a hybrid");
            }
        });
    }

    /// Model exec's exact per-member seed window over the real persistent
    /// engine: create, chunked writes with no intervening publication, detach
    /// on failure, then one closing commit. Retryable backpressure and a
    /// terminal channel loss are different backend facts, but neither may leave the
    /// already-written prefix discoverable after reboot.
    #[test]
    fn seed_member_failures_recover_without_a_partial_name() {
        on_large_stack(|| {
            for (failure, expected) in [
                (BlockError::Backpressure, ProviderError::Backpressure),
                (BlockError::Lost, ProviderError::Lost),
            ] {
                let session = SessionId::new(0);
                let mut volume = crate::block_on(Volume::format_with_era(
                    provider(),
                    session,
                    [0x71; ERA_BYTES],
                ))
                .unwrap();
                let root = volume
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let app = crate::block_on(volume.engine_mut().create(
                    session,
                    root.handle,
                    &name("app"),
                    ObjectKind::Directory,
                ))
                .unwrap();
                crate::block_on(volume.commit(session)).unwrap();

                let member = crate::block_on(volume.engine_mut().create(
                    session,
                    app.handle,
                    &name("tool"),
                    ObjectKind::File,
                ))
                .unwrap();
                let revision = crate::block_on(volume.engine_mut().resize(
                    session,
                    member.handle,
                    8192,
                    member.revision,
                ))
                .unwrap();
                let revision = crate::block_on(volume.engine_mut().write_at(
                    session,
                    member.handle,
                    0,
                    &[0x11; 4096],
                    revision,
                ))
                .unwrap();
                volume
                    .engine()
                    .provider()
                    .control(|device| device.inject(failure))
                    .unwrap();
                assert_eq!(
                    crate::block_on(volume.engine_mut().write_at(
                        session,
                        member.handle,
                        4096,
                        &[0x22; 4096],
                        revision,
                    )),
                    Err(FsError::ProviderFailure(expected))
                );

                // This is write_member's failure branch before it closes the
                // deferral window. Commit the cleaned namespace, then model a
                // cold reboot rather than trusting the live engine's tables.
                crate::block_on(volume.engine_mut().detach_edge(
                    session,
                    app.handle,
                    &name("tool"),
                ))
                .unwrap();
                crate::block_on(volume.commit(session)).unwrap();

                let mut recovered = crate::block_on(volume.restart(session)).unwrap();
                let root = recovered
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let app = recovered
                    .engine_mut()
                    .open_at(session, root.handle, &name("app"), crate::Rights::ALL)
                    .unwrap();
                assert_eq!(
                    recovered.engine_mut().open_at(
                        session,
                        app.handle,
                        &name("tool"),
                        crate::Rights::READ,
                    ),
                    Err(FsError::NotFound),
                    "{failure:?} exposed a partially copied seed after reboot"
                );
            }
        });
    }

    /// Extend the generic root-publication cut matrix through exec's seed
    /// shape. The closing commit may recover as the old root (no name) or the
    /// new root (the complete member), but no cut may publish one written
    /// prefix as a seeded executable.
    #[test]
    fn every_seed_publication_cut_recovers_absent_or_complete() {
        on_large_stack(|| {
            const FIRST: [u8; 4096] = [0x31; 4096];
            const SECOND: [u8; 4096] = [0x32; 4096];
            for stage in [
                CommitStage::BeforeMetadata,
                CommitStage::AfterMetadata,
                CommitStage::AfterMetadataFlush,
                CommitStage::AfterRootWrite,
                CommitStage::AfterRootFlush,
            ] {
                let session = SessionId::new(0);
                let mut volume = crate::block_on(Volume::format_with_era(
                    provider(),
                    session,
                    [0x72; ERA_BYTES],
                ))
                .unwrap();
                let root = volume
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let app = crate::block_on(volume.engine_mut().create(
                    session,
                    root.handle,
                    &name("app"),
                    ObjectKind::Directory,
                ))
                .unwrap();
                crate::block_on(volume.commit(session)).unwrap();

                let member = crate::block_on(volume.engine_mut().create(
                    session,
                    app.handle,
                    &name("tool"),
                    ObjectKind::File,
                ))
                .unwrap();
                let revision = crate::block_on(volume.engine_mut().resize(
                    session,
                    member.handle,
                    (FIRST.len() + SECOND.len()) as u64,
                    member.revision,
                ))
                .unwrap();
                let revision = crate::block_on(volume.engine_mut().write_at(
                    session,
                    member.handle,
                    0,
                    &FIRST,
                    revision,
                ))
                .unwrap();
                crate::block_on(volume.engine_mut().write_at(
                    session,
                    member.handle,
                    FIRST.len() as u64,
                    &SECOND,
                    revision,
                ))
                .unwrap();
                assert_eq!(
                    crate::block_on(volume.commit_with_cut(session, Some(stage))),
                    Err(PersistenceError::Interrupted(stage))
                );

                let mut recovered = crate::block_on(volume.restart(session)).unwrap();
                let root = recovered
                    .engine_mut()
                    .open_root(session, crate::Rights::ALL)
                    .unwrap();
                let app = recovered
                    .engine_mut()
                    .open_at(session, root.handle, &name("app"), crate::Rights::ALL)
                    .unwrap();
                let member = recovered.engine_mut().open_at(
                    session,
                    app.handle,
                    &name("tool"),
                    crate::Rights::READ,
                );
                let Ok(member) = member else {
                    assert_eq!(member, Err(FsError::NotFound));
                    continue;
                };
                assert_eq!(member.len, (FIRST.len() + SECOND.len()) as u64);
                let mut bytes = [0u8; FIRST.len() + SECOND.len()];
                assert_eq!(
                    crate::block_on(recovered.engine_mut().read(
                        session,
                        member.handle,
                        0,
                        &mut bytes[..FIRST.len()],
                    )),
                    Ok(FIRST.len())
                );
                assert_eq!(
                    crate::block_on(recovered.engine_mut().read(
                        session,
                        member.handle,
                        FIRST.len() as u64,
                        &mut bytes[FIRST.len()..],
                    )),
                    Ok(SECOND.len())
                );
                assert_eq!(
                    &bytes[..FIRST.len()],
                    &FIRST,
                    "{stage:?} published a bad prefix"
                );
                assert_eq!(
                    &bytes[FIRST.len()..],
                    &SECOND,
                    "{stage:?} published a bad suffix"
                );
            }
        });
    }

    #[test]
    fn a_root_write_cut_freezes_every_extent_it_may_have_published() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("state"),
                ObjectKind::File,
            ))
            .unwrap();
            let old_revision = crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"old",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            let new_revision = crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"new",
                old_revision,
            ))
            .unwrap();
            assert_eq!(
                crate::block_on(volume.commit_with_cut(session, Some(CommitStage::AfterRootWrite))),
                Err(PersistenceError::Interrupted(CommitStage::AfterRootWrite))
            );

            // The interrupted publisher cannot know whether the new root is
            // durable. This third, unpublished revision must therefore COW
            // rather than overwrite bytes either recoverable root may name.
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"bad",
                new_revision,
            ))
            .unwrap();

            let mut recovered = crate::block_on(volume.restart(session)).unwrap();
            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("state"), crate::Rights::ALL)
                .unwrap();
            let mut bytes = [0; 3];
            crate::block_on(
                recovered
                    .engine_mut()
                    .read(session, file.handle, 0, &mut bytes),
            )
            .unwrap();
            assert!(bytes == *b"old" || bytes == *b"new");
        });
    }

    #[test]
    fn a_file_saved_empty_recovers() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let engine = volume.engine_mut();
            let root = engine.open_root(session, crate::Rights::ALL).unwrap();
            let file = crate::block_on(engine.create(session, root.handle, &name("note"), ObjectKind::File)).unwrap();
            crate::block_on(engine.write_at(session, file.handle, 0, b"text", crate::ANY_REVISION)).unwrap();
            // Saved empty, as the editor saves: a replacement of no bytes,
            // whose placement holds no blocks and so no address.
            let stage = crate::block_on(engine.begin_replace(session, file.handle, 0, crate::ANY_REVISION)).unwrap();
            crate::block_on(engine.commit_replace(session, &stage)).unwrap();
            engine.close(session, file.handle).unwrap();
            // A second empty file: two addressless extents side by side are
            // not an overlap.
            let other = crate::block_on(engine.create(session, root.handle, &name("other"), ObjectKind::File)).unwrap();
            let stage = crate::block_on(engine.begin_replace(session, other.handle, 0, crate::ANY_REVISION)).unwrap();
            crate::block_on(engine.commit_replace(session, &stage)).unwrap();
            engine.close(session, other.handle).unwrap();
            // Both roots carry the empty files, so neither may refuse them.
            crate::block_on(volume.commit(session)).unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            let provider = volume.engine.into_provider();
            let bootstrap = bootstrap_data_start(provider.geometry().block_size).unwrap();
            let provider = provider.restart_with_data_start(bootstrap).unwrap();
            let mut recovered = match crate::block_on(Volume::recover_witnessed(provider, session)) {
                Ok(recovered) => recovered,
                Err(failure) => panic!("recovery refused: {failure:?} (witness {:#x})", failure.witness()),
            };
            assert_eq!(recovered.status().mode, MountMode::ReadWrite);
            // The file is there, empty, and takes bytes again.
            let engine = recovered.engine_mut();
            let root = engine.open_root(session, crate::Rights::ALL).unwrap();
            let file = engine.open_at(session, root.handle, &name("note"), crate::Rights::ALL).unwrap();
            assert_eq!(engine.stat(session, file.handle).unwrap().len, 0);
            crate::block_on(engine.write_at(session, file.handle, 0, b"again", crate::ANY_REVISION)).unwrap();
            crate::block_on(recovered.commit(session)).unwrap();
        });
    }

    #[test]
    fn a_wide_and_deep_directory_tree_recovers() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let engine = volume.engine_mut();
            let root = engine.open_root(session, crate::Rights::ALL).unwrap();
            // Wide: one directory holding many directories, each with a
            // child directory, so restore links populated subtrees.
            let top = crate::block_on(engine.create(session, root.handle, &name("top"), ObjectKind::Directory)).unwrap();
            for i in 0..120 {
                let dir = crate::block_on(engine.create(
                    session,
                    top.handle,
                    &name(&std::format!("d{i}")),
                    ObjectKind::Directory,
                ))
                .unwrap();
                let inner = crate::block_on(engine.create(session, dir.handle, &name("inner"), ObjectKind::Directory)).unwrap();
                engine.close(session, inner.handle).unwrap();
                engine.close(session, dir.handle).unwrap();
            }
            engine.close(session, top.handle).unwrap();
            // Deep: a chain of directories.
            let mut at = crate::block_on(engine.create(session, root.handle, &name("deep"), ObjectKind::Directory)).unwrap();
            for i in 0..crate::bounds::MAX_PATH_DEPTH - 2 {
                let next = crate::block_on(engine.create(
                    session,
                    at.handle,
                    &name(&std::format!("c{i}")),
                    ObjectKind::Directory,
                ))
                .unwrap();
                engine.close(session, at.handle).unwrap();
                at = next;
            }
            engine.close(session, at.handle).unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            let provider = volume.engine.into_provider();
            let bootstrap = bootstrap_data_start(provider.geometry().block_size).unwrap();
            let provider = provider.restart_with_data_start(bootstrap).unwrap();
            if let Err(failure) = crate::block_on(Volume::recover_witnessed(provider, session)) {
                panic!("recovery refused: {failure:?} (witness {:#x})", failure.witness());
            }
        });
    }

    #[test]
    fn a_refused_recovery_names_its_stage_and_cause() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            // Both root records damaged: each root falls back to the other,
            // and the last refusal is the one reported.
            for root in [16, 512 + 16] {
                volume
                    .engine()
                    .provider()
                    .control(|device| device.corrupt_byte(root, 0x80))
                    .unwrap()
                    .unwrap();
            }
            let provider = volume.engine.into_provider();
            let bootstrap = bootstrap_data_start(provider.geometry().block_size).unwrap();
            let provider = provider.restart_with_data_start(bootstrap).unwrap();
            let failure = match crate::block_on(Volume::recover_witnessed(provider, session)) {
                Err(failure) => failure,
                Ok(_) => panic!("two damaged roots must refuse the mount"),
            };
            assert_eq!(failure.stage, RecoverStage::RootDecode);
            let [stage, kind, _, _] = failure.witness().to_le_bytes();
            assert_eq!(stage, RecoverStage::RootDecode as u8);
            assert_eq!(kind, match failure.error {
                PersistenceError::Corrupt => 3,
                PersistenceError::Incompatible => 2,
                PersistenceError::Unformatted => 1,
                other => panic!("a damaged root record is not {other:?}"),
            });
        });
        // A filesystem refusal carries its wire status, and a provider
        // failure inside it its provider code beside it.
        let failure = RecoverFailure::at(
            RecoverStage::Edges,
            PersistenceError::Filesystem(FsError::ProviderFailure(ProviderError::Integrity)),
        );
        assert_eq!(
            failure.witness().to_le_bytes(),
            [
                RecoverStage::Edges as u8,
                7,
                kore_dagfs::wire::fs_status_code(FsError::ProviderFailure(ProviderError::Integrity)),
                kore_dagfs::wire::provider_code(ProviderError::Integrity),
            ]
        );
        let interrupted = RecoverFailure::at(RecoverStage::Extents, PersistenceError::Interrupted(CommitStage::BeforeMetadata));
        assert_eq!(interrupted.witness().to_le_bytes(), [RecoverStage::Extents as u8, 5, 1, 0]);
    }

    #[test]
    fn corrupt_newest_root_falls_back_without_losing_old_payload() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("state"),
                ObjectKind::File,
            ))
            .unwrap();
            let old_revision = crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"old",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap(); // generation 2, slot 0
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"new",
                old_revision,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap(); // generation 3, slot 1
            volume
                .engine()
                .provider()
                .control(|device| device.corrupt_byte(512 + 16, 0x80))
                .unwrap()
                .unwrap();
            let mut recovered = crate::block_on(volume.restart(session)).unwrap();
            assert_eq!(
                recovered.status(),
                MountStatus {
                    generation: 2,
                    used_fallback_root: true,
                    mode: MountMode::DegradedReadOnly,
                    era: [0u8; ERA_BYTES],
                }
            );
            assert_eq!(
                crate::block_on(recovered.commit(session)),
                Err(PersistenceError::Provider(ProviderError::ReadOnly))
            );
            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("state"), crate::Rights::ALL)
                .unwrap();
            let mut bytes = [0; 3];
            crate::block_on(
                recovered
                    .engine_mut()
                    .read(session, file.handle, 0, &mut bytes),
            )
            .unwrap();
            assert_eq!(bytes, *b"old");
            assert_eq!(
                recovered
                    .engine()
                    .stat(session, file.handle)
                    .unwrap()
                    .revision,
                old_revision
            );
        });
    }

    #[test]
    fn parser_rejects_stale_duplicate_cyclic_overlapping_and_out_of_range_records() {
        let sid = |value| StableObjectId::new(value).unwrap();
        let root = DiskObject {
            stable: sid(1),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: 1,
            extent: None,
        };
        let file = |stable, lba| DiskObject {
            stable: sid(stable),
            kind: ObjectKind::File,
            subtype: ObjectSubtype::PLAIN,
            len: 512,
            revision: 2,
            extent: Some(DurableExtent {
                lba,
                blocks: 1,
                len: 512,
            }),
        };
        let edge = |parent, child, label| DiskEdge {
            parent: sid(parent),
            child: sid(child),
            name: name(label),
        };

        let (image, len, mut record) = fixture(&[root], &[]);
        record.generation += 1;
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));

        let (image, len, record) = fixture(&[root, root], &[]);
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));

        let dir = DiskObject {
            stable: sid(2),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: 1,
            extent: None,
        };
        let (image, len, record) = fixture(&[root, dir], &[edge(1, 2, "down"), edge(2, 1, "up")]);
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));

        let (image, len, record) = fixture(
            &[
                root,
                file(2, LEGACY_DATA_START_LBA),
                file(3, LEGACY_DATA_START_LBA),
            ],
            &[edge(1, 2, "a"), edge(1, 3, "b")],
        );
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));

        let (image, len, record) = fixture(&[root, file(2, 256)], &[edge(1, 2, "past-end")]);
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));

        let (mut image, len, record) = fixture(&[root], &[]);
        image[IMAGE_HEADER + 3] ^= 0x40;
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));

        let (image, len, record) = fixture(
            &[root, file(2, LEGACY_DATA_START_LBA)],
            &[edge(2, 1, "file-is-parent")],
        );
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));
    }

    #[test]
    fn parser_accepts_directory_diamonds_but_rejects_a_back_edge() {
        let sid = |value| StableObjectId::new(value).unwrap();
        let dir = |stable| DiskObject {
            stable: sid(stable),
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            len: 0,
            revision: 1,
            extent: None,
        };
        let edge = |parent, child, label| DiskEdge {
            parent: sid(parent),
            child: sid(child),
            name: name(label),
        };
        let objects = [dir(1), dir(2), dir(3), dir(4)];
        let diamond = [
            edge(1, 2, "left"),
            edge(1, 3, "right"),
            edge(2, 4, "shared"),
            edge(3, 4, "shared"),
        ];
        let (image, len, record) = fixture(&objects, &diamond);
        assert!(decode_image(&image[..len], record, 512, 256).is_ok());

        let cyclic = [
            edge(1, 2, "left"),
            edge(1, 3, "right"),
            edge(2, 4, "shared"),
            edge(3, 4, "shared"),
            edge(4, 2, "back"),
        ];
        let (image, len, record) = fixture(&objects, &cyclic);
        assert!(matches!(
            decode_image(&image[..len], record, 512, 256),
            Err(PersistenceError::Corrupt)
        ));
    }

    #[test]
    fn provider_loss_never_acknowledges_and_old_root_recovers() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("state"),
                ObjectKind::File,
            ))
            .unwrap();
            let old_revision = crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"old",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"new",
                old_revision,
            ))
            .unwrap();
            volume
                .engine()
                .provider()
                .control(|device| device.inject(kore_driver::block::BlockError::Lost))
                .unwrap();
            assert!(matches!(
                crate::block_on(volume.commit(session)),
                Err(PersistenceError::Provider(ProviderError::Lost))
            ));
            let mut recovered = crate::block_on(volume.restart(session)).unwrap();
            let root = recovered
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = recovered
                .engine_mut()
                .open_at(session, root.handle, &name("state"), crate::Rights::ALL)
                .unwrap();
            let mut bytes = [0; 3];
            crate::block_on(
                recovered
                    .engine_mut()
                    .read(session, file.handle, 0, &mut bytes),
            )
            .unwrap();
            assert_eq!(bytes, *b"old");
        });
    }

    #[test]
    fn replacing_the_fallback_root_reclaims_only_then_unreachable_extents() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("state"),
                ObjectKind::File,
            ))
            .unwrap();
            let mut revision = crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"one",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            revision = crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"two",
                revision,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            assert_eq!(volume.engine().provider().stats().orphan_reclaims, 0);
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"tri",
                revision,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            assert!(volume.engine().provider().stats().orphan_reclaims > 0);
        });
    }

    /// The digest a lend computed is persisted in the root's sidecar and
    /// answers the same lend after a cold restart without a single read of
    /// the file — the boot-time rehash the sidecar exists to remove.
    /// A lent version's grant names its whole blocks in namespace
    /// coordinates, never the partial last block, and stays on the same
    /// blocks after the file is rewritten: the lease pins the placement,
    /// and the rewrite lands in fresh blocks a new lend names instead.
    #[test]
    fn a_lent_version_grants_its_whole_blocks_and_keeps_them_across_a_rewrite() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("blob"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                &[0x11u8; 3000],
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            let owner = crate::extraction::ExtractionOwner::new(7);
            let (lease, _) =
                crate::block_on(volume.engine_mut().lend_version(owner, session, file.handle))
                    .unwrap();
            let pinned = volume.engine().lent_blocks(owner, lease).unwrap().unwrap();
            let block = u64::from(pinned.block_size);
            assert_eq!(pinned.blocks, 3000 / block, "whole blocks only, never the tail");
            assert!(pinned.blocks * block <= 3000);

            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                &[0x22u8; 3000],
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            assert_eq!(
                volume.engine().lent_blocks(owner, lease).unwrap(),
                Some(pinned),
                "the lease keeps the version's blocks"
            );
            let other = crate::extraction::ExtractionOwner::new(8);
            let (fresh, _) =
                crate::block_on(volume.engine_mut().lend_version(other, session, file.handle))
                    .unwrap();
            let rewritten = volume.engine().lent_blocks(other, fresh).unwrap().unwrap();
            assert_ne!(rewritten.lba, pinned.lba, "the rewrite landed in fresh blocks");
            assert!(
                volume.engine().lent_blocks(other, lease).is_err(),
                "a lease answers only its owner"
            );
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
            crate::block_on(volume.engine_mut().release_lent_version(other, fresh)).unwrap();
            assert!(volume.engine().lent_blocks(owner, lease).is_err(), "released");

            // A lease pins bytes, not independent read authority. Closing the
            // view before grant/read was the inference attachment regression.
            let (closed, identity) =
                crate::block_on(volume.engine().lend_version(owner, session, file.handle)).unwrap();
            volume.engine().close(session, file.handle).unwrap();
            assert!(matches!(volume.engine().lent_blocks(owner, closed),
                Err(crate::extraction::ExtractionError::Filesystem(FsError::InvalidHandle))));
            let mut tail = [0u8; 1];
            assert!(crate::block_on(volume.engine().read_lent_version(owner, closed, identity.len - 1, &mut tail)).is_err());
            crate::block_on(volume.engine().release_lent_version(owner, closed)).unwrap();

            // A refused snapshot must not poison the session: reopen, keep
            // its view through the body grant and tail, release, then close.
            let reopened = volume.engine().open_at(session, root.handle, &name("blob"), crate::Rights::READ).unwrap();
            let (retry, _) =
                crate::block_on(volume.engine().lend_version(owner, session, reopened.handle)).unwrap();
            assert!(volume.engine().lent_blocks(owner, retry).unwrap().is_some());
            assert_eq!(crate::block_on(volume.engine().read_lent_version(owner, retry, identity.len - 1, &mut tail)).unwrap(), 1);
            assert_eq!(tail, [0x22]);
            crate::block_on(volume.engine().release_lent_version(owner, retry)).unwrap();
            volume.engine().close(session, reopened.handle).unwrap();
        });
    }

    #[test]
    fn a_lend_after_restart_answers_its_digest_from_the_sidecar() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("blob"),
                ObjectKind::File,
            ))
            .unwrap();
            let bytes = [0x5au8; 3000];
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                &bytes,
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            let owner = crate::extraction::ExtractionOwner::new(7);
            let (lease, first) =
                crate::block_on(volume.engine_mut().lend_version(owner, session, file.handle))
                    .unwrap();
            assert_eq!(first.digest, *blake3::hash(&bytes).as_bytes());
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            assert!(volume.state.sidecar.is_some(), "the commit wrote the sidecar");

            let recovered = SessionId::new(2);
            let mut volume = crate::block_on(volume.restart(recovered)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(recovered, crate::Rights::ALL)
                .unwrap();
            let file = volume
                .engine_mut()
                .open_at(recovered, root.handle, &name("blob"), crate::Rights::READ)
                .unwrap();
            let reads = volume.engine().provider().stats().reads;
            let (lease, again) =
                crate::block_on(volume.engine_mut().lend_version(owner, recovered, file.handle))
                    .unwrap();
            assert_eq!(again.digest, first.digest);
            assert_eq!(
                volume.engine().provider().stats().reads,
                reads,
                "answered from the sidecar, not by rereading the file"
            );
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
        });
    }

    /// Ruling 3: a version-4 volume mounts as it is, keeps its contents, and
    /// its next commit publishes a version-5 root; the upgraded volume
    /// mounts again with everything in place.
    #[test]
    fn a_version_4_volume_is_upgraded_to_5_by_its_next_commit() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("kept"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            rewrite_active_as_legacy(&volume, SUBTYPE_FORMAT_VERSION);

            let active_version = |volume: &Volume| {
                let provider = volume.engine().provider();
                let block = provider.geometry().block_size as usize;
                let mut head = [0u8; 4096];
                let slot = (volume.status().generation & 1) as u64;
                crate::block_on(provider.read_metadata(slot, &mut head[..block])).unwrap();
                decode_root(
                    &head[..block],
                    provider.geometry().block_size,
                    provider.geometry().block_count,
                )
                .unwrap()
                .version
            };

            let mut volume = crate::block_on(volume.restart(session)).unwrap();
            assert_eq!(volume.status().mode, MountMode::ReadWrite);
            assert_eq!(active_version(&volume), SUBTYPE_FORMAT_VERSION);
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("added"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            assert_eq!(active_version(&volume), DIGEST_FORMAT_VERSION);

            let mut volume = crate::block_on(volume.restart(session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            for kept in ["kept", "added"] {
                volume
                    .engine_mut()
                    .open_at(session, root.handle, &name(kept), crate::Rights::READ)
                    .unwrap();
            }
        });
    }

    /// A sidecar whose bytes are damaged is derived data gone wrong: the
    /// volume mounts, the lend computes the digest from the file again (and
    /// gets it right), and the next commit replaces the sidecar.
    #[test]
    fn a_damaged_sidecar_is_ignored_and_replaced() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let mut volume = crate::block_on(Volume::format(provider(), session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("blob"),
                ObjectKind::File,
            ))
            .unwrap();
            let bytes = [0x33u8; 1500];
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                &bytes,
                crate::ANY_REVISION,
            ))
            .unwrap();
            let owner = crate::extraction::ExtractionOwner::new(9);
            let (lease, _) =
                crate::block_on(volume.engine_mut().lend_version(owner, session, file.handle))
                    .unwrap();
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            let (_, sidecar) = volume.state.sidecar.expect("written");
            // One flipped bit inside the first record: the CRC must catch it.
            volume
                .engine()
                .provider()
                .control(|device| {
                    device.corrupt_byte(
                        sidecar.extent.lba as usize * 512 + crate::digests::HEADER_BYTES + 3,
                        0x80,
                    )
                })
                .unwrap()
                .unwrap();

            let mut volume = crate::block_on(volume.restart(session)).unwrap();
            assert_eq!(volume.status().mode, MountMode::ReadWrite);
            assert!(volume.engine().digests().borrow().is_empty());
            assert!(volume.engine().digests().borrow().dirty(), "marked for replacement");
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = volume
                .engine_mut()
                .open_at(session, root.handle, &name("blob"), crate::Rights::READ)
                .unwrap();
            let reads = volume.engine().provider().stats().reads;
            let (lease, input) =
                crate::block_on(volume.engine_mut().lend_version(owner, session, file.handle))
                    .unwrap();
            assert_eq!(input.digest, *blake3::hash(&bytes).as_bytes());
            assert!(volume.engine().provider().stats().reads > reads, "recomputed");
            crate::block_on(volume.engine_mut().release_lent_version(owner, lease)).unwrap();
            crate::block_on(volume.commit(session)).unwrap();
            let (_, fresh) = volume.state.sidecar.expect("replaced");
            assert_ne!(fresh.extent.lba, sidecar.extent.lba, "a fresh extent, not the damaged one");
        });
    }
}
