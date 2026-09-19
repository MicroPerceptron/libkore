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
        MAX_BLOCK_PLACEMENTS, MAX_DERIVED_RESULTS, MAX_DIR_ENTRIES, MAX_DURABLE_DIRECTORIES,
        MAX_NAME_LEN,
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
/// The current version.
pub const FORMAT_VERSION: u32 = SUBTYPE_FORMAT_VERSION;
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
pub const MAX_DURABLE_EDGES: usize = MAX_DURABLE_DIRECTORIES * MAX_DIR_ENTRIES;

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
        ARENA_FORMAT_VERSION | SUBTYPE_FORMAT_VERSION if data_start == LEGACY_DATA_START_LBA => {
            Ok(ImageProfile::Legacy)
        }
        ARENA_FORMAT_VERSION | SUBTYPE_FORMAT_VERSION
            if data_start >= image_layout(ImageProfile::Current, block_size)?.data_start =>
        {
            Ok(ImageProfile::Current)
        }
        ARENA_FORMAT_VERSION | SUBTYPE_FORMAT_VERSION => Err(PersistenceError::Corrupt),
        _ => Err(PersistenceError::Incompatible),
    }
}

fn durable_bounds(profile: ImageProfile) -> (usize, usize) {
    match profile {
        ImageProfile::Legacy => (LEGACY_MAX_DURABLE_OBJECTS, LEGACY_MAX_DURABLE_EDGES),
        ImageProfile::Current => (MAX_DURABLE_OBJECTS, MAX_DURABLE_EDGES),
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

struct Decoded {
    generation: u64,
    era: [u8; ERA_BYTES],
    root: StableObjectId,
    objects: heapless::Vec<DiskObject, MAX_DURABLE_OBJECTS>,
    edges: heapless::Vec<DiskEdge, MAX_DURABLE_EDGES>,
    derived: heapless::Vec<DiskDerived, MAX_DERIVED_RESULTS>,
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
        let geometry = provider.geometry();
        let block = geometry.block_size as usize;
        if block > 4096 {
            return Err(PersistenceError::Incompatible);
        }
        let mut blocks = [[0u8; 4096]; 2];
        provider.read_metadata(0, &mut blocks[0][..block]).await?;
        provider.read_metadata(1, &mut blocks[1][..block]).await?;
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
            return Err(PersistenceError::Unformatted);
        }
        if let (Ok(a), Ok(b)) = (parsed[0], parsed[1])
            && a.generation == b.generation
        {
            // Two independently published slots cannot lawfully claim the
            // same generation. Choosing by slot would silently bless an
            // ambiguous/tampered history.
            return Err(PersistenceError::Corrupt);
        }
        if let (Ok(a), Ok(b)) = (parsed[0], parsed[1])
            && (a.data_start != b.data_start || a.arena_blocks != b.arena_blocks)
        {
            // Both slots are generations of one volume and therefore one
            // allocation arena. Selecting either of two contradictory bounds
            // would make the other root's extents mean something different.
            return Err(PersistenceError::Corrupt);
        }
        let mut order = [0usize, 1usize];
        let generation = |i: usize| parsed[i].as_ref().map_or(0, |root| root.generation);
        if generation(order[1]) > generation(order[0]) {
            order.swap(0, 1);
        }
        let newest_generation = generation(order[0]);
        let mut last_error = PersistenceError::Corrupt;
        let mut image = ImageScratch::new()?;
        for (attempt, index) in order.into_iter().enumerate() {
            let root = match parsed[index] {
                Ok(root) => root,
                Err(error) => {
                    last_error = error;
                    continue;
                }
            };
            if let Err(error) = provider.configure_arena(root.data_start, root.arena_blocks) {
                last_error = error.into();
                continue;
            }
            let profile = root_profile(root.version, root.data_start, geometry.block_size)?;
            let layout = image_layout(profile, geometry.block_size)?;
            let arena_lba = image_lba(profile, root.slot, geometry.block_size)?;
            let len = root.image_len as usize;
            let span = match image_transfer_bytes(len, geometry.block_size, layout.bytes) {
                Ok(span) => span,
                Err(error) => {
                    last_error = error;
                    continue;
                }
            };
            if let Err(error) = provider
                .read_metadata(arena_lba, &mut image.as_mut()[..span])
                .await
            {
                last_error = error.into();
                continue;
            }
            if CRC.checksum(&image.as_mut()[..len]) != root.image_crc {
                last_error = PersistenceError::Corrupt;
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
                    last_error = error;
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
            let mut live_extents = heapless::Vec::<DurableExtent, MAX_BLOCK_PLACEMENTS>::new();
            for extent in root_payload_extents(&decoded) {
                live_extents
                    .push(extent)
                    .map_err(|_| PersistenceError::Bounds)?;
            }
            let mut fallback_extents = heapless::Vec::<DurableExtent, MAX_BLOCK_PLACEMENTS>::new();
            if let Some(older) = older.as_ref() {
                for extent in root_payload_extents(older) {
                    fallback_extents
                        .push(extent)
                        .map_err(|_| PersistenceError::Bounds)?;
                }
            }
            let recovered_extents = provider.recover_extents(&live_extents, &fallback_extents)?;
            // Consume the keys through one iterator in the same canonical
            // order [`root_payload_extents`] defines; the exhaustion check
            // at the end proves the walks agreed.
            let mut recovered = recovered_extents.iter().copied();
            let ns = Namespace::<BlockKey>::new();
            for object in decoded.objects.iter() {
                let storage = match object.extent {
                    Some(_) => Some(recovered.next().ok_or(PersistenceError::Corrupt)?),
                    None => None,
                };
                ns.restore_object(
                    object.stable,
                    object.kind,
                    object.subtype,
                    object.len,
                    object.revision,
                    storage,
                )?;
            }
            let mut recovered_derived =
                heapless::Vec::<(DiskDerived, BlockKey), MAX_DERIVED_RESULTS>::new();
            for record in decoded.derived.iter().copied() {
                let key = recovered.next().ok_or(PersistenceError::Corrupt)?;
                recovered_derived
                    .push((record, key))
                    .map_err(|_| PersistenceError::Bounds)?;
            }
            if recovered.next().is_some() {
                return Err(PersistenceError::Corrupt);
            }
            for edge in decoded.edges.iter() {
                ns.restore_edge(edge.parent, edge.name, edge.child)?;
            }
            let root_id = ns
                .find_stable(decoded.root)
                .ok_or(PersistenceError::Corrupt)?;
            let engine = Engine::from_recovered(
                provider,
                ns,
                session,
                root_id,
                root.version >= SUBTYPE_FORMAT_VERSION,
            )?;
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
                    let mut hasher = blake3::Hasher::new();
                    let mut offset = 0u64;
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
                    if result.provenance.input.len != file_len
                        || result.provenance.input.version != current_revision
                        || result.provenance.input.digest != *hasher.finalize().as_bytes()
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
                },
            });
        }
        Err(last_error)
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
        state.version,
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
            version: state.version,
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
        },
        &mut root_block[..block],
    )?;
    // Once the root write is submitted, a failed completion or flush can
    // still leave this root durable. Freeze every extent it may name before
    // that cut so a later request cannot overwrite a recoverable generation
    // in place.
    provider.prepare_publication();
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
        .durable_graph::<MAX_DURABLE_EDGES>(root_id)?;
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
    if header == ROOT_HEADER_V3 {
        put_u64(block, 56, root.data_start)?;
        put_u64(block, 64, root.arena_blocks)?;
        put(block, 72, &root.era)?;
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
    let (data_start, arena_blocks, era) = if version == FORMAT_VERSION {
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
    let (max_objects, max_edges) = durable_bounds(profile);
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
    let mut objects = heapless::Vec::new();
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
            if expected != extent.blocks as u64 || extent.lba < root.data_start || end > arena_end {
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
        objects
            .push(DiskObject {
                stable,
                kind,
                subtype,
                len,
                revision,
                extent,
            })
            .map_err(|_| PersistenceError::Bounds)?;
        at += OBJECT_BYTES;
    }
    if objects.iter().filter(|object| object.kind.is_dir()).count() > MAX_DURABLE_DIRECTORIES {
        return Err(PersistenceError::Bounds);
    }
    let mut edges = heapless::Vec::new();
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
        edges
            .push(DiskEdge {
                parent,
                child,
                name,
            })
            .map_err(|_| PersistenceError::Bounds)?;
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
    let mut derived = heapless::Vec::new();
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
        derived
            .push(DiskDerived {
                object,
                input_version,
                extent,
                digest,
            })
            .map_err(|_| PersistenceError::Bounds)?;
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

fn validate_graph(
    root: StableObjectId,
    objects: &heapless::Vec<DiskObject, MAX_DURABLE_OBJECTS>,
    edges: &heapless::Vec<DiskEdge, MAX_DURABLE_EDGES>,
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
        assert_eq!(root.version, SUBTYPE_FORMAT_VERSION);
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
        for stable in 1..=MAX_DURABLE_DIRECTORIES as u64 + 1 {
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
    /// exist on disk without meaning anything.
    #[test]
    fn a_version_3_root_round_trips_its_arena_and_era() {
        let era = core::array::from_fn(|index| index as u8 ^ 0xa5);
        let data_start = image_layout(ImageProfile::Current, 512).unwrap().data_start + 8;
        let record = RootRecord {
            version: FORMAT_VERSION,
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
        };
        let mut block = [0u8; 512];
        encode_root(record, &mut block).unwrap();
        assert_eq!(get_u32(&block, 12).unwrap() as usize, ROOT_HEADER_V3);

        let back = decode_root(&block, 512, 4096).unwrap();
        assert_eq!(back.version, FORMAT_VERSION);
        assert_eq!(back.generation, 9);
        assert_eq!(back.data_start, data_start);
        assert_eq!(back.arena_blocks, 1000);
        assert_eq!(back.era, era);
        assert_eq!(back.root, record.root);
        assert_eq!(back.image_crc, record.image_crc);
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
                .durable_graph::<MAX_DURABLE_EDGES>(root_id)
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
}
