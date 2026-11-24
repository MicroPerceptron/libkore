//! Compiled-blob records: keep what a host provider compiled, bound by hash,
//! so a runtime without that provider's toolchain (kore) can load the exact
//! bytes later.
//!
//! kore consumes accelerator programs only as offline-compiled artifacts it
//! embeds itself: `FORMAT_VPUX_ELF` on its iVPU row, and Intel GT zebin ELFs
//! pinned by a hash manifest on its Xe compute path. Neither path compiles
//! anything at run time, and kore deliberately carries no SPIR-V frontend or
//! vendor runtime. The bridge is therefore *capture*: run the same `.ax`
//! partition through a host provider that owns a real compiler (the OpenVINO
//! GPU and NPU plugins, which reuse Intel's oneDNN/IGC and VPUX pipelines),
//! export the compiled model, and record it here together with the hashes of
//! everything that produced it. The record is metadata, not authority: kore's
//! `ArtifactBuild` admits a record from its trusted package path and the
//! provider re-validates residency at load.
//!
//! What this module knows about blob layouts is deliberately shallow and
//! self-verifying: an ELF64 header is parsed only far enough to bound the
//! image and read its machine type. An OpenVINO GPU export is an opaque
//! stream that *embeds* one zebin ELF per compiled kernel; an NPU export is
//! one VPUX ELF at offset zero. Nothing here interprets kernel descriptors —
//! that is the consuming driver's job, and kore's `xe_compute_blob.py` already
//! does it from the zebin's own metadata.

use core::fmt;

/// `e_machine` of Intel Graphics Technology zebin ELFs (`EM_INTELGT`).
pub const EM_INTELGT: u16 = 205;

/// `e_machine` of a VPUX compiled model (`EM_NONE`). Both a blob captured
/// here from the OpenVINO NPU plugin and the one kore commits under
/// `drivers/ivpu/model/` declare it.
pub const EM_NONE: u16 = 0;

/// kore's registered artifact format number for a VPUX ELF (`FORMAT_VPUX_ELF`).
pub const KORE_FORMAT_VPUX_ELF: u32 = 1;

/// `ELFCLASS64`.
const ELF_CLASS_64: u8 = 2;
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELF64_HEADER_LEN: usize = 64;

/// One ELF64 image found inside a blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedElf<'a> {
    /// Byte offset of the ELF magic inside the enclosing blob.
    pub offset: usize,
    /// `e_machine` of the image.
    pub machine: u16,
    /// The image bytes, bounded by its own program/section header tables
    /// and clipped to the enclosing blob.
    pub bytes: &'a [u8],
}

/// The layout a captured blob was recognised as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobKind {
    /// One `EM_NONE` ELF at offset zero: an NPU (VPUX) compiled model,
    /// kore's `FORMAT_VPUX_ELF`.
    VpuxElf,
    /// One Intel GT zebin ELF at offset zero: a bare kernel binary as kore's
    /// Xe compute path embeds.
    IntelGtZebin,
    /// An opaque container (an OpenVINO GPU export) embedding `kernels`
    /// Intel GT zebin ELFs. Only the embedded kernels are portable; the
    /// container framing belongs to the exporting runtime.
    GpuExport { kernels: usize },
    /// An ELF at offset zero of some other machine type. Recognised as an
    /// ELF and nothing more: no kore format is claimed for it, because
    /// "not Intel GT" is not evidence that a blob is a VPUX model.
    OtherElf { machine: u16 },
    /// No recognised ELF content.
    Unknown,
}

impl fmt::Display for BlobKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VpuxElf => formatter.write_str("vpux-elf"),
            Self::IntelGtZebin => formatter.write_str("intelgt-zebin"),
            Self::GpuExport { kernels } => write!(formatter, "gpu-export/{kernels}"),
            Self::OtherElf { machine } => write!(formatter, "elf/machine-{machine}"),
            Self::Unknown => formatter.write_str("unknown"),
        }
    }
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut word = [0; 8];
    word.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(word)
}

/// Parse the ELF64 image starting at `offset`, if a well-formed little-endian
/// header is there. The image length is the far end of the program and
/// section header tables (at least the header itself), clipped to the blob.
fn elf_at(blob: &[u8], offset: usize) -> Option<EmbeddedElf<'_>> {
    let rest = blob.get(offset..)?;
    if rest.len() < ELF64_HEADER_LEN
        || rest[..4] != ELF_MAGIC
        || rest[4] != ELF_CLASS_64
        || rest[5] != 1
    {
        return None;
    }
    let machine = read_u16(rest, 18);
    // The image ends where its header tables end or where the last segment /
    // section's file contents end, whichever is farther; entries that point
    // past the blob are clipped, never trusted.
    let mut end = ELF64_HEADER_LEN as u64;
    let mut extend_over_table =
        |table_offset: usize, entsize_at: usize, count_at: usize, contents: fn(&[u8]) -> u64| {
            let offset = read_u64(rest, table_offset);
            let entsize = u64::from(read_u16(rest, entsize_at));
            let count = u64::from(read_u16(rest, count_at));
            end = end.max(offset.saturating_add(entsize.saturating_mul(count)));
            for index in 0..count {
                let Some(start) = offset
                    .saturating_add(entsize.saturating_mul(index))
                    .try_into()
                    .ok()
                    .filter(|start: &usize| {
                        entsize >= 40 && start.checked_add(40).is_some_and(|end| end <= rest.len())
                    })
                else {
                    break;
                };
                end = end.max(contents(&rest[start..start + 40]));
            }
        };
    // Program headers: p_offset at 8, p_filesz at 32.
    extend_over_table(32, 54, 56, |entry| {
        read_u64(entry, 8).saturating_add(read_u64(entry, 32))
    });
    // Section headers: sh_type at 4 (8 = SHT_NOBITS occupies no file bytes),
    // sh_offset at 24, sh_size at 32.
    extend_over_table(40, 58, 60, |entry| {
        if u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]) == 8 {
            0
        } else {
            read_u64(entry, 24).saturating_add(read_u64(entry, 32))
        }
    });
    let len = usize::try_from(end).map_or(rest.len(), |end| end.min(rest.len()));
    Some(EmbeddedElf {
        offset,
        machine,
        bytes: &rest[..len],
    })
}

/// Every ELF64 image found by scanning `blob` for the ELF magic. Images are
/// reported in offset order; an image found inside another's declared range
/// is still reported, because container formats are free to nest.
pub fn embedded_elfs(blob: &[u8]) -> Vec<EmbeddedElf<'_>> {
    let mut found = Vec::new();
    let mut at = 0;
    while at + ELF64_HEADER_LEN <= blob.len() {
        if blob[at..at + 4] == ELF_MAGIC {
            if let Some(elf) = elf_at(blob, at) {
                found.push(elf);
            }
        }
        at += 1;
    }
    found
}

/// Recognise a captured blob's layout.
pub fn classify(blob: &[u8]) -> BlobKind {
    match elf_at(blob, 0) {
        Some(elf) if elf.machine == EM_INTELGT => BlobKind::IntelGtZebin,
        Some(elf) if elf.machine == EM_NONE => BlobKind::VpuxElf,
        Some(elf) => BlobKind::OtherElf {
            machine: elf.machine,
        },
        None => {
            let kernels = embedded_elfs(blob)
                .into_iter()
                .filter(|elf| elf.machine == EM_INTELGT)
                .count();
            if kernels == 0 {
                BlobKind::Unknown
            } else {
                BlobKind::GpuExport { kernels }
            }
        }
    }
}

/// kore's artifact format number for a blob, when one is registered. Intel GT
/// kernels have no registered format yet: kore's Xe compute path embeds them
/// from a manifest at build time rather than loading them through the
/// accelerator protocol.
pub fn kore_format(kind: BlobKind) -> Option<u32> {
    match kind {
        BlobKind::VpuxElf => Some(KORE_FORMAT_VPUX_ELF),
        BlobKind::IntelGtZebin
        | BlobKind::GpuExport { .. }
        | BlobKind::OtherElf { .. }
        | BlobKind::Unknown => None,
    }
}

/// A BLAKE3 digest, the hash kore's `ArtifactBuild` binds records with.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.hex())
    }
}

/// One embedded kernel image inside a captured blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRecord {
    pub offset: usize,
    pub len: usize,
    pub machine: u16,
    pub blake3: Digest,
}

/// Which side of a program boundary a port is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
}

impl fmt::Display for Direction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Input => "input",
            Self::Output => "output",
        })
    }
}

/// One boundary tensor of the compiled partition, in the order the provider
/// binds it. kore's `ArtifactBuild` binds device buffers by this slot index,
/// so a record without ports cannot be admitted there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortRecord {
    pub direction: Direction,
    /// Slot index within its direction, the order the provider binds.
    pub index: usize,
    /// `.ax` graph tensor this port carries.
    pub tensor: u32,
    /// Storage encoding, e.g. `f32le`.
    pub dtype: String,
    pub shape: Vec<u32>,
    pub bytes: u64,
}

/// One captured compiled program: what was compiled (the `.ax` partition and
/// the TOSA artifact authored from it), who compiled it, the exact bytes that
/// came back, and the boundary it binds.
///
/// Field names follow kore's `kore-ax-npu-build-v1` record so the two can be
/// compared directly: `artifact_*` is the *compiled* blob, and the TOSA input
/// the provider was handed is `tosa_*`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobRecord {
    /// Caller-chosen name (a lane, a partition index, a graph name).
    pub name: String,
    /// Provider that compiled the program (`openvino`, `vulkan`, …).
    pub provider: String,
    /// The provider's resolved device name (`NPU`, `GPU.0`, an adapter name).
    pub device: String,
    /// Provider runtime/compiler version string, when the provider reports one.
    pub compiler: String,
    /// BLAKE3 of the `.ax` graph bytes the partition came from.
    pub graph_blake3: Digest,
    /// BLAKE3 of the TOSA artifact the provider was handed.
    pub tosa_blake3: Digest,
    /// BLAKE3 of the compiled blob as exported.
    pub artifact_blake3: Digest,
    pub artifact_bytes: usize,
    pub kind: BlobKind,
    /// kore artifact format for `kind`, when registered.
    pub kore_format: Option<u32>,
    pub kernels: Vec<KernelRecord>,
    /// Boundary ports in binding order. Empty means the capture did not
    /// record one, not that the program has none.
    pub ports: Vec<PortRecord>,
}

impl BlobRecord {
    /// Bind a captured blob to its inputs. `ports` may be empty when the
    /// caller has no boundary metadata; [`with_ports`](Self::with_ports)
    /// attaches it.
    pub fn new(
        name: impl Into<String>,
        provider: impl Into<String>,
        device: impl Into<String>,
        compiler: impl Into<String>,
        graph: &[u8],
        tosa: &[u8],
        blob: &[u8],
    ) -> Self {
        let kind = classify(blob);
        let kernels = embedded_elfs(blob)
            .into_iter()
            .map(|elf| KernelRecord {
                offset: elf.offset,
                len: elf.bytes.len(),
                machine: elf.machine,
                blake3: Digest::of(elf.bytes),
            })
            .collect();
        Self {
            name: name.into(),
            provider: provider.into(),
            device: device.into(),
            compiler: compiler.into(),
            graph_blake3: Digest::of(graph),
            tosa_blake3: Digest::of(tosa),
            artifact_blake3: Digest::of(blob),
            artifact_bytes: blob.len(),
            kind,
            kore_format: kore_format(kind),
            kernels,
            ports: Vec::new(),
        }
    }

    /// Attach the boundary ports the provider binds, in binding order.
    pub fn with_ports(mut self, ports: Vec<PortRecord>) -> Self {
        self.ports = ports;
        self
    }

    /// Whether `blob` is the exact bytes this record was made from.
    pub fn verify(&self, blob: &[u8]) -> bool {
        blob.len() == self.artifact_bytes && Digest::of(blob) == self.artifact_blake3
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Render records as the JSON manifest shape kore's Xe kernel directory
/// uses: hashes as lowercase hex, one entry per captured blob. No JSON
/// dependency is pulled in for a fixed, flat schema.
pub fn manifest_json(records: &[BlobRecord]) -> String {
    let mut out =
        String::from("{\n  \"schema\": \"axnn-vaccel-blob-manifest/2\",\n  \"blobs\": [\n");
    for (index, record) in records.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": {},\n", json_string(&record.name)));
        out.push_str(&format!(
            "      \"provider\": {},\n",
            json_string(&record.provider)
        ));
        out.push_str(&format!(
            "      \"device\": {},\n",
            json_string(&record.device)
        ));
        out.push_str(&format!(
            "      \"compiler\": {},\n",
            json_string(&record.compiler)
        ));
        out.push_str(&format!(
            "      \"graph_blake3\": \"{}\",\n",
            record.graph_blake3.hex()
        ));
        out.push_str(&format!(
            "      \"tosa_blake3\": \"{}\",\n",
            record.tosa_blake3.hex()
        ));
        out.push_str(&format!(
            "      \"artifact_blake3\": \"{}\",\n",
            record.artifact_blake3.hex()
        ));
        out.push_str(&format!(
            "      \"artifact_bytes\": {},\n",
            record.artifact_bytes
        ));
        out.push_str(&format!(
            "      \"kind\": {},\n",
            json_string(&record.kind.to_string())
        ));
        match record.kore_format {
            Some(format) => out.push_str(&format!("      \"kore_format\": {format},\n")),
            None => out.push_str("      \"kore_format\": null,\n"),
        }
        out.push_str("      \"ports\": [");
        for (slot, port) in record.ports.iter().enumerate() {
            if slot > 0 {
                out.push(',');
            }
            let shape: Vec<String> = port.shape.iter().map(u32::to_string).collect();
            out.push_str(&format!(
                "\n        {{ \"direction\": \"{}\", \"index\": {}, \"tensor\": {}, \"dtype\": {}, \"shape\": [{}], \"bytes\": {} }}",
                port.direction,
                port.index,
                port.tensor,
                json_string(&port.dtype),
                shape.join(", "),
                port.bytes
            ));
        }
        out.push_str(if record.ports.is_empty() {
            "],\n"
        } else {
            "\n      ],\n"
        });
        out.push_str("      \"kernels\": [");
        for (slot, kernel) in record.kernels.iter().enumerate() {
            if slot > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "\n        {{ \"offset\": {}, \"len\": {}, \"machine\": {}, \"blake3\": \"{}\" }}",
                kernel.offset,
                kernel.len,
                kernel.machine,
                kernel.blake3.hex()
            ));
        }
        out.push_str(if record.kernels.is_empty() {
            "]\n"
        } else {
            "\n      ]\n"
        });
        out.push_str(if index + 1 == records.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }
    out.push_str("  ]\n}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal ELF64 image: the header, one 64-byte section header at
    /// offset 64 describing `body` bytes of contents that follow it.
    fn elf(machine: u16, body: usize) -> Vec<u8> {
        let mut image = vec![0; ELF64_HEADER_LEN + 64 + body];
        image[..4].copy_from_slice(&ELF_MAGIC);
        image[4] = ELF_CLASS_64;
        image[5] = 1;
        image[18..20].copy_from_slice(&machine.to_le_bytes());
        // e_shoff = 64, e_shentsize = 64, e_shnum = 1.
        image[40..48].copy_from_slice(&(ELF64_HEADER_LEN as u64).to_le_bytes());
        image[58..60].copy_from_slice(&64_u16.to_le_bytes());
        image[60..62].copy_from_slice(&1_u16.to_le_bytes());
        // sh_type PROGBITS, sh_offset = 128, sh_size = body.
        image[64 + 4..64 + 8].copy_from_slice(&1_u32.to_le_bytes());
        image[64 + 24..64 + 32].copy_from_slice(&128_u64.to_le_bytes());
        image[64 + 32..64 + 40].copy_from_slice(&(body as u64).to_le_bytes());
        image
    }

    #[test]
    fn classifies_bare_and_embedded_images() {
        let vpux = elf(EM_NONE, 32);
        assert_eq!(classify(&vpux), BlobKind::VpuxElf);
        assert_eq!(kore_format(classify(&vpux)), Some(KORE_FORMAT_VPUX_ELF));

        // "not Intel GT" is not evidence of a VPUX model: an ELF of any other
        // machine type is recognised as an ELF and claims no kore format.
        let x86 = elf(62, 32);
        assert_eq!(classify(&x86), BlobKind::OtherElf { machine: 62 });
        assert_eq!(kore_format(classify(&x86)), None);

        let zebin = elf(EM_INTELGT, 16);
        assert_eq!(classify(&zebin), BlobKind::IntelGtZebin);

        let mut container = vec![0xaa; 37];
        container.extend_from_slice(&zebin);
        container.extend_from_slice(&[0x55; 9]);
        container.extend_from_slice(&elf(EM_INTELGT, 8));
        assert_eq!(classify(&container), BlobKind::GpuExport { kernels: 2 });
        let found = embedded_elfs(&container);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].offset, 37);
        assert_eq!(found[0].bytes.len(), zebin.len());
        assert_eq!(found[1].bytes.len(), ELF64_HEADER_LEN + 64 + 8);
        assert_eq!(classify(&[0u8; 100]), BlobKind::Unknown);

        // A section declared as NOBITS occupies no file bytes.
        let mut nobits = elf(EM_NONE, 4096);
        nobits[64 + 4..64 + 8].copy_from_slice(&8_u32.to_le_bytes());
        assert_eq!(embedded_elfs(&nobits)[0].bytes.len(), 128);
    }

    #[test]
    fn truncated_images_are_clipped_not_overread() {
        let mut image = elf(EM_INTELGT, 1000);
        image.truncate(200);
        let found = embedded_elfs(&image);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].bytes.len(), 200);
        assert!(embedded_elfs(&image[..63]).is_empty());
        // A header whose tables point past the blob is clipped, not trusted.
        let mut wild = elf(EM_NONE, 16);
        wild[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(embedded_elfs(&wild)[0].bytes.len(), wild.len());
    }

    #[test]
    fn records_bind_inputs_and_render_manifest() {
        let blob = elf(0, 4);
        let record = BlobRecord::new(
            "lighting", "openvino", "NPU", "2026.4", b"graph", b"tosa", &blob,
        );
        assert!(record.verify(&blob));
        assert!(!record.verify(&blob[..blob.len() - 1]));
        assert_eq!(record.kind, BlobKind::VpuxElf);
        assert_eq!(record.kore_format, Some(1));
        assert_eq!(record.kernels.len(), 1);
        let json = manifest_json(&[record.clone(), record]);
        assert!(json.contains("\"kind\": \"vpux-elf\""));
        assert!(json.contains("\"kore_format\": 1"));
        assert!(json.contains(&Digest::of(b"tosa").hex()));
        assert_eq!(json.matches("\"name\": \"lighting\"").count(), 2);
    }
}
