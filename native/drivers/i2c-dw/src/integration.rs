//! **Per-integration data** — what differs between one DesignWare core and
//! another, expressed as a value rather than as branches in the core.
//!
//! The same Synopsys IP is wired into machines that need different things
//! done to it before it will answer:
//!
//! - **Intel LPSS** wraps the core in a private register block at `+0x200`
//!   whose reset must be released *first*, before any DW register reads
//!   back meaningfully.
//! - **AMD's `AMDI0010`** is the bare core with no such block. Performing
//!   Intel's reset sequence on it would write into whatever occupies
//!   `+0x204` on that part.
//!
//! Making this data rather than a `match` is the point: the transfer state
//! machine is identical on both, and a core that carried an `if intel`
//! would grow a second one for every integration after it.

/// The Intel LPSS private register block, relative to the window base.
pub const LPSS_PRIVATE: usize = 0x200;
/// LPSS resets register, within the private block.
pub const LPSS_RESETS: usize = LPSS_PRIVATE + 0x04;
/// LPSS resets: release the function.
pub const LPSS_RESETS_FUNC: u32 = 1 << 2;
/// LPSS resets: release the integrated DMA block.
///
/// Released even though this driver is PIO-only: the two share one reset
/// register, and holding iDMA in reset while the function runs is not a
/// configuration Intel documents or tests.
pub const LPSS_RESETS_IDMA: u32 = 0b11;

/// How one DesignWare core is wired into its machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Integration {
    /// Name, for the witness line.
    pub name: &'static str,
    /// Whether an Intel LPSS private block sits at
    /// [`LPSS_PRIVATE`] and must be taken out of reset first.
    pub lpss_reset: bool,
    /// The core's input clock in Hz — used only when firmware declared no
    /// timing counts.
    pub input_clock_hz: u32,
    /// Whether `IC_INTR_MASK` reads back what was written.
    ///
    /// On AMD's integration it does not, so a read-modify-write of the
    /// mask silently loses bits. The core therefore always writes the
    /// mask whole rather than merging — a discipline that is correct
    /// everywhere and only *necessary* here, which is why it is recorded
    /// as a property rather than applied conditionally.
    pub intr_mask_reads_back: bool,
}

impl Integration {
    /// Intel LPSS, PCI-mode or ACPI-mode — the Lunar Lake shape.
    ///
    /// The input clock is the derivation fallback only: Intel firmware
    /// declares `SSCN`/`FMCN`, the scout confirms their presence per
    /// controller, and those counts take precedence whenever present.
    pub const INTEL_LPSS: Self = Self {
        name: "intel-lpss",
        lpss_reset: true,
        input_clock_hz: 100_000_000,
        intr_mask_reads_back: true,
    };

    /// A bare core with no vendor wrapper — AMD's `AMDI0010`.
    ///
    /// Unverified on hardware: no AMD firmware dump is in-tree, so both
    /// the clock and the mask behaviour here are from the published
    /// integration rather than from this fleet's machine. The topology
    /// scout settles it in one boot, and this constant should be
    /// re-confirmed against that boot before the AMD row is founded.
    pub const AMD_PLAIN: Self = Self {
        name: "amd-plain",
        lpss_reset: false,
        input_clock_hz: 133_000_000,
        intr_mask_reads_back: false,
    };
}
