//! Read-only I2C admission evidence, paged through the existing device broker.
//! These are descriptions, not controller, BAR, GPIO or interrupt authority.

use zerocopy::byteorder::{LittleEndian, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const MAX_I2C_ADMISSION_CANDIDATES: usize = 8;
pub const I2C_ADMISSION_PAGES_PER_CANDIDATE: u8 = 3;
/// Global build evidence is appended, never substituted for candidate eight.
pub const I2C_ADMISSION_BUILD_ROW: u8 =
    1 + MAX_I2C_ADMISSION_CANDIDATES as u8 * I2C_ADMISSION_PAGES_PER_CANDIDATE;
pub const I2C_ADMISSION_ROWS: u8 = I2C_ADMISSION_BUILD_ROW + 1;
pub const I2C_ADMISSION_NO_GSI: u32 = u32::MAX;
/// The broker adds its own claim-ledger fact; all lower bits are DeviceRecord flags.
pub const I2C_ADMISSION_CLAIMED: u8 = 1 << 7;

impl super::DeviceRecord {
    /// The native driver profile is Intel LNL / PTL-P LPSS DesignWare. This
    /// identifies candidates only; correlation, BAR and line checks still
    /// belong to Claim. Inspection must remain possible when those fail.
    pub fn is_native_i2c_hid_candidate(&self) -> bool {
        self.species == super::DeviceSpecies::Pci
            && (self.class, self.subclass, self.progif) == (0x0c, 0x80, 0)
            && self.vendor == 0x8086
            // Linux intel-lpss-pci.c maps these LNL-M and PTL-P I2C
            // functions to ehl_i2c_info (100 MHz). Adjacent UART/SPI
            // functions must not receive DesignWare/LPSS I2C resets.
            // https://github.com/torvalds/linux/blob/master/drivers/mfd/intel-lpss-pci.c
            && matches!(
                self.device,
                0xa850 | 0xa851 | 0xa878..=0xa87b | 0xe450 | 0xe451 | 0xe478..=0xe47b
            )
    }
}

macro_rules! diagnostic_codes {
    ($name:ident { $($variant:ident = $value:literal => $label:literal),+ $(,)? }) => {
        #[repr(u8)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name { $($variant = $value),+ }
        impl $name {
            pub const fn from_bits(value: u8) -> Option<Self> {
                match value { $($value => Some(Self::$variant),)+ _ => None }
            }
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $label,)+ }
            }
        }
    };
}

diagnostic_codes!(I2cInventoryState {
    NotPublished = 0 => "initialized inventory not published",
    Initialized = 1 => "initialized inventory published",
    TablesUnavailable = 2 => "AML tables/interpreter unavailable",
    NamespaceRefused = 3 => "namespace initialization refused",
    NamespaceBudget = 4 => "namespace initialization budget exhausted",
    EcConnectionBudget = 5 => "EC connection budget exhausted",
    EcUnavailable = 6 => "EC activation unavailable",
    LidDiscoveryBudget = 7 => "lid discovery budget exhausted",
    DiscoveryBudget = 8 => "initialized I2C discovery budget exhausted",
    PublicationConflict = 9 => "interpreter publication refused",
    BuildRefused = 10 => "AML construction refused",
});

diagnostic_codes!(AmlBuildStage {
    None = 0 => "unreported",
    FadtLookup = 1 => "FADT lookup",
    FixedRegisters = 2 => "fixed registers",
    DsdtMetadata = 3 => "DSDT metadata",
    DsdtLoad = 4 => "DSDT load",
    SsdtLoad = 5 => "SSDT load",
});

// Stable wire categories, not Rust error-enum discriminants. Opaque error
// strings, namespace allocations and physical addresses never leave firmware.
diagnostic_codes!(AmlBuildErrorClass {
    None = 0 => "unreported",
    TableMissing = 1 => "no usable table",
    InvalidAddress = 2 => "invalid address",
    InvalidLength = 3 => "invalid length",
    RegisterShape = 4 => "register shape",
    Stream = 5 => "truncated stream",
    Opcode = 6 => "illegal opcode",
    NamespaceMissing = 7 => "namespace missing",
    ObjectMissing = 8 => "object missing",
    NameCollision = 9 => "name collision",
    InvalidName = 10 => "invalid name",
    Type = 11 => "type mismatch",
    Unsupported = 12 => "unsupported",
    RegionHandler = 13 => "region handler absent",
    Mutex = 14 => "mutex timeout",
    Budget = 15 => "evaluation budget",
    Firmware = 16 => "firmware error",
    Host = 17 => "host error",
    Interpreter = 18 => "interpreter error",
    Reference = 19 => "invalid reference",
    Fatal = 20 => "firmware fatal",
});

/// One failed persistent-interpreter construction. All zero means that no
/// additional detail was supplied, NOT that construction succeeded. Row zero's
/// `BuildRefused` state advertises the appended page. Legacy `TablesUnavailable`
/// does not: an old decoder can silently drop an unknown row, so it must never
/// be probed. The ordinal is one-based in the firmware SSDT iterator,
/// not a filesystem name; non-SSDT stages must use zero.
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct AmlBuildDiagnostic {
    stage: u8,
    error_class: u8,
    detail: U16<LittleEndian>,
    ssdt_ordinal: U32<LittleEndian>,
}

impl AmlBuildDiagnostic {
    pub fn failure(
        stage: AmlBuildStage,
        error_class: AmlBuildErrorClass,
        ssdt_ordinal: u32,
        detail: u16,
    ) -> Option<Self> {
        let value = Self {
            stage: stage as u8,
            error_class: error_class as u8,
            detail: detail.into(),
            ssdt_ordinal: ssdt_ordinal.into(),
        };
        (value.is_present() && value.valid()).then_some(value)
    }

    pub fn valid(self) -> bool {
        let (Some(stage), Some(error)) = (
            AmlBuildStage::from_bits(self.stage),
            AmlBuildErrorClass::from_bits(self.error_class),
        ) else {
            return false;
        };
        if stage == AmlBuildStage::None || error == AmlBuildErrorClass::None {
            return self == Self::default();
        }
        (stage == AmlBuildStage::SsdtLoad) == (self.ssdt_ordinal.get() != 0)
    }

    pub fn is_present(self) -> bool {
        self.stage != 0
    }

    pub fn stage(self) -> Option<AmlBuildStage> {
        AmlBuildStage::from_bits(self.stage)
    }

    pub fn error_class(self) -> Option<AmlBuildErrorClass> {
        AmlBuildErrorClass::from_bits(self.error_class)
    }

    /// Error-specific bounded integer, for example the rejected opcode.
    pub fn detail(self) -> u16 {
        self.detail.get()
    }

    pub fn ssdt_ordinal(self) -> u32 {
        self.ssdt_ordinal.get()
    }

    pub fn words(self) -> [u32; 2] {
        [
            u32::from(self.stage)
                | u32::from(self.error_class) << 8
                | u32::from(self.detail.get()) << 16,
            self.ssdt_ordinal.get(),
        ]
    }

    pub fn from_words(words: [u32; 2]) -> Option<Self> {
        let value = Self {
            stage: words[0] as u8,
            error_class: (words[0] >> 8) as u8,
            detail: ((words[0] >> 16) as u16).into(),
            ssdt_ordinal: words[1].into(),
        };
        value.valid().then_some(value)
    }
}

const _: () = assert!(core::mem::size_of::<AmlBuildDiagnostic>() == 8);

diagnostic_codes!(I2cBindingState {
    NoInitializedChildren = 0 => "no initialized HID children",
    Bound = 1 => "child bindings available",
    ControllerUnresolved = 2 => "controller unresolved",
    ControllerNotAdmitted = 3 => "initialized controller not admitted",
    DsmAbsent = 4 => "descriptor _DSM absent",
    DsmNotOffered = 5 => "descriptor _DSM function not offered",
    DsmUnreadable = 6 => "descriptor _DSM unreadable",
    ClockAbsent = 7 => "input clock absent",
    ClockUnreadable = 8 => "input clock unreadable",
    SlaveAddress = 9 => "slave address out of range",
    TenBitAddress = 10 => "10-bit slave addressing unsupported",
    DescriptorWidth = 11 => "descriptor register wider than 16 bits",
    BusSpeed = 12 => "bus speed unsupported",
    ControllerWindow = 13 => "controller window invalid",
    ZeroTimingCount = 14 => "zero SCL timing count",
    NotPresent = 15 => "child _STA says absent",
    NotEnabled = 16 => "child _STA says disabled",
    StatusUnreadable = 17 => "child _STA unreadable",
    DeviceInitiated = 18 => "device-initiated connection unsupported",
    WrongHidId = 19 => "I2C child does not identify as HID",
});

diagnostic_codes!(I2cLineState {
    NotResolved = 0 => "completion route not evaluated",
    Routed = 1 => "controller completion route available",
    InterruptModelRefused = 2 => "_PIC interrupt-model notification refused",
    NotEnumerated = 3 => "controller not enumerated",
    NoInterruptPin = 4 => "PCI interrupt pin absent",
    InvalidInterruptPin = 5 => "PCI interrupt pin invalid",
    TruncatedPath = 6 => "controller namespace path truncated",
    InvalidPath = 7 => "controller namespace path invalid",
    ControllerNotReady = 8 => "controller initialization/status not ready",
    ControllerAddressMismatch = 9 => "controller namespace/PCI address mismatch",
    NoParentBus = 10 => "controller has no parent PCI namespace",
    PrtAbsent = 11 => "parent _PRT absent",
    PrtRefused = 12 => "parent _PRT evaluation/decoding refused",
    NoMatchingRoute = 13 => "_PRT has no matching address/pin",
    AmbiguousRoute = 14 => "_PRT matching source is ambiguous or invalid",
    LinkNotReady = 15 => "interrupt link initialization/status not ready",
    LinkResourcesRefused = 16 => "interrupt link resources refused",
    NoSingleGsi = 17 => "route does not identify one GSI",
    ConflictingRoutes = 18 => "HID children disagree on controller route",
});

/// One fixed-size broker response. Row zero is the authoritative summary;
/// candidate rows are identity/admission, route/counts, then inventory/PCI IDs.
/// The final global build row has no candidate identity or claim state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I2cAdmissionPage {
    pub row: u8,
    pub words: [u32; 2],
}

impl I2cAdmissionPage {
    pub fn valid(self) -> bool {
        if self.row >= I2C_ADMISSION_ROWS {
            return false;
        }
        if self.row == I2C_ADMISSION_BUILD_ROW {
            return AmlBuildDiagnostic::from_words(self.words).is_some();
        }
        if self.row == 0 {
            let retained = (self.words[0] >> 8) as u8;
            return I2cInventoryState::from_bits(self.words[0] as u8).is_some()
                && usize::from(retained) <= MAX_I2C_ADMISSION_CANDIDATES
                && u16::from(retained) <= (self.words[0] >> 16) as u16;
        }
        if (self.row - 1) % I2C_ADMISSION_PAGES_PER_CANDIDATE == 0 {
            return I2cBindingState::from_bits((self.words[1] >> 16) as u8).is_some()
                && I2cLineState::from_bits((self.words[1] >> 24) as u8).is_some();
        }
        true
    }
}

#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct I2cAdmissionCandidate {
    pub segment: U16<LittleEndian>,
    /// bus << 8 | device << 3 | function; never an address to dereference.
    pub locator: U16<LittleEndian>,
    pub flags: u8,
    pub bound_children: u8,
    pub binding_state: u8,
    pub line_state: u8,
    /// Controller completion GSI, NOT a HID child GPIO. NO_GSI means absent.
    pub gsi: U32<LittleEndian>,
    pub seen_children: U16<LittleEndian>,
    pub refused_children: U16<LittleEndian>,
    pub inventory_index: U32<LittleEndian>,
    pub vendor: U16<LittleEndian>,
    pub device: U16<LittleEndian>,
}

/// One broker/root's immutable startup observation. A caller fetching every
/// advertised page decides availability; inserting a page never claims success.
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct I2cAdmissionSnapshot {
    pub inventory_state: u8,
    pub candidates: u8,
    /// May exceed candidates: truncation is explicit, never silent absence.
    pub total_candidates: U16<LittleEndian>,
    pub not_initialized: U16<LittleEndian>,
    pub crs_refused: U16<LittleEndian>,
    pub candidate: [I2cAdmissionCandidate; MAX_I2C_ADMISSION_CANDIDATES],
    pub build: AmlBuildDiagnostic,
}

impl I2cAdmissionSnapshot {
    pub fn page(&self, row: u8) -> Option<I2cAdmissionPage> {
        if !self.validate() || row >= I2C_ADMISSION_ROWS {
            return None;
        }
        let words = if row == I2C_ADMISSION_BUILD_ROW {
            self.build.words()
        } else if row == 0 {
            [
                u32::from(self.inventory_state)
                    | u32::from(self.candidates) << 8
                    | u32::from(self.total_candidates.get()) << 16,
                u32::from(self.not_initialized.get()) | u32::from(self.crs_refused.get()) << 16,
            ]
        } else {
            let index = usize::from((row - 1) / I2C_ADMISSION_PAGES_PER_CANDIDATE);
            if index >= usize::from(self.candidates) {
                return None;
            }
            let candidate = &self.candidate[index];
            match (row - 1) % I2C_ADMISSION_PAGES_PER_CANDIDATE {
                0 => [
                    u32::from(candidate.segment.get()) | u32::from(candidate.locator.get()) << 16,
                    u32::from(candidate.flags)
                        | u32::from(candidate.bound_children) << 8
                        | u32::from(candidate.binding_state) << 16
                        | u32::from(candidate.line_state) << 24,
                ],
                1 => [
                    candidate.gsi.get(),
                    u32::from(candidate.seen_children.get())
                        | u32::from(candidate.refused_children.get()) << 16,
                ],
                _ => [
                    candidate.inventory_index.get(),
                    u32::from(candidate.vendor.get()) | u32::from(candidate.device.get()) << 16,
                ],
            }
        };
        Some(I2cAdmissionPage { row, words })
    }

    pub fn insert(&mut self, page: I2cAdmissionPage) -> bool {
        if !page.valid() {
            return false;
        }
        if page.row == I2C_ADMISSION_BUILD_ROW {
            let Some(build) = AmlBuildDiagnostic::from_words(page.words) else {
                return false;
            };
            if build.is_present() && self.inventory_state != I2cInventoryState::BuildRefused as u8 {
                return false;
            }
            self.build = build;
            return true;
        }
        let [a, b] = page.words;
        if page.row == 0 {
            if self.build.is_present() && a as u8 != I2cInventoryState::BuildRefused as u8 {
                return false;
            }
            self.inventory_state = a as u8;
            self.candidates = (a >> 8) as u8;
            self.total_candidates.set((a >> 16) as u16);
            self.not_initialized.set(b as u16);
            self.crs_refused.set((b >> 16) as u16);
            return true;
        }
        let index = usize::from((page.row - 1) / I2C_ADMISSION_PAGES_PER_CANDIDATE);
        if index >= usize::from(self.candidates) {
            return false;
        }
        let candidate = &mut self.candidate[index];
        match (page.row - 1) % I2C_ADMISSION_PAGES_PER_CANDIDATE {
            0 => {
                candidate.segment.set(a as u16);
                candidate.locator.set((a >> 16) as u16);
                candidate.flags = b as u8;
                candidate.bound_children = (b >> 8) as u8;
                candidate.binding_state = (b >> 16) as u8;
                candidate.line_state = (b >> 24) as u8;
            }
            1 => {
                candidate.gsi.set(a);
                candidate.seen_children.set(b as u16);
                candidate.refused_children.set((b >> 16) as u16);
            }
            _ => {
                candidate.inventory_index.set(a);
                candidate.vendor.set(b as u16);
                candidate.device.set((b >> 16) as u16);
            }
        }
        true
    }

    pub fn validate(&self) -> bool {
        I2cInventoryState::from_bits(self.inventory_state).is_some()
            && self.build.valid()
            && (!self.build.is_present()
                || self.inventory_state == I2cInventoryState::BuildRefused as u8)
            && usize::from(self.candidates) <= MAX_I2C_ADMISSION_CANDIDATES
            && u16::from(self.candidates) <= self.total_candidates.get()
            && self.candidate[..usize::from(self.candidates)]
                .iter()
                .all(|candidate| {
                    I2cBindingState::from_bits(candidate.binding_state).is_some()
                        && I2cLineState::from_bits(candidate.line_state).is_some()
                })
    }
}

const _: () = assert!(core::mem::size_of::<I2cAdmissionSnapshot>() == 208);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_words_preserve_stage_category_opcode_and_full_width_ssdt_ordinal() {
        assert_eq!(
            AmlBuildDiagnostic::from_words([0; 2]),
            Some(AmlBuildDiagnostic::default())
        );
        for stage in (1..=u8::MAX).filter_map(AmlBuildStage::from_bits) {
            for class in (1..=u8::MAX).filter_map(AmlBuildErrorClass::from_bits) {
                let ordinal = if stage == AmlBuildStage::SsdtLoad {
                    u32::MAX
                } else {
                    0
                };
                let detail = AmlBuildDiagnostic::failure(stage, class, ordinal, u16::MAX).unwrap();
                assert!(detail.is_present());
                assert_eq!(detail.stage(), Some(stage));
                assert_eq!(detail.error_class(), Some(class));
                assert_eq!(detail.ssdt_ordinal(), ordinal);
                assert_eq!(detail.detail(), u16::MAX);
                assert_eq!(AmlBuildDiagnostic::from_words(detail.words()), Some(detail));
            }
        }
        assert!(
            AmlBuildDiagnostic::failure(AmlBuildStage::None, AmlBuildErrorClass::Stream, 0, 0)
                .is_none()
        );
        assert!(
            AmlBuildDiagnostic::failure(AmlBuildStage::DsdtLoad, AmlBuildErrorClass::None, 0, 0)
                .is_none()
        );
    }

    #[test]
    fn build_page_is_global_even_with_no_candidates_and_does_not_reinterpret_old_rows() {
        let mut snapshot = I2cAdmissionSnapshot::default();
        snapshot.inventory_state = I2cInventoryState::BuildRefused as u8;
        let summary = snapshot.page(0).unwrap();
        let detail = AmlBuildDiagnostic::failure(
            AmlBuildStage::SsdtLoad,
            AmlBuildErrorClass::Opcode,
            7,
            0x5bff,
        )
        .unwrap();
        let page = I2cAdmissionPage {
            row: I2C_ADMISSION_BUILD_ROW,
            words: detail.words(),
        };
        assert!(snapshot.insert(page));
        assert_eq!(snapshot.page(I2C_ADMISSION_BUILD_ROW), Some(page));
        assert_eq!(snapshot.page(0), Some(summary));
        assert_eq!(snapshot.candidates, 0);
        assert_eq!(snapshot.build, detail);
        snapshot.candidates = MAX_I2C_ADMISSION_CANDIDATES as u8;
        snapshot
            .total_candidates
            .set(MAX_I2C_ADMISSION_CANDIDATES as u16);
        let last_candidate = I2cAdmissionPage {
            row: I2C_ADMISSION_BUILD_ROW - 1,
            words: [u32::MAX; 2],
        };
        assert!(snapshot.insert(last_candidate));
        assert_eq!(snapshot.page(last_candidate.row), Some(last_candidate));
        assert_eq!(snapshot.build, detail);
    }

    #[test]
    fn malformed_build_evidence_is_rejected_without_mutating_valid_inventory() {
        let mut snapshot = I2cAdmissionSnapshot::default();
        snapshot.inventory_state = I2cInventoryState::BuildRefused as u8;
        for words in [
            [0xff, 0],
            [0xff00, 0],
            [0x100, 0],
            [1, 0],
            [0x1_0000, 0],
            [0, 1],
            [
                AmlBuildStage::SsdtLoad as u32 | (AmlBuildErrorClass::Stream as u32) << 8,
                0,
            ],
            [
                AmlBuildStage::DsdtLoad as u32 | (AmlBuildErrorClass::Stream as u32) << 8,
                1,
            ],
        ] {
            let before = snapshot;
            let page = I2cAdmissionPage {
                row: I2C_ADMISSION_BUILD_ROW,
                words,
            };
            assert!(!page.valid());
            assert!(!snapshot.insert(page));
            assert_eq!(snapshot, before);
        }
        let detail =
            AmlBuildDiagnostic::failure(AmlBuildStage::DsdtLoad, AmlBuildErrorClass::Budget, 0, 0)
                .unwrap();
        let page = I2cAdmissionPage {
            row: I2C_ADMISSION_BUILD_ROW,
            words: detail.words(),
        };
        assert!(snapshot.insert(page));
        assert!(snapshot.validate());
        let before = snapshot;
        assert!(!snapshot.insert(I2cAdmissionPage {
            row: 0,
            words: [I2cInventoryState::Initialized as u32, 0],
        }));
        assert_eq!(snapshot, before);
        snapshot.inventory_state = I2cInventoryState::Initialized as u8;
        assert!(!snapshot.validate());
        snapshot.build = AmlBuildDiagnostic::default();
        assert!(!snapshot.insert(page));
        assert!(snapshot.validate());
    }

    #[test]
    fn pages_keep_second_controller_identity_and_explicit_truncation() {
        let mut snapshot = I2cAdmissionSnapshot::default();
        assert!(snapshot.insert(I2cAdmissionPage {
            row: 0,
            words: [1 | 2 << 8 | 10 << 16, 7 | 3 << 16]
        }));
        for page in [
            I2cAdmissionPage {
                row: 4,
                words: [0x00c9 << 16, 1 << 8 | 1 << 16 | 4 << 24],
            },
            I2cAdmissionPage {
                row: 5,
                words: [I2C_ADMISSION_NO_GSI, 1],
            },
            I2cAdmissionPage {
                row: 6,
                words: [19, 0xa851_8086],
            },
        ] {
            assert!(snapshot.insert(page));
        }
        assert_eq!(snapshot.total_candidates.get(), 10);
        assert_eq!(snapshot.not_initialized.get(), 7);
        assert_eq!(snapshot.candidate[1].locator.get(), 0x00c9);
        assert_eq!(snapshot.candidate[1].device.get(), 0xa851);
        assert_eq!(
            snapshot.candidate[1].line_state,
            I2cLineState::NoInterruptPin as u8
        );
        assert!(snapshot.validate());
    }

    #[test]
    fn invalid_pages_do_not_mutate_the_snapshot() {
        let mut snapshot = I2cAdmissionSnapshot::default();
        for page in [
            I2cAdmissionPage {
                row: I2C_ADMISSION_ROWS,
                words: [0, 0],
            },
            I2cAdmissionPage {
                row: 0,
                words: [0xff, 0],
            },
            I2cAdmissionPage {
                row: 0,
                words: [9 << 8 | 9 << 16, 0],
            },
            I2cAdmissionPage {
                row: 1,
                words: [0, 0],
            },
            I2cAdmissionPage {
                row: 1,
                words: [0, 0xff << 24],
            },
        ] {
            assert!(!snapshot.insert(page));
            assert_eq!(snapshot, I2cAdmissionSnapshot::default());
        }
    }
}
