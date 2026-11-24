//! `.ax` dtype specializations for semantic geometry precision classes.

use ax::DType;

use crate::precision::{
    PrecisionBatchLayout, PrecisionClass, PrecisionEstimate, ProjectionPrecisionPolicy,
};

/// Storage, compute, and reduction types for one graph specialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrecisionTier {
    pub storage: DType,
    pub compute: DType,
    pub accumulate: DType,
}

impl PrecisionTier {
    pub const FP32: Self = Self {
        storage: DType::F32,
        compute: DType::F32,
        accumulate: DType::F32,
    };

    pub const FP16: Self = Self {
        storage: DType::F16,
        compute: DType::F16,
        accumulate: DType::F32,
    };

    /// B8 coefficients expanded once at partition ingress for F16 compute.
    pub const B8_TO_F16: Self = Self {
        storage: DType::B8,
        compute: DType::F16,
        accumulate: DType::F32,
    };

    /// Range-oriented F8 coefficients expanded for B16 compute.
    pub const F8_TO_B16: Self = Self {
        storage: DType::F8,
        compute: DType::B16,
        accumulate: DType::F32,
    };

    pub const fn is_valid(self) -> bool {
        self.storage.is_float()
            && matches!(
                self.compute,
                DType::F64 | DType::F32 | DType::F16 | DType::B16
            )
            && matches!(self.accumulate, DType::F64 | DType::F32)
    }

    #[inline]
    pub const fn expands_storage(self) -> bool {
        self.storage as u8 != self.compute as u8
    }
}

/// Maps backend-neutral projection classes to cached `.ax` dtypes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxProjectionPrecisionPolicy {
    pub classifier: ProjectionPrecisionPolicy,
    pub sensitive: PrecisionTier,
    pub near: PrecisionTier,
    pub far: PrecisionTier,
}

/// One caller-owned work list paired with its cached `.ax` specialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxPrecisionBatch<'a> {
    pub indices: &'a [u32],
    pub tier: PrecisionTier,
}

impl AxProjectionPrecisionPolicy {
    pub const DEFAULT: Self = Self {
        classifier: ProjectionPrecisionPolicy::DEFAULT,
        sensitive: PrecisionTier::FP32,
        near: PrecisionTier::FP16,
        far: PrecisionTier::B8_TO_F16,
    };

    pub const fn is_valid(self) -> bool {
        self.classifier.is_valid()
            && self.sensitive.is_valid()
            && self.near.is_valid()
            && self.far.is_valid()
    }

    pub const fn tier(self, class: PrecisionClass) -> PrecisionTier {
        match class {
            PrecisionClass::Sensitive => self.sensitive,
            PrecisionClass::Near => self.near,
            PrecisionClass::Far => self.far,
        }
    }

    pub fn choose(self, estimate: PrecisionEstimate) -> Option<PrecisionTier> {
        if !self.is_valid() {
            return None;
        }
        self.classifier
            .classify(estimate)
            .map(|class| self.tier(class))
    }

    /// Pair compacted class lists with the storage/compute tier that a runtime
    /// uses to select existing mixed-precision graphs or CPU kernels. Returns
    /// `None` if `compacted` is too short for the supplied layout.
    pub fn batches<'a>(
        self,
        compacted: &'a [u32],
        layout: &PrecisionBatchLayout,
    ) -> Option<[AxPrecisionBatch<'a>; 3]> {
        Some([
            AxPrecisionBatch {
                indices: layout.indices(compacted, PrecisionClass::Sensitive)?,
                tier: self.sensitive,
            },
            AxPrecisionBatch {
                indices: layout.indices(compacted, PrecisionClass::Near)?,
                tier: self.near,
            },
            AxPrecisionBatch {
                indices: layout.indices(compacted, PrecisionClass::Far)?,
                tier: self.far,
            },
        ])
    }
}

impl Default for AxProjectionPrecisionPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::compact_precision_indices;

    #[test]
    fn default_maps_far_work_to_b8_storage_and_f32_accumulation() {
        let policy = AxProjectionPrecisionPolicy::DEFAULT;
        let estimate = PrecisionEstimate {
            screen_error_px: 0.25,
            condition: 1.0,
            boundary_margin: 1.0,
        };
        assert_eq!(policy.choose(estimate), Some(PrecisionTier::B8_TO_F16));
        assert!(PrecisionTier::B8_TO_F16.expands_storage());
        assert_eq!(PrecisionTier::B8_TO_F16.accumulate, DType::F32);
    }

    #[test]
    fn f8_b16_tier_is_explicit_but_keeps_f32_reductions() {
        let tier = PrecisionTier::F8_TO_B16;
        assert!(tier.is_valid());
        assert_eq!(tier.storage, DType::F8);
        assert_eq!(tier.compute, DType::B16);
        assert_eq!(tier.accumulate, DType::F32);
    }

    #[test]
    fn compacted_work_lists_map_directly_to_cached_dtype_tiers() {
        let estimates = [
            PrecisionEstimate {
                screen_error_px: 9.0,
                condition: 1.0,
                boundary_margin: 1.0,
            },
            PrecisionEstimate {
                screen_error_px: 2.0,
                condition: 1.0,
                boundary_margin: 1.0,
            },
            PrecisionEstimate {
                screen_error_px: 0.25,
                condition: 1.0,
                boundary_margin: 1.0,
            },
        ];
        let policy = AxProjectionPrecisionPolicy::DEFAULT;
        let mut compacted = [0; 3];
        let layout =
            compact_precision_indices(policy.classifier, &estimates, &mut compacted).unwrap();
        let batches = policy.batches(&compacted, &layout).unwrap();
        assert_eq!(batches[0].tier, PrecisionTier::FP32);
        assert_eq!(batches[1].tier, PrecisionTier::FP16);
        assert_eq!(batches[2].tier, PrecisionTier::B8_TO_F16);
        assert_eq!(
            batches.map(|batch| batch.indices),
            [&[0][..], &[1][..], &[2][..]]
        );
        assert_eq!(policy.batches(&compacted[..2], &layout), None);
    }
}
