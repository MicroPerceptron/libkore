//! Backend-neutral precision classification for projected geometry.
//!
//! Tensor dtype is fixed per graph specialization, so dynamic precision is a
//! tile/batch decision: estimate projected error and conditioning, classify
//! work, compact stable per-class index lists, then dispatch a cached
//! specialization for that class.

use core::ops::Range;

/// Semantic precision class used for compaction and graph specialization.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrecisionClass {
    Sensitive,
    Near,
    Far,
}

/// Cheap projected-geometry error estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecisionEstimate {
    /// Conservative screen-space error or footprint in pixels.
    pub screen_error_px: f32,
    /// Sensitivity multiplier for inverses, steep gradients, or poor
    /// covariance conditioning. Values below one are treated as one.
    pub condition: f32,
    /// Distance from a classification/event boundary in the estimator's
    /// native units. Nonpositive margins are always sensitive.
    pub boundary_margin: f32,
}

impl PrecisionEstimate {
    pub fn is_valid(self) -> bool {
        self.screen_error_px.is_finite()
            && self.screen_error_px >= 0.0
            && self.condition.is_finite()
            && self.condition >= 0.0
            && self.boundary_margin.is_finite()
    }

    #[inline]
    pub fn weighted_error(self) -> f32 {
        self.screen_error_px * self.condition.max(1.0)
    }
}

/// Semantic thresholds for near/far projection batches.
///
/// This core type deliberately does not name tensor dtypes. A graph/runtime
/// adapter maps the returned class to storage, compute, and accumulation
/// specializations supported by its backend set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionPrecisionPolicy {
    /// Weighted error at or above which full sensitive precision is used.
    pub sensitive_threshold_px: f32,
    /// Weighted error at or above which the near tier is used.
    pub near_threshold_px: f32,
    /// Margin at or below which event/classification decisions stay sensitive.
    pub boundary_margin: f32,
}

impl ProjectionPrecisionPolicy {
    /// Conservative default for projection and field evaluation: F32 near
    /// sensitive boundaries, FP16 for visible near-field work, and FP8 storage
    /// widened once for subpixel far-field work.
    pub const DEFAULT: Self = Self {
        sensitive_threshold_px: 8.0,
        near_threshold_px: 1.0,
        boundary_margin: 0.01,
    };

    pub const fn is_valid(self) -> bool {
        self.sensitive_threshold_px.is_finite()
            && self.near_threshold_px.is_finite()
            && self.boundary_margin.is_finite()
            && self.sensitive_threshold_px >= self.near_threshold_px
            && self.near_threshold_px >= 0.0
            && self.boundary_margin >= 0.0
    }

    pub fn classify(self, estimate: PrecisionEstimate) -> Option<PrecisionClass> {
        if !self.is_valid() || !estimate.is_valid() {
            return None;
        }
        let error = estimate.weighted_error();
        Some(
            if estimate.boundary_margin <= self.boundary_margin
                || error >= self.sensitive_threshold_px
            {
                PrecisionClass::Sensitive
            } else if error >= self.near_threshold_px {
                PrecisionClass::Near
            } else {
                PrecisionClass::Far
            },
        )
    }
}

impl Default for ProjectionPrecisionPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Contiguous class ranges produced by [`compact_precision_indices`].
///
/// The fields are private so every instance is guaranteed to describe three
/// adjacent, non-overlapping ranges within the compacted output slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrecisionBatchLayout {
    sensitive_end: usize,
    near_end: usize,
    len: usize,
}

impl PrecisionBatchLayout {
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn range(&self, class: PrecisionClass) -> Range<usize> {
        match class {
            PrecisionClass::Sensitive => 0..self.sensitive_end,
            PrecisionClass::Near => self.sensitive_end..self.near_end,
            PrecisionClass::Far => self.near_end..self.len,
        }
    }

    /// Borrow one class from the compacted output used to create this layout.
    /// Returns `None` when the supplied slice is shorter than that class range.
    #[inline]
    pub fn indices<'a>(&self, compacted: &'a [u32], class: PrecisionClass) -> Option<&'a [u32]> {
        compacted.get(self.range(class))
    }
}

/// Failure to form dtype-specialized geometry batches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrecisionCompactionError {
    InvalidPolicy,
    InvalidEstimate { index: usize },
    TooManyItems(usize),
    OutputTooSmall { required: usize, actual: usize },
}

/// Stably compact geometry indices into sensitive, near, and far batches.
///
/// This is the allocation-free bridge from scalar error classification to
/// mixed-precision graph/kernel dispatch. `compacted` must have at least one
/// element per estimate. The function makes two linear passes, writes each
/// source index exactly once, and preserves source order within every class.
/// Consumers can therefore dispatch three cached dtype specializations without
/// moving the full geometry records or allocating per-class vectors.
pub fn compact_precision_indices(
    policy: ProjectionPrecisionPolicy,
    estimates: &[PrecisionEstimate],
    compacted: &mut [u32],
) -> Result<PrecisionBatchLayout, PrecisionCompactionError> {
    if !policy.is_valid() {
        return Err(PrecisionCompactionError::InvalidPolicy);
    }
    if estimates.len() > u32::MAX as usize {
        return Err(PrecisionCompactionError::TooManyItems(estimates.len()));
    }
    if compacted.len() < estimates.len() {
        return Err(PrecisionCompactionError::OutputTooSmall {
            required: estimates.len(),
            actual: compacted.len(),
        });
    }

    let mut counts = [0_usize; 3];
    for (index, &estimate) in estimates.iter().enumerate() {
        let class = policy
            .classify(estimate)
            .ok_or(PrecisionCompactionError::InvalidEstimate { index })?;
        counts[class as usize] += 1;
    }
    let sensitive_end = counts[PrecisionClass::Sensitive as usize];
    let near_end = sensitive_end + counts[PrecisionClass::Near as usize];
    let mut write = [0, sensitive_end, near_end];
    for (index, &estimate) in estimates.iter().enumerate() {
        // The first pass already validated every estimate and the policy.
        let class = policy
            .classify(estimate)
            .expect("validated precision estimate");
        let slot = &mut write[class as usize];
        compacted[*slot] = index as u32;
        *slot += 1;
    }
    Ok(PrecisionBatchLayout {
        sensitive_end,
        near_end,
        len: estimates.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(error: f32, condition: f32, margin: f32) -> PrecisionEstimate {
        PrecisionEstimate {
            screen_error_px: error,
            condition,
            boundary_margin: margin,
        }
    }

    #[test]
    fn default_policy_classifies_projected_error() {
        let policy = ProjectionPrecisionPolicy::DEFAULT;
        assert!(policy.is_valid());
        assert_eq!(
            policy.classify(estimate(12.0, 1.0, 1.0)),
            Some(PrecisionClass::Sensitive)
        );
        assert_eq!(
            policy.classify(estimate(2.0, 1.0, 1.0)),
            Some(PrecisionClass::Near)
        );
        assert_eq!(
            policy.classify(estimate(0.25, 1.0, 1.0)),
            Some(PrecisionClass::Far)
        );
    }

    #[test]
    fn conditioning_and_boundary_margin_promote_sensitive_work() {
        let policy = ProjectionPrecisionPolicy::DEFAULT;
        assert_eq!(
            policy.classify(estimate(0.5, 20.0, 1.0)),
            Some(PrecisionClass::Sensitive)
        );
        assert_eq!(
            policy.classify(estimate(0.01, 1.0, 0.0)),
            Some(PrecisionClass::Sensitive)
        );
    }

    #[test]
    fn invalid_estimates_and_policies_do_not_silently_classify() {
        let policy = ProjectionPrecisionPolicy::DEFAULT;
        assert_eq!(policy.classify(estimate(f32::NAN, 1.0, 1.0)), None);
        let invalid = ProjectionPrecisionPolicy {
            sensitive_threshold_px: 0.5,
            near_threshold_px: 1.0,
            ..policy
        };
        assert!(!invalid.is_valid());
        assert_eq!(invalid.classify(estimate(1.0, 1.0, 1.0)), None);
    }

    #[test]
    fn compaction_forms_stable_caller_owned_batches() {
        let estimates = [
            estimate(0.25, 1.0, 1.0),
            estimate(12.0, 1.0, 1.0),
            estimate(2.0, 1.0, 1.0),
            estimate(0.5, 20.0, 1.0),
            estimate(0.1, 1.0, 1.0),
            estimate(3.0, 1.0, 1.0),
        ];
        let mut indices = [u32::MAX; 6];
        let layout =
            compact_precision_indices(ProjectionPrecisionPolicy::DEFAULT, &estimates, &mut indices)
                .unwrap();
        assert_eq!(layout.len(), estimates.len());
        assert_eq!(
            layout.indices(&indices, PrecisionClass::Sensitive),
            Some(&[1, 3][..])
        );
        assert_eq!(
            layout.indices(&indices, PrecisionClass::Near),
            Some(&[2, 5][..])
        );
        assert_eq!(
            layout.indices(&indices, PrecisionClass::Far),
            Some(&[0, 4][..])
        );
        assert_eq!(layout.indices(&indices[..2], PrecisionClass::Far), None);
    }

    #[test]
    fn compaction_rejects_invalid_input_before_writing() {
        let estimates = [estimate(0.25, 1.0, 1.0), estimate(f32::NAN, 1.0, 1.0)];
        let mut indices = [7_u32; 2];
        assert_eq!(
            compact_precision_indices(ProjectionPrecisionPolicy::DEFAULT, &estimates, &mut indices,),
            Err(PrecisionCompactionError::InvalidEstimate { index: 1 })
        );
        assert_eq!(indices, [7, 7]);
    }
}
