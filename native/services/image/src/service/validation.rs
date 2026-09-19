//! Registration validation keeps the source alive until its mapping closes.

use kore_rt::cap::{MappedLeaseView, ReturnedCapability, UncertainLeaseMap};

#[derive(Debug)]
pub(super) enum ValidationMapping {
    Known(MappedLeaseView),
    Uncertain(UncertainLeaseMap),
}

impl ValidationMapping {
    fn try_retire(&mut self) -> bool {
        match self {
            Self::Known(view) => view.try_retire_outcome().is_closed(),
            Self::Uncertain(view) => view.try_retire(),
        }
    }
}

#[derive(Default, Debug)]
pub(super) struct ValidationCleanup {
    pub mapping: Option<ValidationMapping>,
    /// Present when admission did not move this returned lease into a
    /// catalogue entry. The service, not ServeHost, must keep it through
    /// an incomplete validation unshare, including an unknown map result.
    pub source: Option<ReturnedCapability>,
}

impl ValidationCleanup {
    pub fn is_pending(&self) -> bool {
        self.mapping.is_some() || self.source.is_some()
    }

    pub fn try_retire(&mut self) -> bool {
        retire_mapping_before_source(
            &mut self.mapping,
            &mut self.source,
            ValidationMapping::try_retire,
            ReturnedCapability::try_retire,
        )
    }
}

impl Drop for ValidationCleanup {
    fn drop(&mut self) {
        if !self.try_retire() {
            // A failed exact-unshare cannot fall through to a parent revoke
            // from an incidental Rust drop. Preserve that ancestor until task
            // teardown; ordinary service operation keeps this owner for retry.
            if let Some(source) = self.source.take() {
                core::mem::forget(source);
            }
        }
    }
}

/// A completed mapping phase is not replayed when source retirement refuses.
/// The closures allow host tests to exercise the ordering without fabricating
/// native owners or pretending that a host transport implements kernel verbs.
fn retire_mapping_before_source<M, S>(
    mapping: &mut Option<M>,
    source: &mut Option<S>,
    close_mapping: impl FnOnce(&mut M) -> bool,
    close_source: impl FnOnce(&mut S) -> bool,
) -> bool {
    if let Some(view) = mapping.as_mut() {
        if !close_mapping(view) {
            return false;
        }
        *mapping = None;
    }
    if let Some(parent) = source.as_mut() {
        if !close_source(parent) {
            return false;
        }
        *source = None;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::retire_mapping_before_source;

    #[test]
    fn validation_refused_mapping_keeps_the_original_source_untouched() {
        let mut mapping = Some(17);
        let mut source = Some(29);
        for _ in 0..3 {
            assert!(!retire_mapping_before_source(
                &mut mapping,
                &mut source,
                |view| {
                    assert_eq!(*view, 17);
                    false
                },
                |_| panic!("source retirement before exact mapping closure"),
            ));
            assert_eq!(mapping, Some(17));
            assert_eq!(source, Some(29));
        }
        let phase = core::cell::Cell::new(0);
        assert!(retire_mapping_before_source(
            &mut mapping,
            &mut source,
            |_| {
                phase.set(1);
                true
            },
            |_| {
                assert_eq!(phase.get(), 1);
                phase.set(2);
                true
            },
        ));
        assert_eq!(phase.get(), 2);
        assert_eq!(mapping, None);
        assert_eq!(source, None);
    }

    #[test]
    fn validation_source_retry_does_not_replay_a_closed_mapping() {
        let mut mapping = Some(17);
        let mut source = Some(29);
        assert!(!retire_mapping_before_source(
            &mut mapping,
            &mut source,
            |_| true,
            |_| false,
        ));
        assert_eq!(mapping, None);
        assert_eq!(source, Some(29));
        assert!(retire_mapping_before_source(
            &mut mapping,
            &mut source,
            |_| panic!("closed mapping replayed"),
            |parent| {
                assert_eq!(*parent, 29);
                true
            },
        ));
        assert!(retire_mapping_before_source(
            &mut mapping,
            &mut source,
            |_| panic!("idle mapping"),
            |_| panic!("idle source"),
        ));
    }
}
