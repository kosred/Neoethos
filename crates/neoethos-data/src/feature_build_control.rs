//! Per-request cooperative control and optional explicit indicator execution policy.
//! The selected lane must still pass its existing admission and exact-authority checks.
//!
//! This does not abort threads. Owners must join their workers before releasing
//! CPU/source/scratch leases. An in-flight third-party kernel must return before
//! its next checkpoint; no partial frame may escape a cancelled build.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::core::hpc_ta::IndicatorComputePolicy;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureBuildProgress {
    pub timeframe: String,
    pub stage: &'static str,
    pub item: String,
    /// Work units within this named stage, not percent of the entire search.
    pub completed: usize,
    pub total: usize,
}

#[derive(Clone, Default)]
pub struct FeatureBuildControl {
    cancelled: Option<Arc<AtomicBool>>,
    observer: Option<Arc<dyn Fn(FeatureBuildProgress) + Send + Sync>>,
    timeframe: String,
    indicator_compute_policy: Option<IndicatorComputePolicy>,
}

#[derive(Debug)]
pub struct FeatureBuildCancelled;

impl std::fmt::Display for FeatureBuildCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("__FEATURE_BUILD_CANCELLED__: operator stopped feature preparation")
    }
}

impl std::error::Error for FeatureBuildCancelled {}

impl FeatureBuildCancelled {
    pub fn matches(error: &anyhow::Error) -> bool {
        // Vortex converts Write errors into its own error payload. Retain a
        // unique marker at that FFI/library boundary; ordinary failures remain
        // failures even if Stop races with them.
        error.chain().any(|cause| {
            cause.is::<Self>() || cause.to_string().contains("__FEATURE_BUILD_CANCELLED__")
        })
    }
}

impl FeatureBuildControl {
    pub fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self {
            cancelled: Some(cancelled),
            ..Self::default()
        }
    }

    pub fn with_observer(
        mut self,
        observer: impl Fn(FeatureBuildProgress) + Send + Sync + 'static,
    ) -> Self {
        self.observer = Some(Arc::new(observer));
        self
    }

    pub fn for_timeframe(&self, timeframe: &str) -> Self {
        Self {
            timeframe: timeframe.to_owned(),
            ..self.clone()
        }
    }

    /// Pin this build and its cloned timeframe controls without changing process policy.
    pub fn with_indicator_compute_policy(mut self, policy: IndicatorComputePolicy) -> Self {
        self.indicator_compute_policy = Some(policy);
        self
    }

    pub fn indicator_compute_policy(&self) -> Option<IndicatorComputePolicy> {
        self.indicator_compute_policy
    }

    /// Unspecified controls retain the existing process-default behavior.
    pub fn resolved_indicator_compute_policy(&self) -> IndicatorComputePolicy {
        self.indicator_compute_policy
            .unwrap_or_else(crate::core::hpc_ta::resolved_indicator_compute_policy)
    }

    pub fn checkpoint(&self) -> anyhow::Result<()> {
        if self
            .cancelled
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire))
        {
            return Err(FeatureBuildCancelled.into());
        }
        Ok(())
    }

    pub fn report(
        &self,
        stage: &'static str,
        item: impl Into<String>,
        completed: usize,
        total: usize,
    ) -> anyhow::Result<()> {
        self.checkpoint()?;
        if let Some(observer) = &self.observer {
            observer(FeatureBuildProgress {
                timeframe: self.timeframe.clone(),
                stage,
                item: item.into(),
                completed,
                total,
            });
        }
        self.checkpoint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_indicator_policy_is_request_local_and_survives_timeframe_cloning() {
        let default = FeatureBuildControl::default();
        let process_policy = crate::core::hpc_ta::resolved_indicator_compute_policy();
        assert_eq!(default.indicator_compute_policy(), None);
        assert_eq!(default.resolved_indicator_compute_policy(), process_policy);
        for policy in [
            IndicatorComputePolicy::Auto,
            IndicatorComputePolicy::CpuOnly,
            IndicatorComputePolicy::GpuOnly,
        ] {
            let request = default.clone().with_indicator_compute_policy(policy);
            let child = request.for_timeframe("M5");
            assert_eq!(request.indicator_compute_policy(), Some(policy));
            assert_eq!(child.indicator_compute_policy(), Some(policy));
            assert_eq!(child.resolved_indicator_compute_policy(), policy);
            assert_eq!(default.indicator_compute_policy(), None);
        }
        assert_eq!(
            crate::core::hpc_ta::resolved_indicator_compute_policy(),
            process_policy
        );
    }

    #[test]
    fn cancellation_is_shared_with_children_but_not_other_requests() {
        let flag = Arc::new(AtomicBool::new(false));
        let control = FeatureBuildControl::new(Arc::clone(&flag));
        let child = control.for_timeframe("M5");
        flag.store(true, Ordering::Release);
        assert!(FeatureBuildCancelled::matches(
            &child.checkpoint().unwrap_err()
        ));
        assert!(control.checkpoint().is_err());
        assert!(FeatureBuildControl::default().checkpoint().is_ok());
        assert!(
            FeatureBuildControl::new(Arc::new(AtomicBool::new(false)))
                .checkpoint()
                .is_ok()
        );
        assert!(!FeatureBuildCancelled::matches(&anyhow::anyhow!(
            "disk full"
        )));
    }
}
