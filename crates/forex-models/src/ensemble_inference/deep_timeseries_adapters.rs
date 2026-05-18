//! [`super::ExpertModel`] adapters for the deep TIME-SERIES
//! expert families.
//!
//! Phase D1.2.3. This module covers the seven time-series-aware
//! deep architectures: **NBEATS**, **TiDE**, **Transformer**,
//! **PatchTST**, **TimesNet**, **NBeatsx_NF**, **TiDE_NF**.
//!
//! ## Wait — aren't these "forecasters"?
//!
//! Despite the architectural lineage (NBEATS, TiDE, PatchTST, and
//! TimesNet are well-known time-series **forecasters** in the
//! literature), in THIS codebase they are configured with
//! `with_n_classes(3)` at construction (see
//! `deep_models.rs::default_*_config` for each — 7 call sites at
//! lines 646, 654, 661, 667, 674, 689, 695, 703, 715, 723). They
//! emit 3-class softmax probabilities over `[sell, neutral, buy]`
//! exactly like the D1.2.2 classifiers, just via a time-series
//! architecture rather than a feed-forward / KAN / TabNet one.
//!
//! In other words: the architecture is "time-series", the OUTPUT
//! HEAD is 3-class classification. The "forecast" framing only
//! applies to the model's INPUT (a sliding window of past bars)
//! and its internal attention/decomposition modules, not to its
//! output type.
//!
//! Consequently every adapter here uses
//! [`super::ExpertOutputKind::Classification3`] (not `Forecast1`)
//! and the same `classification3_per_row` helper that the tree +
//! deep-classifier adapters use.
//!
//! ## Continuous-output forecasters land elsewhere
//!
//! If a future expert is added that produces a single continuous
//! forecast (e.g. predicted close-to-close return), it would use
//! [`super::ExpertOutputKind::Forecast1`] and a different
//! per-row validator. Until that lands the `Forecast1` variant
//! only exists in the trait taxonomy for completeness — no
//! production expert family uses it yet.

use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::DataFrame;

use super::{ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction};
use crate::base::ExpertModel as BaseExpertModel;
use crate::deep_models::{
    NBeatsExpert, NBeatsxNfExpert, PatchTSTExpert, TiDEExpert, TiDENfExpert, TimesNetExpert,
    TransformerExpert,
};
use crate::runtime::capabilities::ModelFamily;
// Shared row-validator from tree_adapters (pub(super) — same
// helper every Classification3 family uses).
use super::tree_adapters::classification3_per_row;

// ---------------------------------------------------------------------------
// Adapter generation macro
// ---------------------------------------------------------------------------
//
// The seven time-series adapters are IDENTICAL except for:
//   - the canonical name (string literal)
//   - the inner expert type
// We collapse them into a macro to avoid 7× boilerplate copies of
// the same trait impl. Each invocation expands into:
//   - the adapter struct (single `inner` field of the typed expert)
//   - unsafe impl Sync (Burn OnceCell-after-init contract, same as
//     D1.2.2 deep classifiers — see the module-level SAFETY
//     comment in deep_classification_adapters.rs)
//   - ExpertModel impl (delegates predict_proba to inner)
//   - paired Loader struct
//   - ExpertLoader impl

macro_rules! define_deep_timeseries_adapter {
    (
        $adapter:ident,
        $loader:ident,
        $inner:ty,
        $name_literal:literal,
        $doc:expr
    ) => {
        #[doc = $doc]
        ///
        /// SAFETY: see [`super::deep_classification_adapters`]
        /// module doc for the OnceCell-after-init Sync contract
        /// (identical for every Burn-backed deep adapter).
        pub struct $adapter {
            inner: $inner,
        }

        impl $adapter {
            pub fn new(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn inner(&self) -> &$inner {
                &self.inner
            }
        }

        // SAFETY: see deep_classification_adapters.rs module doc.
        unsafe impl Sync for $adapter {}

        impl ExpertModel for $adapter {
            fn name(&self) -> &str {
                $name_literal
            }
            fn family(&self) -> ModelFamily {
                ModelFamily::Deep
            }
            fn output_kind(&self) -> ExpertOutputKind {
                ExpertOutputKind::Classification3
            }
            fn feature_columns(&self) -> &[String] {
                self.inner.feature_columns()
            }
            fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
                let probs = self
                    .inner
                    .predict_proba(df)
                    .with_context(|| format!("{} predict_proba failed", $name_literal))?;
                classification3_per_row(&probs)
            }
        }

        #[doc = concat!("Loader for [`", stringify!($adapter), "`].")]
        pub struct $loader;

        impl ExpertLoader for $loader {
            fn name(&self) -> &str {
                $name_literal
            }
            fn load(&self, artifact_dir: &Path) -> Result<Box<dyn ExpertModel>> {
                let mut inner = <$inner>::new(42, None);
                inner.load(artifact_dir).with_context(|| {
                    format!(
                        "{}::load({}) failed",
                        stringify!($inner),
                        artifact_dir.display()
                    )
                })?;
                Ok(Box::new($adapter::new(inner)))
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Seven concrete adapters
// ---------------------------------------------------------------------------

define_deep_timeseries_adapter!(
    NBeatsAdapter,
    NBeatsLoader,
    NBeatsExpert,
    "nbeats",
    "[`ExpertModel`] adapter for [`NBeatsExpert`] (Neural Basis Expansion Analysis for Time Series). 3-class softmax classifier head."
);

define_deep_timeseries_adapter!(
    NBeatsxNfAdapter,
    NBeatsxNfLoader,
    NBeatsxNfExpert,
    "nbeatsx_nf",
    "[`ExpertModel`] adapter for [`NBeatsxNfExpert`] (NBEATSx with neural-features adaptation). 3-class softmax classifier head."
);

define_deep_timeseries_adapter!(
    TiDEAdapter,
    TiDELoader,
    TiDEExpert,
    "tide",
    "[`ExpertModel`] adapter for [`TiDEExpert`] (Time-series Dense Encoder). 3-class softmax classifier head."
);

define_deep_timeseries_adapter!(
    TiDENfAdapter,
    TiDENfLoader,
    TiDENfExpert,
    "tide_nf",
    "[`ExpertModel`] adapter for [`TiDENfExpert`] (TiDE with neural-features adaptation). 3-class softmax classifier head."
);

define_deep_timeseries_adapter!(
    TransformerAdapter,
    TransformerLoader,
    TransformerExpert,
    "transformer",
    "[`ExpertModel`] adapter for [`TransformerExpert`] (standard encoder-only transformer). 3-class softmax classifier head."
);

define_deep_timeseries_adapter!(
    PatchTstAdapter,
    PatchTstLoader,
    PatchTSTExpert,
    "patchtst",
    "[`ExpertModel`] adapter for [`PatchTSTExpert`] (Patch Time-Series Transformer). 3-class softmax classifier head."
);

define_deep_timeseries_adapter!(
    TimesNetAdapter,
    TimesNetLoader,
    TimesNetExpert,
    "timesnet",
    "[`ExpertModel`] adapter for [`TimesNetExpert`] (period-decomposition 2D-conv time-series model). 3-class softmax classifier head."
);

// ---------------------------------------------------------------------------
// Convenience: register-all-deep-timeseries-loaders
// ---------------------------------------------------------------------------

/// Register every deep-time-series loader (7 canonical names) on
/// the supplied registry. Same shape as the tree-family and
/// deep-classifier counterparts; the forex-app bootstrap will
/// call all three to install the full ~17-model deep+tree
/// foundation in one pass.
pub fn register_deep_timeseries_loaders(registry: &mut super::ExpertRegistry) -> Result<()> {
    registry.register(Box::new(NBeatsLoader))?;
    registry.register(Box::new(NBeatsxNfLoader))?;
    registry.register(Box::new(TiDELoader))?;
    registry.register(Box::new(TiDENfLoader))?;
    registry.register(Box::new(TransformerLoader))?;
    registry.register(Box::new(PatchTstLoader))?;
    registry.register(Box::new(TimesNetLoader))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble_inference::ExpertRegistry;

    // --- Adapter construction + naming invariants -----------------------

    #[test]
    fn nbeats_adapter_name_family_kind() {
        let inner = NBeatsExpert::new(42, None);
        let adapter = NBeatsAdapter::new(inner);
        assert_eq!(adapter.name(), "nbeats");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn nbeatsx_nf_adapter_name_family_kind() {
        let inner = NBeatsxNfExpert::new(42, None);
        let adapter = NBeatsxNfAdapter::new(inner);
        assert_eq!(adapter.name(), "nbeatsx_nf");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn tide_adapter_name_family_kind() {
        let inner = TiDEExpert::new(42, None);
        let adapter = TiDEAdapter::new(inner);
        assert_eq!(adapter.name(), "tide");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn tide_nf_adapter_name_family_kind() {
        let inner = TiDENfExpert::new(42, None);
        let adapter = TiDENfAdapter::new(inner);
        assert_eq!(adapter.name(), "tide_nf");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn transformer_adapter_name_family_kind() {
        let inner = TransformerExpert::new(42, None);
        let adapter = TransformerAdapter::new(inner);
        assert_eq!(adapter.name(), "transformer");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn patchtst_adapter_name_family_kind() {
        let inner = PatchTSTExpert::new(42, None);
        let adapter = PatchTstAdapter::new(inner);
        assert_eq!(adapter.name(), "patchtst");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    #[test]
    fn timesnet_adapter_name_family_kind() {
        let inner = TimesNetExpert::new(42, None);
        let adapter = TimesNetAdapter::new(inner);
        assert_eq!(adapter.name(), "timesnet");
        assert_eq!(adapter.family(), ModelFamily::Deep);
        assert_eq!(adapter.output_kind(), ExpertOutputKind::Classification3);
    }

    // --- Loader names match adapter names -------------------------------

    #[test]
    fn loader_names_match_their_adapter_names() {
        assert_eq!(NBeatsLoader.name(), "nbeats");
        assert_eq!(NBeatsxNfLoader.name(), "nbeatsx_nf");
        assert_eq!(TiDELoader.name(), "tide");
        assert_eq!(TiDENfLoader.name(), "tide_nf");
        assert_eq!(TransformerLoader.name(), "transformer");
        assert_eq!(PatchTstLoader.name(), "patchtst");
        assert_eq!(TimesNetLoader.name(), "timesnet");
    }

    // --- register_deep_timeseries_loaders: all 7 land -------------------

    #[test]
    fn register_deep_timeseries_loaders_installs_seven_names() {
        let mut reg = ExpertRegistry::new();
        register_deep_timeseries_loaders(&mut reg).expect("register");
        let mut names = reg.registered_names();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "nbeats",
                "nbeatsx_nf",
                "patchtst",
                "tide",
                "tide_nf",
                "timesnet",
                "transformer",
            ]
        );
    }

    #[test]
    fn register_deep_timeseries_loaders_rejects_double_registration() {
        let mut reg = ExpertRegistry::new();
        register_deep_timeseries_loaders(&mut reg).expect("first call");
        let err = register_deep_timeseries_loaders(&mut reg).expect_err("second call must error");
        assert!(err.to_string().contains("already registered"));
    }

    // --- Full deep + tree coexistence ----------------------------------
    //
    // After D1.2.1 + D1.2.2 + D1.2.3, the registry can hold all 17
    // tree+deep adapters at once: 7 tree + 3 deep-classifier + 7
    // deep-time-series.

    #[test]
    fn all_seventeen_tree_and_deep_loaders_coexist() {
        let mut reg = ExpertRegistry::new();
        super::super::tree_adapters::register_tree_loaders(&mut reg).expect("trees");
        super::super::deep_classification_adapters::register_deep_classification_loaders(&mut reg)
            .expect("deep-cls");
        register_deep_timeseries_loaders(&mut reg).expect("deep-ts");

        let names = reg.registered_names();
        assert_eq!(names.len(), 17);
        for required in [
            // Tree (7)
            "lightgbm",
            "xgboost",
            "xgboost_rf",
            "xgboost_dart",
            "catboost",
            "catboost_alt",
            "sklears_tree",
            // Deep classifier (3)
            "mlp",
            "kan",
            "tabnet",
            // Deep time-series (7)
            "nbeats",
            "nbeatsx_nf",
            "tide",
            "tide_nf",
            "transformer",
            "patchtst",
            "timesnet",
        ] {
            assert!(
                reg.has_loader(required),
                "registry missing loader for '{required}'"
            );
        }
    }

    // --- feature_columns proxies through to inner ----------------------

    #[test]
    fn transformer_adapter_feature_columns_proxies_inner_state() {
        let inner = TransformerExpert::new(42, None);
        let adapter = TransformerAdapter::new(inner);
        assert!(adapter.feature_columns().is_empty());
    }
}
