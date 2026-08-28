//! One installation point for every process-wide runtime override this crate
//! reads, plus the report that names every env var it has STOPPED reading.
//!
//! # Why this module exists
//!
//! Before 2026-08-10 `neoethos-models` read 22 `std::env::var` sites spread
//! across ten files. Each one was a second way to set something — a device, a
//! precision, a thread count, a budget, a threshold — and none of them was in
//! the run artifact. A run could therefore be steered by a shell export that
//! left no trace anywhere, which is the `NEOETHOS_GPU_F64` failure mode.
//!
//! Every one of those sites is now either routed to a `Settings` field or
//! deleted. The fields are installed here, once, from the operator's resolved
//! `Settings`.
//!
//! # The retired-name report (non-negotiable #4)
//!
//! A deleted env var that is still exported must never fail silently. At
//! install time [`report_retired_env_vars`] walks [`RETIRED_ENV_VARS`], and for
//! every name still present in the environment it logs, at WARN, the name, the
//! field that replaced it, and the fact that the exported value was IGNORED.
//! Silence is the disease; a run that ignores an operator's export and says so
//! is honest, a run that ignores it quietly is not.

use std::sync::Once;

/// Every env var `neoethos-models` used to read and no longer does, with the
/// configuration that replaced it.
///
/// `(env name, what now decides it)`
///
/// Keep this list APPEND-ONLY. A name removed from here is a name the operator
/// can export again without being told it does nothing.
pub const RETIRED_ENV_VARS: &[(&str, &str)] = &[
    // --- per-model / per-subsystem CUDA kernel kill-switches -------------
    (
        "NEOETHOS_BOT_STATISTICAL_CUDA_KERNEL",
        "models.statistical_device (set it to `cpu` for the CPU path)",
    ),
    (
        "NEOETHOS_BOT_NEAT_CUDA_KERNEL",
        "the NEAT expert's configured device policy",
    ),
    (
        "NEOETHOS_BOT_NEURO_EVO_CUDA_KERNEL",
        "the neuro-evo expert's configured device policy",
    ),
    (
        "NEOETHOS_BOT_ELASTICNET_CUDA_KERNEL",
        "models.statistical_device",
    ),
    (
        "NEOETHOS_BOT_LOGISTIC_CUDA_KERNEL",
        "models.statistical_device",
    ),
    // --- CUDA ordinal overrides that outranked the policy string ---------
    (
        "NEOETHOS_BOT_STATISTICAL_CUDA_DEVICE",
        "the ordinal in models.statistical_device (`gpu:1`)",
    ),
    (
        "NEOETHOS_BOT_NEAT_CUDA_DEVICE",
        "the ordinal in the NEAT device policy (`gpu:1`)",
    ),
    (
        "NEOETHOS_BOT_NEURO_EVO_CUDA_DEVICE",
        "the ordinal in the neuro-evo device policy (`gpu:1`)",
    ),
    // --- kernel launch geometry ------------------------------------------
    (
        "NEOETHOS_BOT_STATISTICAL_KERNEL_UNITS",
        "the device's own max_units_per_cube (hardware, not config)",
    ),
    (
        "NEOETHOS_BOT_NEAT_KERNEL_UNITS",
        "the device's own max_units_per_cube (hardware, not config)",
    ),
    (
        "NEOETHOS_BOT_NEURO_EVO_KERNEL_UNITS",
        "the device's own max_units_per_cube (hardware, not config)",
    ),
    // --- device policy ----------------------------------------------------
    (
        "NEOETHOS_BOT_META_DEVICE",
        "models.statistical_device, and system.device for the runtime report",
    ),
    // --- training precision ----------------------------------------------
    (
        "NEOETHOS_BOT_TRAIN_PRECISION",
        "system.hardware.training_precision",
    ),
    (
        "FOREX_TRAIN_PRECISION",
        "system.hardware.training_precision",
    ),
    (
        "FOREX_BURN_MODEL_SUPPORTS_BF16",
        "the backend's own B::supports_dtype probe (a capability, not a knob)",
    ),
    // --- budgets and thresholds ------------------------------------------
    (
        "NEOETHOS_BOT_DRIFT_THRESHOLD",
        "risk.feature_drift_threshold, passed into detect_feature_drift",
    ),
    (
        "FOREX_GENETIC_MAX_LABEL_EVALS",
        "a compile-time budget constant in genetic.rs (never operator config)",
    ),
    (
        "FOREX_GENETIC_MAX_DISCOVERY_CANDIDATES",
        "a compile-time budget constant in genetic.rs (never operator config)",
    ),
    (
        "FOREX_NEURO_EVO_MAX_EVALS",
        "a compile-time budget constant in crfmnes_impl.rs (never operator config)",
    ),
    (
        "NEOETHOS_BOT_PROP_MIN_EMBARGO_BARS",
        "the fixed 20-bar leak floor; widen the gap with models.embargo_minutes",
    ),
    // --- paths -------------------------------------------------------------
    (
        "NEOETHOS_BOT_DATA_ROOT",
        "system.data_dir, or TrainingOrchestrator::with_data_root",
    ),
    (
        "CATBOOST_EXECUTABLE",
        "NEOETHOS_BOT_CATBOOST_EXECUTABLE (one name for the CatBoost CLI)",
    ),
    ("GIT_COMMIT_HASH", "NEOETHOS_SOURCE_COMMIT"),
];

static REPORTED: Once = Once::new();

/// Log every retired env var that is still set in this process.
///
/// Runs at most once per process. Called by
/// [`install_model_runtime_from_settings`], which every startup path reaches.
pub fn report_retired_env_vars() {
    REPORTED.call_once(|| {
        let mut still_set = 0usize;
        for (name, replacement) in RETIRED_ENV_VARS {
            let Ok(value) = std::env::var(name) else {
                continue;
            };
            still_set += 1;
            tracing::warn!(
                target: "neoethos_models::config",
                env_var = name,
                exported_value = %value,
                replaced_by = replacement,
                "RETIRED ENV VAR IS STILL SET — its value was IGNORED. \
                 neoethos-models no longer reads this name; set the named \
                 configuration instead, then unset the variable."
            );
        }
        if still_set > 0 {
            tracing::warn!(
                target: "neoethos_models::config",
                count = still_set,
                "{still_set} retired neoethos-models env var(s) are exported in \
                 this environment and changed nothing in this run"
            );
        }
    });
}

/// Install every process-wide override `neoethos-models` reads, from the
/// operator's resolved `Settings`, and report retired env names.
///
/// First install wins (each registry is a `OnceLock`), so calling this more
/// than once — the app, the desktop shell and the CLI each call an installer —
/// is safe and idempotent.
pub fn install_model_runtime_from_settings(settings: &neoethos_core::Settings) {
    crate::tree_models::config::set_tree_runtime(settings);
    crate::statistical::common::set_statistical_device(settings);
    crate::runtime::capabilities::set_model_device_registry(settings);
    report_retired_env_vars();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retired_env_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for (name, _) in RETIRED_ENV_VARS {
            assert!(seen.insert(*name), "duplicate retired env name {name}");
        }
    }

    #[test]
    fn every_retired_name_states_what_replaced_it() {
        for (name, replacement) in RETIRED_ENV_VARS {
            assert!(
                !replacement.trim().is_empty(),
                "{name} must name what decides it now — 'it does nothing' is not a report"
            );
        }
    }

    /// Names this crate STILL reads must not be reported as retired: claiming
    /// we ignore a variable we honour is the same lie as ignoring one we claim
    /// to honour, pointing the other way.
    #[test]
    fn names_still_read_are_not_listed_as_retired() {
        for still_read in [
            // toolchain locator for an external binary
            "NEOETHOS_BOT_CATBOOST_EXECUTABLE",
            // artifact provenance
            "NEOETHOS_SOURCE_COMMIT",
            "GITHUB_SHA",
            // driver-level device mask, read as a hardware fact
            "CUDA_VISIBLE_DEVICES",
        ] {
            assert!(
                !RETIRED_ENV_VARS.iter().any(|(name, _)| *name == still_read),
                "{still_read} is still read by this crate and must not be reported as retired"
            );
        }
    }

    /// `RAYON_NUM_THREADS` is no longer read by THIS crate — the budget comes
    /// from `models.backtest_runtime.rayon_threads` — but rayon's own global
    /// pool still honours it, so telling the operator his export "was IGNORED"
    /// would be false.
    #[test]
    fn rayon_num_threads_is_not_reported_as_retired() {
        assert!(
            !RETIRED_ENV_VARS
                .iter()
                .any(|(name, _)| *name == "RAYON_NUM_THREADS")
        );
    }

    #[test]
    fn reporting_is_safe_when_a_retired_name_is_set() {
        unsafe {
            std::env::set_var("NEOETHOS_BOT_DRIFT_THRESHOLD", "0.4");
        }
        report_retired_env_vars();
        unsafe {
            std::env::remove_var("NEOETHOS_BOT_DRIFT_THRESHOLD");
        }
    }
}
