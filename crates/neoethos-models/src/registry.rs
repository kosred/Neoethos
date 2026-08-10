// Model Registry
//
// SCOPE (2026-08-09, batch D4): name -> capability record, and the per-model
// GPU-capability table. Nothing else.
//
// What was here and is gone: a whole catalogue/recommender API —
// `get_model_info`, `list_models_by_category`, `is_valid_model`,
// `register_model` (plus the dynamic `Mutex<HashMap<..>>` registry it was the
// only writer for), `ModelInfo`, `ModelCategory`, `get_recommended_device`,
// `get_recommended_device_with_plan`, `get_recommended_precision_with_plan`
// and their private helpers. Every one of them resolved, outside its own
// definition, only into `#[cfg(test)]` blocks or `tests/integration_test.rs`.
// The three `get_recommended_*` functions had no caller at all, test included.
// The device decision is made by `neoethos_core::HardwareExecutionPlan`
// (`training_orchestrator.rs` calls it on every train); a second recommender
// that nothing consulted was a second answer to a settled question.
//
// The one live export is [`get_model_capability`], reached from
// `runtime/dispatch.rs:33` inside `build_dispatch_plan` on every train.

use crate::runtime::capabilities::{ModelCapability, ModelFamily, model_capability};

/// Resolve the capability record for a known model name.
///
/// LIVE: `runtime::dispatch::build_dispatch_plan` calls this for every model
/// in a training run.
pub fn get_model_capability(name: &str) -> Option<ModelCapability> {
    model_capability(name)
}

/// Whether this build contains a GPU code path that `name` can actually use.
///
/// NOT WIRED YET — held deliberately. This is a correct per-model capability
/// table (it reads the real Cargo features, so it cannot drift from what was
/// compiled), and task #35's per-model "device self-report" on the training
/// summary is exactly the consumer it lacks. Until that lands, the operator
/// has no way to see that e.g. the DQN's CUDA path is compiled but the Deep
/// models' is not. Deleting the table would delete the answer as well as the
/// missing question.
///
/// Do not add an `#[allow(dead_code)]` here if a caller disappears — either
/// wire it or delete it.
pub fn supports_gpu_for_model(name: &str, family: ModelFamily) -> bool {
    match name {
        "lightgbm" => cfg!(feature = "lightgbm-gpu"),
        "xgboost" | "xgboost_rf" | "xgboost_dart" => cfg!(feature = "xgboost"),
        "catboost" | "catboost_alt" => cfg!(feature = "catboost"),
        "dqn" => cfg!(feature = "reinforcement-learning-cuda"),
        // SAC runs on the Burn backend (like Deep/Exit), not rlkit/CUDA.
        "sac" => cfg!(feature = "burn-wgpu-backend"),
        _ => match family {
            ModelFamily::Deep | ModelFamily::Exit => cfg!(feature = "burn-wgpu-backend"),
            _ => false,
        },
    }
}

/// Whether the GPU path is the one this model SHOULD take when a card is
/// present. Same hold as [`supports_gpu_for_model`] — see its note.
pub fn prefers_gpu_for_model(name: &str, family: ModelFamily) -> bool {
    match name {
        "lightgbm" => cfg!(feature = "lightgbm-gpu"),
        "xgboost" | "xgboost_rf" | "xgboost_dart" => cfg!(feature = "xgboost"),
        "catboost" | "catboost_alt" => cfg!(feature = "catboost"),
        "dqn" => cfg!(feature = "reinforcement-learning-cuda"),
        // SAC runs on the Burn backend (like Deep/Exit), not rlkit/CUDA.
        "sac" => cfg!(feature = "burn-wgpu-backend"),
        _ => match family {
            ModelFamily::Deep | ModelFamily::Exit => cfg!(feature = "burn-wgpu-backend"),
            _ => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::capabilities::{CapabilityState, KNOWN_MODEL_NAMES};
    use neoethos_core::Settings;

    #[test]
    fn known_configured_models_resolve_to_capabilities() {
        let expectations = [
            ("lightgbm", ModelFamily::Tree, CapabilityState::Verified),
            ("xgboost", ModelFamily::Tree, CapabilityState::Verified),
            ("xgboost_rf", ModelFamily::Tree, CapabilityState::Verified),
            ("xgboost_dart", ModelFamily::Tree, CapabilityState::Verified),
            ("catboost", ModelFamily::Tree, CapabilityState::Verified),
            ("catboost_alt", ModelFamily::Tree, CapabilityState::Verified),
            ("sklears_tree", ModelFamily::Tree, CapabilityState::Verified),
            ("mlp", ModelFamily::Deep, CapabilityState::Verified),
            ("elasticnet", ModelFamily::Meta, CapabilityState::Verified),
            ("logistic", ModelFamily::Meta, CapabilityState::Verified),
            ("bayes_logit", ModelFamily::Meta, CapabilityState::Verified),
            ("meta_blender", ModelFamily::Meta, CapabilityState::Verified),
            (
                "probability_calibrator",
                ModelFamily::Meta,
                CapabilityState::Verified,
            ),
            (
                "conformal_gate",
                ModelFamily::Meta,
                CapabilityState::Verified,
            ),
            ("meta_stack", ModelFamily::Meta, CapabilityState::Verified),
            (
                "genetic",
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
            ),
            ("exit_agent", ModelFamily::Exit, CapabilityState::Verified),
            (
                "online_pa",
                ModelFamily::Adaptive,
                CapabilityState::Verified,
            ),
            (
                "online_hoeffding",
                ModelFamily::Adaptive,
                CapabilityState::Verified,
            ),
            (
                "isolation_forest",
                ModelFamily::Anomaly,
                CapabilityState::Verified,
            ),
            ("dqn", ModelFamily::Rl, CapabilityState::Verified),
            ("transformer", ModelFamily::Deep, CapabilityState::Verified),
            ("nbeats", ModelFamily::Deep, CapabilityState::Verified),
            ("tide", ModelFamily::Deep, CapabilityState::Verified),
            ("tabnet", ModelFamily::Deep, CapabilityState::Verified),
            ("kan", ModelFamily::Deep, CapabilityState::Verified),
            ("patchtst", ModelFamily::Deep, CapabilityState::Verified),
            ("timesnet", ModelFamily::Deep, CapabilityState::Verified),
            ("nbeatsx_nf", ModelFamily::Deep, CapabilityState::Verified),
            ("tide_nf", ModelFamily::Deep, CapabilityState::Verified),
            (
                "swarm_forecaster",
                ModelFamily::Forecasting,
                CapabilityState::Verified,
            ),
            (
                "neuro_evo",
                ModelFamily::Evolutionary,
                CapabilityState::Implemented,
            ),
            ("neat", ModelFamily::Evolutionary, CapabilityState::Verified),
        ];

        for (name, family, state) in expectations {
            let capability = get_model_capability(name)
                .unwrap_or_else(|| panic!("missing capability for {name}"));

            assert_eq!(capability.name, name);
            assert_eq!(capability.family, family);
            assert_eq!(capability.state, state);
        }
    }

    #[test]
    fn all_default_configured_model_names_have_capabilities() {
        let settings = Settings::default();

        for name in &settings.models.ml_models {
            assert!(
                get_model_capability(name).is_some(),
                "configured model {name} should resolve to a capability"
            );
        }
    }

    #[test]
    fn known_model_names_are_unique_and_resolve() {
        let mut seen = std::collections::HashSet::new();
        for name in KNOWN_MODEL_NAMES {
            assert!(seen.insert(*name), "duplicate known model name {name}");
            assert!(
                get_model_capability(name).is_some(),
                "known model name {name} should resolve"
            );
        }
    }

    #[test]
    fn unknown_model_names_do_not_resolve() {
        assert!(get_model_capability("nonexistent").is_none());
        assert!(get_model_capability("").is_none());
    }

    #[test]
    fn gpu_capability_table_tracks_the_compiled_features() {
        // The table must report what THIS build actually contains, not what
        // some other build might. Pinning it against `cfg!` here is what makes
        // it usable as a device self-report.
        for name in KNOWN_MODEL_NAMES {
            let Some(capability) = get_model_capability(name) else {
                continue;
            };
            let supports = supports_gpu_for_model(&capability.name, capability.family);
            let prefers = prefers_gpu_for_model(&capability.name, capability.family);
            assert_eq!(
                supports, prefers,
                "{name}: supports/prefers diverged — a model that can use the card but \
                 should not, or vice versa, needs its own documented reason"
            );
        }

        assert_eq!(
            supports_gpu_for_model("lightgbm", ModelFamily::Tree),
            cfg!(feature = "lightgbm-gpu")
        );
        assert_eq!(
            supports_gpu_for_model("dqn", ModelFamily::Rl),
            cfg!(feature = "reinforcement-learning-cuda")
        );
        assert_eq!(
            supports_gpu_for_model("mlp", ModelFamily::Deep),
            cfg!(feature = "burn-wgpu-backend")
        );
        // Meta / Adaptive / Anomaly families have no GPU lane in any build.
        assert!(!supports_gpu_for_model("logistic", ModelFamily::Meta));
        assert!(!supports_gpu_for_model(
            "isolation_forest",
            ModelFamily::Anomaly
        ));
    }
}
