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

/// Every named production surface reachable through the normal `gpu-cuda`
/// build whose training implementation can execute NVIDIA CUDA work.
///
/// This is a census, not a promise that every model prefers CUDA under the
/// current runtime settings. LightGBM has a separate operator opt-in and the
/// statistical models ship with an explicit CPU policy.
pub const CUDA_CAPABLE_MODEL_NAMES: &[&str] = &[
    "catboost",
    "catboost_alt",
    "conformal_gate",
    "dqn",
    "elasticnet",
    "lightgbm",
    "logistic",
    "meta_blender",
    "meta_stack",
    "neat",
    "neuro_evo",
    "probability_calibrator",
    "xgboost",
    "xgboost_dart",
    "xgboost_rf",
];

/// Resolve the capability record for a known model name.
///
/// LIVE: `runtime::dispatch::build_dispatch_plan` calls this for every model
/// in a training run.
pub fn get_model_capability(name: &str) -> Option<ModelCapability> {
    model_capability(name)
}

/// Whether this build contains a GPU code path that `name` can actually use.
///
/// The full-NVIDIA training preflight consumes this table before launching any
/// model. It reads the real Cargo features, so a build cannot advertise a CUDA
/// implementation that was not compiled. Runtime artifacts and training
/// summaries then record the requested/effective device selected by each
/// model's own strict execution boundary.
///
/// Do not add an `#[allow(dead_code)]` here if a caller disappears — either
/// wire it or delete it.
pub fn supports_gpu_for_model(name: &str, family: ModelFamily) -> bool {
    match name {
        "lightgbm" => cfg!(feature = "lightgbm-gpu"),
        "xgboost"
        | "xgboost_rf"
        | "xgboost_dart"
        | "meta_blender"
        | "probability_calibrator"
        | "conformal_gate"
        | "meta_stack" => cfg!(feature = "xgboost"),
        "catboost" | "catboost_alt" => cfg!(feature = "catboost"),
        "dqn" => cfg!(feature = "reinforcement-learning-cuda"),
        "neat" | "neuro_evo" => cfg!(feature = "neuro-evolution-gpu"),
        "logistic" | "elasticnet" => cfg!(feature = "statistical-gpu"),
        // SAC runs on the selected Burn GPU backend (like Deep/Exit), not the
        // rlkit/Candle DQN path.
        "sac" => cfg!(any(
            feature = "burn-wgpu-backend",
            feature = "burn-cuda-backend"
        )),
        _ => match family {
            ModelFamily::Deep | ModelFamily::Exit => cfg!(any(
                feature = "burn-wgpu-backend",
                feature = "burn-cuda-backend"
            )),
            _ => false,
        },
    }
}

/// Whether the GPU path is the one this model SHOULD take when a card is
/// present under the installed runtime policy.
pub fn prefers_gpu_for_model(name: &str, family: ModelFamily) -> bool {
    match name {
        "lightgbm" => {
            cfg!(feature = "lightgbm-gpu")
                && crate::tree_models::config::lightgbm_gpu_allowed()
                && !matches!(
                    crate::common::parse_cuda_device_policy(
                        &crate::tree_models::config::current_tree_runtime().device
                    ),
                    Ok(crate::common::CudaDevicePolicy::Cpu) | Err(_)
                )
        }
        "xgboost"
        | "xgboost_rf"
        | "xgboost_dart"
        | "meta_blender"
        | "probability_calibrator"
        | "conformal_gate"
        | "meta_stack" => cfg!(feature = "xgboost"),
        "catboost" | "catboost_alt" => cfg!(feature = "catboost"),
        "dqn" => cfg!(feature = "reinforcement-learning-cuda"),
        "neat" | "neuro_evo" => cfg!(feature = "neuro-evolution-gpu"),
        "logistic" | "elasticnet" => {
            cfg!(feature = "statistical-gpu")
                && !matches!(
                    crate::common::parse_cuda_device_policy(
                        crate::statistical::common::configured_statistical_device()
                    ),
                    Ok(crate::common::CudaDevicePolicy::Cpu) | Err(_)
                )
        }
        // SAC runs on the selected Burn GPU backend (like Deep/Exit), not the
        // rlkit/Candle DQN path.
        "sac" => cfg!(any(
            feature = "burn-wgpu-backend",
            feature = "burn-cuda-backend"
        )),
        _ => match family {
            ModelFamily::Deep | ModelFamily::Exit => cfg!(any(
                feature = "burn-wgpu-backend",
                feature = "burn-cuda-backend"
            )),
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
            assert!(
                !prefers || supports,
                "{name}: a model cannot prefer a GPU path absent from this build"
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
        for (name, family) in [
            ("mlp", ModelFamily::Deep),
            ("exit_agent", ModelFamily::Exit),
            ("sac", ModelFamily::Rl),
        ] {
            let burn_gpu_compiled = cfg!(any(
                feature = "burn-wgpu-backend",
                feature = "burn-cuda-backend"
            ));
            assert_eq!(
                supports_gpu_for_model(name, family),
                burn_gpu_compiled,
                "{name} must report every compiled Burn GPU backend"
            );
            assert_eq!(
                prefers_gpu_for_model(name, family),
                burn_gpu_compiled,
                "{name} must prefer the explicitly compiled Burn GPU backend"
            );
        }
        assert_eq!(
            supports_gpu_for_model("logistic", ModelFamily::Meta),
            cfg!(feature = "statistical-gpu")
        );
        assert!(
            !prefers_gpu_for_model("logistic", ModelFamily::Meta),
            "the shipped statistical_device=cpu policy must remain an explicit opt-out"
        );
        assert!(
            !prefers_gpu_for_model("lightgbm", ModelFamily::Tree),
            "the shipped lightgbm_gpu=false gate must remain an explicit opt-out"
        );
        // Adaptive / Anomaly families have no GPU lane in any build.
        assert!(!supports_gpu_for_model(
            "isolation_forest",
            ModelFamily::Anomaly
        ));

        for name in CUDA_CAPABLE_MODEL_NAMES {
            let capability = get_model_capability(name)
                .unwrap_or_else(|| panic!("CUDA census name `{name}` lacks a capability"));
            assert_eq!(
                supports_gpu_for_model(name, capability.family),
                match *name {
                    "lightgbm" => cfg!(feature = "lightgbm-gpu"),
                    "catboost" | "catboost_alt" => cfg!(feature = "catboost"),
                    "dqn" => cfg!(feature = "reinforcement-learning-cuda"),
                    "neat" | "neuro_evo" => cfg!(feature = "neuro-evolution-gpu"),
                    "logistic" | "elasticnet" => cfg!(feature = "statistical-gpu"),
                    _ => cfg!(feature = "xgboost"),
                },
                "CUDA census feature mapping drifted for {name}"
            );
        }
    }
}
