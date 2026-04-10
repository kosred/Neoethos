use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    Tree,
    Deep,
    Forecasting,
    Meta,
    Evolutionary,
    Exit,
    Adaptive,
    Anomaly,
    Rl,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Tree => "tree",
            Self::Deep => "deep",
            Self::Forecasting => "forecasting",
            Self::Meta => "meta",
            Self::Evolutionary => "evolutionary",
            Self::Exit => "exit",
            Self::Adaptive => "adaptive",
            Self::Anomaly => "anomaly",
            Self::Rl => "rl",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityState {
    Planned,
    Implemented,
    Verified,
}

impl fmt::Display for CapabilityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Planned => "planned",
            Self::Implemented => "implemented",
            Self::Verified => "verified",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapability {
    pub name: String,
    pub family: ModelFamily,
    pub state: CapabilityState,
}

impl ModelCapability {
    pub fn new(name: impl Into<String>, family: ModelFamily, state: CapabilityState) -> Self {
        let name = name.into();
        assert!(
            !name.trim().is_empty(),
            "model capability requires a non-empty name"
        );
        Self {
            name,
            family,
            state,
        }
    }
}

pub const KNOWN_MODEL_NAMES: &[&str] = &[
    "lightgbm",
    "xgboost",
    "xgboost_rf",
    "xgboost_dart",
    "catboost",
    "catboost_alt",
    "sklears_tree",
    "mlp",
    "nbeats",
    "tide",
    "tabnet",
    "kan",
    "transformer",
    "patchtst",
    "timesnet",
    "nbeatsx_nf",
    "tide_nf",
    "swarm_forecaster",
    "elasticnet",
    "bayes_logit",
    "meta_blender",
    "probability_calibrator",
    "conformal_gate",
    "meta_stack",
    "genetic",
    "neuro_evo",
    "neat",
    "exit_agent",
    "online_pa",
    "online_hoeffding",
    "isolation_forest",
    "dqn",
];

pub fn model_capability(name: &str) -> Option<ModelCapability> {
    let is_transformer_replica = name
        .strip_prefix("transformer_")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()));

    let capability = match name {
        // Tree models
        "lightgbm" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Implemented),
        "xgboost" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Implemented),
        "xgboost_rf" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Implemented),
        "xgboost_dart" => {
            ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Implemented)
        }
        "catboost" => ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Implemented),
        "catboost_alt" => {
            ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Implemented)
        }
        "sklears_tree" => {
            ModelCapability::new(name, ModelFamily::Tree, CapabilityState::Implemented)
        }

        // Deep models
        "mlp" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "nbeats" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "tide" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "tabnet" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "kan" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "transformer" => {
            ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented)
        }
        _ if is_transformer_replica => {
            ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented)
        }
        "patchtst" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "timesnet" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "nbeatsx_nf" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),
        "tide_nf" => ModelCapability::new(name, ModelFamily::Deep, CapabilityState::Implemented),

        // Forecasting models
        "swarm_forecaster" => {
            ModelCapability::new(name, ModelFamily::Forecasting, CapabilityState::Implemented)
        }

        // Meta models
        "elasticnet" => ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Implemented),
        "bayes_logit" => {
            ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Implemented)
        }
        "meta_blender" => {
            ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Implemented)
        }
        "probability_calibrator" => {
            ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Implemented)
        }
        "conformal_gate" => {
            ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Implemented)
        }
        "meta_stack" => ModelCapability::new(name, ModelFamily::Meta, CapabilityState::Implemented),

        // Evolutionary models
        "genetic" => ModelCapability::new(
            name,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
        ),
        "neuro_evo" => ModelCapability::new(
            name,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
        ),
        "neat" => ModelCapability::new(
            name,
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
        ),

        // Exit models
        "exit_agent" => ModelCapability::new(name, ModelFamily::Exit, CapabilityState::Implemented),

        // Adaptive models
        "online_pa" => {
            ModelCapability::new(name, ModelFamily::Adaptive, CapabilityState::Implemented)
        }
        "online_hoeffding" => {
            ModelCapability::new(name, ModelFamily::Adaptive, CapabilityState::Implemented)
        }

        // Anomaly models
        "isolation_forest" => {
            ModelCapability::new(name, ModelFamily::Anomaly, CapabilityState::Implemented)
        }

        // Reinforcement-learning models
        "dqn" => ModelCapability::new(name, ModelFamily::Rl, CapabilityState::Implemented),

        _ => return None,
    };

    Some(capability)
}

#[cfg(test)]
mod tests {
    use super::{model_capability, CapabilityState, ModelCapability, ModelFamily};
    use std::collections::HashSet;

    #[test]
    fn model_family_has_expected_variants() {
        assert_eq!(ModelFamily::Tree.to_string(), "tree");
        assert_eq!(ModelFamily::Deep.to_string(), "deep");
        assert_eq!(ModelFamily::Forecasting.to_string(), "forecasting");
        assert_eq!(ModelFamily::Meta.to_string(), "meta");
        assert_eq!(ModelFamily::Evolutionary.to_string(), "evolutionary");
        assert_eq!(ModelFamily::Exit.to_string(), "exit");
        assert_eq!(ModelFamily::Adaptive.to_string(), "adaptive");
        assert_eq!(ModelFamily::Anomaly.to_string(), "anomaly");
        assert_eq!(ModelFamily::Rl.to_string(), "rl");
    }

    #[test]
    fn capability_state_has_expected_variants() {
        assert_eq!(CapabilityState::Planned.to_string(), "planned");
        assert_eq!(CapabilityState::Implemented.to_string(), "implemented");
        assert_eq!(CapabilityState::Verified.to_string(), "verified");
    }

    #[test]
    fn model_capability_can_be_constructed() {
        let capability =
            ModelCapability::new("lightgbm", ModelFamily::Tree, CapabilityState::Planned);

        assert_eq!(capability.name, "lightgbm");
        assert_eq!(capability.family, ModelFamily::Tree);
        assert_eq!(capability.state, CapabilityState::Planned);
    }

    #[test]
    fn known_configured_model_names_resolve_to_capabilities() {
        let capability = model_capability("transformer").expect("transformer should resolve");
        assert_eq!(capability.family, ModelFamily::Deep);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("sklears_tree").expect("sklears_tree should resolve");
        assert_eq!(capability.family, ModelFamily::Tree);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("patchtst").expect("patchtst should resolve");
        assert_eq!(capability.family, ModelFamily::Deep);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability =
            model_capability("swarm_forecaster").expect("swarm_forecaster should resolve");
        assert_eq!(capability.family, ModelFamily::Forecasting);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("online_pa").expect("online_pa should resolve");
        assert_eq!(capability.family, ModelFamily::Adaptive);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("elasticnet").expect("elasticnet should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("meta_blender").expect("meta_blender should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("probability_calibrator")
            .expect("probability_calibrator should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("conformal_gate").expect("conformal_gate should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("meta_stack").expect("meta_stack should resolve");
        assert_eq!(capability.family, ModelFamily::Meta);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("genetic").expect("genetic should resolve");
        assert_eq!(capability.family, ModelFamily::Evolutionary);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("neuro_evo").expect("neuro_evo should resolve");
        assert_eq!(capability.family, ModelFamily::Evolutionary);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("neat").expect("neat should resolve");
        assert_eq!(capability.family, ModelFamily::Evolutionary);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("exit_agent").expect("exit_agent should resolve");
        assert_eq!(capability.family, ModelFamily::Exit);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability =
            model_capability("isolation_forest").expect("isolation_forest should resolve");
        assert_eq!(capability.family, ModelFamily::Anomaly);
        assert_eq!(capability.state, CapabilityState::Implemented);

        let capability = model_capability("dqn").expect("dqn should resolve");
        assert_eq!(capability.family, ModelFamily::Rl);
        assert_eq!(capability.state, CapabilityState::Implemented);
    }

    #[test]
    fn known_model_names_are_unique_and_resolve() {
        let mut seen = HashSet::new();
        for name in KNOWN_MODEL_NAMES {
            assert!(seen.insert(*name), "duplicate known model name {name}");
            assert!(
                model_capability(name).is_some(),
                "known model name {name} should resolve"
            );
        }
    }

    #[test]
    #[should_panic(expected = "non-empty name")]
    fn model_capability_new_rejects_blank_name() {
        let _ = ModelCapability::new("   ", ModelFamily::Tree, CapabilityState::Implemented);
    }
}
