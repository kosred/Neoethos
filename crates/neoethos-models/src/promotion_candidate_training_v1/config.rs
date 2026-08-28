use super::{
    PromotionCandidateTrainingRefusalCodeV1, PromotionCandidateTrainingRefusalV1,
    SCHEMA_VERSION_V1, hash_json_v1, refusal_v1, validate_model_names_v1,
};
use crate::TrainingOrchestrator;
use neoethos_core::{HardwareExecutionPlan, Settings};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const CONFIG_IDENTITY_SCHEMA_V1: &str = "neoethos.promotion-candidate-training-config.v1";
const RUNTIME_CONFIG_IDENTITY_DOMAIN_V1: &[u8] =
    b"neoethos.promotion-candidate-training-runtime-config.v1\0";
const MODEL_CONFIG_IDENTITY_DOMAIN_V1: &[u8] =
    b"neoethos.promotion-candidate-training-model-config.v1\0";

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionCandidateTrainingConfigIdentityV1 {
    schema: String,
    version: u16,
    pub(super) runtime_config_sha256: String,
    pub(super) model_config_sha256: String,
    pub(super) planned_models: Vec<String>,
    #[serde(default)]
    sealed_hardware_plan_v1: Option<HardwareExecutionPlan>,
}

impl PromotionCandidateTrainingConfigIdentityV1 {
    pub fn checked_new(
        runtime_config_sha256: String,
        model_config_sha256: String,
        planned_models: Vec<String>,
    ) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        Self::checked_new_inner(
            runtime_config_sha256,
            model_config_sha256,
            planned_models,
            None,
        )
    }

    fn checked_new_with_sealed_hardware_plan_v1(
        runtime_config_sha256: String,
        model_config_sha256: String,
        planned_models: Vec<String>,
        sealed_hardware_plan_v1: HardwareExecutionPlan,
    ) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        Self::checked_new_inner(
            runtime_config_sha256,
            model_config_sha256,
            planned_models,
            Some(sealed_hardware_plan_v1),
        )
    }

    fn checked_new_inner(
        runtime_config_sha256: String,
        model_config_sha256: String,
        planned_models: Vec<String>,
        sealed_hardware_plan_v1: Option<HardwareExecutionPlan>,
    ) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        let value = Self {
            schema: CONFIG_IDENTITY_SCHEMA_V1.to_owned(),
            version: SCHEMA_VERSION_V1,
            runtime_config_sha256,
            model_config_sha256,
            planned_models,
            sealed_hardware_plan_v1,
        };
        value.validate()?;
        Ok(value)
    }

    pub(super) fn validate(&self) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        if self.schema != CONFIG_IDENTITY_SCHEMA_V1 || self.version != SCHEMA_VERSION_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "unsupported training-config identity schema/version",
            ));
        }
        super::validate_sha256_v1(&self.runtime_config_sha256, "runtime config")?;
        super::validate_sha256_v1(&self.model_config_sha256, "model config")?;
        validate_model_names_v1(&self.planned_models)
    }

    pub fn runtime_config_sha256(&self) -> &str {
        &self.runtime_config_sha256
    }

    pub fn model_config_sha256(&self) -> &str {
        &self.model_config_sha256
    }

    pub fn planned_models(&self) -> &[String] {
        &self.planned_models
    }

    pub fn sealed_hardware_plan_v1(&self) -> Option<&HardwareExecutionPlan> {
        self.sealed_hardware_plan_v1.as_ref()
    }
}

pub fn resolve_promotion_candidate_training_config_identity_v1(
    settings: &Settings,
) -> Result<PromotionCandidateTrainingConfigIdentityV1, PromotionCandidateTrainingRefusalV1> {
    resolve_with_orchestrator_v1(
        settings,
        TrainingOrchestrator::new(settings.clone(), PathBuf::new()),
    )
}

pub(super) fn resolve_promotion_candidate_training_config_with_plan_v1(
    settings: &Settings,
    sealed_hardware_plan_v1: &HardwareExecutionPlan,
) -> Result<PromotionCandidateTrainingConfigIdentityV1, PromotionCandidateTrainingRefusalV1> {
    resolve_with_orchestrator_v1(
        settings,
        TrainingOrchestrator::new(settings.clone(), PathBuf::new())
            .with_sealed_hardware_plan_v1(sealed_hardware_plan_v1.clone()),
    )
}

fn resolve_with_orchestrator_v1(
    settings: &Settings,
    orchestrator: TrainingOrchestrator,
) -> Result<PromotionCandidateTrainingConfigIdentityV1, PromotionCandidateTrainingRefusalV1> {
    let (planned_models, hardware_plan, model_plan) = orchestrator
        .promotion_candidate_training_plan_material_v1()
        .map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch,
                format!("resolve effective configured model plan: {error}"),
            )
        })?;
    let runtime_config_sha256 = hash_json_v1(
        RUNTIME_CONFIG_IDENTITY_DOMAIN_V1,
        &serde_json::json!({
            "system_config": &settings.system,
            "hardware_execution_plan": &hardware_plan,
        }),
        PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch,
    )?;
    let model_config_sha256 = hash_json_v1(
        MODEL_CONFIG_IDENTITY_DOMAIN_V1,
        &model_plan,
        PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch,
    )?;
    PromotionCandidateTrainingConfigIdentityV1::checked_new_with_sealed_hardware_plan_v1(
        runtime_config_sha256,
        model_config_sha256,
        planned_models,
        hardware_plan,
    )
}
