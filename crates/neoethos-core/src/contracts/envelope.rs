use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use super::{ArtifactContractError, ArtifactKind, ArtifactProvenance, LiveExecutionContract};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEnvelope<T> {
    pub provenance: ArtifactProvenance,
    pub payload: T,
}

impl<T> ArtifactEnvelope<T> {
    pub fn new(provenance: ArtifactProvenance, payload: T) -> Result<Self, ArtifactContractError> {
        provenance.validate()?;
        Ok(Self {
            provenance,
            payload,
        })
    }

    pub fn require_kind(&self, expected: ArtifactKind) -> Result<(), ArtifactContractError> {
        require_artifact_kind(self.provenance.artifact_kind, expected)
    }

    pub fn require_live_ready(&self) -> Result<(), ArtifactContractError> {
        require_live_ready_provenance(&self.provenance)
    }

    pub fn require_live_execution_contract(
        &self,
        contract: &LiveExecutionContract,
    ) -> Result<(), ArtifactContractError> {
        contract.validate_provenance(&self.provenance)
    }
}

pub trait ArtifactContractKind {
    const KIND: ArtifactKind;
    const REQUIRES_LIVE_READY: bool = false;

    fn contract_name() -> &'static str;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrainingModelArtifactContract;

impl ArtifactContractKind for TrainingModelArtifactContract {
    const KIND: ArtifactKind = ArtifactKind::TrainingModel;

    fn contract_name() -> &'static str {
        "training_model_artifact"
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SearchCheckpointArtifactContract;

impl ArtifactContractKind for SearchCheckpointArtifactContract {
    const KIND: ArtifactKind = ArtifactKind::SearchCheckpoint;

    fn contract_name() -> &'static str {
        "search_checkpoint_artifact"
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortfolioSelectionArtifactContract;

impl ArtifactContractKind for PortfolioSelectionArtifactContract {
    const KIND: ArtifactKind = ArtifactKind::PortfolioSelection;

    fn contract_name() -> &'static str {
        "portfolio_selection_artifact"
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRuntimeArtifactContract;

impl ArtifactContractKind for ModelRuntimeArtifactContract {
    const KIND: ArtifactKind = ArtifactKind::ModelRuntime;

    fn contract_name() -> &'static str {
        "model_runtime_artifact"
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LiveReadyStrategyArtifactContract;

impl ArtifactContractKind for LiveReadyStrategyArtifactContract {
    const KIND: ArtifactKind = ArtifactKind::LiveReadyStrategy;
    const REQUIRES_LIVE_READY: bool = true;

    fn contract_name() -> &'static str {
        "live_ready_strategy_artifact"
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedArtifactEnvelope<C, T>
where
    C: ArtifactContractKind,
{
    pub provenance: ArtifactProvenance,
    pub payload: T,
    #[serde(skip)]
    contract: PhantomData<C>,
}

impl<C, T> TypedArtifactEnvelope<C, T>
where
    C: ArtifactContractKind,
{
    pub fn new(provenance: ArtifactProvenance, payload: T) -> Result<Self, ArtifactContractError> {
        provenance.validate()?;
        require_artifact_kind(provenance.artifact_kind, C::KIND)?;
        if C::REQUIRES_LIVE_READY {
            require_live_ready_provenance(&provenance)?;
        }
        Ok(Self {
            provenance,
            payload,
            contract: PhantomData,
        })
    }

    pub fn contract_kind(&self) -> ArtifactKind {
        C::KIND
    }

    pub fn contract_name(&self) -> &'static str {
        C::contract_name()
    }

    pub fn require_live_ready(&self) -> Result<(), ArtifactContractError> {
        require_live_ready_provenance(&self.provenance)
    }

    pub fn require_live_execution_contract(
        &self,
        contract: &LiveExecutionContract,
    ) -> Result<(), ArtifactContractError> {
        contract.validate_provenance(&self.provenance)
    }

    pub fn into_untyped(self) -> ArtifactEnvelope<T> {
        ArtifactEnvelope {
            provenance: self.provenance,
            payload: self.payload,
        }
    }
}

pub type TrainingModelArtifact<T> = TypedArtifactEnvelope<TrainingModelArtifactContract, T>;
pub type SearchCheckpointArtifact<T> = TypedArtifactEnvelope<SearchCheckpointArtifactContract, T>;
pub type PortfolioSelectionArtifact<T> =
    TypedArtifactEnvelope<PortfolioSelectionArtifactContract, T>;
pub type ModelRuntimeArtifact<T> = TypedArtifactEnvelope<ModelRuntimeArtifactContract, T>;
pub type LiveReadyStrategyArtifact<T> = TypedArtifactEnvelope<LiveReadyStrategyArtifactContract, T>;

pub(super) fn require_artifact_kind(
    actual: ArtifactKind,
    expected: ArtifactKind,
) -> Result<(), ArtifactContractError> {
    if actual != expected {
        return Err(ArtifactContractError::WrongArtifactKind { actual, expected });
    }
    Ok(())
}

pub(super) fn require_live_ready_provenance(
    provenance: &ArtifactProvenance,
) -> Result<(), ArtifactContractError> {
    provenance.validate()?;
    if !provenance.artifact_kind.is_live_eligible() {
        return Err(ArtifactContractError::LiveRejectedArtifactKind(
            provenance.artifact_kind,
        ));
    }
    if !provenance.runtime_mode.is_live_safe() || provenance.backend_kind.is_degraded() {
        return Err(ArtifactContractError::LiveRejectedRuntimeMode {
            mode: provenance.runtime_mode,
            backend: provenance.backend_kind,
        });
    }
    Ok(())
}
