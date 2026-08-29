//! Typed GPU capability inventory and fail-fast pipeline preflight.

use crate::backend::EvaluationBackend;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PipelineStage {
    FeaturePreparation,
    GaGenerationSelection,
    PopulationEvaluation,
    SignalAndMinTradeFilter,
    QualityScreen,
    MonteCarlo,
    PropFirmWindow,
    CandidateCorrelation,
    WalkForward,
    Cpcv,
    Pbo,
    RobustnessPermutationPlateau,
    RiskDiagnostics,
    CanonicalReplay,
    ForwardTailReplay,
    SurvivorRanking,
}

impl PipelineStage {
    pub const FULL_DISCOVERY: [Self; 16] = [
        Self::FeaturePreparation,
        Self::GaGenerationSelection,
        Self::PopulationEvaluation,
        Self::SignalAndMinTradeFilter,
        Self::QualityScreen,
        Self::MonteCarlo,
        Self::PropFirmWindow,
        Self::CandidateCorrelation,
        Self::WalkForward,
        Self::Cpcv,
        Self::Pbo,
        Self::RobustnessPermutationPlateau,
        Self::RiskDiagnostics,
        Self::CanonicalReplay,
        Self::ForwardTailReplay,
        Self::SurvivorRanking,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageGpuCapability {
    /// The stage has a path that can complete with zero candidate-dependent CPU
    /// execution and no hidden CPU fallback.
    StrictGpu,
    /// A GPU implementation exists, but the current production path still
    /// depends on a CPU lane, CPU completion, or unconditional CPU fallback.
    HybridOnly,
    /// Candidate-dependent implementation is currently CPU-only.
    CpuOnly,
    /// The selected engine/backend cannot implement this requested stage or
    /// strategy feature.
    Unsupported,
}

impl StageGpuCapability {
    pub fn satisfies_strict_gpu(self) -> bool {
        self == Self::StrictGpu
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageCapability {
    pub stage: PipelineStage,
    pub capability: StageGpuCapability,
    pub detail: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCapabilityManifest {
    entries: Vec<StageCapability>,
}

impl GpuCapabilityManifest {
    pub fn new(entries: impl IntoIterator<Item = StageCapability>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    /// Accurate inventory of the repository at Stage 1 start. Existing GPU
    /// kernels are marked `HybridOnly` when the caller can still execute a CPU
    /// lane or silently recompute on CPU.
    pub fn stage1_baseline() -> Self {
        use PipelineStage as S;
        use StageGpuCapability as C;
        Self::new([
            StageCapability {
                stage: S::FeaturePreparation,
                capability: C::CpuOnly,
                detail: "feature/SMC preparation remains host-side",
            },
            StageCapability {
                stage: S::GaGenerationSelection,
                capability: C::CpuOnly,
                detail: "generation, selection, mutation, crossover, dedup and archive are CPU-side",
            },
            StageCapability {
                stage: S::PopulationEvaluation,
                capability: C::StrictGpu,
                detail: "gpu_required routes the complete logical population through CubeCL with bounded GPU-only rebatching and no CPU fallback",
            },
            StageCapability {
                stage: S::SignalAndMinTradeFilter,
                capability: C::CpuOnly,
                detail: "post-GA signal synthesis and minimum-trade filtering use CPU/Rayon",
            },
            StageCapability {
                stage: S::QualityScreen,
                capability: C::CpuOnly,
                detail: "trade simulation and quality/regime diagnostics remain CPU-side",
            },
            StageCapability {
                stage: S::MonteCarlo,
                capability: C::HybridOnly,
                detail: "GPU population evaluation exists but failure and surrounding reductions remain CPU-routed",
            },
            StageCapability {
                stage: S::PropFirmWindow,
                capability: C::CpuOnly,
                detail: "prop-firm rolling-window simulation is CPU-side",
            },
            StageCapability {
                stage: S::CandidateCorrelation,
                capability: C::CpuOnly,
                detail: "candidate Pearson/Spearman pruning is CPU-side",
            },
            StageCapability {
                stage: S::WalkForward,
                capability: C::HybridOnly,
                detail: "GPU metric evaluation exists but host gathering and CPU diagnostics remain",
            },
            StageCapability {
                stage: S::Cpcv,
                capability: C::HybridOnly,
                detail: "GPU fold evaluation exists but host gathers/reductions remain",
            },
            StageCapability {
                stage: S::Pbo,
                capability: C::CpuOnly,
                detail: "PBO ranking and fold selection remain host-side",
            },
            StageCapability {
                stage: S::RobustnessPermutationPlateau,
                capability: C::CpuOnly,
                detail: "permutation and parameter plateau checks use CPU/Rayon",
            },
            StageCapability {
                stage: S::RiskDiagnostics,
                capability: C::CpuOnly,
                detail: "canonical risk diagnostics repeat CPU simulation",
            },
            StageCapability {
                stage: S::CanonicalReplay,
                capability: C::CpuOnly,
                detail: "canonical artifact replay is CPU-side",
            },
            StageCapability {
                stage: S::ForwardTailReplay,
                capability: C::CpuOnly,
                detail: "forward-tail replay is CPU-side",
            },
            StageCapability {
                stage: S::SurvivorRanking,
                capability: C::CpuOnly,
                detail: "final ranking/selection remains host-side",
            },
        ])
    }

    pub fn entries(&self) -> &[StageCapability] {
        &self.entries
    }

    pub fn capability(&self, stage: PipelineStage) -> Option<&StageCapability> {
        self.entries.iter().find(|entry| entry.stage == stage)
    }

    pub fn with_capability(
        mut self,
        stage: PipelineStage,
        capability: StageGpuCapability,
        detail: &'static str,
    ) -> Self {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.stage == stage) {
            entry.capability = capability;
            entry.detail = detail;
        } else {
            self.entries.push(StageCapability {
                stage,
                capability,
                detail,
            });
        }
        self
    }
}

pub fn gpu_pipeline_preflight(
    _backend: EvaluationBackend,
    manifest: &GpuCapabilityManifest,
    requested_stages: &[PipelineStage],
) -> Result<(), GpuPipelinePreflightError> {
    let mut unsupported = Vec::new();
    for stage in requested_stages {
        match manifest.capability(*stage) {
            Some(capability) if capability.capability.satisfies_strict_gpu() => {}
            Some(capability) => unsupported.push(capability.clone()),
            None => unsupported.push(StageCapability {
                stage: *stage,
                capability: StageGpuCapability::Unsupported,
                detail: "stage missing from the capability manifest",
            }),
        }
    }

    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(GpuPipelinePreflightError { unsupported })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPipelinePreflightError {
    pub unsupported: Vec<StageCapability>,
}

impl fmt::Display for GpuPipelinePreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GPU-required pipeline preflight failed: {} requested stages are not strict-GPU capable:",
            self.unsupported.len()
        )?;
        for item in &self.unsupported {
            writeln!(
                f,
                "- {:?}: {:?} ({})",
                item.stage, item.capability, item.detail
            )?;
        }
        Ok(())
    }
}

impl Error for GpuPipelinePreflightError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_backend_cannot_bypass_strict_preflight() {
        let error = gpu_pipeline_preflight(
            EvaluationBackend::AUTO,
            &GpuCapabilityManifest::stage1_baseline(),
            &PipelineStage::FULL_DISCOVERY,
        )
        .unwrap_err();
        assert!(!error.unsupported.is_empty());
    }

    #[test]
    fn strict_full_pipeline_fails_before_work_starts() {
        let error = gpu_pipeline_preflight(
            EvaluationBackend::GPU_REQUIRED,
            &GpuCapabilityManifest::stage1_baseline(),
            &PipelineStage::FULL_DISCOVERY,
        )
        .unwrap_err();

        assert!(!error.unsupported.is_empty());
        assert!(error.unsupported.iter().any(|item| {
            item.stage == PipelineStage::QualityScreen
                && item.capability == StageGpuCapability::CpuOnly
        }));
        assert!(
            !error
                .unsupported
                .iter()
                .any(|item| item.stage == PipelineStage::PopulationEvaluation)
        );
    }

    #[test]
    fn engine_only_preflight_passes_once_engine_declares_strict_capability() {
        let manifest = GpuCapabilityManifest::stage1_baseline().with_capability(
            PipelineStage::PopulationEvaluation,
            StageGpuCapability::StrictGpu,
            "test engine is device-resident and has CPU fallback disabled",
        );
        gpu_pipeline_preflight(
            EvaluationBackend::GPU_REQUIRED,
            &manifest,
            &[PipelineStage::PopulationEvaluation],
        )
        .unwrap();
    }

    #[test]
    fn missing_manifest_entry_is_reported_as_unsupported() {
        let manifest = GpuCapabilityManifest::new([]);
        let error = gpu_pipeline_preflight(
            EvaluationBackend::GPU_REQUIRED,
            &manifest,
            &[PipelineStage::Pbo],
        )
        .unwrap_err();
        assert_eq!(error.unsupported.len(), 1);
        assert_eq!(
            error.unsupported[0].capability,
            StageGpuCapability::Unsupported
        );
    }
}
