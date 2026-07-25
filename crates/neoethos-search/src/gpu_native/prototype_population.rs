//! Shared, pre-execution workload and eligibility contract for Prototypes B/C.

use crate::gpu_native::prototype_a::{
    PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeARebatchPlan, PrototypeAScenarioUpload,
    PrototypeAUploadError,
};
use crate::gpu_native::prototype_bc::PrototypeKind;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrototypePopulationWorkload {
    pub dataset: PrototypeADatasetUpload,
    pub genes: PrototypeAGeneUpload,
    pub scenarios: PrototypeAScenarioUpload,
    pub requirements: PrototypeBcRequirements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrototypeBcRequirements {
    pub prop_firm_state: PropFirmRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropFirmRequirement {
    NotRequested,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrototypeBcIneligibilityReason {
    GlobalTrailingOrBreakEven,
    PropFirmStateRequired,
    NonFiniteStaticLevels,
    AdaptiveAtEntryUnavailable,
    UnsupportedScenario,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonBcEligibility {
    pub supported_candidate_ids: Vec<u64>,
    pub supported_gene_indices: Vec<usize>,
    pub unsupported: BTreeMap<PrototypeBcIneligibilityReason, Vec<u64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonBcCoverage {
    pub total_candidates: usize,
    pub supported_candidates: usize,
    pub unsupported_candidates: usize,
}

impl CommonBcEligibility {
    pub fn coverage(&self) -> CommonBcCoverage {
        let unsupported_ids = self
            .unsupported
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        CommonBcCoverage {
            total_candidates: self.supported_candidate_ids.len() + unsupported_ids.len(),
            supported_candidates: self.supported_candidate_ids.len(),
            unsupported_candidates: unsupported_ids.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrototypePopulationError {
    Upload(PrototypeAUploadError),
    DuplicateCandidateId(u64),
    EmptySupportedPartition,
    EligibilityIndexOutOfBounds(usize),
    EligibilityCandidateMismatch {
        index: usize,
        expected: u64,
        actual: u64,
    },
    DuplicateEligibilityIndex(usize),
}

impl fmt::Display for PrototypePopulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upload(error) => write!(f, "{error}"),
            Self::DuplicateCandidateId(candidate_id) => {
                write!(f, "Prototype B/C candidate ID {candidate_id} is duplicated")
            }
            Self::EmptySupportedPartition => {
                write!(f, "Prototype B/C eligibility has no supported candidates")
            }
            Self::EligibilityIndexOutOfBounds(index) => {
                write!(
                    f,
                    "Prototype B/C eligibility index {index} is out of bounds"
                )
            }
            Self::EligibilityCandidateMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "Prototype B/C eligibility index {index} targets candidate {actual}, expected {expected}"
            ),
            Self::DuplicateEligibilityIndex(index) => {
                write!(f, "Prototype B/C eligibility index {index} is duplicated")
            }
        }
    }
}

impl Error for PrototypePopulationError {}

impl From<PrototypeAUploadError> for PrototypePopulationError {
    fn from(error: PrototypeAUploadError) -> Self {
        Self::Upload(error)
    }
}

impl PrototypePopulationWorkload {
    pub fn from_uploads(
        dataset: PrototypeADatasetUpload,
        genes: PrototypeAGeneUpload,
        scenarios: PrototypeAScenarioUpload,
        requirements: PrototypeBcRequirements,
    ) -> Result<Self, PrototypePopulationError> {
        validate_dataset_shape(&dataset)?;
        PrototypeARebatchPlan::new(&genes, &scenarios, genes.population())?;

        let mut candidate_ids = BTreeSet::new();
        for candidate_id in genes.candidate_ids.iter().copied() {
            if !candidate_ids.insert(candidate_id) {
                return Err(PrototypePopulationError::DuplicateCandidateId(candidate_id));
            }
        }

        Ok(Self {
            dataset,
            genes,
            scenarios,
            requirements,
        })
    }

    pub fn common_bc_intersection(&self, _prototype: PrototypeKind) -> CommonBcEligibility {
        let candidate_ids = &self.genes.candidate_ids;
        let mut unsupported = BTreeMap::<PrototypeBcIneligibilityReason, Vec<u64>>::new();
        let mut full_workload_rejections = Vec::new();

        if self.dataset.settings.trailing_enabled {
            full_workload_rejections
                .push(PrototypeBcIneligibilityReason::GlobalTrailingOrBreakEven);
        }
        if self.requirements.prop_firm_state == PropFirmRequirement::Required {
            full_workload_rejections.push(PrototypeBcIneligibilityReason::PropFirmStateRequired);
        }
        if !full_workload_rejections.is_empty() {
            for reason in full_workload_rejections.iter().copied() {
                unsupported.insert(reason, candidate_ids.clone());
            }
            return CommonBcEligibility {
                supported_candidate_ids: Vec::new(),
                supported_gene_indices: Vec::new(),
                unsupported,
            };
        }

        let mut supported_candidate_ids = Vec::with_capacity(candidate_ids.len());
        let mut supported_gene_indices = Vec::with_capacity(candidate_ids.len());
        for (index, candidate_id) in candidate_ids.iter().copied().enumerate() {
            let mut candidate_supported = true;
            let stop_vol_multiplier = self.genes.stop_vol_multipliers[index];
            if stop_vol_multiplier == 0.0 {
                let stop = self.genes.stop_pips[index];
                let target = self.genes.target_pips[index];
                if !stop.is_finite() || !target.is_finite() {
                    insert_unsupported(
                        &mut unsupported,
                        PrototypeBcIneligibilityReason::NonFiniteStaticLevels,
                        candidate_id,
                    );
                    candidate_supported = false;
                }
            } else if !self.adaptive_at_entry_is_supported(stop_vol_multiplier) {
                insert_unsupported(
                    &mut unsupported,
                    PrototypeBcIneligibilityReason::AdaptiveAtEntryUnavailable,
                    candidate_id,
                );
                candidate_supported = false;
            }

            if !self.scenario_is_supported(index) {
                insert_unsupported(
                    &mut unsupported,
                    PrototypeBcIneligibilityReason::UnsupportedScenario,
                    candidate_id,
                );
                candidate_supported = false;
            }

            if candidate_supported {
                supported_candidate_ids.push(candidate_id);
                supported_gene_indices.push(index);
            }
        }

        CommonBcEligibility {
            supported_candidate_ids,
            supported_gene_indices,
            unsupported,
        }
    }

    pub fn supported_partition(
        &self,
        eligibility: &CommonBcEligibility,
    ) -> Result<Self, PrototypePopulationError> {
        if eligibility.supported_gene_indices.is_empty() {
            return Err(PrototypePopulationError::EmptySupportedPartition);
        }
        if eligibility.supported_gene_indices.len() != eligibility.supported_candidate_ids.len() {
            return Err(PrototypePopulationError::Upload(
                PrototypeAUploadError::ShapeMismatch {
                    field: "supported_candidate_ids",
                    expected: eligibility.supported_gene_indices.len(),
                    actual: eligibility.supported_candidate_ids.len(),
                },
            ));
        }

        let mut seen_indices = BTreeSet::new();
        for (&index, &candidate_id) in eligibility
            .supported_gene_indices
            .iter()
            .zip(&eligibility.supported_candidate_ids)
        {
            if !seen_indices.insert(index) {
                return Err(PrototypePopulationError::DuplicateEligibilityIndex(index));
            }
            let actual = self
                .genes
                .candidate_ids
                .get(index)
                .copied()
                .ok_or(PrototypePopulationError::EligibilityIndexOutOfBounds(index))?;
            if actual != candidate_id {
                return Err(PrototypePopulationError::EligibilityCandidateMismatch {
                    index,
                    expected: candidate_id,
                    actual,
                });
            }
        }

        let mut offsets = Vec::with_capacity(eligibility.supported_gene_indices.len() + 1);
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        offsets.push(0_i32);
        for &index in &eligibility.supported_gene_indices {
            let start = self.genes.offsets[index] as usize;
            let end = self.genes.offsets[index + 1] as usize;
            indices.extend_from_slice(&self.genes.indices[start..end]);
            weights.extend_from_slice(&self.genes.weights[start..end]);
            offsets.push(indices.len() as i32);
        }

        let selected = &eligibility.supported_gene_indices;
        let genes = PrototypeAGeneUpload {
            candidate_ids: select(&self.genes.candidate_ids, selected),
            offsets,
            indices,
            weights,
            long_thresholds: select(&self.genes.long_thresholds, selected),
            short_thresholds: select(&self.genes.short_thresholds, selected),
            stop_pips: select(&self.genes.stop_pips, selected),
            target_pips: select(&self.genes.target_pips, selected),
            stop_vol_multipliers: select(&self.genes.stop_vol_multipliers, selected),
            smc_flags: select(&self.genes.smc_flags, selected),
            smc_weights: self.genes.smc_weights,
            gate_threshold: self.genes.gate_threshold,
        };
        let scenarios = PrototypeAScenarioUpload {
            scenarios: select(&self.scenarios.scenarios, selected),
        };

        Self::from_uploads(self.dataset.clone(), genes, scenarios, self.requirements)
    }

    fn adaptive_at_entry_is_supported(&self, stop_vol_multiplier: f64) -> bool {
        stop_vol_multiplier.is_finite()
            && stop_vol_multiplier > 0.0
            && self.dataset.settings.adaptive_rr.is_finite()
            && self.dataset.settings.adaptive_rr > 0.0
            && self
                .dataset
                .settings
                .adaptive_base_pips
                .as_ref()
                .is_some_and(|base| {
                    base.len() == self.dataset.bars()
                        && base.iter().all(|value| value.is_finite() && *value > 0.0)
                })
    }

    fn scenario_is_supported(&self, index: usize) -> bool {
        let scenario = &self.scenarios.scenarios[index];
        scenario.window_offset == 0
            && scenario.window_len as usize == self.dataset.bars()
            && scenario.scenario_type == 0
            && scenario.spread_ticks == 0
            && scenario.slippage_ticks == 0
            && scenario.commission_micros == 0
            && scenario.perturbation_count == 0
    }
}

fn insert_unsupported(
    unsupported: &mut BTreeMap<PrototypeBcIneligibilityReason, Vec<u64>>,
    reason: PrototypeBcIneligibilityReason,
    candidate_id: u64,
) {
    unsupported.entry(reason).or_default().push(candidate_id);
}

fn validate_dataset_shape(
    dataset: &PrototypeADatasetUpload,
) -> Result<(), PrototypePopulationError> {
    let bars = dataset.bars();
    if bars == 0 {
        return Err(PrototypeAUploadError::EmptyDataset.into());
    }
    if dataset.feature_count == 0 {
        return Err(PrototypeAUploadError::ZeroFeatureCount.into());
    }
    for (field, actual) in [
        ("high", dataset.high.len()),
        ("low", dataset.low.len()),
        ("months", dataset.months.len()),
        ("days", dataset.days.len()),
        ("timestamps", dataset.timestamps.len()),
        ("smc_data", dataset.smc_data.len()),
    ] {
        if actual != bars {
            return Err(PrototypeAUploadError::ShapeMismatch {
                field,
                expected: bars,
                actual,
            }
            .into());
        }
    }
    let expected_indicators = dataset.feature_count.saturating_mul(bars);
    if dataset.indicators.len() != expected_indicators {
        return Err(PrototypeAUploadError::ShapeMismatch {
            field: "indicators",
            expected: expected_indicators,
            actual: dataset.indicators.len(),
        }
        .into());
    }
    Ok(())
}

fn select<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&index| values[index].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_native::population_fixture::TinyPopulationFixture;
    use crate::gpu_native::prototype_bc::{
        PrototypeKind, StrategyPathProfile, SubgroupSupport, route_strategy,
    };

    fn mixed_fixed_and_adaptive_workload() -> PrototypePopulationWorkload {
        let fixture = TinyPopulationFixture::new(2, 64, 4);
        let (mut dataset, mut genes, mut scenarios) = fixture.prototype_a_uploads();
        dataset.settings.adaptive_base_pips = Some(vec![12.0; dataset.bars()]);
        dataset.settings.adaptive_rr = 2.0;
        genes.stop_vol_multipliers = vec![0.0, 1.5];
        scenarios.scenarios[0].scenario_id = 101;
        scenarios.scenarios[0].rng_counter = 7;
        scenarios.scenarios[1].scenario_id = 202;
        scenarios.scenarios[1].rng_counter = 11;

        PrototypePopulationWorkload::from_uploads(
            dataset,
            genes,
            scenarios,
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            },
        )
        .unwrap()
    }

    #[test]
    fn fixed_and_adaptive_candidates_share_one_bc_intersection() {
        let workload = mixed_fixed_and_adaptive_workload();

        let b = workload.common_bc_intersection(PrototypeKind::BWarpCooperative);
        let c = workload.common_bc_intersection(PrototypeKind::CSparseFirstHit);

        assert_eq!(b, c);
        assert_eq!(b.supported_candidate_ids, vec![0, 1]);
        assert_eq!(b.supported_gene_indices, vec![0, 1]);
        assert_eq!(b.coverage().total_candidates, 2);
        assert_eq!(b.coverage().supported_candidates, 2);
        assert_eq!(b.coverage().unsupported_candidates, 0);
    }

    #[test]
    fn supported_partition_preserves_ids_csr_order_scenarios_and_rng_counters() {
        let workload = mixed_fixed_and_adaptive_workload();
        let eligibility = workload.common_bc_intersection(PrototypeKind::BWarpCooperative);
        let original_offsets = workload.genes.offsets.clone();
        let original_indices = workload.genes.indices.clone();
        let original_weights = workload.genes.weights.clone();
        let original_scenarios = workload.scenarios.scenarios.clone();

        let partition = workload.supported_partition(&eligibility).unwrap();

        assert_eq!(partition.genes.candidate_ids, vec![0, 1]);
        assert_eq!(partition.genes.offsets, original_offsets);
        assert_eq!(partition.genes.indices, original_indices);
        assert_eq!(partition.genes.weights, original_weights);
        assert_eq!(
            partition
                .scenarios
                .scenarios
                .iter()
                .map(|scenario| (
                    scenario.base_candidate_id,
                    scenario.scenario_id,
                    scenario.rng_counter
                ))
                .collect::<Vec<_>>(),
            original_scenarios
                .iter()
                .map(|scenario| (
                    scenario.base_candidate_id,
                    scenario.scenario_id,
                    scenario.rng_counter
                ))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn global_trailing_and_break_even_reject_the_complete_workload() {
        let fixture = TinyPopulationFixture::new(2, 64, 4);
        let (mut dataset, genes, scenarios) = fixture.prototype_a_uploads();
        dataset.settings.trailing_enabled = true;
        dataset.settings.trailing_be_trigger_r = 0.75;
        let workload = PrototypePopulationWorkload::from_uploads(
            dataset,
            genes,
            scenarios,
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            },
        )
        .unwrap();

        let eligibility = workload.common_bc_intersection(PrototypeKind::CSparseFirstHit);

        assert!(eligibility.supported_candidate_ids.is_empty());
        assert_eq!(
            eligibility
                .unsupported
                .get(&PrototypeBcIneligibilityReason::GlobalTrailingOrBreakEven),
            Some(&vec![0, 1])
        );
        assert_eq!(eligibility.coverage().total_candidates, 2);
        assert_eq!(eligibility.coverage().unsupported_candidates, 2);
    }

    #[test]
    fn requested_prop_firm_state_rejects_the_complete_workload() {
        let fixture = TinyPopulationFixture::new(2, 64, 4);
        let (dataset, genes, scenarios) = fixture.prototype_a_uploads();
        let workload = PrototypePopulationWorkload::from_uploads(
            dataset,
            genes,
            scenarios,
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::Required,
            },
        )
        .unwrap();

        let eligibility = workload.common_bc_intersection(PrototypeKind::BWarpCooperative);

        assert!(eligibility.supported_candidate_ids.is_empty());
        assert_eq!(
            eligibility
                .unsupported
                .get(&PrototypeBcIneligibilityReason::PropFirmStateRequired),
            Some(&vec![0, 1])
        );
    }

    #[test]
    fn unsupported_scenario_is_partitioned_before_device_execution() {
        let fixture = TinyPopulationFixture::new(2, 64, 4);
        let (dataset, genes, mut scenarios) = fixture.prototype_a_uploads();
        scenarios.scenarios[1].scenario_type = 99;
        let workload = PrototypePopulationWorkload::from_uploads(
            dataset,
            genes,
            scenarios,
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            },
        )
        .unwrap();

        let eligibility = workload.common_bc_intersection(PrototypeKind::BWarpCooperative);
        let partition = workload.supported_partition(&eligibility).unwrap();

        assert_eq!(eligibility.supported_candidate_ids, vec![0]);
        assert_eq!(eligibility.supported_gene_indices, vec![0]);
        assert_eq!(
            eligibility
                .unsupported
                .get(&PrototypeBcIneligibilityReason::UnsupportedScenario),
            Some(&vec![1])
        );
        assert_eq!(partition.genes.candidate_ids, vec![0]);
        assert_eq!(partition.scenarios.scenarios[0].base_candidate_id, 0);
    }

    #[test]
    fn non_finite_fixed_levels_and_missing_adaptive_base_are_coverage_gaps() {
        let fixture = TinyPopulationFixture::new(2, 64, 4);
        let (dataset, mut genes, scenarios) = fixture.prototype_a_uploads();
        genes.stop_pips[0] = f64::NAN;
        genes.stop_vol_multipliers[1] = 1.0;
        let workload = PrototypePopulationWorkload::from_uploads(
            dataset,
            genes,
            scenarios,
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            },
        )
        .unwrap();

        let eligibility = workload.common_bc_intersection(PrototypeKind::CSparseFirstHit);

        assert!(eligibility.supported_candidate_ids.is_empty());
        assert_eq!(
            eligibility
                .unsupported
                .get(&PrototypeBcIneligibilityReason::NonFiniteStaticLevels),
            Some(&vec![0])
        );
        assert_eq!(
            eligibility
                .unsupported
                .get(&PrototypeBcIneligibilityReason::AdaptiveAtEntryUnavailable),
            Some(&vec![1])
        );
    }

    #[test]
    fn observed_event_density_can_route_b_or_c_but_cannot_change_eligibility() {
        let workload = mixed_fixed_and_adaptive_workload();
        let eligibility = workload.common_bc_intersection(PrototypeKind::BWarpCooperative);

        let sparse_route = route_strategy(
            StrategyPathProfile {
                fixed_at_entry: true,
                event_density: 0.01,
                ..StrategyPathProfile::default()
            },
            SubgroupSupport::known(32),
        );
        let dense_route = route_strategy(
            StrategyPathProfile {
                fixed_at_entry: true,
                event_density: 0.5,
                ..StrategyPathProfile::default()
            },
            SubgroupSupport::known(32),
        );

        assert_eq!(sparse_route.prototype, PrototypeKind::CSparseFirstHit);
        assert_eq!(dense_route.prototype, PrototypeKind::BWarpCooperative);
        assert_eq!(
            eligibility,
            workload.common_bc_intersection(PrototypeKind::CSparseFirstHit)
        );
    }
}
