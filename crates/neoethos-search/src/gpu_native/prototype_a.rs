//! Honest status and measured transfer evidence for the existing fused CubeCL baseline.

use crate::eval::SmcRow;
use crate::gpu_native::engine::{EngineCapabilities, EngineStatus, TransferSnapshot};
use crate::gpu_native::snapshot_fixture::SnapshotSettingsDto;
use neoethos_gpu_contracts::device::ScenarioDescriptor;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::ops::Range;

#[cfg(feature = "gpu")]
pub use crate::gpu_native::prototype_a_engine::create_prototype_a_engine;

pub const PROTOTYPE_A_UPLOAD_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Serialize, Deserialize)]
struct PrototypeAUploadEnvelope<T> {
    schema_version: u32,
    payload: T,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrototypeADatasetUpload {
    pub close: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    /// Feature-major contiguous `[feature][bar]` values.
    pub indicators: Vec<f64>,
    pub feature_count: usize,
    pub months: Vec<i64>,
    pub days: Vec<i64>,
    pub timestamps: Vec<i64>,
    pub smc_data: Vec<SmcRow>,
    pub settings: SnapshotSettingsDto,
}

impl PrototypeADatasetUpload {
    pub fn bars(&self) -> usize {
        self.close.len()
    }

    fn validate(&self) -> Result<(), PrototypeAUploadError> {
        let bars = self.bars();
        if bars == 0 {
            return Err(PrototypeAUploadError::EmptyDataset);
        }
        if self.feature_count == 0 {
            return Err(PrototypeAUploadError::ZeroFeatureCount);
        }
        for (field, actual) in [
            ("high", self.high.len()),
            ("low", self.low.len()),
            ("months", self.months.len()),
            ("days", self.days.len()),
            ("timestamps", self.timestamps.len()),
            ("smc_data", self.smc_data.len()),
        ] {
            if actual != bars {
                return Err(PrototypeAUploadError::ShapeMismatch {
                    field,
                    expected: bars,
                    actual,
                });
            }
        }
        let expected_indicators = self.feature_count.saturating_mul(bars);
        if self.indicators.len() != expected_indicators {
            return Err(PrototypeAUploadError::ShapeMismatch {
                field: "indicators",
                expected: expected_indicators,
                actual: self.indicators.len(),
            });
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, PrototypeAUploadError> {
        encode_upload(self)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PrototypeAUploadError> {
        let decoded: Self = decode_upload(bytes)?;
        decoded.validate()?;
        Ok(decoded)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrototypeAGeneUpload {
    pub candidate_ids: Vec<u64>,
    pub offsets: Vec<i32>,
    pub indices: Vec<i32>,
    pub weights: Vec<f64>,
    pub long_thresholds: Vec<f64>,
    pub short_thresholds: Vec<f64>,
    pub stop_pips: Vec<f64>,
    pub target_pips: Vec<f64>,
    pub stop_vol_multipliers: Vec<f64>,
    pub smc_flags: Vec<[i8; 11]>,
    pub smc_weights: [f64; 11],
    pub gate_threshold: f64,
}

impl PrototypeAGeneUpload {
    pub fn population(&self) -> usize {
        self.candidate_ids.len()
    }

    fn validate(&self) -> Result<(), PrototypeAUploadError> {
        let population = self.population();
        if population == 0 {
            return Err(PrototypeAUploadError::EmptyGeneBatch);
        }
        for (field, actual) in [
            ("long_thresholds", self.long_thresholds.len()),
            ("short_thresholds", self.short_thresholds.len()),
            ("stop_pips", self.stop_pips.len()),
            ("target_pips", self.target_pips.len()),
            ("stop_vol_multipliers", self.stop_vol_multipliers.len()),
            ("smc_flags", self.smc_flags.len()),
        ] {
            if actual != population {
                return Err(PrototypeAUploadError::ShapeMismatch {
                    field,
                    expected: population,
                    actual,
                });
            }
        }
        if self.offsets.len() != population + 1 {
            return Err(PrototypeAUploadError::ShapeMismatch {
                field: "offsets",
                expected: population + 1,
                actual: self.offsets.len(),
            });
        }
        if self.indices.len() != self.weights.len() {
            return Err(PrototypeAUploadError::ShapeMismatch {
                field: "weights",
                expected: self.indices.len(),
                actual: self.weights.len(),
            });
        }
        if self.offsets.first().copied() != Some(0)
            || self.offsets.last().copied() != Some(self.indices.len() as i32)
            || self
                .offsets
                .windows(2)
                .any(|window| window[0] < 0 || window[0] > window[1])
        {
            return Err(PrototypeAUploadError::InvalidOffsets);
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, PrototypeAUploadError> {
        encode_upload(self)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PrototypeAUploadError> {
        let decoded: Self = decode_upload(bytes)?;
        decoded.validate()?;
        Ok(decoded)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrototypeAScenarioUpload {
    pub scenarios: Vec<ScenarioDescriptor>,
}

impl PrototypeAScenarioUpload {
    fn validate(&self) -> Result<(), PrototypeAUploadError> {
        if self.scenarios.is_empty() {
            return Err(PrototypeAUploadError::EmptyScenarioBatch);
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, PrototypeAUploadError> {
        encode_upload(self)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PrototypeAUploadError> {
        let decoded: Self = decode_upload(bytes)?;
        decoded.validate()?;
        Ok(decoded)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrototypeARebatchPlan {
    ranges: Vec<Range<usize>>,
}

impl PrototypeARebatchPlan {
    pub fn new(
        genes: &PrototypeAGeneUpload,
        scenarios: &PrototypeAScenarioUpload,
        max_batch: usize,
    ) -> Result<Self, PrototypeAUploadError> {
        genes.validate()?;
        if max_batch == 0 {
            return Err(PrototypeAUploadError::ZeroBatchSize);
        }
        let population = genes.population();
        if scenarios.scenarios.len() != population {
            return Err(PrototypeAUploadError::ShapeMismatch {
                field: "scenarios",
                expected: population,
                actual: scenarios.scenarios.len(),
            });
        }
        for (index, (candidate_id, scenario)) in genes
            .candidate_ids
            .iter()
            .copied()
            .zip(scenarios.scenarios.iter())
            .enumerate()
        {
            // TWO MEANINGS OF ONE FIELD, and this is the older one.
            //
            // Prototype A requires `base_candidate_id` to equal the gene's own
            // `candidate_id` — a VALUE match — because A pairs one scenario per
            // gene positionally and uses the equality as a self-check. The
            // Prototype B population lane reads the same field as an INDEX into
            // the uploaded gene array (`upload_scenarios` refuses anything
            // outside `0..population`), which is what lets 174 genes carry
            // 17 574 scenarios.
            //
            // Both hold for every work list in the tree because every producer
            // numbers genes `0..population`, so index == id. A caller that
            // numbers genes by anything else would satisfy exactly one of the
            // two. A is the f32 engine measured 54 % wrong at 200 k bars and is
            // slated for deletion; when it goes, this meaning goes with it.
            if scenario.base_candidate_id != candidate_id {
                return Err(PrototypeAUploadError::ScenarioCandidateMismatch {
                    index,
                    expected: candidate_id,
                    actual: scenario.base_candidate_id,
                });
            }
        }
        let ranges = (0..population)
            .step_by(max_batch)
            .map(|start| start..(start + max_batch).min(population))
            .collect();
        Ok(Self { ranges })
    }

    pub fn ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        self.ranges.iter().cloned()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrototypeAUploadError {
    EmptyDataset,
    ZeroFeatureCount,
    EmptyGeneBatch,
    EmptyScenarioBatch,
    ZeroBatchSize,
    ShapeMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidOffsets,
    UnsupportedSchema {
        expected: u32,
        actual: u32,
    },
    Codec(String),
    ScenarioCandidateMismatch {
        index: usize,
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for PrototypeAUploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(f, "Prototype A dataset upload is empty"),
            Self::ZeroFeatureCount => {
                write!(f, "Prototype A dataset upload has no features")
            }
            Self::EmptyGeneBatch => write!(f, "Prototype A gene upload is empty"),
            Self::EmptyScenarioBatch => write!(f, "Prototype A scenario upload is empty"),
            Self::ZeroBatchSize => write!(f, "Prototype A rebatch size must be non-zero"),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "Prototype A {field} length {actual} does not match expected {expected}"
            ),
            Self::InvalidOffsets => write!(f, "Prototype A gene CSR offsets are invalid"),
            Self::UnsupportedSchema { expected, actual } => write!(
                f,
                "Prototype A upload schema {actual} is unsupported; expected {expected}"
            ),
            Self::Codec(message) => write!(f, "Prototype A upload codec error: {message}"),
            Self::ScenarioCandidateMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "Prototype A scenario {index} targets candidate {actual}, expected {expected}"
            ),
        }
    }
}

impl Error for PrototypeAUploadError {}

fn encode_upload<T: Serialize>(payload: &T) -> Result<Vec<u8>, PrototypeAUploadError> {
    serde_json::to_vec(&PrototypeAUploadEnvelope {
        schema_version: PROTOTYPE_A_UPLOAD_SCHEMA_VERSION,
        payload,
    })
    .map_err(|error| PrototypeAUploadError::Codec(error.to_string()))
}

fn decode_upload<T>(bytes: &[u8]) -> Result<T, PrototypeAUploadError>
where
    T: for<'de> Deserialize<'de>,
{
    let envelope: PrototypeAUploadEnvelope<T> = serde_json::from_slice(bytes)
        .map_err(|error| PrototypeAUploadError::Codec(error.to_string()))?;
    if envelope.schema_version != PROTOTYPE_A_UPLOAD_SCHEMA_VERSION {
        return Err(PrototypeAUploadError::UnsupportedSchema {
            expected: PROTOTYPE_A_UPLOAD_SCHEMA_VERSION,
            actual: envelope.schema_version,
        });
    }
    Ok(envelope.payload)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrototypeATelemetry {
    pub gpu_calls: u64,
    pub resident_cache_hits: u64,
    pub resident_cache_misses: u64,
    pub resident_upload_bytes: u64,
    pub streamed_dataset_upload_bytes: u64,
    pub gene_uploads: u64,
    pub gene_upload_bytes: u64,
    pub full_readbacks: u64,
    pub full_readback_bytes: u64,
    pub compact_readbacks: u64,
    pub compact_readback_bytes: u64,
    pub chained_reuploads: u64,
    pub synchronization_events: u64,
}

impl PrototypeATelemetry {
    pub fn transfer_snapshot(self) -> TransferSnapshot {
        TransferSnapshot {
            dataset_uploads: u64::from(self.resident_upload_bytes > 0),
            gene_uploads: self.gene_uploads,
            scenario_uploads: 0,
            full_d2h_readbacks: self.full_readbacks,
            compact_d2h_readbacks: self.compact_readbacks,
            chained_reuploads: self.chained_reuploads,
            synchronization_events: self.synchronization_events,
            workspace_allocations: 0,
            h2d_bytes: self
                .resident_upload_bytes
                .saturating_add(self.streamed_dataset_upload_bytes)
                .saturating_add(self.gene_upload_bytes)
                .saturating_add(if self.chained_reuploads > 0 {
                    self.full_readback_bytes
                } else {
                    0
                }),
            d2h_bytes: self
                .full_readback_bytes
                .saturating_add(self.compact_readback_bytes),
        }
    }

    pub fn satisfies_no_dense_roundtrip(self) -> bool {
        self.full_readbacks == 0 && self.chained_reuploads == 0
    }
}

pub fn prototype_a_status() -> EngineStatus {
    #[cfg(feature = "gpu")]
    {
        EngineStatus::NotBenchmarked
    }
    #[cfg(not(feature = "gpu"))]
    {
        EngineStatus::UnsupportedCapability
    }
}

pub fn prototype_a_capabilities() -> EngineCapabilities {
    EngineCapabilities {
        fixed_stops: true,
        adaptive_stops: true,
        break_even: true,
        trailing: true,
        prop_firm_state: true,
        device_filtering: false,
        compact_readback: true,
    }
}

pub fn is_known_no_adapter_error(message: &str) -> bool {
    message.contains("No possible adapter available")
        || message.contains("No Discrete GPU device found")
        || message.contains("No Integrated GPU device found")
        || message.contains("No Virtual GPU device found")
}

pub fn disable_prototype_a_telemetry() {
    #[cfg(feature = "gpu")]
    crate::cubecl_eval::disable_cubecl_transfer_telemetry();
}

pub fn reset_prototype_a_telemetry() {
    #[cfg(feature = "gpu")]
    crate::cubecl_eval::reset_cubecl_transfer_telemetry();
}

pub fn prototype_a_telemetry() -> PrototypeATelemetry {
    #[cfg(feature = "gpu")]
    {
        let raw = crate::cubecl_eval::cubecl_transfer_telemetry_snapshot();
        PrototypeATelemetry {
            gpu_calls: raw.gpu_calls,
            resident_cache_hits: raw.resident_cache_hits,
            resident_cache_misses: raw.resident_cache_misses,
            resident_upload_bytes: raw.resident_upload_bytes,
            streamed_dataset_upload_bytes: raw.streamed_dataset_upload_bytes,
            gene_uploads: raw.gene_uploads,
            gene_upload_bytes: raw.gene_upload_bytes,
            full_readbacks: raw.full_readbacks,
            full_readback_bytes: raw.full_readback_bytes,
            compact_readbacks: raw.compact_readbacks,
            compact_readback_bytes: raw.compact_readback_bytes,
            chained_reuploads: raw.chained_reuploads,
            synchronization_events: raw.synchronization_events,
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        PrototypeATelemetry::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_native::snapshot_fixture::SnapshotSettingsDto;
    use neoethos_gpu_contracts::device::ScenarioDescriptor;

    fn settings() -> SnapshotSettingsDto {
        SnapshotSettingsDto {
            session_spread_profile: None,
            max_hold_bars: 12,
            min_hold_bars: 0,
            max_trades_per_day: 20,
            gap_threshold_ms: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: 0.0001,
            spread_pips: 0.8,
            commission_per_trade: 3.0,
            pip_value_per_lot: 10.0,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: 0.1,
            pnl_conversion_fee_rate: 0.005,
            risk_based_sizing: true,
            risk_per_trade_min: 0.005,
            risk_per_trade_max: 0.01,
            high_quality_confidence: 0.65,
            adaptive_base_pips: None,
            adaptive_rr: 2.0,
        }
    }

    #[test]
    fn transfer_mapping_never_fabricates_scenario_uploads() {
        let telemetry = PrototypeATelemetry {
            resident_upload_bytes: 100,
            gene_uploads: 2,
            gene_upload_bytes: 50,
            compact_readbacks: 2,
            compact_readback_bytes: 20,
            ..PrototypeATelemetry::default()
        };
        let transfers = telemetry.transfer_snapshot();
        assert_eq!(transfers.dataset_uploads, 1);
        assert_eq!(transfers.scenario_uploads, 0);
        assert_eq!(transfers.h2d_bytes, 150);
        assert_eq!(transfers.d2h_bytes, 20);
    }

    #[test]
    fn dense_roundtrip_is_an_explicit_acceptance_failure() {
        let telemetry = PrototypeATelemetry {
            full_readbacks: 1,
            chained_reuploads: 1,
            ..PrototypeATelemetry::default()
        };
        assert!(!telemetry.satisfies_no_dense_roundtrip());
    }

    #[test]
    fn no_adapter_classifier_matches_only_the_known_cubecl_absence_signature() {
        assert!(is_known_no_adapter_error(
            "No possible adapter available, requested_backends: Backends(VULKAN)"
        ));
        assert!(is_known_no_adapter_error(
            "No Integrated GPU device found for index 99"
        ));
        assert!(is_known_no_adapter_error(
            "No Discrete GPU device found for index 99"
        ));
        assert!(is_known_no_adapter_error(
            "No Virtual GPU device found for index 99"
        ));
        assert!(!is_known_no_adapter_error(
            "wgpu validation error; requested_backends: Backends(VULKAN)"
        ));
        assert!(!is_known_no_adapter_error(
            "buffer offset is not aligned to min_storage_buffer_offset_alignment"
        ));
    }

    #[test]
    fn rebatching_preserves_candidate_scenario_and_rng_identity() {
        let genes = PrototypeAGeneUpload {
            candidate_ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 3],
            indices: vec![0, 1, 2],
            weights: vec![0.5, -0.25, 0.75],
            long_thresholds: vec![0.2; 3],
            short_thresholds: vec![-0.2; 3],
            stop_pips: vec![20.0; 3],
            target_pips: vec![40.0; 3],
            stop_vol_multipliers: vec![0.0; 3],
            smc_flags: vec![[0; 11]; 3],
            smc_weights: [0.0; 11],
            gate_threshold: 0.0,
        };
        let scenarios = PrototypeAScenarioUpload {
            scenarios: vec![
                ScenarioDescriptor {
                    base_candidate_id: 10,
                    scenario_id: 100,
                    rng_counter: 7,
                    ..ScenarioDescriptor::default()
                },
                ScenarioDescriptor {
                    base_candidate_id: 20,
                    scenario_id: 200,
                    rng_counter: 8,
                    ..ScenarioDescriptor::default()
                },
                ScenarioDescriptor {
                    base_candidate_id: 30,
                    scenario_id: 300,
                    rng_counter: 9,
                    ..ScenarioDescriptor::default()
                },
            ],
        };

        let plan = PrototypeARebatchPlan::new(&genes, &scenarios, 2).unwrap();
        let identities = plan
            .ranges()
            .flat_map(|range| {
                range.map(|index| {
                    (
                        genes.candidate_ids[index],
                        scenarios.scenarios[index].scenario_id,
                        scenarios.scenarios[index].rng_counter,
                    )
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(identities, vec![(10, 100, 7), (20, 200, 8), (30, 300, 9)]);
    }

    #[test]
    fn scenario_candidate_mismatch_is_typed() {
        let genes = PrototypeAGeneUpload {
            candidate_ids: vec![10, 20],
            offsets: vec![0, 1, 2],
            indices: vec![0, 1],
            weights: vec![0.5, -0.25],
            long_thresholds: vec![0.2; 2],
            short_thresholds: vec![-0.2; 2],
            stop_pips: vec![20.0; 2],
            target_pips: vec![40.0; 2],
            stop_vol_multipliers: vec![0.0; 2],
            smc_flags: vec![[0; 11]; 2],
            smc_weights: [0.0; 11],
            gate_threshold: 0.0,
        };
        let scenarios = PrototypeAScenarioUpload {
            scenarios: vec![
                ScenarioDescriptor {
                    base_candidate_id: 10,
                    ..ScenarioDescriptor::default()
                },
                ScenarioDescriptor {
                    base_candidate_id: 99,
                    ..ScenarioDescriptor::default()
                },
            ],
        };

        assert!(matches!(
            PrototypeARebatchPlan::new(&genes, &scenarios, 2),
            Err(PrototypeAUploadError::ScenarioCandidateMismatch {
                index: 1,
                expected: 20,
                actual: 99,
            })
        ));
    }

    #[test]
    fn upload_codec_round_trips_exactly_and_rejects_unknown_schema() {
        let genes = PrototypeAGeneUpload {
            candidate_ids: vec![10],
            offsets: vec![0, 1],
            indices: vec![2],
            weights: vec![0.75],
            long_thresholds: vec![0.2],
            short_thresholds: vec![-0.2],
            stop_pips: vec![20.0],
            target_pips: vec![40.0],
            stop_vol_multipliers: vec![0.0],
            smc_flags: vec![[0; 11]],
            smc_weights: [0.0; 11],
            gate_threshold: 0.0,
        };

        let encoded = genes.encode().unwrap();
        assert_eq!(PrototypeAGeneUpload::decode(&encoded).unwrap(), genes);

        let mut envelope: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        envelope["schema_version"] = serde_json::json!(999);
        let unsupported = serde_json::to_vec(&envelope).unwrap();
        assert!(matches!(
            PrototypeAGeneUpload::decode(&unsupported),
            Err(PrototypeAUploadError::UnsupportedSchema {
                expected: PROTOTYPE_A_UPLOAD_SCHEMA_VERSION,
                actual: 999,
            })
        ));
    }

    #[test]
    fn dataset_codec_validates_shape_before_device_upload() {
        let dataset = PrototypeADatasetUpload {
            close: vec![1.0, 1.1],
            high: vec![1.2],
            low: vec![0.9, 1.0],
            indicators: vec![0.1, 0.2, 0.3, 0.4],
            feature_count: 2,
            months: vec![1, 1],
            days: vec![10, 10],
            timestamps: vec![1_000, 2_000],
            smc_data: vec![[0; 11]; 2],
            settings: settings(),
        };
        let encoded = dataset.encode().unwrap();

        assert!(matches!(
            PrototypeADatasetUpload::decode(&encoded),
            Err(PrototypeAUploadError::ShapeMismatch {
                field: "high",
                expected: 2,
                actual: 1,
            })
        ));
    }

    #[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
    #[test]
    fn direct_prototype_a_engine_is_resident_and_matches_cpu_fixture() {
        use crate::backend::EvaluationBackend;
        use crate::gpu_native::cpu_strategy::CpuStrategyAuditContext;
        use crate::gpu_native::engine::{BacktestEngine, DeviceFilterPolicy};
        use crate::gpu_native::population_fixture::TinyPopulationFixture;

        if std::env::var("NEOETHOS_RUN_CUDA_SEARCH_TESTS").as_deref() != Ok("1") {
            eprintln!(
                "SKIPPED direct_prototype_a_engine_is_resident_and_matches_cpu_fixture — set \
                 NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 on a real GPU host"
            );
            return;
        }

        let fixture = TinyPopulationFixture::new(4, 128, 4);
        let reference = fixture
            .evaluate_test_oracle(
                EvaluationBackend::CPU_CANONICAL,
                &CpuStrategyAuditContext::validation_reference(91),
            )
            .unwrap();
        let (dataset, genes, scenarios) = fixture.prototype_a_uploads();

        let mut engine = create_prototype_a_engine(None, 9001, 2)
            .unwrap_or_else(|error| panic!("Prototype A engine creation failed: {error}"));
        let dataset_handle = engine.upload_dataset(&dataset.encode().unwrap()).unwrap();
        assert!(matches!(
            engine.upload_dataset(&dataset.encode().unwrap()),
            Err(
                crate::gpu_native::engine::EngineError::UnsupportedCapability {
                    operation: "dataset_reupload",
                    ..
                }
            )
        ));
        let gene_handle = engine.upload_genes(&genes.encode().unwrap()).unwrap();
        let scenario_handle = engine
            .upload_scenarios(&scenarios.encode().unwrap())
            .unwrap();
        let (metrics, evaluated) = engine
            .evaluate(dataset_handle, gene_handle, scenario_handle, None)
            .unwrap();
        assert!(matches!(
            engine.filter(metrics, DeviceFilterPolicy::TopK(2), evaluated),
            Err(
                crate::gpu_native::engine::EngineError::UnsupportedCapability {
                    operation: "device_filtering",
                    ..
                }
            )
        ));
        let (selection, filtered) = engine
            .filter(metrics, DeviceFilterPolicy::All, evaluated)
            .unwrap();
        let summary = engine.readback_compact(selection, filtered).unwrap();

        assert_eq!(summary.candidate_ids, genes.candidate_ids);
        assert_eq!(
            summary.scenario_ids,
            scenarios
                .scenarios
                .iter()
                .map(|scenario| scenario.scenario_id)
                .collect::<Vec<_>>()
        );
        let parity = TinyPopulationFixture::compare_final_metrics(&reference, &summary.metrics);
        assert!(
            parity.is_match(),
            "Prototype A metric parity failed: {:?}",
            parity.first_divergence
        );
        let transfers = engine.session().transfer_snapshot();
        transfers.assert_device_resident_chain().unwrap();
        assert_eq!(transfers.dataset_uploads, 1);
        assert_eq!(transfers.gene_uploads, 1);
        assert_eq!(transfers.scenario_uploads, 1);
        assert_eq!(transfers.full_d2h_readbacks, 0);
        assert_eq!(transfers.compact_d2h_readbacks, 1);
        assert_eq!(transfers.chained_reuploads, 0);
        assert!(transfers.synchronization_events > 0);

        let first_workspace_allocations = transfers.workspace_allocations;
        assert!(first_workspace_allocations > 0);
        let (metrics, evaluated) = engine
            .evaluate(dataset_handle, gene_handle, scenario_handle, None)
            .unwrap();
        let (selection, filtered) = engine
            .filter(metrics, DeviceFilterPolicy::All, evaluated)
            .unwrap();
        let second_summary = engine.readback_compact(selection, filtered).unwrap();
        assert_eq!(second_summary.metrics, summary.metrics);
        let second_transfers = engine.session().transfer_snapshot();
        assert_eq!(
            second_transfers.workspace_allocations,
            first_workspace_allocations
        );
        assert_eq!(second_transfers.dataset_uploads, 1);
        assert_eq!(second_transfers.gene_uploads, 1);
        assert_eq!(second_transfers.scenario_uploads, 1);
    }
}
