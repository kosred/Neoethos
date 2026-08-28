use super::execution_receipt_continuation_v1_tests::{
    financial_contract_v1, native_receipt_from_value_v1, native_receipt_value_v1,
};
use super::payload_v1_tests::{gene_for_metrics_v1, metric_row_v1, valid_evaluation_config_v1};
#[allow(unused_imports)]
use super::*;
use crate::canonical_native_discovery_request_v1::{
    CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1,
    CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1, CanonicalNativeCostBandStatusV1,
    CanonicalNativeExecutionScopeV1, MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1,
    MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1, MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1, MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1,
    MAX_CANONICAL_NATIVE_GEN0_TERMS_V1, MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1,
};
use crate::genetic::SearchResult;
use crate::historical_research::{
    HistoricalResearchArtifactClassV1, HistoricalResearchPromotionEligibilityV1,
};
use neoethos_gpu_cuda::{PopulationMetricsOnlyPlanV1, PopulationResidencyCountersV1};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{self, Write};

const EVIDENCE_IDENTITY_PLACEHOLDER_V1: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Default)]
struct FailAfterV1 {
    remaining: usize,
}

impl Write for FailAfterV1 {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("injected writer failure"));
        }
        let written = self.remaining.min(bytes.len());
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn request_evidence_v1() -> CanonicalNativeGenerationZeroRequestEvidenceV1 {
    let raw_generations = 1_usize;
    let clamped_generations = 1_usize;
    let cost_band = Some((-f64::MAX, f64::MAX));
    let mut scope_digest = Sha256::new();
    scope_digest.update(b"neoethos.canonical-native.gen0-scope.v1\0");
    scope_digest.update((raw_generations as u64).to_le_bytes());
    scope_digest.update((clamped_generations as u64).to_le_bytes());
    for value in cost_band.into_iter().flat_map(|pair| [pair.0, pair.1]) {
        scope_digest.update(value.to_bits().to_le_bytes());
    }
    CanonicalNativeGenerationZeroRequestEvidenceV1 {
        execution_scope: CanonicalNativeExecutionScopeV1::GenerationZeroOnly,
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        authorization_issued: false,
        contract_artifact_reference_schema: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1
            .to_owned(),
        contract_artifact_reference_version: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1,
        contract_artifact_relative_path: "contracts/a.json".to_owned(),
        contract_artifact_expected_sha256: "a".repeat(64),
        contract_artifact_exact_file_sha256: "a".repeat(64),
        contract_artifact_exact_file_byte_count: 4_096,
        contract_domain_identity_sha256: String::new(),
        startup_settings_id: "b".repeat(64),
        runtime_install_receipt_id: "c".repeat(64),
        generation_zero_runtime_authority_id: "d".repeat(64),
        unused_full_search_scope_id: format!("{:x}", scope_digest.finalize()),
        raw_generations,
        clamped_generations,
        cost_band_status: CanonicalNativeCostBandStatusV1::UnusedGenerationZero,
        cost_band,
        configured_population_cap: MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1,
        resolved_population_cap: MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1,
        term_cap: MAX_CANONICAL_NATIVE_GEN0_TERMS_V1,
        string_bytes_cap: MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1,
        vector_elements_cap: MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1,
        source_count_cap: MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1,
        result_bytes_cap: MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    }
}

#[derive(Default)]
struct CountingSha256WriterV1 {
    byte_count: u64,
    sha256: Sha256,
}

impl Write for CountingSha256WriterV1 {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.byte_count = self
            .byte_count
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("test counting writer overflow"))?;
        self.sha256.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl CountingSha256WriterV1 {
    fn finish(self) -> (u64, String) {
        (self.byte_count, format!("{:x}", self.sha256.finalize()))
    }
}

fn independently_stream_compact_json_v1(value: &impl Serialize) -> (u64, String) {
    let mut writer = CountingSha256WriterV1::default();
    value
        .serialize(&mut serde_json::Serializer::new(&mut writer))
        .unwrap();
    writer.finish()
}

fn independently_stream_identity_v1(value: &impl Serialize) -> String {
    let mut writer = CountingSha256WriterV1::default();
    writer
        .write_all(CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_IDENTITY_DOMAIN_V1)
        .unwrap();
    value
        .serialize(&mut serde_json::Serializer::new(&mut writer))
        .unwrap();
    writer.finish().1
}

struct NeedleMutationSha256WriterV1<'a> {
    inner: CountingSha256WriterV1,
    needle: &'a [u8],
    matched: usize,
    mutate_next: bool,
    mutated: bool,
}

impl<'a> NeedleMutationSha256WriterV1<'a> {
    fn new(needle: &'a [u8]) -> Self {
        Self {
            inner: CountingSha256WriterV1::default(),
            needle,
            matched: 0,
            mutate_next: false,
            mutated: false,
        }
    }
}

impl Write for NeedleMutationSha256WriterV1<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        for &byte in bytes {
            let output = if self.mutate_next {
                self.mutate_next = false;
                self.mutated = true;
                byte ^ 1
            } else {
                if byte == self.needle[self.matched] {
                    self.matched += 1;
                    if self.matched == self.needle.len() {
                        self.matched = 0;
                        self.mutate_next = true;
                    }
                } else {
                    self.matched = usize::from(byte == self.needle[0]);
                }
                byte
            };
            self.inner.write_all(&[output])?;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn independently_stream_identity_with_mutation_v1(value: &impl Serialize, needle: &[u8]) -> String {
    let mut writer = NeedleMutationSha256WriterV1::new(needle);
    writer
        .inner
        .write_all(CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_IDENTITY_DOMAIN_V1)
        .unwrap();
    value
        .serialize(&mut serde_json::Serializer::new(&mut writer))
        .unwrap();
    assert!(writer.mutated, "identity mutation needle was absent");
    writer.inner.finish().1
}

fn maximum_evaluation_snapshot_v1(
    maximum_general_string: &str,
) -> CanonicalNativeGenerationZeroEvaluationSnapshotV1 {
    CanonicalNativeGenerationZeroEvaluationSnapshotV1 {
        symbol: maximum_general_string.to_owned(),
        account_currency: maximum_general_string.to_owned(),
        max_hold_bars: usize::MAX,
        trailing_enabled: false,
        trailing_atr_multiplier: -f64::MAX,
        trailing_be_trigger_r: -f64::MAX,
        trailing_min_lock_pips: -f64::MAX,
        pip_value: -f64::MAX,
        spread_pips: -f64::MAX,
        commission_per_trade: -f64::MAX,
        pip_value_per_lot: -f64::MAX,
        swap_long_pips_per_day: -f64::MAX,
        swap_short_pips_per_day: -f64::MAX,
        pnl_conversion_fee_rate: -f64::MAX,
        smc_gate_threshold: -f64::MAX,
        smc_weight_ob: -f64::MAX,
        smc_weight_fvg: -f64::MAX,
        smc_weight_liq: -f64::MAX,
        smc_weight_mtf: -f64::MAX,
        smc_weight_premium: -f64::MAX,
        smc_weight_inducement: -f64::MAX,
        smc_weight_bos: -f64::MAX,
        smc_weight_choch: -f64::MAX,
        smc_weight_eqh: -f64::MAX,
        smc_weight_eql: -f64::MAX,
        smc_weight_displacement: -f64::MAX,
        growth_objective: false,
    }
}

fn maximum_empty_result_wire_v1<'a>(
    contract: &'a crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    relative_path: &'a str,
    native_receipt_v3: &'a crate::data_selection::CanonicalGpuResidentSearchInputReceiptV3,
    sizing_receipt_v2: &'a crate::resident_population_auto_sizing_receipt_v2::ResidentPopulationAutoSizingReceiptV2,
    evaluation_snapshot: &'a CanonicalNativeGenerationZeroEvaluationSnapshotV1,
    lower_hex_sha256: &'a str,
) -> ResultWireV1<'a> {
    ResultWireV1 {
        identity_material: IdentityMaterialV1 {
            schema: CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1,
            version: u16::MAX,
            scope: CanonicalNativeExecutionScopeV1::GenerationZeroOnly,
            artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
            promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
            authorization_issued: false,
            contract_artifact: CanonicalNativeGenerationZeroContractArtifactWireV1 {
                reference: CanonicalNativeGenerationZeroArtifactReferenceWireV1 {
                    schema: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1,
                    version: u16::MAX,
                    relative_path,
                    expected_sha256: lower_hex_sha256,
                },
                exact_file_sha256: lower_hex_sha256,
                exact_file_byte_count: u64::MAX,
                contract_domain_identity_sha256: lower_hex_sha256,
            },
            runtime_authority: CanonicalNativeGenerationZeroRuntimeAuthorityWireV1 {
                startup_settings_id: lower_hex_sha256,
                runtime_install_receipt_id: lower_hex_sha256,
                generation_zero_runtime_authority_id: lower_hex_sha256,
            },
            unused_full_search: CanonicalNativeGenerationZeroUnusedFullSearchWireV1 {
                scope_id: lower_hex_sha256,
                raw_generations: usize::MAX,
                clamped_generations: usize::MAX,
            },
            cost_band_status: CanonicalNativeGenerationZeroCostBandWireV1 {
                status: CanonicalNativeCostBandStatusV1::UnusedGenerationZero,
                cost: Some((-f64::MAX, -f64::MAX)),
            },
            limits: CanonicalNativeGenerationZeroLimitsWireV1 {
                configured_population_cap: usize::MAX,
                resolved_population_cap: usize::MAX,
                term_cap: usize::MAX,
                string_bytes_cap: usize::MAX,
                vector_elements_cap: usize::MAX,
                source_count_cap: usize::MAX,
                result_bytes_cap: u64::MAX,
            },
            financial_provenance_only: CanonicalNativeGenerationZeroFinancialProvenanceWireV1 {
                contract,
                cpu_receipt_id: lower_hex_sha256,
            },
            evaluated_native_input: CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1 {
                receipt_v3: native_receipt_v3,
                receipt_id: lower_hex_sha256,
            },
            population_sizing: CanonicalNativeGenerationZeroPopulationSizingWireV1 {
                receipt_v2: sizing_receipt_v2,
                receipt_id: lower_hex_sha256,
                prepared_feature_count: usize::MAX,
                raw_configured_max_indicators: usize::MAX,
                resolved_max_indicators: usize::MAX,
                term_cap: usize::MAX,
                configured_population: usize::MAX,
                resolved_population: usize::MAX,
                population_cap: usize::MAX,
                hard_growth_cap: usize::MAX,
                max_concurrent_scenario_count: usize::MAX,
                stage1_row_start: usize::MAX,
                stage1_row_end: usize::MAX,
                selected_device_ordinal: u32::MAX,
                metrics_receipt_identities_sha256: LowerHexSha256SliceV1(&[]),
                adaptive_token_identity_sha256: Some(LowerHexSha256V1([u8::MAX; 32])),
            },
            generation_zero_evaluation: CanonicalNativeGenerationZeroEvaluationWireV1 {
                snapshot_v1: evaluation_snapshot,
                snapshot_identity_sha256: lower_hex_sha256,
                scoring_objective:
                    CanonicalNativeGenerationZeroScoringObjectiveV1::RiskyKellyGrowthV5,
                effective_smc_gate_threshold: -f64::MAX,
                effective_smc_gate_source:
                    EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1,
                genes: &[],
                metrics: &[],
            },
            residency_counters: CanonicalNativeGenerationZeroResidencyCountersSnapshotV1 {
                parent_upload_count: u64::MAX,
                parent_upload_bytes: u64::MAX,
                view_binding_count: u64::MAX,
                full_binding_count: u64::MAX,
                range_binding_count: u64::MAX,
                ordered_binding_count: u64::MAX,
                ordered_index_upload_bytes: u64::MAX,
                adaptive_upload_bytes: u64::MAX,
                stream_creation_count: u64::MAX,
                explicit_synchronization_count: u64::MAX,
                metric_rows_readback_count: u64::MAX,
                metric_rows_readback_rows: u64::MAX,
                metric_rows_readback_bytes: u64::MAX,
                diagnostic_readback_count: u64::MAX,
                diagnostic_readback_rows: u64::MAX,
                diagnostic_readback_bytes: u64::MAX,
                accepted_trade_total_readback_count: u64::MAX,
                accepted_trade_total_readback_bytes: u64::MAX,
            },
            completion: CanonicalNativeGenerationZeroCompletionWireV1 {
                engine: "CudaNativeF64",
                consumer_completion_confirmed: false,
            },
            replay: CanonicalNativeGenerationZeroReplayWireV1 {
                replay_identity_sealed: false,
            },
        },
        evidence_identity_sha256: lower_hex_sha256,
    }
}

pub(super) fn with_fully_populated_sealed_result_v1(
    check: impl FnOnce(
        &CanonicalNativeGenerationZeroResearchResultViewV1<'_>,
        &CanonicalNativeGenerationZeroCompactJsonSealV1,
    ),
) {
    let financial_contract = financial_contract_v1();
    let mut request_evidence = request_evidence_v1();
    request_evidence.contract_domain_identity_sha256 =
        financial_contract.identity_sha256().unwrap();

    let native_value = native_receipt_value_v1(&financial_contract);
    let source_count = native_value["source_bindings"].as_array().unwrap().len();
    let total_source_segment_count = native_value["source_bindings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|binding| binding["segments"].as_array().unwrap().len())
        .sum();
    let resident_parent_rows = native_value["row_count"].as_u64().unwrap();
    assert!(resident_parent_rows > 0);
    let native_receipt = native_receipt_from_value_v1(&native_value);
    let native_identity = native_receipt.identity_sha256().unwrap();

    let projection = crate::resident_population_auto_sizing_receipt_v2::
        canonical_pinned_source_projection_from_search_receipt_v1(
            financial_contract.input_receipt(),
        )
        .unwrap();
    let sizing_receipt = crate::resident_population_auto_sizing_receipt_v2::tests::
        canonical_result_fixture_receipt_v2(
            financial_contract.identity_sha256().unwrap(),
            financial_contract.input_receipt_sha256().to_owned(),
            projection.identity_sha256(),
            0,
            resident_parent_rows,
            0,
            resident_parent_rows,
            4,
        );

    let mut evaluation_config = valid_evaluation_config_v1(false);
    evaluation_config.pip_value = financial_contract.pip_size();
    evaluation_config.pip_value_per_lot = financial_contract.pip_value_per_lot();
    evaluation_config.spread_pips =
        financial_contract.screening_spread_and_slippage_round_trip_pips();
    evaluation_config.commission_per_trade =
        financial_contract.round_trip_commission_account_per_lot();
    evaluation_config.swap_long_pips_per_day = financial_contract.swap_long_pips_per_day();
    evaluation_config.swap_short_pips_per_day = financial_contract.swap_short_pips_per_day();
    evaluation_config.pnl_conversion_fee_rate = financial_contract.pnl_conversion_fee_rate();
    let evaluation_evidence =
        CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
            &evaluation_config,
            crate::discovery::DiscoveryMode::PropFirm,
        )
        .unwrap();

    let population = sizing_receipt.resolved_population();
    let scenario_count = sizing_receipt.max_concurrent_scenario_count();
    let launch_count = population.div_ceil(scenario_count);
    assert!(launch_count > 1);
    let mut genes = Vec::with_capacity(population);
    let mut metrics = Vec::with_capacity(population);
    for candidate in 0..population {
        let mut row = metric_row_v1();
        row[0] += candidate as f64;
        row[2] += candidate as f64;
        let mut gene = gene_for_metrics_v1(&row, false);
        gene.strategy_id = format!("gen0-strategy-{candidate}");
        genes.push(gene);
        metrics.push(row);
    }
    let search_result = SearchResult {
        genes,
        metrics,
        effective_smc_gate_threshold: 0.5,
    };

    let metric_bytes = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
        population,
        u32::try_from(sizing_receipt.month_capacity()).unwrap(),
    )
    .unwrap()
    .metric_rows_bytes();
    let raw_counters = [
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        launch_count as u64,
        launch_count as u64,
        population as u64,
        metric_bytes,
        0,
        0,
        0,
        0,
        0,
    ];
    // SAFETY: the schema/source RED independently freezes repr(C), exactly 18
    // ordered u64 fields, size 144, and alignment 8 for this test-only value.
    let residency_counters =
        unsafe { std::mem::transmute::<[u64; 18], PopulationResidencyCountersV1>(raw_counters) };
    let metric_receipts = (0..launch_count)
        .map(|launch| [u8::try_from(launch + 1).unwrap(); 32])
        .collect();
    let milestone =
        crate::prepared_discovery_run_input_v3::ResidentGenerationZeroMilestoneV1::test_fixture_v1(
            0,
            native_identity,
            sizing_receipt.identity_sha256().to_owned(),
            population,
            sizing_receipt.term_cap(),
            sizing_receipt.stage1_row_start(),
            sizing_receipt.stage1_row_end(),
            metric_receipts,
            None,
            residency_counters,
            search_result,
        );

    let contract_compact_json_bytes =
        checked_compact_json_byte_count_v1(&financial_contract).unwrap();
    let contract_artifact_relative_path_compact_json_bytes =
        checked_compact_json_string_byte_count_v1(
            &request_evidence.contract_artifact_relative_path,
        )
        .unwrap();
    let preflight = checked_preflight_from_fixed_metadata_shape_v1(
        sizing_receipt.feature_count(),
        sizing_receipt.requested_max_indicators(),
        sizing_receipt.configured_population(),
        CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
            contract_compact_json_bytes,
            contract_artifact_relative_path_compact_json_bytes,
            source_count,
            total_source_segment_count,
        },
    )
    .unwrap();

    let (view, seal) =
        checked_seal_canonical_native_generation_zero_research_result_from_evidence_v1(
            preflight,
            request_evidence,
            financial_contract,
            native_receipt,
            sizing_receipt,
            evaluation_evidence,
            &milestone,
        )
        .unwrap();
    check(&view, &seal);
}

#[test]
fn actual_maximum_empty_result_wire_equals_the_independent_analytic_preflight_bound() {
    let contract = financial_contract_v1();
    let relative_path = "contracts/\".json";
    let maximum_general_string = "\0".repeat(MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1);
    let native_receipt = crate::data_selection::canonical_result_maximum_json_receipt_v3_for_test(
        &maximum_general_string,
        1,
        1,
    );
    let sizing_receipt = crate::resident_population_auto_sizing_receipt_v2::tests::
        canonical_result_maximum_json_receipt_v2_for_test(&maximum_general_string);
    let evaluation_snapshot = maximum_evaluation_snapshot_v1(&maximum_general_string);
    let lower_hex_sha256 = "f".repeat(64);
    let mut wire = maximum_empty_result_wire_v1(
        &contract,
        relative_path,
        &native_receipt,
        &sizing_receipt,
        &evaluation_snapshot,
        &lower_hex_sha256,
    );
    wire.evidence_identity_sha256 = EVIDENCE_IDENTITY_PLACEHOLDER_V1;

    let contract_bytes = independently_stream_compact_json_v1(&contract).0;
    let relative_path_json_bytes = independently_stream_compact_json_v1(&relative_path).0;
    let v2_bytes = independently_stream_compact_json_v1(&sizing_receipt).0;
    let v3_bytes = independently_stream_compact_json_v1(&native_receipt).0;
    assert_eq!(v2_bytes, 7_080_504);
    assert_eq!(v3_bytes, 393_995 + 1_966_378 + 148);
    assert_eq!(
        independently_stream_compact_json_v1(&EVIDENCE_IDENTITY_PLACEHOLDER_V1).0,
        66
    );

    let analytic_b_empty = 8_266_104 + contract_bytes + relative_path_json_bytes + 1_966_378 + 148;
    assert_eq!(
        independently_stream_compact_json_v1(&wire).0,
        analytic_b_empty
    );
    let shape = CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        contract_compact_json_bytes: contract_bytes,
        contract_artifact_relative_path_compact_json_bytes: relative_path_json_bytes,
        source_count: 1,
        total_source_segment_count: 1,
    };
    assert_eq!(
        checked_fixed_metadata_upper_bound_with_empty_arrays_bytes_v1(shape).unwrap(),
        analytic_b_empty
    );
    let preflight = checked_preflight_from_fixed_metadata_shape_v1(5, 5, 10, shape).unwrap();
    assert_eq!(
        preflight.fixed_metadata_upper_bound_with_empty_arrays_bytes(),
        analytic_b_empty
    );
}

#[test]
fn actual_sealed_view_writer_is_deterministic_counted_capped_and_io_fail_closed() {
    with_fully_populated_sealed_result_v1(|view, sealed_count| {
        let mut first = Vec::new();
        let first_count = write_canonical_native_generation_zero_research_result_v1(
            view,
            &mut first,
            sealed_count.byte_count(),
        )
        .unwrap();
        assert_eq!(&first_count, sealed_count);
        assert_eq!(first_count.byte_count(), first.len() as u64);
        assert_eq!(
            first_count.sha256(),
            format!("{:x}", Sha256::digest(&first))
        );

        let mut second = Vec::new();
        let repeated = write_canonical_native_generation_zero_research_result_v1(
            view,
            &mut second,
            sealed_count.byte_count(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(repeated, first_count);

        let mut capped = Vec::new();
        assert!(
            write_canonical_native_generation_zero_research_result_v1(
                view,
                &mut capped,
                sealed_count.byte_count() - 1,
            )
            .is_err()
        );
        let mut failing = FailAfterV1 { remaining: 8 };
        assert!(
            write_canonical_native_generation_zero_research_result_v1(
                view,
                &mut failing,
                MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
            )
            .is_err()
        );

        let population = view.milestone().search_result().genes.len();
        let planned = view
            .preflight()
            .checked_upper_bound_for_population(population)
            .unwrap();
        let mut empty_wire = view.result_wire_v1();
        empty_wire
            .identity_material
            .population_sizing
            .metrics_receipt_identities_sha256 = LowerHexSha256SliceV1(&[]);
        empty_wire
            .identity_material
            .generation_zero_evaluation
            .genes = &[];
        empty_wire
            .identity_material
            .generation_zero_evaluation
            .metrics = &[];
        empty_wire.evidence_identity_sha256 = EVIDENCE_IDENTITY_PLACEHOLDER_V1;
        let independent_empty_count = independently_stream_compact_json_v1(&empty_wire).0;
        assert_eq!(
            independent_empty_count,
            view.checked_fixed_metadata_with_empty_arrays_byte_count_v1()
                .unwrap()
        );
        assert!(
            independent_empty_count
                <= view
                    .preflight()
                    .fixed_metadata_upper_bound_with_empty_arrays_bytes()
        );
        assert!(sealed_count.byte_count() <= planned);
        assert!(planned <= MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1);
    });
}

#[test]
fn actual_identity_material_binds_every_group_and_population_order_excluding_only_self_sha() {
    with_fully_populated_sealed_result_v1(|view, _| {
        assert_eq!(
            CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_IDENTITY_DOMAIN_V1.last(),
            Some(&0),
            "production identity domain must retain its NUL separator"
        );
        let material = view.identity_material_v1();
        assert_eq!(
            independently_stream_identity_v1(&material),
            view.evidence_identity_sha256()
        );

        for field in [
            b"\"schema\":".as_slice(),
            b"\"version\":".as_slice(),
            b"\"scope\":".as_slice(),
            b"\"artifact_class\":".as_slice(),
            b"\"promotion_eligibility\":".as_slice(),
            b"\"authorization_issued\":".as_slice(),
            b"\"contract_artifact\":".as_slice(),
            b"\"runtime_authority\":".as_slice(),
            b"\"unused_full_search\":".as_slice(),
            b"\"cost_band_status\":".as_slice(),
            b"\"limits\":".as_slice(),
            b"\"financial_provenance_only\":".as_slice(),
            b"\"evaluated_native_input\":".as_slice(),
            b"\"population_sizing\":".as_slice(),
            b"\"generation_zero_evaluation\":".as_slice(),
            b"\"residency_counters\":".as_slice(),
            b"\"completion\":".as_slice(),
            b"\"replay\":".as_slice(),
        ] {
            assert_ne!(
                independently_stream_identity_with_mutation_v1(&material, field),
                view.evidence_identity_sha256(),
                "identity omitted a declared group"
            );
        }

        for population_element in [
            b"\"metrics_receipt_identities_sha256\":[".as_slice(),
            b"\"genes\":[".as_slice(),
            b"\"metrics\":[".as_slice(),
        ] {
            assert_ne!(
                independently_stream_identity_with_mutation_v1(&material, population_element),
                view.evidence_identity_sha256(),
                "identity omitted a population element"
            );
        }

        let mut order_changed = view.identity_material_v1();
        let original_receipts = order_changed
            .population_sizing
            .metrics_receipt_identities_sha256
            .0;
        assert!(original_receipts.len() > 1);
        let mut reversed_receipts = Vec::with_capacity(original_receipts.len());
        for receipt in original_receipts.iter().rev() {
            reversed_receipts.push(*receipt);
        }
        order_changed
            .population_sizing
            .metrics_receipt_identities_sha256 = LowerHexSha256SliceV1(&reversed_receipts);
        assert_ne!(
            independently_stream_identity_v1(&order_changed),
            view.evidence_identity_sha256(),
            "identity omitted population receipt order"
        );

        let mut reversed_genes = view.milestone().search_result().genes.clone();
        reversed_genes.reverse();
        assert!(reversed_genes.len() > 1);
        let mut gene_order_changed = view.identity_material_v1();
        gene_order_changed.generation_zero_evaluation.genes = &reversed_genes;
        assert_ne!(
            independently_stream_identity_v1(&gene_order_changed),
            view.evidence_identity_sha256(),
            "identity omitted gene order"
        );

        let mut reversed_metrics = view.milestone().search_result().metrics.clone();
        reversed_metrics.reverse();
        assert!(reversed_metrics.len() > 1);
        let mut metric_order_changed = view.identity_material_v1();
        metric_order_changed.generation_zero_evaluation.metrics = &reversed_metrics;
        assert_ne!(
            independently_stream_identity_v1(&metric_order_changed),
            view.evidence_identity_sha256(),
            "identity omitted metric-row order"
        );
    });
}

#[test]
fn sha256_receipt_wires_are_borrowed_quoted_lowerhex_and_never_integer_arrays() {
    let identities = [[0xab_u8; 32], [0_u8; 32]];
    let encoded = serde_json::to_string(&LowerHexSha256SliceV1(&identities)).unwrap();
    assert_eq!(
        encoded,
        format!("[\"{}\",\"{}\"]", "ab".repeat(32), "00".repeat(32))
    );
    assert!(!encoded.contains("171,171"));

    let adaptive = Some(LowerHexSha256V1([0xcd_u8; 32]));
    assert_eq!(
        serde_json::to_string(&adaptive).unwrap(),
        format!("\"{}\"", "cd".repeat(32))
    );
    let absent: Option<LowerHexSha256V1> = None;
    assert_eq!(serde_json::to_string(&absent).unwrap(), "null");
}
