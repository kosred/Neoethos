use super::*;
use crate::canonical_native_discovery_request_v1::{
    MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1, MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1,
};
use crate::genetic::Gene;

fn worst_escape_strategy_id_v1() -> String {
    (0..128)
        .map(|index| if index % 2 == 0 { '"' } else { '\\' })
        .collect()
}

#[test]
fn private_result_api_is_present_without_exporting_raw_fixed_metadata_authority() {
    let _preflight = preflight_canonical_native_generation_zero_result_v1;
    let _sealer = seal_canonical_native_generation_zero_research_result_v1;
    let _writer = write_canonical_native_generation_zero_research_result_v1::<Vec<u8>>;
    let _: Option<CanonicalNativeGenerationZeroResultPreflightV1> = None;
    let _: Option<CanonicalNativeGenerationZeroResearchResultViewV1<'static>> = None;
    let _: Option<CanonicalNativeGenerationZeroCompactJsonSealV1> = None;
    let _: Option<CanonicalNativeGenerationZeroResultErrorV1> = None;
}

#[test]
fn pre_v5_fixed_metadata_bound_constructs_pcap_without_post_run_receipts() {
    let shape = CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        contract_compact_json_bytes: 4_096,
        contract_artifact_relative_path_compact_json_bytes: 34,
        source_count: 2,
        total_source_segment_count: 3,
    };
    let preflight = checked_preflight_from_fixed_metadata_shape_v1(5, 0, 10, shape)
        .expect("pre-V5 request/F facts must derive Pcap");
    assert_eq!(preflight.prepared_feature_count(), 5);
    assert_eq!(preflight.raw_configured_max_indicators(), 0);
    assert_eq!(preflight.resolved_max_indicators(), 5);
    assert_eq!(preflight.term_cap(), 5);
    assert_eq!(preflight.configured_population(), 10);
    assert!(preflight.population_cap() >= 10);
    assert_eq!(EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1, 3);
    assert_eq!(COST_BAND_OPTION_JSON_UPPER_BOUND_BYTES_V1, 51);
    assert_eq!(ADAPTIVE_TOKEN_OPTION_JSON_UPPER_BOUND_BYTES_V1, 66);
    assert_eq!(EVIDENCE_IDENTITY_JSON_STRING_BYTES_V1, 66);
    assert!(
        preflight.fixed_metadata_upper_bound_with_empty_arrays_bytes()
            >= EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1
    );
    assert!(
        preflight
            .checked_upper_bound_for_population(preflight.population_cap())
            .unwrap()
            <= MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1
    );

    let more_segments = checked_preflight_from_fixed_metadata_shape_v1(
        5,
        0,
        10,
        CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
            total_source_segment_count: 4,
            ..shape
        },
    )
    .unwrap();
    assert_eq!(
        more_segments.fixed_metadata_upper_bound_with_empty_arrays_bytes()
            - preflight.fixed_metadata_upper_bound_with_empty_arrays_bytes(),
        NATIVE_V3_SOURCE_SEGMENT_JSON_UPPER_BOUND_BYTES_V1
    );
}

#[derive(serde::Serialize)]
struct CompactContractCountProbeV1<'a> {
    schema: &'a str,
    version: u16,
    nested: (&'a str, u64, bool),
}

fn independent_compact_object_len_v1(fields: &[(&str, u64)]) -> u64 {
    let separators = fields.len().saturating_sub(1) as u64;
    2 + separators
        + fields
            .iter()
            .map(|(key, value_len)| key.len() as u64 + 3 + value_len)
            .sum::<u64>()
}

#[test]
fn grouped_b_empty_equation_is_exact_stream_counted_and_escape_aware() {
    let lower_hex_string = 66;
    let general_string = MAX_RESULT_STRING_JSON_CONTENT_BYTES_V1 + 2;
    let u16_max = 5;
    let u32_max = 10;
    let integer_max = 20;
    let finite_f64_max = 24;
    let false_json = 5;

    let artifact_reference = independent_compact_object_len_v1(&[
        ("schema", 54),
        ("version", u16_max),
        ("relative_path", 0),
        ("expected_sha256", lower_hex_string),
    ]);
    assert_eq!(artifact_reference, 183);
    let contract_artifact = independent_compact_object_len_v1(&[
        ("reference", artifact_reference),
        ("exact_file_sha256", lower_hex_string),
        ("exact_file_byte_count", integer_max),
        ("contract_domain_identity_sha256", lower_hex_string),
    ]);
    assert_eq!(contract_artifact, 430);
    let runtime_authority = independent_compact_object_len_v1(&[
        ("startup_settings_id", lower_hex_string),
        ("runtime_install_receipt_id", lower_hex_string),
        ("generation_zero_runtime_authority_id", lower_hex_string),
    ]);
    assert_eq!(runtime_authority, 292);
    let unused_full_search = independent_compact_object_len_v1(&[
        ("scope_id", lower_hex_string),
        ("raw_generations", integer_max),
        ("clamped_generations", integer_max),
    ]);
    assert_eq!(unused_full_search, 161);
    let cost_band = independent_compact_object_len_v1(&[
        ("status", 24),
        ("cost", COST_BAND_OPTION_JSON_UPPER_BOUND_BYTES_V1),
    ]);
    assert_eq!(cost_band, 94);
    let limits = independent_compact_object_len_v1(&[
        ("configured_population_cap", integer_max),
        ("resolved_population_cap", integer_max),
        ("term_cap", integer_max),
        ("string_bytes_cap", integer_max),
        ("vector_elements_cap", integer_max),
        ("source_count_cap", integer_max),
        ("result_bytes_cap", integer_max),
    ]);
    assert_eq!(limits, 292);
    let financial_provenance =
        independent_compact_object_len_v1(&[("contract", 0), ("cpu_receipt_id", lower_hex_string)]);
    assert_eq!(financial_provenance, 97);
    let evaluated_native_input =
        independent_compact_object_len_v1(&[("receipt_v3", 0), ("receipt_id", lower_hex_string)]);
    assert_eq!(evaluated_native_input, 95);
    let population_sizing = independent_compact_object_len_v1(&[
        ("receipt_v2", 0),
        ("receipt_id", lower_hex_string),
        ("prepared_feature_count", integer_max),
        ("raw_configured_max_indicators", integer_max),
        ("resolved_max_indicators", integer_max),
        ("term_cap", integer_max),
        ("configured_population", integer_max),
        ("resolved_population", integer_max),
        ("population_cap", integer_max),
        ("hard_growth_cap", integer_max),
        ("max_concurrent_scenario_count", integer_max),
        ("stage1_row_start", integer_max),
        ("stage1_row_end", integer_max),
        ("selected_device_ordinal", u32_max),
        ("metrics_receipt_identities_sha256", 2),
        (
            "adaptive_token_identity_sha256",
            ADAPTIVE_TOKEN_OPTION_JSON_UPPER_BOUND_BYTES_V1,
        ),
    ]);
    assert_eq!(population_sizing, 745);
    let evaluation_snapshot = independent_compact_object_len_v1(&[
        ("symbol", general_string),
        ("account_currency", general_string),
        ("max_hold_bars", integer_max),
        ("trailing_enabled", false_json),
        ("trailing_atr_multiplier", finite_f64_max),
        ("trailing_be_trigger_r", finite_f64_max),
        ("trailing_min_lock_pips", finite_f64_max),
        ("pip_value", finite_f64_max),
        ("spread_pips", finite_f64_max),
        ("commission_per_trade", finite_f64_max),
        ("pip_value_per_lot", finite_f64_max),
        ("swap_long_pips_per_day", finite_f64_max),
        ("swap_short_pips_per_day", finite_f64_max),
        ("pnl_conversion_fee_rate", finite_f64_max),
        ("smc_gate_threshold", finite_f64_max),
        ("smc_weight_ob", finite_f64_max),
        ("smc_weight_fvg", finite_f64_max),
        ("smc_weight_liq", finite_f64_max),
        ("smc_weight_mtf", finite_f64_max),
        ("smc_weight_premium", finite_f64_max),
        ("smc_weight_inducement", finite_f64_max),
        ("smc_weight_bos", finite_f64_max),
        ("smc_weight_choch", finite_f64_max),
        ("smc_weight_eqh", finite_f64_max),
        ("smc_weight_eql", finite_f64_max),
        ("smc_weight_displacement", finite_f64_max),
        ("growth_objective", false_json),
    ]);
    assert_eq!(evaluation_snapshot, 787_554);
    let generation_zero_evaluation = independent_compact_object_len_v1(&[
        ("snapshot_v1", evaluation_snapshot),
        ("snapshot_identity_sha256", lower_hex_string),
        ("scoring_objective", 23),
        ("effective_smc_gate_threshold", finite_f64_max),
        ("effective_smc_gate_source", 49),
        ("genes", 2),
        ("metrics", 2),
    ]);
    assert_eq!(generation_zero_evaluation, 787_866);
    let residency_counters = independent_compact_object_len_v1(&[
        ("parent_upload_count", integer_max),
        ("parent_upload_bytes", integer_max),
        ("view_binding_count", integer_max),
        ("full_binding_count", integer_max),
        ("range_binding_count", integer_max),
        ("ordered_binding_count", integer_max),
        ("ordered_index_upload_bytes", integer_max),
        ("adaptive_upload_bytes", integer_max),
        ("stream_creation_count", integer_max),
        ("explicit_synchronization_count", integer_max),
        ("metric_rows_readback_count", integer_max),
        ("metric_rows_readback_rows", integer_max),
        ("metric_rows_readback_bytes", integer_max),
        ("diagnostic_readback_count", integer_max),
        ("diagnostic_readback_rows", integer_max),
        ("diagnostic_readback_bytes", integer_max),
        ("accepted_trade_total_readback_count", integer_max),
        ("accepted_trade_total_readback_bytes", integer_max),
    ]);
    assert_eq!(residency_counters, 866);
    let completion = independent_compact_object_len_v1(&[
        ("engine", 15),
        ("consumer_completion_confirmed", false_json),
    ]);
    assert_eq!(completion, 64);
    let replay = independent_compact_object_len_v1(&[("replay_identity_sealed", false_json)]);
    assert_eq!(replay, 32);

    let independently_counted_grouped_static = independent_compact_object_len_v1(&[
        ("schema", 62),
        ("version", 5),
        ("scope", 22),
        ("artifact_class", 15),
        ("promotion_eligibility", 24),
        ("authorization_issued", 5),
        ("contract_artifact", contract_artifact),
        ("runtime_authority", runtime_authority),
        ("unused_full_search", unused_full_search),
        ("cost_band_status", cost_band),
        ("limits", limits),
        ("financial_provenance_only", financial_provenance),
        ("evaluated_native_input", evaluated_native_input),
        ("population_sizing", population_sizing),
        ("generation_zero_evaluation", generation_zero_evaluation),
        ("residency_counters", residency_counters),
        ("completion", completion),
        ("replay", replay),
        ("evidence_identity_sha256", lower_hex_string),
    ]);
    assert_eq!(independently_counted_grouped_static, 791_605);

    let contract = CompactContractCountProbeV1 {
        schema: "neoethos.test.contract.v1",
        version: 1,
        nested: ("financial", 17, false),
    };
    let contract_bytes = checked_compact_json_byte_count_v1(&contract).unwrap();
    assert_eq!(
        contract_bytes,
        serde_json::to_vec(&contract).unwrap().len() as u64
    );

    let plain_path = "contracts/a.json";
    let escaped_path = "contracts/\".json";
    assert_eq!(plain_path.len(), escaped_path.len());
    let plain_json = checked_compact_json_string_byte_count_v1(plain_path).unwrap();
    let escaped_json = checked_compact_json_string_byte_count_v1(escaped_path).unwrap();
    assert_eq!((plain_json, escaped_json), (18, 19));

    let checked = |path_json_string_bytes| {
        checked_fixed_metadata_upper_bound_with_empty_arrays_bytes_v1(
            CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
                contract_compact_json_bytes: contract_bytes,
                contract_artifact_relative_path_compact_json_bytes: path_json_string_bytes,
                source_count: 2,
                total_source_segment_count: 3,
            },
        )
        .unwrap()
    };
    let expected_plain =
        8_266_104_u64 + contract_bytes + plain_json + 1_966_378_u64 * 2 + 148_u64 * 3;
    assert_eq!(
        GROUPED_FIXED_METADATA_STATIC_JSON_BYTES_V1,
        independently_counted_grouped_static
    );
    assert_eq!(
        GROUPED_FIXED_METADATA_BASE_WITH_V2_V3_JSON_BYTES_V1,
        8_266_104
    );
    assert_eq!(checked(plain_json), expected_plain);
    assert_eq!(checked(escaped_json), expected_plain + 1);
    assert_eq!(EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1, 3);
    assert_eq!(EMPTY_POPULATION_ARRAYS_COMPACT_JSON_BYTES_V1, 6);

    let preflight = checked_preflight_from_fixed_metadata_shape_v1(
        5,
        5,
        10,
        CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
            contract_compact_json_bytes: contract_bytes,
            contract_artifact_relative_path_compact_json_bytes: plain_json,
            source_count: 2,
            total_source_segment_count: 3,
        },
    )
    .unwrap();
    let per_population = 1_442_u64 + 46 * 5;
    for population in [1_usize, 10, preflight.population_cap()] {
        assert_eq!(
            preflight
                .checked_upper_bound_for_population(population)
                .unwrap(),
            expected_plain - 3 + u64::try_from(population).unwrap() * per_population
        );
    }
    assert!(
        preflight
            .checked_upper_bound_for_population(preflight.population_cap() + 1)
            .is_err()
    );
}

#[test]
fn wrapper_option_and_identity_bounds_cover_their_largest_compact_forms() {
    let cost_band = Some((-f64::MAX, -f64::MAX));
    assert_eq!(
        serde_json::to_vec(&cost_band).unwrap().len() as u64,
        COST_BAND_OPTION_JSON_UPPER_BOUND_BYTES_V1
    );
    let adaptive = Some("a".repeat(64));
    assert_eq!(
        serde_json::to_vec(&adaptive).unwrap().len() as u64,
        ADAPTIVE_TOKEN_OPTION_JSON_UPPER_BOUND_BYTES_V1
    );
    assert_eq!(
        serde_json::to_vec(&"f".repeat(64)).unwrap().len() as u64,
        EVIDENCE_IDENTITY_JSON_STRING_BYTES_V1
    );
    assert_eq!(EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1, 3);
}

#[test]
fn pre_v5_unknown_receipt_bounds_match_the_reviewed_analytic_census() {
    assert_eq!(
        RESIDENT_POPULATION_SIZING_RECEIPT_V2_JSON_UPPER_BOUND_BYTES_V1,
        7_080_504
    );
    for (sources, segments) in [(1, 1), (2, 2), (2, 3), (12, 1_000_000)] {
        assert_eq!(
            checked_native_v3_receipt_json_upper_bound_bytes_v1(sources, segments).unwrap(),
            393_995_u64
                + 1_966_378_u64 * u64::try_from(sources).unwrap()
                + 148_u64 * u64::try_from(segments).unwrap()
        );
    }
    for (sources, segments) in [
        (0, 0),
        (1, 0),
        (2, 1),
        (MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1 + 1, 15),
        (1, 1_000_001),
        (1, usize::MAX),
    ] {
        assert!(checked_native_v3_receipt_json_upper_bound_bytes_v1(sources, segments).is_err());
    }
}

#[test]
fn receipt_schema_census_prevents_magic_upper_bounds_from_hiding_new_fields() {
    let v2_source = include_str!("resident_population_auto_sizing_receipt_v2.rs");
    let v2_fields = v2_source
        .split_once("pub struct ResidentPopulationAutoSizingReceiptV2 {")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    let count = |suffix: &str| {
        v2_fields
            .lines()
            .filter(|line| line.trim_end().ends_with(suffix))
            .count()
    };
    assert_eq!(count(": String,"), 18);
    assert_eq!(count(": u64,"), 32);
    assert_eq!(count(": bool,"), 4);
    assert_eq!(count(": u16,"), 1);
    assert_eq!(count(": u32,"), 1);
    assert_eq!(count(": [u8; 32],"), 2);
    assert_eq!(18 + 32 + 4 + 1 + 1 + 2, 58);
    assert_eq!(
        RESIDENT_POPULATION_SIZING_RECEIPT_V2_FIXED_JSON_BYTES_V1,
        2_616
    );
    assert_eq!(
        RESIDENT_POPULATION_SIZING_RECEIPT_V2_JSON_UPPER_BOUND_BYTES_V1,
        RESIDENT_POPULATION_SIZING_RECEIPT_V2_FIXED_JSON_BYTES_V1
            + 18 * MAX_RESULT_STRING_JSON_CONTENT_BYTES_V1
    );

    let v3_source = include_str!("data_selection.rs");
    let v3 = v3_source
        .split_once("pub struct CanonicalGpuResidentSearchInputReceiptV3 {")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    assert_eq!(v3.matches(": String,").count(), 6);
    assert_eq!(v3.matches(": u64,").count(), 2);
    assert_eq!(NATIVE_V3_FIXED_JSON_UPPER_BOUND_BYTES_V1, 393_995);
    assert_eq!(
        NATIVE_V3_SOURCE_BINDING_JSON_UPPER_BOUND_BYTES_V1,
        1_966_378
    );
    assert_eq!(NATIVE_V3_SOURCE_SEGMENT_JSON_UPPER_BOUND_BYTES_V1, 148);
}

#[test]
fn gene_schema_census_ratchets_every_bounded_field_and_serde_attribute() {
    let source = include_str!("genetic/strategy_gene.rs");
    let body = source
        .split_once("pub struct Gene {")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    let expected = [
        "pub indices: Vec<usize>,",
        "pub weights: Vec<f64>,",
        "pub long_threshold: f64,",
        "pub short_threshold: f64,",
        "pub fitness: f64,",
        "pub sharpe_ratio: f64,",
        "pub win_rate: f64,",
        "pub max_drawdown: f64,",
        "pub profit_factor: f64,",
        "pub expectancy: f64,",
        "pub trades_count: usize,",
        "pub generation: usize,",
        "pub strategy_id: String,",
        "pub use_ob: bool,",
        "pub use_fvg: bool,",
        "pub use_liq_sweep: bool,",
        "pub mtf_confirmation: bool,",
        "pub use_premium_discount: bool,",
        "pub use_inducement: bool,",
        "pub use_bos: bool,",
        "pub use_choch: bool,",
        "pub use_eqh: bool,",
        "pub use_eql: bool,",
        "pub use_displacement: bool,",
        "pub tp_pips: f64,",
        "pub sl_pips: f64,",
        "pub slice_pass_rate: f64,",
        "pub consistency: f64,",
        "pub stop_vol_mult: f64,",
    ];
    let field_lines: Vec<_> = body
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("pub ") && line.ends_with(','))
        .collect();
    assert_eq!(field_lines, expected);
    assert_eq!(body.matches("Vec<").count(), 2);
    assert_eq!(body.matches(": f64,").count(), 13);
    assert_eq!(body.matches(": usize,").count(), 2);
    assert_eq!(body.matches(": String,").count(), 1);
    assert_eq!(body.matches(": bool,").count(), 11);
    assert_eq!(body.matches("#[serde(default)]").count(), 6);
    for forbidden in ["serde(skip", "serde(flatten", "serde(rename"] {
        assert!(!body.contains(forbidden), "Gene wire contains {forbidden}");
    }
}

#[test]
fn direct_population_schema_bounds_use_finite_f64_extrema_and_lowerhex_receipts() {
    assert_eq!(serde_json::to_vec(&-f64::MAX).unwrap().len(), 24);
    assert_eq!(MAX_FINITE_F64_JSON_BYTES_V1, 24);

    let metrics = [-f64::MAX; 11];
    assert_eq!(serde_json::to_vec(&metrics).unwrap().len(), 276);
    assert_eq!(METRIC_ROW_JSON_UPPER_BOUND_BYTES_V1, 276);

    let receipt = "a".repeat(64);
    assert_eq!(serde_json::to_vec(&receipt).unwrap().len(), 66);
    assert_eq!(METRIC_RECEIPT_LOWER_HEX_STRING_UPPER_BOUND_BYTES_V1, 66);

    for term_cap in [1_usize, 5, 4_096] {
        let gene = checked_gene_json_upper_bound_bytes_v1(term_cap).unwrap();
        assert_eq!(gene, 1_097 + 46 * term_cap as u64);
        assert_eq!(
            checked_per_population_json_upper_bound_bytes_v1(term_cap).unwrap(),
            gene + 276 + 66 + 3
        );
        assert_eq!(
            checked_per_population_json_upper_bound_bytes_v1(term_cap).unwrap(),
            1_442 + 46 * term_cap as u64
        );
    }
}

#[test]
fn strategy_id_worst_escape_is_258_bytes_and_stays_inside_the_gene_bound() {
    let strategy_id = worst_escape_strategy_id_v1();
    assert_eq!(strategy_id.len(), 128);
    assert!(strategy_id.bytes().all(|byte| byte.is_ascii_graphic()));
    assert_eq!(serde_json::to_vec(&strategy_id).unwrap().len(), 258);
    validate_strategy_id_v1(&strategy_id).expect("128 graphic bytes are admitted");

    let term_cap = 5;
    let mut gene = Gene {
        indices: vec![usize::MAX; term_cap],
        weights: vec![-f64::MAX; term_cap],
        long_threshold: -f64::MAX,
        short_threshold: -f64::MAX,
        fitness: -f64::MAX,
        sharpe_ratio: -f64::MAX,
        win_rate: -f64::MAX,
        max_drawdown: -f64::MAX,
        profit_factor: -f64::MAX,
        expectancy: -f64::MAX,
        trades_count: usize::MAX,
        generation: 0,
        strategy_id,
        tp_pips: -f64::MAX,
        sl_pips: -f64::MAX,
        slice_pass_rate: -f64::MAX,
        consistency: -f64::MAX,
        stop_vol_mult: -f64::MAX,
        ..Gene::default()
    };
    assert!(
        serde_json::to_vec(&gene).unwrap().len()
            <= checked_gene_json_upper_bound_bytes_v1(term_cap).unwrap() as usize
    );
    gene.strategy_id.push('x');
    assert!(validate_strategy_id_v1(&gene.strategy_id).is_err());
    for invalid in ["", "bad id", "bad\nid", "στρατηγική"] {
        assert!(validate_strategy_id_v1(invalid).is_err());
    }
}
