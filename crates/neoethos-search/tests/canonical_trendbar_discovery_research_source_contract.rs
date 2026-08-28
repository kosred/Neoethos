use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn function_body<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing function marker {marker:?}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function has an opening brace");
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function {marker:?} has no closing brace")
}

#[test]
fn canonical_trendbar_research_contract_v3_is_receipt_bound_and_never_a_broker_permit() {
    let source = read("src/canonical_trendbar_research.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("canonical trendbar research production source");

    for required in [
        "pub struct CanonicalTrendbarResearchExecutionContractV3",
        "CanonicalSearchInputReceiptV2",
        "input_receipt_sha256",
        "HistoricalResearchArtifactClassV1::ResearchOnly",
        "HistoricalResearchPromotionEligibilityV1::NotPromotionEligible",
        "pip_size",
        "pip_value_per_lot",
        "full_spread_pips_assumption",
        "slippage_pips_per_fill_assumption",
        "commission_account_per_lot_per_fill_assumption",
        "screening_spread_and_slippage_round_trip_pips",
        "round_trip_commission_account_per_lot",
        "screening_round_trip_cost_pips",
        "swap_long_pips_per_day",
        "swap_short_pips_per_day",
        "pnl_conversion_fee_rate",
        "pub struct CanonicalTrendbarResearchDiscoveryResultV3",
        "pub fn validate_against_input(",
        "pub(crate) fn active_canonical_trendbar_research_execution_v3(",
        "active canonical-trendbar research execution already exists",
    ] {
        assert!(
            source.contains(required),
            "canonical research contract is missing `{required}`"
        );
    }

    for forbidden in [
        "BrokerFinancialTruthCapabilityV1",
        "BrokerFinancialTruthPermitV1",
        "current_broker_financial_truth_capability_v1",
        "tick",
        "BidAsk",
        "resample",
        "NotPromotionEligible,\n        permit_issued: true",
        "CanonicalTrendbarResearchExecutionContractV1",
        "CanonicalTrendbarResearchCostAssumptionsV1",
        "round_trip_commission_per_trade",
    ] {
        assert!(
            !production.contains(forbidden),
            "canonical research contract contains forbidden authority/data path `{forbidden}`"
        );
    }
}

#[test]
fn full_discovery_has_a_separate_research_only_entrypoint_while_broker_entrypoints_stay_closed() {
    let discovery = read("src/discovery.rs");
    let library = read("src/lib.rs");

    for required in [
        "pub fn run_canonical_trendbar_research_discovery_with_holdout_and_progress",
        "CanonicalTrendbarResearchExecutionContractV3",
        "CanonicalTrendbarResearchDiscoveryResultV3",
        "install_canonical_trendbar_research_execution_v3",
    ] {
        assert!(
            discovery.contains(required) || library.contains(required),
            "full discovery research boundary is missing `{required}`"
        );
    }

    let broker_entry = function_body(
        &discovery,
        "pub fn run_discovery_cycle_with_holdout_and_progress<F>(",
    );
    assert!(
        broker_entry.contains("current_broker_financial_truth_capability_v1"),
        "the existing broker-real discovery entrypoint no longer fails closed"
    );

    let research_entry = function_body(
        &discovery,
        "pub fn run_canonical_trendbar_research_discovery_with_holdout_and_progress<F>(",
    );
    assert!(
        research_entry.contains("validate_against_input")
            && research_entry.contains("install_canonical_trendbar_research_execution_v3")
            && research_entry.contains("run_discovery_cycle_with_holdout_and_progress_authorized"),
        "research discovery does not validate/install its exact authority before arithmetic"
    );
    assert!(
        !research_entry.contains("current_broker_financial_truth_capability_v1"),
        "research discovery still asks for the forbidden tick/BidAsk capability"
    );
}

#[test]
fn numerical_search_workers_accept_only_broker_truth_or_the_active_exact_research_scope() {
    let authority = read("src/historical_evaluation_authority.rs");
    for required in [
        "pub(crate) fn require_historical_evaluation_authority_v1(",
        "current_broker_financial_truth_capability_v1",
        "active_canonical_trendbar_research_execution_v3",
    ] {
        assert!(
            authority.contains(required),
            "combined historical-evaluation authority is missing `{required}`"
        );
    }

    for (path, marker) in [
        (
            "src/backend.rs",
            "pub fn evaluate_population_core_with_backend_and_audit(",
        ),
        ("src/eval.rs", "fn require_historical_evaluation_authority("),
        (
            "src/validation.rs",
            "pub fn embargoed_walkforward_backtest(",
        ),
        (
            "src/genetic/search_engine.rs",
            "pub fn validation_genes_population(",
        ),
        (
            "src/gpu_native/prototype_population_oracle.rs",
            "pub fn evaluate_population_oracle(",
        ),
    ] {
        let source = read(path);
        let body = function_body(&source, marker);
        assert!(
            body.contains("require_historical_evaluation_authority_v1"),
            "{path}::{marker} does not require the broker-or-exact-research authority"
        );
    }

    let discovery = read("src/discovery.rs");
    let faithful_oos = function_body(&discovery, "pub fn faithful_oos_eval(");
    assert!(
        faithful_oos.contains("current_broker_financial_truth_capability_v1"),
        "the live-portfolio faithful OOS path was weakened to research authority"
    );
}

#[test]
fn search_input_can_be_built_only_from_the_exact_selected_series_generations() {
    let source = read("src/data_selection.rs");
    let body = function_body(&source, "pub fn from_exact_series_receipt(");

    for required in [
        "CanonicalDatasetSeriesReceiptV1",
        "CanonicalTimeframe",
        ".validate()",
        "direct_timeframes()",
        "load_exact_dataset_series_receipt",
        "prepare_multitimeframe_features_with_options",
        "CanonicalSearchInput",
    ] {
        assert!(
            body.contains(required),
            "exact-series search-input builder is missing `{required}`"
        );
    }
    for forbidden in [
        "ExactCanonicalSeries::open",
        "load_canonical_timeframe(",
        "load_symbol_dataset(",
        "discover_",
        "current_generation",
        "resample",
    ] {
        assert!(
            !body.contains(forbidden),
            "exact-series search-input builder contains forbidden path `{forbidden}`"
        );
    }
}

#[test]
fn research_evidence_identity_hashes_the_complete_serialized_discovery_result() {
    let discovery = read("src/discovery.rs");
    assert!(
        discovery.contains("#[derive(Debug, Clone, Serialize)]\npub struct DiscoveryResult"),
        "DiscoveryResult is not a serializable evidence payload"
    );

    let contract = read("src/canonical_trendbar_research.rs");
    let identity = function_body(&contract, "fn result_identity_sha256(");
    for required in ["serde_json::to_vec(result)", "push_bytes(&mut bytes"] {
        assert!(
            identity.contains(required),
            "research result identity omits `{required}`"
        );
    }
}

#[test]
fn normalization_and_holdout_use_one_exact_public_split_boundary() {
    let discovery = read("src/discovery.rs");
    let boundary = function_body(
        &discovery,
        "pub fn canonical_discovery_normalization_training_rows(",
    );
    for required in [
        "DEFAULT_OOS_HOLDOUT_FRACTION",
        "0..split_at",
        "split_at >= 64",
    ] {
        assert!(
            boundary.contains(required),
            "canonical normalization boundary is missing `{required}`"
        );
    }
    let holdout = function_body(&discovery, "fn with_holdout(");
    assert!(
        holdout.contains("canonical_discovery_normalization_training_rows"),
        "holdout split duplicates or diverges from the normalization boundary"
    );
}

#[test]
fn research_sensitivity_commission_cannot_be_cheaper_than_screening_baseline() {
    let source = read("src/discovery.rs");
    let apply = function_body(&source, "fn apply_research_contract_to_discovery_config(");
    assert!(
        apply.contains("config.sensitivity_commission_per_lot")
            && apply.contains(".max(contract.round_trip_commission_account_per_lot())"),
        "research contract can install a baseline commission above its sensitivity stress"
    );
}

#[test]
fn screening_contract_installs_two_slippage_fills_and_two_commission_fills() {
    let source = read("src/discovery.rs");
    let apply = function_body(&source, "fn apply_research_contract_to_discovery_config(");
    assert!(
        apply.contains("contract.screening_spread_and_slippage_round_trip_pips()")
            && apply.contains("contract.round_trip_commission_account_per_lot()"),
        "discovery does not consume the V2 per-fill screening-cost semantics"
    );
}

#[test]
fn screening_spread_assumption_never_claims_protooa_symbol_spread_provenance() {
    let metadata = read("../neoethos-core/src/symbol_metadata.rs");
    let search = read("src/genetic/strategy_gene.rs");
    for forbidden in [
        "ProtoOASymbol::spread",
        "ProtoOASymbol.spread",
        "broker-authoritative typical spread",
        "real broker spread",
        "ολα τα νουμερα",
    ] {
        assert!(
            !metadata.contains(forbidden) && !search.contains(forbidden),
            "screening spread retains false broker provenance `{forbidden}`"
        );
    }
}
