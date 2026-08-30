use std::fs;
use std::path::PathBuf;

fn read(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

#[test]
fn canonical_contract_build_seals_exact_cpu_feature_receipt_without_starting_research() {
    let main = read("src/main.rs");
    let source = read("src/canonical_full_run.rs");

    assert!(
        main.contains(
            "\"canonical-contract-build\" => canonical_full_run::build_contract(tail, settings),"
        ),
        "CLI does not expose the standalone canonical contract builder"
    );
    for flag in [
        "--authority-root",
        "--data-root",
        "--plan-sha256",
        "--matrix-sha256",
        "--symbol",
        "--base-timeframe",
        "--cost-assumptions",
        "--broker-symbol-contract",
        "--settings-source",
        "--contract-out",
        "--receipt-out",
    ] {
        assert!(
            source.contains(flag),
            "canonical contract builder omits required flag {flag}",
        );
    }

    let start = source
        .find("pub fn build_contract(")
        .expect("standalone canonical contract builder");
    let end = source[start..]
        .find("\npub fn ")
        .map_or(source.len(), |offset| start + offset);
    let builder = &source[start..end];

    for required in [
        "validate_contract_build_args(args)?",
        "ensure_distinct_output_targets(",
        "store.open_plan(&plan_receipt)?",
        "store.open_matrix(&data_root, &plan_receipt, &matrix_receipt)?",
        "canonical_feature_options(settings, base_timeframe)?",
        "CanonicalSearchInput::from_exact_series_receipt_gpu_exact_parity_cpu_reference_v3(",
        "search_input.receipt()?",
        "CanonicalTrendbarResearchExecutionContractV3::new(",
        "contract.validate_against_receipt(&receipt)?",
        "write_json_atomic(&contract_out, &contract)",
        "write_json_atomic(&receipt_out, &receipt)",
        "CanonicalTrendbarResearchExecutionContractV3 =",
        "CanonicalSearchInputReceiptV2::from_json_bytes(",
        "contract_sha256",
        "receipt_sha256",
        "canonical_discovery_normalization_training_rows(",
        "base_row_count",
        "oos_from_ms",
    ] {
        assert!(
            builder.contains(required),
            "canonical contract builder is missing `{required}`",
        );
    }
    for forbidden in [
        "prepare_canonical_discovery_run_input",
        "TrainingOrchestrator",
        "run_discovery",
        "Cuda",
        "CUDA",
    ] {
        assert!(
            !builder.contains(forbidden),
            "evidence-only contract builder must not invoke {forbidden}",
        );
    }

    let receipt_write = builder
        .find("write_json_atomic(&receipt_out, &receipt)")
        .expect("receipt publication");
    let contract_write = builder
        .find("write_json_atomic(&contract_out, &contract)")
        .expect("contract publication");
    assert!(
        receipt_write < contract_write,
        "the contract must be the final published member of the two-file handoff",
    );

    let training_start = source
        .find("pub fn train_receipt_bound(")
        .expect("canonical training entry");
    let training_end = source[training_start..]
        .find("\npub fn ")
        .map_or(source.len(), |offset| training_start + offset);
    let training = &source[training_start..training_end];
    for required in [
        "canonical_discovery_normalization_training_rows(",
        "exact_training_oos_from_ms",
        "training_oos_from_ms == exact_training_oos_from_ms",
    ] {
        assert!(
            training.contains(required),
            "canonical training does not bind the exact OOS boundary with `{required}`",
        );
    }
}
