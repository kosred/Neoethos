use std::fs;
use std::path::PathBuf;

fn workspace_read(relative: &str) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root");
    let path = workspace.join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

#[test]
fn cpu_discover_research_is_exact_receipt_bound_and_never_exports_a_live_portfolio() {
    let main = workspace_read("crates/neoethos-cli/src/main.rs");
    let module = workspace_read("crates/neoethos-cli/src/canonical_full_run.rs");
    let research = workspace_read("crates/neoethos-search/src/canonical_trendbar_research.rs");

    for required in [
        "const CANONICAL_CPU_RESEARCH_FLAGS",
        "--research-authority-root",
        "--research-plan-sha256",
        "--research-matrix-sha256",
        "--research-cost-assumptions",
        "--research-broker-symbol-contract",
        "--research-settings-source",
        "positions.len() == 1",
        "canonical_cpu_research.is_some() || has_flag(args, \"--stream-sweep\")",
    ] {
        assert!(
            main.contains(required),
            "CPU research CLI omits `{required}`"
        );
    }

    let branch = main
        .split("if let Some(research_inputs) = canonical_cpu_research.as_ref() {")
        .nth(1)
        .expect("CPU research discover branch")
        .split("#[cfg(not(feature = \"gpu-nvidia\"))]\n        let (result, streaming)")
        .next()
        .expect("end of CPU research discover branch");
    for required in [
        "canonical_discovery_normalization_training_rows",
        "StreamingPlan::streaming(stream_max_batches)",
        "prepare_multitimeframe_features_batch_with_options",
        "CanonicalSearchRunInputV2::from_fresh_feature_frame",
        "seal_cpu_research_contract_for_input",
        "try_from_settings_for_canonical_trendbar_research",
        "run_canonical_trendbar_research_discovery_with_holdout_and_progress",
        "completed_without_survivors",
        "write_json_atomic(&path, research)",
        "neoethos.canonical-cpu-research-streaming-index.v1",
        "artifact_class=ResearchOnly",
        "promotion_eligibility=NotPromotionEligible",
        "authorization_issued=false",
    ] {
        assert!(
            branch.contains(required),
            "CPU research discover branch omits `{required}`"
        );
    }
    for forbidden in ["save_portfolio_json", "save_live_portfolio_json"] {
        assert!(
            !branch.contains(forbidden),
            "CPU research branch can leak into `{forbidden}`"
        );
    }
    assert!(
        !main.contains("canonical CPU research does not support --stream-sweep"),
        "canonical CPU research still rejects the adaptive working-set sweep"
    );

    let seal = module
        .split("pub(crate) fn seal_cpu_research_contract_for_input(")
        .nth(1)
        .expect("CPU research contract sealer");
    for required in [
        "store.open_plan",
        "store.open_matrix",
        "ensure_unique_series",
        "validate_input_receipt_against_series",
        "validate_settings_source",
        "validate_broker_symbol_contract",
        "validate_costs",
        "CanonicalTrendbarResearchExecutionContractV3::new",
        "contract.validate_against_receipt",
    ] {
        assert!(
            seal.contains(required),
            "CPU research contract sealer omits `{required}`"
        );
    }

    assert!(
        research.contains("#[derive(Debug, Clone, Serialize)]")
            && research.contains("pub struct CanonicalTrendbarResearchDiscoveryResultV3"),
        "the complete research-only result envelope is not serializable"
    );
}
