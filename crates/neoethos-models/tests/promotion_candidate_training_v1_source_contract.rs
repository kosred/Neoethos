use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("models crate is below workspace root")
        .to_path_buf()
}

fn read(relative: &str) -> String {
    let path = workspace_root().join(relative);
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
fn promotion_candidate_training_uses_only_exact_receipts_and_the_frozen_cutoff() {
    let source = read("crates/neoethos-models/src/promotion_candidate_training_v1.rs");
    let run = function_body(
        &source,
        "pub fn train_and_deploy_promotion_candidate_v1<R>(",
    );
    for required in [
        "validate_against_settings_v1",
        "with_data_root(data_root)",
        "with_oos_lock_from_ms(handoff.oos_cutoff_ms())",
        "with_sealed_hardware_plan_v1",
        "train_canonical_series_receipt_with_progress",
        "handoff.canonical_series()",
        "handoff.base_timeframe()",
        "handoff.search_input_receipt()",
        "handoff.screening_contract()",
        "install_promotion_candidate_model_tree_v1",
        "PromotionCandidateTrainingTerminalV1::Refused",
    ] {
        assert!(run.contains(required), "P3 training omits `{required}`");
    }
    for forbidden in [
        "load_symbol_dataset",
        "load_canonical_timeframe",
        "current_generation",
        "models/",
        "live_models",
        "JobState::Degraded",
    ] {
        assert!(
            !run.contains(forbidden),
            "P3 training reaches forbidden ambient/live path `{forbidden}`"
        );
    }
}

#[test]
fn general_training_preflight_seals_the_effective_model_inventory_before_training() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let preflight = function_body(&source, "pub fn preflight_configured_training(&self)");
    assert!(
        preflight.contains("self.configured_training_plan_v1()?"),
        "public preflight must use the same exact plan material as P3 identity sealing"
    );
    let sealed_plan = function_body(&source, "fn configured_training_plan_v1(&self)");
    for required in [
        "self.create_dispatch_plan()?",
        "self.validate_dispatch_plan(&dispatch_plan)?",
        "self.build_training_configs_with_hardware_plan(&dispatch_plan, &hardware_plan)?",
        "configured training resolved an empty model plan",
    ] {
        assert!(
            sealed_plan.contains(required),
            "sealed effective plan omits `{required}`"
        );
    }
    let identity_material = function_body(
        &source,
        "pub(crate) fn promotion_candidate_training_plan_material_v1(",
    );
    for required in [
        "self.configured_training_plan_v1()?",
        "BTreeMap",
        "model_type",
        "capability_family",
        "capability_state",
        "params",
    ] {
        assert!(
            identity_material.contains(required),
            "P3 model identity material omits `{required}`"
        );
    }
}

#[test]
fn app_adapter_preserves_move_only_handoff_and_typed_refused_terminal() {
    let source = read("crates/neoethos-app/src/app_services/training.rs");
    let run = function_body(&source, "pub fn run_promotion_candidate_training_v1<R>(");
    for required in [
        "PromotionCandidateTrainingHandoffV1",
        "PromotionCandidateTrainingTerminalV1",
        "train_and_deploy_promotion_candidate_v1",
        "candidate_root",
        "data_root",
        "lease",
    ] {
        assert!(run.contains(required), "app P3 adapter omits `{required}`");
    }
    assert!(
        !run.contains("clone()") && !run.contains("JobState::Degraded"),
        "app P3 adapter must move the handoff and preserve Refused without Degraded"
    );
}

#[test]
fn p3_does_not_modify_live_or_promotion_paths() {
    let source = read("crates/neoethos-models/src/promotion_candidate_training_v1.rs");
    for forbidden in [
        "live_trading",
        "promotion_gate",
        "live_models",
        "authorization_issued: true",
        "PromotionEligible",
    ] {
        assert!(
            !source.contains(forbidden),
            "P3 source contains `{forbidden}`"
        );
    }
}
