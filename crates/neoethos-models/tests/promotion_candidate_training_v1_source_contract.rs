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
fn app_training_moves_selected_handoff_and_preserves_typed_refusal() {
    let source = read("crates/neoethos-app/src/app_services/training.rs");
    let run = function_body(&source, "fn start_training_job_impl(");
    let selected = run
        .split_once("if let Some(handoff) = selected_handoff {")
        .expect("training dispatches the selected move-owned handoff")
        .1
        .split_once("let model_names = manifest")
        .expect("training handles its typed terminal before consuming the manifest")
        .0;
    let selected: String = selected
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    for required in [
        "letterminal=train_and_deploy_promotion_candidate_v1(&settings,handoff,&candidate_root,&settings.system.data_dir,&lease,progress,);",
        "letmanifest=matchterminal{",
        "PromotionCandidateTrainingTerminalV1::Installed(manifest)|PromotionCandidateTrainingTerminalV1::ExistingIdentical(manifest)=>manifest,",
        "PromotionCandidateTrainingTerminalV1::Refused(error)=>{returnErr(error.into());}",
    ] {
        assert!(
            selected.contains(required),
            "selected candidate training omits `{required}`"
        );
    }
    assert_eq!(
        selected
            .matches("train_and_deploy_promotion_candidate_v1(")
            .count(),
        1
    );
    assert!(
        !selected.contains("clone()") && !selected.contains("JobState::Degraded"),
        "selected candidate training must move the handoff and preserve Refused without Degraded"
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

#[test]
fn app_training_reloads_the_exact_installed_candidate_through_the_inference_consumer() {
    let source = read("crates/neoethos-app/src/app_services/training.rs");
    let reload = source
        .find("bootstrap::build_ensemble_for_candidate(")
        .expect("training reloads its exact candidate");
    let call = &source[reload
        ..source[reload..]
            .find(")\n")
            .map(|offset| reload + offset)
            .expect("candidate load call closes")];
    for required in ["&candidate_root", "&manifest", "&settings"] {
        assert!(
            call.contains(required),
            "candidate consumer omits {required}"
        );
    }
    let bootstrap = read("crates/neoethos-models/src/ensemble_inference/bootstrap.rs");
    let shared_call =
        "build_ensemble_from_validated_candidate(candidate_root,manifest,settings,&handoff)";
    let mut guarded_bodies = Vec::new();
    for (marker, required_validation, forbidden_validation) in [
        (
            "pub fn build_ensemble_for_candidate(",
            ".validate_against_settings_v1(settings)",
            ".validate_inference_settings_v1(",
        ),
        (
            "pub fn build_ensemble_for_candidate_inference(",
            ".validate_inference_settings_v1(settings)",
            ".validate_against_settings_v1(",
        ),
    ] {
        let wrapper: String = function_body(&bootstrap, marker)
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        let reopen = wrapper
            .find(".reopen_handoff(candidate_root)")
            .expect("candidate wrapper reopens the exact selected handoff");
        let validation = wrapper
            .find(required_validation)
            .unwrap_or_else(|| panic!("{marker} omits {required_validation}"));
        let dispatch = wrapper
            .find(shared_call)
            .expect("candidate wrapper forwards its exact validated inputs to the shared loader");
        assert!(
            reopen < validation && validation < dispatch,
            "{marker} must reopen and validate before shared loading"
        );
        assert!(
            wrapper[validation..dispatch].ends_with("?;"),
            "{marker} must propagate validation failure before shared loading"
        );
        assert_eq!(wrapper.matches(shared_call).count(), 1);
        assert!(
            wrapper.ends_with(&format!("{shared_call}}}")),
            "{marker} must return the shared loader result directly"
        );
        assert!(
            !wrapper.contains(forbidden_validation),
            "{marker} uses the other caller's validation policy"
        );
        guarded_bodies.push(wrapper);
    }
    let shared: String = function_body(&bootstrap, "fn build_ensemble_from_validated_candidate(")
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let mut remaining = shared.as_str();
    for required in [
        "candidate_root.join(manifest.candidate_relative_dir())",
        "load_model_feature_input_for_handoff_v1(&models_root,handoff,)",
        "handoff.canonical_series().anchor().identity().symbol_name()",
        "load_experts_for_symbol(&models_root,symbol,handoff.base_timeframe().as_str())?",
        "manifest.model_artifacts().iter().map(|artifact|artifact.model_name())",
        "bind_candidate_inventory(&mutoutcome,&expected)?",
        "validate_loaded_model_inputs(&models_root,symbol,handoff.base_timeframe().as_str(),&model_feature_input,&outcome,)?",
        "voting_config(&settings.models.ensemble_voting)?",
        ".bind_model_feature_input(model_feature_input)?",
        ".verify_installed(candidate_root)",
        "Ok(ensemble)",
    ] {
        remaining = remaining
            .split_once(required)
            .unwrap_or_else(|| panic!("shared candidate loader omits or reorders {required}"))
            .1;
    }
    let verification = shared.find(".verify_installed(candidate_root)").unwrap();
    let returned = shared.rfind("Ok(ensemble)").unwrap();
    assert!(
        shared[verification..returned].ends_with("?;"),
        "shared candidate loader must propagate post-load tree verification failure"
    );
    guarded_bodies.push(shared);
    for body in guarded_bodies {
        for forbidden in [
            "Settings::load",
            "Settings::default",
            "voting_config_from_settings",
            "build_ensemble_for_symbol(",
            "symbol_model_feature_input(",
            "role_decisions_from_feature_frame(",
            "read_dir(",
            "latest",
            "live_models",
            "unwrap_or_default(",
            ".or_else(",
        ] {
            assert!(
                !body.contains(forbidden),
                "candidate loading reaches ambient or fallback route {forbidden}"
            );
        }
    }
}
