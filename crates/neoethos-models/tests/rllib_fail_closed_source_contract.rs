const ORCHESTRATOR: &str = include_str!("../src/training_orchestrator.rs");
const CONFIG: &str = include_str!("../../neoethos-core/src/config.rs");
const SHIPPED_CONFIG: &str = include_str!("../../../desktop/src-tauri/resources/config.yaml");

const LEGACY_SUBSTITUTION_WARNING: &str = "rllib/Ray requested for DQN but is unavailable in the pure-Rust build; using rlkit (native) backend instead";
const MIGRATION_ERROR: &str = "legacy RLlib migration boundary v1: `use_rllib_agent`, `auto_enable_rllib`, and non-zero `rllib_num_workers` are unsupported because this build has no Ray runtime; set the flags to false, set `rllib_num_workers: 0`, and use `use_rl_agent: true` for the native rlkit DQN backend";

#[test]
fn shipped_defaults_do_not_auto_request_an_unavailable_rllib_runtime() {
    assert!(CONFIG.contains("auto_enable_rllib: false,"));
    assert!(SHIPPED_CONFIG.contains("  auto_enable_rllib: false"));
    assert!(SHIPPED_CONFIG.contains("  use_rllib_agent: false"));
}

#[test]
fn legacy_rllib_flags_fail_before_dispatch_instead_of_selecting_rlkit() {
    for required in [
        "reject_legacy_rllib_request_v1(&self.settings)?;",
        "fn reject_legacy_rllib_request_v1(settings: &Settings) -> Result<()> {",
        "settings.models.use_rllib_agent",
        "settings.models.auto_enable_rllib",
        "settings.models.rllib_num_workers != 0",
        MIGRATION_ERROR,
        "reject_legacy_rllib_model_params_v1(&entry.name, &params)?;",
        "fn reject_legacy_rllib_model_params_v1(\n    model_name: &str,\n    params: &HashMap<String, String>,\n) -> Result<()> {",
    ] {
        assert!(
            ORCHESTRATOR.contains(required),
            "missing fail-closed RLlib migration boundary {required:?}"
        );
    }

    let guard = ORCHESTRATOR
        .find("reject_legacy_rllib_request_v1(&self.settings)?;")
        .expect("migration guard must exist");
    let dispatch_materialization = ORCHESTRATOR
        .find("let mut requested_models = self.settings.models.ml_models.clone();")
        .expect("dispatch plan materialization must exist");
    assert!(
        guard < dispatch_materialization,
        "legacy RLlib flags must fail before dispatch materialization"
    );
}

#[test]
fn native_rlkit_is_an_explicit_backend_and_never_an_rllib_fallback() {
    assert!(ORCHESTRATOR.contains("(\"backend\".to_string(), \"rlkit\".to_string())"));
    for retired in [
        LEGACY_SUBSTITUTION_WARNING,
        "rllib_unavailable_warn_once",
        "(\"__rllib_requested\".to_string(),",
        "\"auto_rllib\".to_string()",
        "let rllib_auto =",
        "training executed on the native rlkit backend",
    ] {
        assert!(
            !ORCHESTRATOR.contains(retired),
            "stale RLlib-to-rlkit substitution path remains: {retired:?}"
        );
    }
}
