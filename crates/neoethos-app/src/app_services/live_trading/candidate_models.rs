//! Load the models belonging to the selected portfolio, not an ambient symbol folder.
//! Matching training/research artifacts is not broker or promotion permission.

use anyhow::{Context, Result, ensure};
use neoethos_core::Settings;
use neoethos_models::{
    MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1, PromotionCandidateTrainingManifestV1,
    ensemble_inference::{SoftVotingEnsemble, bootstrap::build_ensemble_for_candidate_inference},
};
use std::path::Path;

use crate::app_services::training::{handoff, read_saved_combined_research_reports};

pub(super) struct LoadedCandidate {
    pub ensemble: SoftVotingEnsemble,
    pub training_handoff: String,
    pub candidate_tree: String,
}

/// Resolve only exact portfolio/config matches, never newest/best-performing models.
/// All filesystem/model work happens at engine startup, outside the bar loop.
pub(super) fn load(
    settings: &Settings,
    candidate_root: &Path,
    portfolio_identity: &str,
) -> Result<LoadedCandidate> {
    let (identity, selected_settings) =
        select_handoff(settings, candidate_root, portfolio_identity)?;
    let bytes = neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit(
        &candidate_root.join(format!("{identity}.manifest.json")),
        MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 as u64,
    )?;
    let manifest: PromotionCandidateTrainingManifestV1 = serde_json::from_slice(&bytes)?;
    let installed = manifest.reopen_handoff(candidate_root)?;
    ensure!(
        installed.identity_sha256()? == identity
            && installed.locked_portfolio().identity_sha256() == portfolio_identity,
        "installed model manifest does not belong to the selected portfolio/handoff"
    );
    installed.validate_inference_settings_v1(&selected_settings)?;

    // Reuse the existing saved-report reader. This proves only that the SAME
    // inference settings were exercised with these model bytes and genes. It
    // does not turn research-only results (including repeated final uses) into
    // financial admission or select a model by its held-out profit.
    let research = read_saved_combined_research_reports(
        &settings.system.data_dir,
        candidate_root,
        &identity,
        &manifest,
    )?;
    let inference_settings = serde_json::to_value(&selected_settings.models)?;
    ensure!(
        research
            .reports
            .iter()
            .any(|report| { report.model_inference_settings == inference_settings }),
        "selected candidate has no verified combined research with the current inference/blend settings; completed={}, unavailable={}; do not substitute generic or differently configured models",
        research.reports.len(),
        research.unavailable.len(),
    );
    let ensemble =
        build_ensemble_for_candidate_inference(candidate_root, &manifest, &selected_settings)?;
    Ok(LoadedCandidate {
        ensemble,
        training_handoff: identity,
        candidate_tree: manifest.candidate_tree_sha256().to_owned(),
    })
}

fn select_handoff(
    settings: &Settings,
    candidate_root: &Path,
    portfolio_identity: &str,
) -> Result<(String, Settings)> {
    let root = settings.system.data_dir.join("discovery_targets");
    let mut selected = None;
    let mut config_mismatches = Vec::new();
    let mut unavailable_handoffs = Vec::new();
    for entry in std::fs::read_dir(&root).with_context(|| {
        format!(
            "read portfolio-owned model selections in {}",
            root.display()
        )
    })? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(identity) = name
            .to_str()
            .and_then(|name| name.strip_suffix(".training-handoff.json"))
        else {
            continue;
        };
        // Reuse the canonical bounded reader and selector validation; do not
        // interpret filenames, symbol names or a matching timeframe as ownership.
        let candidate = match handoff::load(&settings.system.data_dir, identity) {
            Ok(candidate) => candidate,
            Err(error) => {
                // A damaged unrelated catalog entry cannot veto a valid exact
                // selection. Invalid entries never become candidates; the chosen
                // manifest, ownership and model tree still undergo strict checks.
                tracing::warn!(handoff = identity, error = %error,
                    "unavailable training handoff excluded from live model selection");
                unavailable_handoffs.push(format!("{identity}: {error}"));
                continue;
            }
        };
        if candidate.locked_portfolio().identity_sha256() != portfolio_identity {
            continue;
        }
        let selected_settings =
            handoff::settings_for_series(settings, candidate.canonical_series());
        if let Err(error) = candidate.validate_inference_settings_v1(&selected_settings) {
            config_mismatches.push(format!("{identity}: {error}"));
            continue;
        }
        if !candidate_root
            .join(format!("{identity}.manifest.json"))
            .try_exists()?
        {
            continue;
        }
        ensure!(
            selected.is_none(),
            "multiple installed model handoffs match this exact portfolio/config; no latest/profit-based automatic choice"
        );
        selected = Some((identity.to_owned(), selected_settings));
    }
    selected.with_context(|| format!(
        "no installed candidate models match portfolio {portfolio_identity} and its current training configuration; complete its selected Training handoff. Configuration mismatches: {}. Unavailable handoffs: {}",
        config_mismatches.join("; "),
        unavailable_handoffs.join("; "),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_settings() -> Settings {
        // Exact settings of the reviewed private Models transport fixture.
        // Its model payloads deliberately cannot perform inference.
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "cpu".to_owned();
        settings.models.label_horizon_bars = 3;
        settings.models.ml_models = vec!["bayes_logit".to_owned()];
        settings.models.phase5_core_models.clear();
        settings.models.regime_router_enabled = false;
        settings.models.phase5_filter_meta_blender = false;
        settings.models.calibration_enabled = false;
        settings.models.use_sac_agent = false;
        settings.models.use_rl_agent = false;
        settings.models.use_neuroevolution = false;
        settings
    }

    #[test]
    fn exact_selection_uses_installed_portfolio_handoff_not_generic_models_or_ambient_symbol() {
        let root = std::env::temp_dir().join(format!(
            "neoethos-live-candidate-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir(&root).unwrap();
        let identity =
            crate::app_services::training::install_saved_research_test_fixture(&root).unwrap();
        let mut settings = fixture_settings();
        settings.system.data_dir = root.join("data");
        settings.system.symbol = "NOT_THE_SELECTED_SYMBOL".to_owned();
        settings.system.base_timeframe = "H4".to_owned();
        let candidate = handoff::load(&settings.system.data_dir, &identity).unwrap();
        let portfolio = candidate.locked_portfolio().identity_sha256();
        let candidates = root.join("models/candidates");
        for unrelated in ["notes".to_owned(), "e".repeat(64)] {
            std::fs::write(
                settings
                    .system
                    .data_dir
                    .join("discovery_targets")
                    .join(format!("{unrelated}.training-handoff.json")),
                b"malformed unrelated catalog entry",
            )
            .unwrap();
        }
        let (resolved, anchored) = select_handoff(&settings, &candidates, portfolio).unwrap();
        assert_eq!(resolved, identity);
        assert_eq!(
            anchored.system.symbol,
            candidate
                .canonical_series()
                .anchor()
                .identity()
                .symbol_name()
        );
        assert_eq!(
            anchored.system.base_timeframe,
            candidate.base_timeframe().as_str()
        );
        assert!(select_handoff(&settings, &candidates, &"f".repeat(64)).is_err());
        settings.models.label_horizon_bars += 1;
        assert!(select_handoff(&settings, &candidates, portfolio).is_err());
        settings.models.label_horizon_bars -= 1;
        let manifest = candidates.join(format!("{identity}.manifest.json"));
        std::fs::remove_file(manifest).unwrap();
        // The generic folder existing is not a substitute for the missing candidate.
        std::fs::create_dir_all(root.join("models/EURUSD/M1/bayes_logit")).unwrap();
        assert!(select_handoff(&settings, &candidates, portfolio).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
