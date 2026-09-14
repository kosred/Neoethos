//! Optional demo forward-test check for the autonomous LIVE path.
//!
//! V6 sizing evidence measures calibration, not an independent final/live
//! benchmark. When this check is enabled, admission refuses until the exact
//! final result for the live execution policy is connected. A reserved final
//! window or a ResearchOnly combined bar report does not supply that result.
//!
//! Disabled by default. Skipping this optional check is not live approval:
//! independent final evidence, broker identity and risk sizing are enforced
//! separately. This module never initiates trading.
//!
//! Admission is checked once by `live_trading::start`; the live loop separately
//! stops if the broker environment changes.

use anyhow::{Context, Result};

use neoethos_core::Settings;
use neoethos_core::broker_config::CTraderBrokerEnvironment;
use neoethos_core::domain::demo_gate::DemoForwardDecision;

use crate::app_services::broker_persistence::load_broker_settings;

/// True when the active broker environment routes to REAL money.
pub fn active_env_is_live() -> bool {
    matches!(
        load_broker_settings().ctrader.environment,
        CTraderBrokerEnvironment::Live
    )
}

/// Validate the artifact before reporting the missing final/live benchmark.
/// Do not substitute calibration metrics or aggregate unrelated journal rows.
fn refuse_unavailable_final_benchmark(
    artifact: &neoethos_search::LivePortfolioArtifact,
) -> Result<DemoForwardDecision> {
    artifact
        .validate()
        .context("validate live portfolio benchmark provenance")?;
    let calibration = artifact
        .sizing_evidence
        .first()
        .context("live portfolio has no calibration sizing evidence")?
        .forward_test
        .scope()
        .evaluated_window();
    let final_window = artifact.final_holdout_scope.evaluated_window();
    anyhow::bail!(
        "live demo-forward benchmark unavailable: V6 sizing evidence is calibration \
         {:?} rows {}..{}, not independent final OOS. Reserved {:?} rows {}..{} \
         identifies a window, not an evaluated live-policy result. Exact independent \
         final evidence bound to the live account sizing, exits and broker execution \
         policy is not connected. Saved combined bar research remains diagnostic-only",
        calibration.role(),
        calibration.row_start(),
        calibration.row_end(),
        final_window.role(),
        final_window.row_start(),
        final_window.row_end(),
    )
}

/// Evaluate the optional check using the saved configuration.
pub fn evaluate_for_portfolio(portfolio_path: &str) -> Result<DemoForwardDecision> {
    let settings = Settings::load().context("load optional demo-check settings")?;
    evaluate_for_portfolio_with_settings(portfolio_path, &settings)
}

/// Evaluate using one explicit configuration snapshot, without ambient defaults.
pub fn evaluate_for_portfolio_with_settings(
    portfolio_path: &str,
    settings: &Settings,
) -> Result<DemoForwardDecision> {
    let gate_config = &settings.models.demo_forward_gate;
    if !gate_config.enabled {
        return Ok(DemoForwardDecision::disabled());
    }
    let artifact = neoethos_search::load_live_portfolio_json(portfolio_path)
        .with_context(|| format!("load live portfolio {portfolio_path}"))?;
    refuse_unavailable_final_benchmark(&artifact)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration_only_fixture() -> neoethos_search::LivePortfolioArtifact {
        // Actual compact V6 Search-writer fixture with synthetic positive
        // calibration summaries. It is not broker or final execution proof.
        neoethos_search::LivePortfolioArtifact::from_persisted_json_bytes(include_bytes!(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../neoethos-search/test_fixtures/strategy_report_v6/",
                "8510cec7a31d6c18b3ed61709a9315fee0ea2e4c9b7c4fb2220f73d314d7f02d.research.live_portfolio.json"
            )
        ))
        .expect("the unchanged Search writer fixture must validate")
    }

    #[test]
    fn positive_v6_calibration_cannot_become_an_independent_final_live_benchmark() {
        use neoethos_search::CanonicalSearchWindowRoleV1;
        let artifact = calibration_only_fixture();
        assert!(!artifact.sizing_evidence.is_empty());
        for evidence in &artifact.sizing_evidence {
            assert_eq!(
                evidence.forward_test.scope().evaluated_window().role(),
                CanonicalSearchWindowRoleV1::SelectionValidation
            );
            assert!(evidence.oos_metrics().net_profit > 0.0);
            assert!(evidence.oos_metrics().trade_count > 0);
        }
        assert_eq!(
            artifact.final_holdout_scope.evaluated_window().role(),
            CanonicalSearchWindowRoleV1::Holdout
        );
        let error = refuse_unavailable_final_benchmark(&artifact)
            .unwrap_err()
            .to_string();
        for expected in [
            "benchmark unavailable",
            "calibration SelectionValidation rows 80..90",
            "not independent final OOS",
            "Reserved Holdout rows 90..100",
            "not an evaluated live-policy result",
            "Saved combined bar research remains diagnostic-only",
        ] {
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn benchmark_refusal_preserves_v6_scope_validation() {
        let mut artifact = calibration_only_fixture();
        artifact.final_holdout_scope = artifact.sizing_evidence[0].forward_test.scope().clone();
        let error = refuse_unavailable_final_benchmark(&artifact).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("validate live portfolio benchmark provenance"),
            "{error:#}"
        );
    }

    #[test]
    fn disabled_optional_check_needs_no_broker_journal_or_portfolio_io() {
        let settings = Settings::default();
        assert!(!settings.models.demo_forward_gate.enabled);
        let decision =
            evaluate_for_portfolio_with_settings("missing.live_portfolio.json", &settings).unwrap();
        assert!(!decision.enabled);
        assert!(decision.eligible);
        assert!(decision.criteria.is_empty());
    }
}
