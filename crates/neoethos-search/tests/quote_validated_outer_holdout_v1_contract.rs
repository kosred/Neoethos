//! RED-only contract for the single post-lock quote-replay seam in discovery.
//!
//! Search, CPCV, walk-forward validation, feature construction, and model inputs
//! remain direct canonical-trendbar research. Historical Bid/Ask evidence is
//! consumed only after the final portfolio and outer holdout are locked. The
//! resulting evidence stays research-only and cannot become promotion authority.

use std::fs;
use std::path::PathBuf;

const PRODUCTION_RELATIVE_PATH: &str = "src/quote_validated_outer_holdout_v1.rs";

fn crate_root() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn production_source() -> String {
    let path = crate_root().join(PRODUCTION_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "RED: missing locked-portfolio quote-validation boundary {}: {error}",
            path.display()
        )
    })
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

fn require_tokens(source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(
            source.contains(token),
            "missing quote-validated outer-holdout contract token `{token}`"
        );
    }
}

#[test]
fn sealed_historical_ledgers_are_the_only_quote_authority() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "pub struct LockedPortfolioOuterHoldoutReplaySetV1",
            "SealedHistoricalQuoteValidatedResearchLedgerV1",
            "QuoteValidatedExecutionEconomicsLedgerV1",
            "QuoteValidatedResearchReplayReceiptV1",
            "QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly",
            "QuoteValidatedOuterHoldoutArtifactClassV1::ResearchOnly",
            "QuoteValidatedOuterHoldoutPromotionEligibilityV1::NotPromotionEligible",
            "pub fn evaluate_locked_portfolio_outer_holdout_v1(",
        ],
    );

    for forbidden in [
        "Vec<QuoteValidatedResearchLedgerV1>",
        "CompleteBidAskQuoteReplayEvidenceV1",
        "UnverifiedCallerSuppliedQuotes",
        "BrokerFinancialTruthCapabilityV1",
        "BrokerFinancialTruthPermitV1",
        "current_broker_financial_truth",
        "unwrap_or_default()",
    ] {
        assert!(
            !source.contains(forbidden),
            "outer-holdout quote validation contains forbidden authority/fallback `{forbidden}`"
        );
    }
}

#[test]
fn replay_set_exact_binds_locked_portfolio_holdout_and_every_receipt() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "ordered_quote_ledgers",
            "ordered_execution_economics_ledgers",
            "canonical_search_input_receipt_sha256",
            "canonical_signal_plan_sha256",
            "portfolio_identity_sha256",
            "search_config_hash",
            "holdout_scope",
            "account_id",
            "symbol_id",
            "locked_evaluation_window",
            "reviewed_replay_rule_identity_sha256",
            "historical_acquisition_link_manifest_sha256",
            "ledger_sha256",
            "MissingReplayReceipt",
            "UnexpectedReplayReceipt",
            "DuplicateReplayReceipt",
            "ReceiptOrderMismatch",
            "BindingMismatch",
        ],
    );
}

#[test]
fn complete_metric_tuple_is_recomputed_from_quote_execution_ledgers() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "pub struct QuoteValidatedOuterHoldoutMetricsV1",
            "net_profit",
            "sharpe",
            "peak_equity",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "expectancy",
            "trade_count",
            "consistency",
            "max_daily_drawdown",
            "fn derive_complete_quote_validated_metrics_v1(",
            "net_pnl_account_currency",
            "entry_unavailable",
        ],
    );
    let derive = function_body(&source, "fn derive_complete_quote_validated_metrics_v1(");
    for forbidden in [
        "ForwardTestSummary",
        "forward_test_validation_artifacts",
        "prop_firm_validation_artifacts",
        ".metrics.clone()",
        "metrics.net_profit =",
        "..legacy",
        "unwrap_or(0.0)",
    ] {
        assert!(
            !derive.contains(forbidden),
            "complete quote metrics reuse or patch legacy OHLC evidence via `{forbidden}`"
        );
    }
}

#[test]
fn legacy_forward_test_v2_and_prop_artifacts_are_diagnostics_only() {
    let source = production_source();
    let discovery = read("src/discovery.rs");
    let validation = function_body(
        &discovery,
        "pub fn validate_complete_promotion_evidence(&self) -> Result<()> ",
    );

    require_tokens(
        &source,
        &[
            "QuoteValidatedOuterHoldoutErrorCodeV1",
            "MissingSealedQuoteValidatedOuterHoldout",
            "LegacyForwardTestV2Insufficient",
            "LegacyPropFirmV2Insufficient",
            "pub struct QuoteValidatedOuterHoldoutReceiptV1",
            "quote_replay_receipts",
        ],
    );
    assert!(
        validation.contains("require_quote_validated_outer_holdout_v1"),
        "legacy canonical/walk-forward/ForwardTest/Prop V2 sets still satisfy complete promotion evidence"
    );
}

#[test]
fn quote_replay_runs_only_after_final_portfolio_lock_and_early_returns() {
    let discovery = read("src/discovery.rs");
    let body = function_body(
        &discovery,
        "fn run_discovery_cycle_with_holdout_and_progress_authorized<F>(",
    );
    let search = body
        .find("run_discovery_cycle_values_with_progress")
        .expect("final trendbar search call");
    let cancelled = body
        .find("search_cancel_requested")
        .expect("cancelled-search early return");
    let empty = body
        .find("result.portfolio.is_empty()")
        .expect("empty-portfolio early return");
    let quote_replay = body
        .find("evaluate_locked_portfolio_outer_holdout_v1")
        .expect("missing quote replay at the locked outer-holdout seam");
    assert!(
        search < cancelled && cancelled < empty && empty < quote_replay,
        "quote replay must occur only after trendbar search, cancellation, and empty-portfolio checks"
    );
    require_tokens(
        body,
        &[
            "quote_validated_outer_holdout",
            "LockedPortfolioOuterHoldoutReplaySetV1",
            "result.holdout_scope()?",
        ],
    );
}

#[test]
fn ga_cpcv_walkforward_features_and_models_remain_trendbar_only() {
    let discovery = read("src/discovery.rs");
    let numerical_search = function_body(
        &discovery,
        "fn run_discovery_cycle_values_with_progress<F>(",
    );
    let genetic = read("src/genetic/search_engine.rs");

    for (name, source) in [
        ("numerical discovery", numerical_search),
        ("genetic search", genetic.as_str()),
    ] {
        for forbidden in [
            "SealedHistoricalQuoteValidatedResearchLedgerV1",
            "QuoteValidatedExecutionEconomicsLedgerV1",
            "evaluate_locked_portfolio_outer_holdout_v1",
            "replay_sealed_quote_validated_research_v1",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} improperly consumes execution-only quote evidence via `{forbidden}`"
            );
        }
    }
}

#[test]
fn library_exports_only_the_versioned_research_boundary() {
    let library = read("src/lib.rs");
    for required in [
        "mod quote_validated_outer_holdout_v1;",
        "LockedPortfolioOuterHoldoutReplaySetV1",
        "QuoteValidatedOuterHoldoutReceiptV1",
        "QuoteValidatedOuterHoldoutResearchEvidenceV1",
        "evaluate_locked_portfolio_outer_holdout_v1",
    ] {
        assert!(
            library.contains(required),
            "search library is missing quote-validated export `{required}`"
        );
    }
}
