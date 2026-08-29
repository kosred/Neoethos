//! RED-only contract for autoresearch's one permitted final OOS quote touch.
//!
//! Sweeps and finalist selection remain canonical-trendbar research. Only the
//! already selected, immutable `PromotionPortfolio` may reach this seam, and
//! legacy OHLC OOS evidence can never substitute for sealed quote replay.

use std::fs;
use std::path::{Path, PathBuf};

const PRODUCTION_RELATIVE_PATH: &str =
    "crates/neoethos-autoresearch/src/quote_validated_oos_touch_v1.rs";

fn repository_root() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        return Path::new(manifest_dir)
            .parent()
            .and_then(Path::parent)
            .expect("neoethos-autoresearch manifest must be under <repo>/crates")
            .to_path_buf();
    }
    std::env::current_dir().expect("standalone source-contract working directory")
}

fn read(relative: &str) -> String {
    let path = repository_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn production_source() -> String {
    let path = repository_root().join(PRODUCTION_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "RED: missing single-touch quote-validation boundary {}: {error}",
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
            "missing quote-validated OOS-touch contract token `{token}`"
        );
    }
}

#[test]
fn oos_touch_evidence_binds_the_exact_finalist_session_and_sealed_replays() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "pub struct QuoteValidatedOosTouchEvidenceV1",
            "pub struct QuoteValidatedOosTouchReceiptV1",
            "SealedHistoricalQuoteValidatedResearchLedgerV1",
            "QuoteValidatedExecutionEconomicsLedgerV1",
            "QuoteValidatedResearchReplayReceiptV1",
            "session_id",
            "sweep",
            "slot",
            "config_hash",
            "dataset_receipt",
            "oos_window",
            "promotion_portfolio_sha256",
            "ordered_quote_ledger_sha256s",
            "ordered_historical_link_manifest_sha256s",
            "QuoteValidatedOosArtifactClassV1::ResearchOnly",
            "QuoteValidatedOosPromotionEligibilityV1::NotPromotionEligible",
        ],
    );
}

#[test]
fn legacy_ohlc_oos_and_forward_test_v2_are_explicitly_insufficient() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "QuoteValidatedOosTouchErrorCodeV1",
            "MissingSealedQuoteReplay",
            "LegacyOhlcOosEvidenceInsufficient",
            "LegacyForwardTestV2Insufficient",
            "ReceiptSetMismatch",
            "PortfolioBindingMismatch",
            "WindowBindingMismatch",
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
            "OOS touch contains forbidden authority/fallback `{forbidden}`"
        );
    }
}

#[test]
fn sweep_executor_returns_quote_validated_evidence_not_detached_ohlc_statistics() {
    let runner = read("crates/neoethos-autoresearch/src/runner.rs");
    let executor = function_body(&runner, "pub trait SweepExecutor {");
    require_tokens(
        executor,
        &[
            "fn evaluate_oos(",
            "Result<QuoteValidatedOosTouchEvidenceV1>",
        ],
    );
    assert!(
        !executor.contains("Result<OosEvidence>"),
        "legacy caller-constructible OHLC OosEvidence still crosses the final OOS trait seam"
    );
}

#[test]
fn single_touch_order_is_preflight_then_durable_spend_then_quote_evaluation_then_judge() {
    let runner = read("crates/neoethos-autoresearch/src/runner.rs");
    let promote = function_body(&runner, "fn promote(");
    let preflight = promote
        .find("executor.oos_preflight(&portfolio)")
        .expect("OOS preflight before spending the touch");
    let spent = promote
        .find("writer.append(Record::OosTouchSpent")
        .expect("durable OOS touch-spent record");
    let evaluate = promote
        .find("executor.evaluate_oos(")
        .expect("single quote-validated OOS evaluation");
    let judge = promote
        .find("crate::judge::promote(")
        .expect("judge after quote-validated evaluation");
    assert!(
        preflight < spent && spent < evaluate && evaluate < judge,
        "the only OOS touch must be preflight -> durable spend -> quote evaluation -> judge"
    );
    assert_eq!(
        promote.matches("executor.evaluate_oos(").count(),
        1,
        "autoresearch may perform exactly one final OOS evaluation"
    );
    assert!(
        promote.contains("QuoteValidatedOosTouchEvidenceV1"),
        "the spent touch is not statically typed as quote-validated evidence"
    );
}

#[test]
fn streaming_finalist_builds_signals_from_bars_but_executes_only_through_quote_replay() {
    let streaming = read("crates/neoethos-autoresearch/src/runner/streaming.rs");
    let evaluate = function_body(&streaming, "fn evaluate_oos(");
    require_tokens(
        evaluate,
        &[
            "signals_for_gene",
            "evaluate_locked_portfolio_outer_holdout_v1",
            "QuoteValidatedOosTouchEvidenceV1",
            "PromotionPortfolio",
        ],
    );
    for forbidden in [
        "simulate_trades_broker_real",
        "base.high",
        "base.low",
        "ForwardTestSummary",
        "compute_discovery_forward_test_artifacts",
    ] {
        assert!(
            !evaluate.contains(forbidden),
            "final OOS execution still uses legacy OHLC path `{forbidden}`"
        );
    }

    let execute = function_body(
        &streaming,
        "fn execute(&mut self, request: &SearchRequest<'_>)",
    );
    for forbidden in [
        "SealedHistoricalQuoteValidatedResearchLedgerV1",
        "evaluate_locked_portfolio_outer_holdout_v1",
        "QuoteValidatedOosTouchEvidenceV1",
    ] {
        assert!(
            !execute.contains(forbidden),
            "mass screening improperly consumes OOS quote evidence via `{forbidden}`"
        );
    }
}

#[test]
fn judge_recomputes_and_consumes_the_complete_quote_validated_oos_tuple() {
    let source = production_source();
    let judge = read("crates/neoethos-autoresearch/src/judge.rs");
    let promote = function_body(&judge, "pub fn promote(");
    require_tokens(
        &source,
        &[
            "per_trade_net_pips",
            "r_multiples",
            "monthly_returns",
            "period_keys",
            "trades_per_day",
            "band_survives",
            "derive_complete_oos_statistics_from_quote_ledgers_v1",
            "net_pnl_account_currency",
            "entry_unavailable",
        ],
    );
    require_tokens(
        promote,
        &[
            "QuoteValidatedOosTouchEvidenceV1",
            ".per_trade_net_pips()",
            ".r_multiples()",
            ".monthly_returns()",
            ".trades_per_day()",
            ".band_survives()",
        ],
    );
    assert!(
        !promote.contains("oos: &OosEvidence"),
        "judge still accepts detached legacy OHLC OosEvidence"
    );
    let derive = function_body(
        &source,
        "fn derive_complete_oos_statistics_from_quote_ledgers_v1(",
    );
    for forbidden in [
        "legacy_oos",
        "ForwardTestSummary",
        ".net_profit =",
        "..legacy",
        "unwrap_or(0.0)",
    ] {
        assert!(
            !derive.contains(forbidden),
            "quote OOS statistics patch/reuse legacy evidence via `{forbidden}`"
        );
    }
}

#[test]
fn module_is_versioned_exported_and_never_mints_promotion_authority() {
    let source = production_source();
    let library = read("crates/neoethos-autoresearch/src/lib.rs");
    for required in [
        "mod quote_validated_oos_touch_v1;",
        "QuoteValidatedOosTouchEvidenceV1",
        "QuoteValidatedOosTouchReceiptV1",
        "evaluate_quote_validated_oos_touch_v1",
    ] {
        assert!(
            library.contains(required),
            "autoresearch library is missing quote-OOS export `{required}`"
        );
    }
    for forbidden in [
        "QuoteValidatedOosPromotionEligibilityV1::PromotionEligible",
        "promotion_eligibility: true",
        "permit_issued: true",
        "BrokerFinancialTruthPermitV1",
        "BrokerFinancialTruthCapabilityV1",
    ] {
        assert!(
            !source.contains(forbidden),
            "research-only quote OOS boundary contains promotion authority `{forbidden}`"
        );
    }
}
