//! Source contracts for the post-lock quote-replay seam in discovery.
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
fn execution_money_is_checked_against_sealed_fills_before_metrics() {
    let source = production_source();
    let legacy = function_body(
        &source,
        "pub fn evaluate_locked_portfolio_outer_holdout_v1(",
    );
    require_tokens(
        legacy,
        &[
            "canonical_signal_plan_sha256_v1(",
            "evaluate_bound_quote_ledgers_v1(",
        ],
    );
    let evaluate = function_body(&source, "fn evaluate_bound_quote_ledgers_v1(");
    let validate = evaluate
        .find(".validate_against_quote_ledger(quote_ledger)")
        .expect("economics must be rebound to the independently held sealed fills");
    let consume_money = evaluate
        .find("execution.net_pnl_account_currency().amount()")
        .expect("quote-validated metric money input");
    assert!(
        validate < consume_money,
        "fill validation must precede using the economics PnL"
    );
}

#[test]
fn closed_trade_metric_tuple_is_recomputed_from_quote_execution_ledgers() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "pub struct QuoteValidatedOuterHoldoutMetricsV1",
            "metric_basis",
            "initial_balance: AccountMoneyV1",
            "ending_balance",
            "net_profit",
            "net_return_fraction",
            "sharpe",
            "sharpe_unavailable_reason",
            "peak_equity",
            "max_drawdown",
            "max_drawdown_fraction",
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
fn discovery_binds_account_money_before_the_holdout_metrics_consumer() {
    let source = production_source();
    let evaluate = function_body(&source, "fn evaluate_bound_quote_ledgers_v1(");
    let currency_check = evaluate
        .find("execution.account_currency() != initial_balance.currency()")
        .expect("execution money must match the run's capital currency");
    let consume_money = evaluate
        .find("execution.net_pnl_account_currency().amount()")
        .expect("execution money consumer");
    assert!(currency_check < consume_money);
    require_tokens(
        evaluate,
        &[
            "initial_balance: AccountMoneyV1",
            "&initial_balance,",
            "derive_complete_quote_validated_metrics_v1",
        ],
    );

    let discovery = read("src/discovery.rs");
    let capital = function_body(&discovery, "pub fn initial_account_balance(");
    require_tokens(
        capital,
        &[
            "neoethos_broker_truth::AccountMoneyV1::new(",
            "self.evaluation_account_currency.clone()",
            "self.initial_balance",
        ],
    );
    let body = function_body(
        &discovery,
        "fn run_discovery_cycle_with_holdout_and_progress_authorized<F>(",
    );
    require_tokens(body, &["config.initial_account_balance()?"]);
    let autoresearch = read("../neoethos-autoresearch/src/runner/streaming.rs");
    let oos = function_body(&autoresearch, "fn evaluate_oos(");
    require_tokens(oos, &["config.initial_account_balance()?"]);
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
        ],
    );
    let compact: String = body.chars().filter(|ch| !ch.is_whitespace()).collect();
    assert!(compact.contains("result.holdout_scope()?"));
}

#[test]
fn post_lock_consumers_share_preparation_per_window_without_reusing_calibration_as_final() {
    let discovery = read("src/discovery.rs");
    let body = function_body(
        &discovery,
        "fn run_discovery_cycle_with_holdout_and_progress_authorized<F>(",
    );
    let (calibration, final_window) = body
        .split_once("drop(prepared);")
        .expect("calibration vectors must be released before optional final preparation");
    for window in [calibration, final_window] {
        assert_eq!(
            window
                .matches("PreparedLockedHoldoutResearch::new_with_policy(")
                .count(),
            1,
            "each independent window must prepare its signals exactly once"
        );
    }
    require_tokens(
        calibration,
        &[
            "calibration.features()",
            "calibration.ohlcv()",
            "&calibration_scope,",
            "rayon::join(",
            "prepared.forward_test_artifacts()",
            "prepared.prop_firm_artifacts(prop_firm_rules)",
        ],
    );
    let calibration_computation = calibration
        .split_once("let prepared = PreparedLockedHoldoutResearch::new_with_policy(")
        .expect("calibration preparation call")
        .1;
    // Earlier row-count telemetry may inspect the final window length. The
    // calibration computation itself must not consume final values/signals.
    assert!(!calibration_computation.contains("holdout.features()"));
    assert!(!calibration_computation.contains("holdout.ohlcv()"));
    require_tokens(
        final_window,
        &[
            "let final_prepared = if prelocked_quote_replay.is_some()",
            "|| quote_validated_outer_holdout.is_some()",
            "holdout.features()",
            "holdout.ohlcv()",
            "&holdout_scope,",
            "replay(&result, prepared)?",
        ],
    );
    assert!(!final_window.contains("calibration.features()"));
    assert!(!final_window.contains("calibration.ohlcv()"));
    assert!(!final_window.contains("prepared.forward_test_artifacts()"));
    assert!(!final_window.contains("prepared.prop_firm_artifacts("));
    require_tokens(
        body,
        &[
            "result.effective_smc_gate_threshold",
            "Some(sealed_policy)",
            "rayon::join(",
            "prepared.forward_test_artifacts()",
            "prepared.prop_firm_artifacts(prop_firm_rules)",
            "let evaluation = &prepared.evaluation",
            "let ordered_signals = &prepared.ordered_signals",
            "replay_provider(&result, ordered_signals, &holdout_scope)",
        ],
    );
    for forbidden in [
        "signals_for_gene_with_config(",
        "SmcGateArrays::build(",
        "ThreadPoolBuilder",
    ] {
        assert!(
            !body.contains(forbidden),
            "holdout runner bypasses shared preparation via {forbidden}"
        );
    }
    let legacy_prepare = function_body(&discovery, "fn locked_holdout_signals_and_confidences(");
    require_tokens(
        legacy_prepare,
        &["locked_holdout_signals_and_confidences_with_policy("],
    );
    let prepare_signals = function_body(
        &discovery,
        "pub fn locked_holdout_signals_and_confidences_with_policy(",
    );
    assert_eq!(prepare_signals.matches("SmcGateArrays::build(").count(), 1);
    require_tokens(
        prepare_signals,
        &[
            ".par_iter()",
            "signals_and_confidence_for_gene_full_with_smc_policy(",
            "smc_gate_disabled: bool",
            "signals.len() == n && confidences.len() == n",
            "Ok(generated.into_iter().unzip())",
        ],
    );
    assert!(!prepare_signals.contains("signals_for_gene_with_config("));
    assert!(!prepare_signals.contains("ThreadPoolBuilder"));
    assert!(!prepare_signals.contains("crate::genetic::smc_gate_disabled()"));
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
            "evaluate_locked_portfolio_outer_holdout_v3",
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

#[test]
fn prelocked_v3_hashes_inputs_before_quotes_and_reuses_the_sealed_money_consumer() {
    let source = read("src/quote_validated_outer_holdout_v2.rs");
    let payload = function_body(&source, "struct SignalPlanPayloadV3<'a>");
    require_tokens(
        payload,
        &[
            "ordered_signals",
            "ordered_confidences",
            "ordered_size_multipliers",
            "account_risk_policy",
            "adaptive_stops_policy",
            "portfolio_identity_sha256",
            "ordered_gene_identity_sha256",
            "exit_policy",
            "bar_open_timestamps",
            "high",
            "low",
        ],
    );
    assert!(!payload.contains("ordered_risk_pips"));
    assert!(!payload.contains("quote_ledgers"));
    require_tokens(
        &source,
        &[
            "neoethos.canonical-bar-signal-plan.v3",
            "ledger.executed_decision()",
            "ledger.executed_plan_sha256()",
            "risk_pips: stop_pips",
            "canonical_source_row",
            "missing eligible decision",
            "evaluate_bound_quote_ledgers_v1(",
        ],
    );
    let compact: String = source.chars().filter(|ch| !ch.is_whitespace()).collect();
    assert!(compact.contains("snapshot.validate_replay_context_v1(binding,policy)"));
    let validate = function_body(&source, "fn validate_lane<'a>(");
    require_tokens(
        validate,
        &[
            "decision.stop_price()",
            "decision.target_price()",
            ".entry_stop_target_pips(lane, index)",
            "locked.exit_policy.max_hold_bars == 0",
            "outcome.time_exit != expected_timer.as_ref()",
            "outcomes.next().is_some()",
        ],
    );
    let entry_geometry = function_body(&source, "pub fn entry_stop_target_pips(");
    require_tokens(
        entry_geometry,
        &[
            "crate::stop_target::resolve_entry_stop_target_pips(",
            "self.adaptive_base_pips.is_some()",
            "self.adaptive_stops_policy.reward_risk_fallback()",
        ],
    );
    let sizing = function_body(&source, "fn size_locked_account_entries<E, F>(");
    require_tokens(
        sizing,
        &[
            "entry_inputs(ledger, origin)",
            "execution_economics(ledger, &sizing)",
            "validate_against_quote_ledger(ledger)",
        ],
    );
}

#[test]
fn canonical_research_v3_exact_maps_shipped_genes_to_prepared_signals_and_sealed_policy() {
    let discovery = read("src/discovery.rs");
    let entry = function_body(
        &discovery,
        "pub fn run_canonical_trendbar_research_with_quote_holdout_v3<F, P>(",
    );
    require_tokens(
        entry,
        &[
            "contract.validate_against_input(input)?",
            "from_explicit_canonical_cpu_research_v1",
            "LivePortfolioArtifact::from_discovery(",
            "canonical_locked_portfolio_identity_sha256_v1(&artifact)",
            "for gene in &artifact.genes",
            "ValidationStrategyIdentityV2::from_gene(gene)?",
            ".validate_against(&prepared.portfolio[index])?",
            "signals.push(prepared.ordered_signals[index].clone())",
            "confidences.push(prepared.ordered_confidences[index].clone())",
            "&artifact.genes",
            "&full_portfolio_hash",
            "&signals",
            "&confidences",
            "&multipliers",
            "prepared.ohlcv",
            "sealed_evaluation_config()?",
            "CanonicalSignalExitPolicyV2::from_evaluation(&evaluation)",
            "CanonicalSignalAccountRiskPolicyV3::from_evaluation(&evaluation)?",
            ".sealed_adaptive_stops_policy()?",
            "replay_provider(&locked)",
            "evaluate_locked_portfolio_outer_holdout_v3(",
            "Some(&mut observe)",
        ],
    );
    assert!(!entry.contains("current_broker_financial_truth"));
    assert!(!entry.contains("signals_and_confidence_for_gene"));
    assert!(
        entry
            .find(".validate_against(&prepared.portfolio[index])?")
            .unwrap()
            < entry.find("LockedCanonicalSignalPlanV3::new(").unwrap()
    );
    let body = function_body(
        &discovery,
        "fn run_discovery_cycle_with_holdout_and_progress_authorized<F>(",
    );
    assert!(
        body.find("result.portfolio.is_empty()").unwrap()
            < body.find("replay(&result, prepared)?").unwrap()
    );
    let acquisition =
        read("../neoethos-broker-truth-acquire/src/finalist_quote_replay_acquisition_v1.rs");
    let consumer = function_body(
        &acquisition,
        "pub fn replay_reviewed_locked_portfolio_v3<E, F>(",
    );
    assert!(
        consumer.find("validate_replay_binding").unwrap()
            < consumer.find("open_reviewed_quote_snapshot_v1").unwrap()
    );
    require_tokens(
        consumer,
        &[
            "replay_locked_canonical_signal_portfolio_v3(",
            "LockedPortfolioOuterHoldoutReplaySetV3::new(",
            "QuoteEntryFinancialInputsV3",
            "QuoteEntrySizingEvidenceV3",
            "entry_inputs",
            "execution_economics",
        ],
    );
    let compact: String = consumer.chars().filter(|ch| !ch.is_whitespace()).collect();
    assert!(compact.contains(
        "LockedPortfolioOuterHoldoutReplaySetV3::new(locked,&self.replay_binding,&self.replay_policy,&snapshot,lanes,entry_inputs,execution_economics,"
    ));
}
