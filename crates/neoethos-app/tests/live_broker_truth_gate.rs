use neoethos_app::app_services::live_trading::{StartRequest, start};
use neoethos_app::server::risky::RiskyScenarioQuery;
use neoethos_app::server::state::AppApiState;
use neoethos_app::server::strategy_lab::PromotionQuery;

const BROKER_TRUTH_UNAVAILABLE: &str = "BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1";

fn assert_broker_truth_refusal(error: &anyhow::Error) {
    let message = format!("{error:#}");
    assert!(
        message.contains(BROKER_TRUTH_UNAVAILABLE),
        "the route must fail at the typed broker-truth boundary, got: {message}"
    );
}

#[test]
fn historical_replay_is_refused_before_data_or_default_pip_math_is_touched() {
    let missing_root = std::path::Path::new("this-path-must-not-be-read-before-broker-truth");
    let result = neoethos_trader::replay_symbol_from_dir(
        missing_root,
        "EURUSD",
        "M5",
        neoethos_trader::EngineConfig::default(),
    );

    let error = match result {
        Ok(_) => panic!("historical replay ran without exact broker evidence"),
        Err(error) => error,
    };
    assert_broker_truth_refusal(&error);
}

#[test]
fn legacy_zero_cost_replay_engine_is_not_a_public_production_bypass() {
    let trader_lib = include_str!("../../neoethos-trader/src/lib.rs");
    let engine = include_str!("../../neoethos-trader/src/engine.rs");
    let replay = include_str!("../../neoethos-trader/src/replay.rs");

    assert!(
        !trader_lib.contains("pub use engine::{\n    AutonomousEngine"),
        "the raw replay engine is still re-exported to production callers"
    );
    assert!(
        !trader_lib.contains("pub use replay::replay;"),
        "the zero-broker replay harness is still re-exported to production callers"
    );
    assert!(
        !trader_lib.contains("pub use execution::{MockExecutionAdapter"),
        "the mock fill adapter is still re-exported to production callers"
    );
    assert!(
        !trader_lib.contains("pub use execution::ReplayCostModel"),
        "the operator-estimated replay cost model is still re-exported"
    );
    assert!(
        engine.contains("pub(crate) struct AutonomousEngine"),
        "the raw financial loop must be crate-private behind the broker gate"
    );
    assert!(
        replay.contains("pub(crate) fn replay"),
        "the zero-broker replay harness must be crate-private"
    );
    let execution = include_str!("../../neoethos-trader/src/execution.rs");
    assert!(
        execution.contains("pub(crate) struct MockExecutionAdapter"),
        "the mock fill adapter must be crate-private"
    );
    assert!(
        execution.contains("pub(crate) struct ReplayCostModel"),
        "the raw replay cost arithmetic type must be crate-private"
    );
    assert!(
        !engine.contains("pub costs: crate::execution::ReplayCostModel"),
        "EngineConfig still lets external callers inject heuristic financial costs"
    );
}

#[test]
fn replay_config_refuses_before_flat_spread_or_operator_commission_is_resolved() {
    let settings = neoethos_core::Settings::default();
    let result =
        neoethos_trader::EngineConfig::try_for_replay_from_settings(Some(&settings), "EURUSD");

    let error = match result {
        Ok(_) => panic!("replay cost configuration was synthesized without broker truth"),
        Err(error) => error,
    };
    assert_broker_truth_refusal(&error);
}

#[tokio::test(flavor = "current_thread")]
async fn live_start_is_refused_before_portfolio_loading_or_loop_spawn() {
    let result = start(StartRequest {
        portfolio_path: "this-portfolio-must-not-be-read-before-broker-truth.json".to_owned(),
        lot_size: 0.01,
        stop_loss_pips: Some(20.0),
        take_profit_pips: Some(40.0),
        warmup_bars: 1_000,
        cull_after_consecutive_losses: 6,
        cull_min_win_rate_pct: 57.0,
        cull_window_trades: 10,
    });

    let error = match result {
        Ok(handle) => {
            handle.stop();
            panic!("live trading started without authoritative broker PnL and deal truth")
        }
        Err(error) => error,
    };
    assert_broker_truth_refusal(&error);
}

async fn response_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("read bounded response body");
    String::from_utf8(bytes.to_vec()).expect("API errors are UTF-8 JSON")
}

#[tokio::test(flavor = "current_thread")]
async fn risky_projection_is_refused_before_default_win_rate_or_reward_math() {
    let response = neoethos_app::server::risky::scenarios(
        axum::extract::State(AppApiState::default()),
        axum::extract::Query(RiskyScenarioQuery {
            starting_usd: None,
            target_usd: None,
            risk_fraction: None,
            win_rate: None,
            reward_to_risk: None,
            trades_per_day: None,
        }),
    )
    .await;
    let body = response_text(response).await;
    assert!(
        body.contains(BROKER_TRUTH_UNAVAILABLE),
        "Risky projection executed heuristic defaults instead of failing closed: {body}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn promotion_status_is_refused_before_loading_heuristic_quality_metrics() {
    let response = neoethos_app::server::strategy_lab::promotion_status(
        axum::extract::State(AppApiState::default()),
        axum::extract::Query(PromotionQuery {
            symbol: Some("EURUSD".to_owned()),
            base_tf: Some("M5".to_owned()),
        }),
    )
    .await;
    let body = response_text(response).await;
    assert!(
        body.contains(BROKER_TRUTH_UNAVAILABLE),
        "promotion evaluated non-broker-real quality metrics: {body}"
    );
}

#[test]
fn live_parity_is_refused_before_default_pip_or_broker_bar_fetch() {
    let result = neoethos_app::app_services::live_parity::run_live_parity_check(
        "this-portfolio-must-not-be-read-before-broker-truth.json",
        1_000,
        3_000,
    );
    let error = match result {
        Ok(_) => panic!("live parity resolved default pip/risk geometry without broker truth"),
        Err(error) => error,
    };
    assert_broker_truth_refusal(&error);
}

#[test]
fn risky_tail_analysis_is_refused_before_loading_trade_returns() {
    let result = neoethos_app::app_services::tail_risk::run_tail_risk(
        "this-portfolio-must-not-be-read-before-broker-truth.json",
        2_000,
        None,
    );
    let error = match result {
        Ok(_) => panic!("tail-risk arithmetic ran without broker truth"),
        Err(error) => error,
    };
    assert_broker_truth_refusal(&error);
}

#[test]
fn prop_firm_challenge_is_refused_before_loading_trade_returns() {
    let result = neoethos_app::app_services::challenge_sim::run_challenge_sim(
        "this-portfolio-must-not-be-read-before-broker-truth.json",
        2_000,
    );
    let error = match result {
        Ok(_) => panic!("Prop-Firm challenge arithmetic ran without broker truth"),
        Err(error) => error,
    };
    assert_broker_truth_refusal(&error);
}

#[test]
fn live_eligibility_gate_is_refused_before_loading_quality_metrics() {
    let result = neoethos_app::app_services::live_gate::evaluate_for_portfolio(
        "this-portfolio-must-not-be-read-before-broker-truth.json",
    );
    let error = match result {
        Ok(_) => panic!("live eligibility consumed unverified financial artifacts"),
        Err(error) => error,
    };
    assert_broker_truth_refusal(&error);
}

#[test]
fn obsolete_local_pnl_fallback_implementation_is_not_shipped() {
    let bridge_source = include_str!("../src/server/bridge.rs");
    let obsolete_module =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/app_services/pnl.rs");
    assert!(
        !obsolete_module.exists(),
        "superseded duplicate PnL request module is still shipped"
    );

    for forbidden in [
        "compute_pnl_pips",
        "trader.unrealized_pnl",
        "pnl_usd=0.0",
        ".unwrap_or(0.0)",
    ] {
        assert!(
            !bridge_source.contains(forbidden),
            "bridge can still invent a financial zero/proxy: {forbidden}"
        );
    }

    let refresh = bridge_source
        .find("async fn refresh_once(")
        .expect("bridge refresh entry point exists");
    let scoped = &bridge_source[refresh..];
    let gate = scoped
        .find("current_broker_financial_truth_capability_v1")
        .expect("bridge refresh has a broker-truth gate");
    let first_work = scoped
        .find("tokio::task::spawn_blocking")
        .expect("bridge refresh has a blocking credential load");
    assert!(
        gate < first_work,
        "bridge loads local/account state before the live financial truth gate"
    );
}

#[test]
fn superseded_local_pnl_knobs_and_fixture_docs_are_not_shipped() {
    let forbidden = [
        "pnl_audit_drift_fraction",
        "pnl_circuit_breaker_fraction",
        "NEOETHOS_BOT_PNL_AUDIT_DRIFT_FRACTION",
        "NEOETHOS_BOT_PNL_CIRCUIT_BREAKER_FRACTION",
    ];
    for (name, source) in [
        (
            "core config",
            include_str!("../../neoethos-core/src/config.rs"),
        ),
        (
            "app runtime accessors",
            include_str!("../src/app_services/env_overrides.rs"),
        ),
        (
            "retired env aliases",
            include_str!("../src/app_services/retired_env.rs"),
        ),
        (
            "knob catalog",
            include_str!("../src/server/knob_catalog.rs"),
        ),
        (
            "desktop advanced screen",
            include_str!("../../../desktop/src/screens/Advanced.tsx"),
        ),
        (
            "shipped config",
            include_str!("../../../desktop/src-tauri/resources/config.yaml"),
        ),
        (
            "active knob reference",
            include_str!("../../../docs/CONFIG-KNOBS-REFERENCE.md"),
        ),
        (
            "active env reference",
            include_str!("../../../docs/ENV-VARS.md"),
        ),
    ] {
        for token in forbidden {
            assert!(
                !source.contains(token),
                "{name} still advertises the superseded local-PnL control {token}"
            );
        }
    }

    let obsolete_fixture_readme = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ctrader/unrealized_pnl/README.md");
    assert!(
        !obsolete_fixture_readme.exists(),
        "fixture documentation for deleted local-PnL drift tests is still shipped"
    );
}

#[test]
fn broker_money_parsers_never_guess_a_missing_scale_or_fallback_to_fiat() {
    let missing =
        neoethos_app::app_services::ctrader_money::required_money_digits(None, "test.moneyDigits")
            .expect_err("an omitted broker scale must fail closed");
    assert!(missing.to_string().contains("test.moneyDigits"));

    let invalid = neoethos_app::app_services::ctrader_money::required_money_digits(
        Some(11),
        "test.moneyDigits",
    )
    .expect_err("an out-of-contract broker scale must fail closed");
    assert!(invalid.to_string().contains("out of spec range"));

    for (name, source) in [
        (
            "ctrader_money.rs",
            include_str!("../src/app_services/ctrader_money.rs"),
        ),
        (
            "ctrader_account.rs",
            include_str!("../src/app_services/ctrader_account.rs"),
        ),
        (
            "ctrader_execution.rs",
            include_str!("../src/app_services/ctrader_execution.rs"),
        ),
    ] {
        for forbidden in [
            "defaulting to 2",
            "falling back to fiat default",
            ".unwrap_or(0.0)",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} still guesses broker money via {forbidden:?}"
            );
        }
    }

    let account_source = include_str!("../src/app_services/ctrader_account.rs");
    assert!(
        account_source.contains("build_get_position_unrealized_pnl_request"),
        "account runtime does not request the broker's unrealized-PnL response"
    );
    assert!(
        account_source.contains("parse_get_position_unrealized_pnl_response"),
        "account runtime does not parse the broker's unrealized-PnL response"
    );
    assert!(
        !account_source.contains("unrealized_pnl: 0.0"),
        "account runtime still fabricates zero unrealized PnL"
    );

    let money_source = include_str!("../src/app_services/ctrader_money.rs");
    assert!(
        !money_source.contains("unscale_to_ctrader_money_int"),
        "unused speculative money conversion survived replacement cleanup"
    );
}

#[test]
fn account_identity_and_margin_views_do_not_use_local_financial_fallbacks() {
    let account_source = include_str!("../src/app_services/ctrader_account.rs");
    assert!(
        account_source.contains("build_asset_list_request"),
        "account runtime must request ProtoOAAssetListRes"
    );
    assert!(
        account_source.contains("resolve_deposit_asset_name"),
        "account runtime must join depositAssetId to the broker asset registry"
    );

    for (name, source) in [
        ("bridge.rs", include_str!("../src/server/bridge.rs")),
        (
            "live_trading.rs",
            include_str!("../src/app_services/live_trading.rs"),
        ),
        (
            "desktop broker.rs",
            include_str!("../../../desktop/src-tauri/src/broker.rs"),
        ),
    ] {
        for forbidden in ["asset_id_to_currency", "fn asset_currency(", "_ => \"EUR\""] {
            assert!(
                !source.contains(forbidden),
                "{name} still maps broker money identity locally via {forbidden:?}"
            );
        }
    }

    let bridge_source = include_str!("../src/server/bridge.rs");
    for forbidden in [
        ".filter_map(|p| p.used_margin)",
        "(equity - used_margin).max(0.0)",
        ".and_then(neoethos_core::symbol_metadata::resolve)",
    ] {
        assert!(
            !bridge_source.contains(forbidden),
            "bridge still hides missing broker margin/contract evidence via {forbidden:?}"
        );
    }
}

#[test]
fn live_journal_requires_complete_broker_deal_financials_and_broker_equity() {
    let source = include_str!("../src/app_services/journal_reconcile.rs");
    for forbidden in [
        "d.gross_profit.unwrap_or(0.0)",
        "d.fee.unwrap_or(0.0)",
        "d.swap.unwrap_or(0.0)",
        ".unwrap_or(d.filled_volume)",
        "equity: balance,",
    ] {
        assert!(
            !source.contains(forbidden),
            "journal still fabricates a broker financial component via {forbidden:?}"
        );
    }
    assert!(
        source.contains("runtime.trader.balance + runtime.unrealized_pnl"),
        "journal equity is not derived from the authoritative broker unrealized-PnL response"
    );
    let entry = source
        .find("pub fn reconcile_best_effort(")
        .expect("journal entry point exists");
    let scoped = &source[entry..];
    let gate = scoped
        .find("current_broker_financial_truth_capability_v1")
        .expect("journal entry point has a broker-truth gate");
    let first_local_read = scoped
        .find("data_dir()")
        .expect("journal resolves its data dir");
    assert!(
        gate < first_local_read,
        "journal touches local state before the broker-truth capability gate"
    );
}

#[test]
fn live_order_units_come_from_the_exact_resolved_broker_symbol() {
    let source = include_str!("../src/app_services/broker_api.rs");
    for forbidden in [
        "neoethos_core::symbol_metadata::resolve(symbol)",
        "meta.lots_to_wire_volume",
        "meta.pip_size",
    ] {
        assert!(
            !source.contains(forbidden),
            "live order preparation still uses local symbol math via {forbidden:?}"
        );
    }
    assert!(
        source.contains("resolved.symbol.pip_position"),
        "live order preparation ignores ProtoOASymbol.pipPosition"
    );
    assert!(
        source.contains("resolved.symbol.lot_size"),
        "live order preparation ignores ProtoOASymbol.lotSize"
    );
}
