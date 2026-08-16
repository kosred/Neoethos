//! Demo forward-test gate enforcement for the autonomous LIVE path.
//!
//! Before the engine is allowed to trade REAL money, the strategy must clear
//! the demo forward-test gate ([`neoethos_core::domain::demo_gate`]): at least
//! N real DEMO fills AND live metrics within tolerance of the backtest it was
//! promoted on. On a Demo account the gate is a no-op pass — a demo account is
//! exactly how you accumulate those fills in the first place.
//!
//! This never *initiates* trading; it only BLOCKS the live start when the
//! strategy hasn't yet proven itself on demo. The operator still presses Start.
//!
//! **Scope (2026-08-09, W2 + the same-day correction):** the demo fills and the
//! equity curve are scoped to rows the journal recorded while the broker
//! environment was **Demo**. See [`scope_to_demo_environment`] — and read the
//! note there before changing it, because the first attempt at this scoping
//! made the gate impossible to pass.
//!
//! **Drawdown scope (2026-08-09, item #197).** Environment scoping was not
//! enough. Trades are additionally filtered to the portfolio's SYMBOL, but an
//! [`EquitySample`] is an account-level fact and cannot be — the account has
//! one balance, not one per instrument. So the drawdown the gate compares
//! against a single strategy's `quality.json` used to be the union of every
//! engine and every manual order on the demo account. It is now reconstructed
//! from the scoped trades' own realised P&L
//! ([`journal_stats::max_drawdown_from_trade_pnl`]). **Sharpe is still taken
//! from the demo-scoped ACCOUNT curve** — see the note in
//! [`evaluate_for_portfolio`] for why that one was deliberately left alone.
//!
//! **Cadence:** this is an ADMISSION decision, evaluated once by
//! `live_trading::start`. The engine additionally captures the broker
//! environment that decision was made against and stops itself if it ever
//! changes — see the environment re-check in `live_trading::run`.

use anyhow::{Context, Result};

use neoethos_core::Settings;
use neoethos_core::broker_config::CTraderBrokerEnvironment;
use neoethos_core::domain::demo_gate::{
    DemoForwardDecision, DemoForwardGateConfig, evaluate_demo_forward_gate,
};
use neoethos_core::domain::promotion_gate::PromotionMetrics;

use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::journal_stats::{compute_stats, max_drawdown_from_trade_pnl};
use crate::app_services::journal_store::{
    ClosedTrade, EquitySample, query_closed_trades, query_equity,
};

/// The label [`CTraderBrokerEnvironment::Demo`] stores on a journal row.
const DEMO_ENVIRONMENT: &str = "Demo";

/// Keep ONLY the journal rows that were recorded while the broker environment
/// was **Demo**. Rows with no environment stamp are excluded.
///
/// # Why this rule, and why the obvious one is wrong
///
/// The audit found a real defect: the gate called
/// `query_closed_trades(&data_dir, None, None)` and filtered on SYMBOL alone,
/// so fills from a previous or foreign cTrader account satisfied
/// `min_demo_trades` for a strategy that had never traded on the account about
/// to go live. The obvious repair — copy `server/journal.rs:95` and scope to
/// [`journal_store::active_account_id`] — was applied first and is WRONG here,
/// in a way that is worth recording so it is not re-applied:
///
/// * this function only runs when the environment is **Live**
///   ([`active_env_is_live`] gates the caller);
/// * on a Live environment the active account — the one flagged
///   `enabled_for_execution`, the one `broker_api::resolve_creds` routes orders
///   to — is by construction the LIVE account (a demo ctid sent to the live
///   host is `CANT_ROUTE_REQUEST`);
/// * the demo fills the gate is supposed to count were journalled under the
///   DEMO account's id.
///
/// So account scoping retained exactly zero demo trades, `min_demo_trades`
/// (default 100) could never be met, and `POST /autonomous/start` refused every
/// strategy on Live forever. Fail-closed in direction, but a gate that can only
/// say no is a gate the operator switches off — which is how a control gets
/// traded for none.
///
/// The fact the gate actually needs is not "which account" but "was this real
/// money or not", and no journal row could express it. It can now:
/// [`ClosedTrade::environment`] / [`EquitySample::environment`] are stamped at
/// reconcile time. This rule reads that stamp directly, so it is immune to
/// account bookkeeping entirely — including `POST /broker/credentials`
/// replacing the `[[ctrader.accounts]]` list wholesale
/// (`server/broker_control.rs:146`), which would have silently emptied any
/// account-set-based scoping too.
///
/// **Legacy rows (`environment: None`) are excluded.** They predate the stamp
/// and cannot be attributed to an environment; counting them would restore the
/// exact defect the audit found. The consequence is stated plainly rather than
/// hidden: an operator whose journal was written before 2026-08-09 starts the
/// demo forward test from zero.
fn scope_to_demo_environment(trades: &mut Vec<ClosedTrade>, equity: &mut Vec<EquitySample>) {
    let is_demo = |env: Option<&str>| env.is_some_and(|e| e.eq_ignore_ascii_case(DEMO_ENVIRONMENT));
    trades.retain(|t| is_demo(t.environment.as_deref()));
    equity.retain(|e| is_demo(e.environment.as_deref()));
}

/// `drawdown_source` label meaning "this strategy's own closed-trade curve".
const SCOPED_DD: &str = "symbol_scoped_trade_pnl";
/// `drawdown_source` label meaning "there were no demo fills to draw down".
const NO_TRADES_DD: &str = "no_demo_trades";
/// `drawdown_source` label meaning "fills exist and the drawdown could not be
/// measured, so the gate refuses".
const UNMEASURABLE: &str = "unmeasurable_refused";

/// Decide the drawdown percent the gate compares, and say where it came from.
///
/// **Fail-closed, and this is the whole point of the function.** The gate's
/// drawdown criterion is a CAP (`live <= backtest * (1 + tolerance)`), so a
/// value of `0.0` is an automatic PASS. A drawdown that could not be measured
/// must therefore never be reported as zero — it is reported as [`f64::MAX`],
/// which fails the cap against any finite backtest figure.
///
/// The one case where zero is honest is an empty scope: no demo fills means no
/// drawdown, and `min_demo_trades` is the criterion that blocks there, with a
/// message that names the shortfall.
fn resolve_gate_drawdown(
    scoped: Option<crate::app_services::journal_stats::TradeDrawdown>,
    scoped_trade_count: usize,
) -> (f64, &'static str) {
    match (scoped, scoped_trade_count) {
        (Some(d), _) => (d.max_drawdown_pct, SCOPED_DD),
        (None, 0) => (0.0, NO_TRADES_DD),
        (None, _) => (f64::MAX, UNMEASURABLE),
    }
}

/// True when the active broker environment routes to REAL money.
pub fn active_env_is_live() -> bool {
    matches!(
        load_broker_settings().ctrader.environment,
        CTraderBrokerEnvironment::Live
    )
}

/// Read the BACKTEST [`PromotionMetrics`] for a `*.live_portfolio.json` from its
/// sibling `*.quality.json` (written by discovery). Units are normalised to
/// match the gate + the journal: `win_rate` as a fraction, `max_drawdown_pct`
/// as a PERCENT (quality.json stores it as a fraction, e.g. 0.118 → 11.8).
fn backtest_metrics_for(portfolio_path: &str) -> Result<PromotionMetrics> {
    let p = std::path::Path::new(portfolio_path);
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    // discovery writes `<base>.json.live_portfolio.json` + `<base>.json.quality.json`
    let stem = name
        .strip_suffix(".live_portfolio.json")
        .unwrap_or(name);
    let dir = p.parent().unwrap_or_else(|| std::path::Path::new("."));
    let qpath = dir.join(format!("{stem}.quality.json"));
    let text = std::fs::read_to_string(&qpath)
        .with_context(|| format!("read backtest quality {}", qpath.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse quality {}", qpath.display()))?;
    // quality.json is an ARRAY of per-gene records; use the most-traded gene as
    // the representative backtest (matches the Strategy Report's choice).
    let trades_of = |x: &serde_json::Value| x.get("total_trades").and_then(|t| t.as_f64()).unwrap_or(0.0);
    let obj = if let Some(arr) = v.as_array() {
        arr.iter()
            .max_by(|a, b| trades_of(a).partial_cmp(&trades_of(b)).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    } else {
        v
    };
    let f = |k: &str| obj.get(k).and_then(|x| x.as_f64());

    let mut win_rate = f("win_rate").unwrap_or(0.0);
    if win_rate > 1.0 {
        win_rate /= 100.0; // defensive: accept a percent too
    }
    let dd_frac = f("max_drawdown_pct").unwrap_or(0.0);

    Ok(PromotionMetrics {
        sharpe: f("sharpe_ratio").unwrap_or(0.0),
        win_rate,
        profit_factor: f("profit_factor").unwrap_or(0.0),
        max_drawdown_pct: dd_frac * 100.0,
        trades: f("total_trades").unwrap_or(0.0) as u64,
    })
}

/// Evaluate the demo forward gate for a live portfolio: demo fills from the
/// journal (filtered to the portfolio's symbol) measured against the backtest
/// metrics the strategy was promoted on.
///
/// # Which curve each metric is measured on — 2026-08-09, item #197
///
/// The previous comment here said *"equity-derived metrics (Sharpe, max
/// drawdown) use the account equity curve, which is exact while the live engine
/// trades a single symbol"*. The premise was false in the operator's actual
/// setup: he runs several engines plus manual orders on one cTrader account, so
/// the account curve is their union — while the `quality.json` it is compared
/// against belongs to ONE strategy.
///
/// * **Max drawdown** is now reconstructed from the SYMBOL-SCOPED trades' own
///   realised P&L ([`max_drawdown_from_trade_pnl`]), laid on the account equity
///   at the start of the demo window for the percentage denominator. Same
///   construct as the backtest figure: a closed-trade equity curve.
/// * **Sharpe** is still taken from the demo-scoped ACCOUNT curve, and this is
///   deliberate, not an oversight. It is polluted the same way — but the
///   backtest `sharpe_ratio` in `quality.json` is on an unknown sampling basis
///   (per trade / per bar), so swapping the live side to a per-trade Sharpe
///   would change the units of one side of a comparison whose other side is
///   unverified. Changing it would be a different defect, not a repair. The
///   pollution is named in the log line below so it cannot be mistaken for a
///   clean measurement.
/// * **Win rate / profit factor / trade count** were already trade-derived and
///   therefore already symbol-scoped.
pub fn evaluate_for_portfolio(portfolio_path: &str) -> Result<DemoForwardDecision> {
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::LiveTrading)
        .map_err(anyhow::Error::new)?;

    let artifact = neoethos_search::load_live_portfolio_json(portfolio_path)
        .with_context(|| format!("load live portfolio {portfolio_path}"))?;
    let symbol = artifact.symbol;
    let backtest = backtest_metrics_for(portfolio_path)?;

    // 2026-08-04: ONE load, used for BOTH the data root and the gate
    // thresholds. This function used to load a `Settings` here, take only
    // `system.data_dir` from it, and then hand
    // `&DemoForwardGateConfig::default()` to the gate twenty-three lines
    // below — so the last gate before real money ran on literals the
    // operator could not reach. `models.demo_forward_gate` is
    // `#[serde(default)]`, so a config without the key yields exactly the
    // previous thresholds.
    let settings = Settings::load().ok();
    let data_dir = settings
        .as_ref()
        .map(|s| s.system.data_dir.clone())
        .unwrap_or_else(|| std::path::PathBuf::from("data"));
    let gate_config: DemoForwardGateConfig = settings
        .as_ref()
        .map(|s| s.models.demo_forward_gate.clone())
        .unwrap_or_default();
    // 2026-08-09: scope BOTH sides to fills recorded on a DEMO environment
    // before measuring anything. Symbol-only filtering let a foreign account's
    // history pay this account's admission fee; account scoping (the first
    // repair) made the gate unpassable. See `scope_to_demo_environment`.
    let mut trades: Vec<ClosedTrade> = query_closed_trades(&data_dir, None, None)
        .into_iter()
        .filter(|t| t.symbol.eq_ignore_ascii_case(&symbol))
        .collect();
    let mut equity = query_equity(&data_dir, None, None);
    let trades_before = trades.len();
    let equity_before = equity.len();
    scope_to_demo_environment(&mut trades, &mut equity);
    let stats = compute_stats(&trades, &equity);

    // ── The drawdown this gate compares (#197) ────────────────────────────
    // The denominator: account equity at the start of the demo window. The
    // account has ONE balance, so this is necessarily account-level — but it
    // is only the denominator. The numerator is this symbol's own realised
    // P&L, which is the part that was previously the union of every engine on
    // the account.
    let baseline_equity = equity
        .first()
        .map(|s| s.equity)
        .filter(|e| e.is_finite() && *e > 0.0);
    let scoped_dd = baseline_equity.and_then(|b| max_drawdown_from_trade_pnl(&trades, b));

    let (live_max_drawdown_pct, drawdown_source) = resolve_gate_drawdown(scoped_dd, trades.len());
    if drawdown_source == UNMEASURABLE {
        tracing::error!(
            target: "neoethos_app::live_gate",
            %symbol,
            demo_trades_in_scope = trades.len(),
            equity_samples_in_scope = equity.len(),
            "demo forward-test gate: {} demo fills exist for {} but the drawdown \
             could not be measured (no usable demo equity sample to serve as the \
             percentage baseline). REFUSING promotion rather than assuming zero \
             drawdown — check that <data_dir>/journal/equity_samples.jsonl exists \
             and carries Demo-stamped rows",
            trades.len(), symbol
        );
    }

    tracing::info!(
        target: "neoethos_app::live_gate",
        %symbol,
        demo_trades_in_scope = trades.len(),
        equity_samples_in_scope = equity.len(),
        rows_excluded_not_demo = trades_before - trades.len(),
        equity_excluded_not_demo = equity_before - equity.len(),
        min_demo_trades = gate_config.min_demo_trades,
        // WHOSE drawdown this is. The old log reported counts only, so an
        // operator could not tell that the figure being gated on belonged to
        // the whole account rather than to this strategy.
        drawdown_source,
        drawdown_baseline_equity = baseline_equity.unwrap_or(f64::NAN),
        strategy_max_drawdown_pct = live_max_drawdown_pct,
        // Kept alongside for contrast: this is what the gate used to compare.
        account_curve_max_drawdown_pct = stats.max_drawdown_pct,
        trades_skipped_non_finite_pnl =
            scoped_dd.map(|d| d.trades_skipped_non_finite).unwrap_or(0),
        sharpe_source = "demo_scoped_ACCOUNT_equity_curve (still the union of every \
                         engine and manual order on the demo account — see #197)",
        "demo forward-test gate: journal scoped to DEMO-environment fills \
         (rows with no environment stamp predate 2026-08-09 and are excluded); \
         drawdown reconstructed from this symbol's own closed trades"
    );

    let live = PromotionMetrics {
        sharpe: stats.sharpe.unwrap_or(0.0),
        win_rate: stats.win_rate_pct / 100.0,
        // No losses yet → PF is None (∞); treat as meeting the backtest floor.
        profit_factor: stats.profit_factor.unwrap_or(backtest.profit_factor),
        max_drawdown_pct: live_max_drawdown_pct,
        trades: stats.total_trades as u64,
    };

    Ok(evaluate_demo_forward_gate(
        stats.total_trades as u64,
        &live,
        &backtest,
        &gate_config,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backtest_metrics_convert_drawdown_fraction_to_percent() {
        let dir = std::env::temp_dir().join(format!("neo_gate_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let q = dir.join("X_H1.json.quality.json");
        // Real format: an ARRAY of per-gene records; the most-traded wins.
        std::fs::write(
            &q,
            r#"[{"win_rate":0.30,"profit_factor":1.0,"sharpe_ratio":0.5,"max_drawdown_pct":0.4,"total_trades":12},
                {"win_rate":0.45,"profit_factor":1.78,"sharpe_ratio":4.28,"max_drawdown_pct":0.118,"total_trades":300}]"#,
        )
        .unwrap();
        let pf = dir.join("X_H1.json.live_portfolio.json");
        let m = backtest_metrics_for(pf.to_str().unwrap()).unwrap();
        assert!((m.win_rate - 0.45).abs() < 1e-9);
        assert!((m.profit_factor - 1.78).abs() < 1e-9);
        assert!((m.max_drawdown_pct - 11.8).abs() < 1e-6, "dd% = {}", m.max_drawdown_pct);
        assert_eq!(m.trades, 300);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn trade(env: Option<&str>) -> ClosedTrade {
        ClosedTrade {
            schema_version: 1,
            recorded_at_unix_ms: 0,
            position_id: 0,
            symbol: "EURUSD".to_string(),
            side: "BUY".to_string(),
            lots: 0.01,
            account_id: Some("111".to_string()),
            environment: env.map(|s| s.to_string()),
            entry_ts_ms: None,
            entry_price: None,
            exit_ts_ms: None,
            exit_price: None,
            gross_profit: 0.0,
            commission: 0.0,
            swap: 0.0,
            net_profit: 0.0,
            balance_after: None,
        }
    }

    fn equity(env: Option<&str>) -> EquitySample {
        EquitySample {
            ts_ms: 0,
            balance: 100.0,
            equity: 100.0,
            account_id: Some("111".to_string()),
            environment: env.map(|s| s.to_string()),
        }
    }

    #[test]
    fn gate_counts_only_demo_environment_fills() {
        // The defect this closes: LIVE fills (real money already at risk) must
        // not pay the admission fee for going live.
        let mut trades = vec![trade(Some("Demo")), trade(Some("Live")), trade(None)];
        let mut eq = vec![equity(Some("Demo")), equity(Some("Live")), equity(None)];
        scope_to_demo_environment(&mut trades, &mut eq);
        assert_eq!(trades.len(), 1, "only Demo-environment trades survive");
        assert_eq!(eq.len(), 1, "only Demo-environment equity survives");
        assert_eq!(trades[0].environment.as_deref(), Some("Demo"));
    }

    /// THE REGRESSION THAT MATTERS. The first repair scoped to the ACTIVE
    /// account, which on a Live environment is the LIVE account — so a
    /// strategy with a full demo history counted zero and could never be
    /// admitted. A gate that can only say no gets switched off.
    #[test]
    fn a_full_demo_history_is_still_eligible_when_the_live_account_is_active() {
        let mut trades: Vec<ClosedTrade> = (0..100).map(|_| trade(Some("Demo"))).collect();
        let mut eq = vec![equity(Some("Demo"))];
        // The live account is active and has traded nothing yet — the exact
        // state at the moment the operator flips to Live and presses Start.
        scope_to_demo_environment(&mut trades, &mut eq);
        assert_eq!(
            trades.len(),
            100,
            "the demo forward test must remain passable once it is passed"
        );
    }

    #[test]
    fn legacy_unstamped_rows_are_excluded() {
        // `environment: None` rows predate the stamp — unattributable to an
        // environment, so they must not pay the admission fee. Fail-closed.
        let mut trades = vec![trade(None), trade(None)];
        let mut eq = vec![equity(None)];
        scope_to_demo_environment(&mut trades, &mut eq);
        assert!(trades.is_empty());
        assert!(eq.is_empty());
    }

    /// #197. The account curve is the union of every engine on the demo
    /// account; the gate compares it against ONE strategy's quality.json. The
    /// scoped curve is what the comparison was always supposed to use.
    #[test]
    fn the_gate_measures_this_strategys_drawdown_not_the_accounts() {
        let t = |net: f64, ts: i64| {
            let mut x = trade(Some("Demo"));
            x.net_profit = net;
            x.exit_ts_ms = Some(ts);
            x
        };
        let trades = vec![t(20.0, 1), t(-20.0, 2), t(20.0, 3)];
        let scoped = max_drawdown_from_trade_pnl(&trades, 1000.0).expect("measurable");
        let (pct, source) = resolve_gate_drawdown(Some(scoped), trades.len());
        assert_eq!(source, SCOPED_DD);
        assert!(pct < 2.0, "this strategy risked ~2%, not the account's: {pct}");
    }

    /// FAIL-CLOSED. The drawdown criterion is a CAP, so an unmeasurable
    /// drawdown reported as 0.0 would be an automatic pass. It must refuse.
    #[test]
    fn an_unmeasurable_drawdown_refuses_instead_of_passing() {
        let (pct, source) = resolve_gate_drawdown(None, 137);
        assert_eq!(source, UNMEASURABLE);
        assert_eq!(pct, f64::MAX);

        // And it must actually block the gate, not merely look alarming.
        let live = PromotionMetrics {
            sharpe: 10.0,
            win_rate: 0.99,
            profit_factor: 99.0,
            max_drawdown_pct: pct,
            trades: 137,
        };
        let backtest = PromotionMetrics {
            sharpe: 1.0,
            win_rate: 0.5,
            profit_factor: 1.5,
            max_drawdown_pct: 11.8,
            trades: 300,
        };
        let decision = evaluate_demo_forward_gate(
            137,
            &live,
            &backtest,
            &DemoForwardGateConfig {
                enabled: true,
                min_demo_trades: 100,
                forward_tolerance: 0.20,
            },
        );
        assert!(
            !decision.eligible,
            "a strategy whose drawdown could not be measured must not be promoted"
        );
    }

    /// An empty scope is the one case where zero is honest — and
    /// `min_demo_trades` is the criterion that blocks there.
    #[test]
    fn no_demo_fills_reports_zero_and_is_blocked_by_the_trade_count() {
        let (pct, source) = resolve_gate_drawdown(None, 0);
        assert_eq!(source, NO_TRADES_DD);
        assert_eq!(pct, 0.0);

        let zero = PromotionMetrics {
            sharpe: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            max_drawdown_pct: 0.0,
            trades: 0,
        };
        let decision = evaluate_demo_forward_gate(
            0,
            &zero,
            &zero,
            &DemoForwardGateConfig::default(),
        );
        assert!(!decision.eligible, "zero fills can never be eligible");
    }

    #[test]
    fn the_demo_label_matches_what_the_broker_config_writes() {
        // The scoping rule compares against a literal; pin it to the enum that
        // `journal_reconcile` actually stamps so a rename cannot silently
        // empty the gate.
        assert_eq!(CTraderBrokerEnvironment::Demo.as_str(), DEMO_ENVIRONMENT);
    }
}
