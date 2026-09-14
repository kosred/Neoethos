use super::*;

fn outcome(timestamp: i64, pnl: f64, currency: &str) -> QuoteValidatedOuterHoldoutTradeOutcomeV1 {
    QuoteValidatedOuterHoldoutTradeOutcomeV1 {
        quote_ledger_sha256: "a".repeat(64),
        execution_economics_ledger_sha256: "b".repeat(64),
        exit_timestamp_unix_ms: timestamp,
        account_currency: currency.to_owned(),
        net_pnl_account_currency: pnl,
        net_pips: pnl / 10.0,
        r_multiple: pnl / 100.0,
    }
}

fn balance(amount: f64) -> AccountMoneyV1 {
    AccountMoneyV1::new("USD", amount).unwrap()
}

#[test]
fn discovery_and_autoresearch_share_validated_account_capital() {
    let mut config = crate::discovery::DiscoveryConfig {
        initial_balance: 2500.0,
        evaluation_account_currency: "EUR".to_owned(),
        ..Default::default()
    };
    assert_eq!(
        config.initial_account_balance().unwrap(),
        AccountMoneyV1::new("EUR", 2500.0).unwrap()
    );
    for amount in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        config.initial_balance = amount;
        assert!(config.initial_account_balance().is_err());
    }
    config.initial_balance = 2500.0;
    config.evaluation_account_currency = "eur".to_owned();
    assert!(config.initial_account_balance().is_err());
}

#[test]
fn initial_capital_is_part_of_the_realized_balance_curve() {
    let trades = [
        outcome(1, 100.0, "USD"),
        outcome(2, -110.0, "USD"),
        outcome(3, 99.0, "USD"),
    ];
    let metrics = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).unwrap();
    // Independently accumulated balances: 1000 -> 1100 -> 990 -> 1089.
    assert_eq!(metrics.peak_equity(), 1100.0);
    assert_eq!(metrics.net_profit(), 89.0);
    assert_eq!(metrics.max_drawdown(), 110.0);
    assert_eq!(metrics.initial_balance(), &balance(1000.0));
    assert_eq!(metrics.ending_balance(), 1089.0);
    assert_eq!(metrics.net_return_fraction(), 89.0 / 1000.0);
    assert_eq!(metrics.max_drawdown_fraction(), 110.0 / 1100.0);
    assert_eq!(metrics.max_daily_drawdown_fraction(), Some(110.0 / 1100.0));
    assert_eq!(metrics.win_rate(), Some(2.0 / 3.0));
    assert_eq!(metrics.profit_factor(), Some(199.0 / 110.0));
    assert_eq!(metrics.expectancy(), Some(89.0 / 3.0));
}

#[test]
fn closed_cash_outcomes_do_not_masquerade_as_periodic_equity_sharpe() {
    let trades = [outcome(1, 100.0, "USD"), outcome(86_400_001, -50.0, "USD")];
    let metrics = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).unwrap();
    assert_eq!(
        metrics.sharpe(),
        None,
        "no regular marked-equity or benchmark return series was supplied"
    );
}

#[test]
fn finite_trades_cannot_seal_non_finite_aggregates() {
    let trades = [outcome(1, f64::MAX, "USD"), outcome(2, f64::MAX, "USD")];
    let result = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0);
    assert!(
        result.is_err(),
        "overflowing finite inputs must not become JSON null metrics"
    );
}

#[test]
fn account_currency_must_match_each_closed_outcome() {
    let trades = [outcome(1, 100.0, "USD"), outcome(2, 100.0, "EUR")];
    let result = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0);
    assert!(result.is_err(), "100 USD plus 100 EUR is not 200 USD");
}

#[test]
fn same_timestamp_closes_do_not_invent_an_intraperiod_peak() {
    for pnls in [[500.0, -500.0], [-500.0, 500.0]] {
        let trades = [outcome(10, pnls[0], "USD"), outcome(10, pnls[1], "USD")];
        let metrics =
            derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).unwrap();
        assert_eq!(metrics.max_drawdown(), 0.0);
        assert_eq!(metrics.max_daily_drawdown(), 0.0);
        assert_eq!(metrics.peak_equity(), 1000.0);
    }
}

#[test]
fn no_fills_preserves_the_starting_balance() {
    let metrics = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &[], 2).unwrap();
    assert_eq!(metrics.peak_equity(), 1000.0);
    assert_eq!(metrics.trade_count(), 0);
    assert_eq!(metrics.entry_unavailable(), 2);
}

#[test]
fn invalid_starting_capital_is_rejected_at_the_metric_boundary() {
    for amount in [0.0, -1000.0] {
        assert!(derive_complete_quote_validated_metrics_v1(&balance(amount), &[], 0).is_err());
    }
}

#[test]
fn closed_outcomes_must_remain_chronological() {
    let trades = [outcome(2, 100.0, "USD"), outcome(1, -50.0, "USD")];
    assert!(derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).is_err());
}

#[test]
fn changing_capital_changes_fractional_risk_not_currency_pnl() {
    let trades = [outcome(1, 100.0, "USD"), outcome(2, -110.0, "USD")];
    let smaller = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).unwrap();
    let larger = derive_complete_quote_validated_metrics_v1(&balance(2000.0), &trades, 0).unwrap();
    assert_eq!(smaller.net_profit(), larger.net_profit());
    assert_eq!(smaller.max_drawdown(), larger.max_drawdown());
    assert_eq!(smaller.net_return_fraction(), -10.0 / 1000.0);
    assert_eq!(larger.net_return_fraction(), -10.0 / 2000.0);
    assert_eq!(smaller.max_drawdown_fraction(), 110.0 / 1100.0);
    assert_eq!(larger.max_drawdown_fraction(), 110.0 / 2100.0);
}

#[test]
fn daily_balance_peak_resets_at_utc_midnight() {
    let trades = [
        outcome(86_399_999, 200.0, "USD"),
        outcome(86_400_000, -120.0, "USD"),
        outcome(86_400_001, 60.0, "USD"),
    ];
    let metrics = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).unwrap();
    assert_eq!(metrics.ending_balance(), 1140.0);
    assert_eq!(metrics.max_daily_drawdown(), 120.0);
    assert_eq!(metrics.max_daily_drawdown_fraction(), Some(120.0 / 1200.0));
    assert_eq!(metrics.consistency(), Some(0.5));
}

#[test]
fn fractional_drawdown_uses_the_peak_at_each_trough_not_the_later_global_peak() {
    let trades = [
        outcome(1, -100.0, "USD"),
        outcome(2, 1100.0, "USD"),
        outcome(3, -150.0, "USD"),
    ];
    let metrics = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).unwrap();
    assert_eq!(metrics.max_drawdown(), 150.0);
    assert_eq!(metrics.peak_equity(), 2000.0);
    // The first drop is 10%, the later larger cash drop is only 7.5%.
    assert_eq!(metrics.max_drawdown_fraction(), 0.1);
}

#[test]
fn cash_balance_crossing_zero_does_not_claim_marked_equity_insolvency() {
    let trades = [
        outcome(1, -1100.0, "USD"),
        outcome(86_400_001, 200.0, "USD"),
    ];
    let metrics = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &trades, 0).unwrap();
    assert_eq!(metrics.ending_balance(), 100.0);
    assert_eq!(metrics.max_drawdown_fraction(), 1.1);
    assert_eq!(metrics.max_daily_drawdown_fraction(), None);
    assert_eq!(metrics.sharpe(), None);
    assert_eq!(
        metrics.metric_basis(),
        "closed_trade_balance_at_exit_timestamps"
    );
}

#[test]
fn malformed_deserialized_currency_is_rejected_even_without_fills() {
    let untrusted: AccountMoneyV1 = serde_json::from_value(serde_json::json!({
        "currency": "usd",
        "amount": 1000.0
    }))
    .unwrap();
    assert!(derive_complete_quote_validated_metrics_v1(&untrusted, &[], 0).is_err());
}

#[test]
fn non_finite_pip_and_r_outcomes_cannot_be_serialized_as_null() {
    for field in ["net_pips", "r_multiple"] {
        let mut trade = outcome(1, 10.0, "USD");
        if field == "net_pips" {
            trade.net_pips = f64::INFINITY;
        } else {
            trade.r_multiple = f64::NAN;
        }
        assert!(derive_complete_quote_validated_metrics_v1(&balance(1000.0), &[trade], 0).is_err());
    }
}

#[test]
fn account_basis_is_serialized_into_the_hashed_metrics_payload() {
    let usd = derive_complete_quote_validated_metrics_v1(&balance(1000.0), &[], 0).unwrap();
    let more_usd = derive_complete_quote_validated_metrics_v1(&balance(2000.0), &[], 0).unwrap();
    let eur = derive_complete_quote_validated_metrics_v1(
        &AccountMoneyV1::new("EUR", 1000.0).unwrap(),
        &[],
        0,
    )
    .unwrap();
    let serialized = serde_json::to_value(&usd).unwrap();
    assert_eq!(serialized["initial_balance"]["currency"], "USD");
    assert_eq!(serialized["initial_balance"]["amount"], 1000.0);
    assert_eq!(serialized["sharpe"], serde_json::Value::Null);
    assert_eq!(
        serialized["sharpe_unavailable_reason"],
        "regular_mark_to_market_and_benchmark_returns_not_supplied"
    );
    assert_ne!(
        stable_sha256("test.metrics", &usd).unwrap(),
        stable_sha256("test.metrics", &more_usd).unwrap()
    );
    assert_ne!(
        stable_sha256("test.metrics", &usd).unwrap(),
        stable_sha256("test.metrics", &eur).unwrap()
    );
}
