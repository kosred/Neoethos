use super::*;

fn utc(month: u32, day: u32) -> i64 {
    Utc.with_ymd_and_hms(2024, month, day, 0, 0, 0)
        .unwrap()
        .timestamp_millis()
}

#[test]
fn monthly_report_reconciles_money_and_uses_exit_month_with_idle_and_partial_months() {
    let trades = [
        (1, 16, 1, 17, 100.0),
        (1, 31, 2, 1, -50.0),
        (2, 2, 2, 3, -25.0),
        (4, 1, 4, 2, 25.0),
    ]
    .map(|(em, ed, xm, xd, pnl)| Trade {
        entry_time: utc(em, ed),
        exit_time: Some(utc(xm, xd)),
        pnl,
        ..Default::default()
    });
    let mut canonical = [0.0; 11];
    canonical[0] = 50.0;
    canonical[1] = 0.1;
    canonical[3] = 0.1;
    canonical[8] = 4.0;
    let metrics = StrategyQualityAnalyzer::default()
        .analyze_strategy_with_evaluation(
            "calendar-ledger",
            &trades,
            1000.0,
            utc(1, 15),
            utc(4, 9),
            &canonical,
        )
        .unwrap();
    let wire = serde_json::to_value(metrics).unwrap();
    let rows = wire["monthly_breakdown"]
        .as_array()
        .expect("each calendar month needs an observation");
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows.iter()
            .map(|r| r["net_profit"].as_f64().unwrap())
            .sum::<f64>(),
        50.0
    );
    assert_eq!(rows[0]["month_utc"], "2024-01");
    assert_eq!(rows[0]["closed_trades"], 1);
    assert_eq!(rows[0]["is_partial"], true);
    assert_eq!(rows[1]["month_utc"], "2024-02");
    assert_eq!(rows[1]["closed_trades"], 2);
    assert_eq!(rows[1]["starting_balance"], 1100.0);
    assert_eq!(rows[1]["ending_balance"], 1025.0);
    assert_eq!(rows[1]["net_profit"], -75.0);
    assert_eq!(rows[1]["max_closed_balance_drawdown_money"], 75.0);
    assert!((rows[1]["return_pct"].as_f64().unwrap() + 75.0 / 11.0).abs() < 1e-12);
    assert_eq!(rows[1]["is_partial"], false);
    assert_eq!(rows[1]["profit_factor"], 0.0);
    assert_eq!(rows[2]["month_utc"], "2024-03");
    assert_eq!(rows[2]["closed_trades"], 0);
    assert_eq!(rows[2]["net_profit"], 0.0);
    assert_eq!(rows[2]["return_pct"], 0.0);
    assert!(rows[2]["win_rate"].is_null());
    assert!(rows[2]["profit_factor"].is_null());
    assert_eq!(rows[3]["ending_balance"], 1050.0);
    assert_eq!(rows[3]["is_partial"], true);
    assert!(
        rows[3]["profit_factor"].is_null(),
        "no losses is undefined, never an invented finite PF"
    );
    assert_eq!(rows[3]["gross_profit"], 25.0);
    assert_eq!(rows[3]["gross_loss"], 0.0);
}

#[test]
fn monthly_report_keeps_an_entire_idle_evaluation_interval() {
    let metrics = StrategyQualityAnalyzer::default()
        .analyze_strategy_with_evaluation(
            "idle-calendar",
            &[],
            1000.0,
            utc(1, 1),
            utc(3, 1),
            &[0.0; 11],
        )
        .unwrap();
    let wire = serde_json::to_value(metrics).unwrap();
    let rows = wire["monthly_breakdown"]
        .as_array()
        .expect("idle months must not disappear");
    assert_eq!(rows.len(), 3);
    for row in rows {
        assert_eq!(row["starting_balance"], 1000.0);
        assert_eq!(row["ending_balance"], 1000.0);
        assert_eq!(row["closed_trades"], 0);
    }
}
