const CHART: &str = include_str!("../src/server/chart.rs");
const INDICATORS: &str = include_str!("../src/server/indicators.rs");
const JOURNAL: &str = include_str!("../src/server/journal.rs");

#[test]
fn local_runtime_reads_require_one_exact_identity_and_generation_receipt() {
    assert!(CHART.contains("struct ExactDatasetReceipt"));
    assert!(CHART.contains("CanonicalDatasetIdentity"));
    assert!(CHART.contains("expected_generation"));
    assert!(CHART.contains("load_canonical_timeframe"));
    assert!(CHART.contains("artifact().generation_id()"));

    for (name, source) in [
        ("chart", CHART),
        ("indicators", INDICATORS),
        ("journal", JOURNAL),
    ] {
        assert!(
            source.contains("ExactDatasetReceipt"),
            "{name} has no typed exact-generation receipt"
        );
        assert!(
            !source.contains("discover_timeframes"),
            "{name} still discovers local data by bare symbol"
        );
        assert!(
            !source.contains("load_symbol_timeframe"),
            "{name} still loads local data by bare symbol/timeframe"
        );
        assert!(
            !source.contains("load_symbol_dataset"),
            "{name} still loads a local dataset by bare symbol"
        );
        assert!(
            !source.to_ascii_lowercase().contains("resample"),
            "{name} must never synthesize a timeframe"
        );
        assert!(!source.contains("ensure_timeframes_with_resample"));
        assert!(!source.contains("resample_ohlcv"));
    }
}

#[test]
fn chart_broker_live_and_exact_local_modes_are_separate_and_fail_closed() {
    assert!(CHART.contains("enum ChartRequest"));
    assert!(CHART.contains("BrokerLive"));
    assert!(CHART.contains("ExactLocal"));
    assert!(CHART.contains("load_broker_chart"));
    assert!(CHART.contains("load_exact_local_chart"));
    assert!(!CHART.contains("falling back to local Vortex cache"));
    assert!(!CHART.contains("or_else(|err|"));
    assert!(!CHART.contains("discover_timeframes"));
}

#[test]
fn indicators_use_only_the_selected_pinned_local_generation() {
    assert!(INDICATORS.contains("ExactDatasetReceipt"));
    assert!(INDICATORS.contains("load_exact_current_frame"));
    assert!(!INDICATORS.contains("fetch_recent_chart_bars_blocking"));
    assert!(!INDICATORS.contains("load_symbol_dataset"));
}

#[test]
fn journal_price_windows_never_guess_a_symbol_timeframe_series() {
    assert!(JOURNAL.contains("Option<ExactDatasetReceipt>"));
    assert!(JOURNAL.contains("CanonicalOhlcvFrame"));
    assert!(JOURNAL.contains("load_exact_current_frame"));
    assert!(JOURNAL.contains("None => journal_analytics::analyse(&trades, None)"));
    assert!(!JOURNAL.contains("discover_timeframes"));
    assert!(!JOURNAL.contains("load_symbol_timeframe"));
}
