use neoethos_broker_history::bulk_cli::CanonicalTrendbarBulkCli;
use neoethos_broker_history::{
    CANONICAL_TRENDBAR_SERIES_FROM_MS_V1, CanonicalTrendbarCheckpointReceiptV1,
};
use neoethos_data::{CTraderEnvironment, CanonicalTimeframe};

const TO_MS_EXCLUSIVE: &str = "1767225600000";
const BULK_MAIN: &str = include_str!("../src/bin/neoethos-canonical-trendbar-bulk.rs");
const BULK_CLI: &str = include_str!("../src/bulk_cli.rs");
const BULK_RUNNER: &str = include_str!("../src/historical_series_runner_v1.rs");

fn required_args(data_root: &str, authority_root: &str) -> Vec<String> {
    vec![
        "neoethos-canonical-trendbar-bulk".to_owned(),
        "--environment".to_owned(),
        "demo".to_owned(),
        "--account-id".to_owned(),
        "42".to_owned(),
        "--symbol".to_owned(),
        "1=EUR/USD".to_owned(),
        "--symbol".to_owned(),
        "2=USD/JPY".to_owned(),
        "--to-ms-exclusive".to_owned(),
        TO_MS_EXCLUSIVE.to_owned(),
        "--data-root".to_owned(),
        data_root.to_owned(),
        "--authority-root".to_owned(),
        authority_root.to_owned(),
    ]
}

#[test]
fn explicit_plan_uses_exact_account_cutoff_symbols_and_all_direct_canonical_timeframes() {
    let data = tempfile::tempdir().expect("data root");
    let authority = tempfile::tempdir().expect("authority root");
    let args = required_args(
        data.path().to_str().expect("data path"),
        authority.path().to_str().expect("authority path"),
    );
    let prepared = CanonicalTrendbarBulkCli::try_parse_from(args)
        .expect("strict bulk CLI")
        .prepare()
        .expect("exact immutable plan");
    let plan = prepared.plan();
    assert_eq!(plan.environment(), CTraderEnvironment::Demo);
    assert_eq!(plan.server(), "demo.ctraderapi.com");
    assert_eq!(plan.account_id(), 42);
    assert_eq!(plan.from_ms(), CANONICAL_TRENDBAR_SERIES_FROM_MS_V1);
    assert_eq!(plan.to_ms_exclusive().to_string(), TO_MS_EXCLUSIVE);
    assert_eq!(
        plan.symbols()
            .iter()
            .map(|symbol| (symbol.symbol_id(), symbol.symbol_name()))
            .collect::<Vec<_>>(),
        vec![(1, "EUR/USD"), (2, "USD/JPY")]
    );
    assert_eq!(plan.timeframes(), CanonicalTimeframe::ALL);
    assert_eq!(plan.cell_count(), 2 * CanonicalTimeframe::ALL.len());
    assert!(prepared.checkpoint_receipt().is_none());
}

#[test]
fn cutoff_environment_account_symbols_and_roots_are_all_required() {
    let data = tempfile::tempdir().expect("data root");
    let authority = tempfile::tempdir().expect("authority root");
    let base = required_args(
        data.path().to_str().expect("data path"),
        authority.path().to_str().expect("authority path"),
    );
    for required in [
        "--environment",
        "--account-id",
        "--symbol",
        "--to-ms-exclusive",
        "--data-root",
        "--authority-root",
    ] {
        let mut args = base.clone();
        while let Some(index) = args.iter().position(|argument| argument == required) {
            args.drain(index..=index + 1);
        }
        assert!(
            CanonicalTrendbarBulkCli::try_parse_from(args).is_err(),
            "{required} must never acquire a default"
        );
    }
}

#[test]
fn resampling_tick_current_and_partial_timeframe_flags_do_not_exist() {
    let data = tempfile::tempdir().expect("data root");
    let authority = tempfile::tempdir().expect("authority root");
    let base = required_args(
        data.path().to_str().expect("data path"),
        authority.path().to_str().expect("authority path"),
    );
    for forbidden in [
        ["--resample-from", "M1"],
        ["--ticks", "true"],
        ["--use-current", "true"],
        ["--timeframe", "H1"],
        ["--from-ms", "1451606400000"],
    ] {
        let mut args = base.clone();
        args.push(forbidden[0].to_owned());
        args.push(forbidden[1].to_owned());
        assert!(CanonicalTrendbarBulkCli::try_parse_from(args).is_err());
    }
}

#[test]
fn resume_accepts_only_an_explicit_exact_checkpoint_digest() {
    let data = tempfile::tempdir().expect("data root");
    let authority = tempfile::tempdir().expect("authority root");
    let mut malformed = required_args(
        data.path().to_str().expect("data path"),
        authority.path().to_str().expect("authority path"),
    );
    malformed.extend(["--checkpoint-sha256".to_owned(), "current".to_owned()]);
    assert!(CanonicalTrendbarBulkCli::try_parse_from(malformed).is_err());

    assert!(CanonicalTrendbarCheckpointReceiptV1::from_sha256("a".repeat(64)).is_ok());
    assert!(CanonicalTrendbarCheckpointReceiptV1::from_sha256("A".repeat(64)).is_err());
}

#[test]
fn binary_reaches_only_the_strict_bulk_plan_runner_and_receipt_renderer() {
    for required in [
        "CanonicalTrendbarBulkCli::try_parse_from",
        ".prepare()",
        "execute_canonical_trendbar_bulk_v1",
        "render_canonical_trendbar_bulk_stdout_v1",
        "initialize_source_seal_before_runtime",
    ] {
        assert!(
            BULK_MAIN.contains(required),
            "missing binary route {required}"
        );
    }
    for forbidden in [
        "capture_historical_generation",
        "load_production_historical_credentials",
        "resample",
        "tick",
        "current",
    ] {
        assert!(
            !BULK_MAIN.contains(forbidden),
            "bulk binary contains forbidden route {forbidden}"
        );
    }
    assert!(
        BULK_RUNNER.contains("load_exact_production_historical_credentials"),
        "production resume must load the plan's exact account"
    );
    assert!(!BULK_RUNNER.contains("load_production_historical_credentials("));
    for source in [BULK_CLI, BULK_RUNNER] {
        for forbidden in ["resample", "tick_data", "capture_tick", "use_current"] {
            assert!(
                !source.contains(forbidden),
                "bulk production source contains forbidden route {forbidden}"
            );
        }
    }
}
