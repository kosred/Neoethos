//! RED-only consumer contract for account-money trader state.
//!
//! The legacy `ExecReport`, `PositionManager`, and `EngineStats` are loop
//! diagnostics built from price-points times requested lots. They must never be
//! promoted in place to monetary authority. V2 consumes the versioned
//! broker-truth economics ledger and accounts actual filled lots only.

use std::fs;
use std::path::PathBuf;

const PRODUCTION_RELATIVE_PATH: &str = "src/engine_money_v2.rs";

fn crate_root() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-trader"))
}

fn read(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read trader source {}: {error}", path.display()))
}

fn production_source() -> String {
    let path = crate_root().join(PRODUCTION_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "RED: missing account-money trader boundary {}: {error}",
            path.display()
        )
    })
}

fn item_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing production item `{signature}`"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing body for production item `{signature}`"));
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
    panic!("unterminated production item `{signature}`")
}

fn require_tokens(source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(
            source.contains(token),
            "missing engine-money V2 contract token `{token}`"
        );
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-9,
        "expected {expected}, got {actual}"
    );
}

#[derive(Default)]
struct FilledLotOracle {
    remaining_lots: f64,
    exit_commission_account_currency: f64,
}

impl FilledLotOracle {
    fn open(actual_filled_lots: f64) -> Self {
        assert!(actual_filled_lots.is_finite() && actual_filled_lots > 0.0);
        Self {
            remaining_lots: actual_filled_lots,
            exit_commission_account_currency: 0.0,
        }
    }

    fn close(&mut self, actual_filled_lots: f64, commission_account_per_lot_per_fill: f64) {
        assert!(actual_filled_lots.is_finite() && actual_filled_lots > 0.0);
        assert!(actual_filled_lots <= self.remaining_lots);
        self.remaining_lots -= actual_filled_lots;
        self.exit_commission_account_currency +=
            actual_filled_lots * commission_account_per_lot_per_fill;
    }
}

#[test]
fn filled_report_requires_actual_fill_volume_price_time_and_identity() {
    let source = production_source();
    let report = item_body(&source, "pub struct FilledExecutionReportV2");
    require_tokens(
        report,
        &[
            "schema_version",
            "fill_identity_sha256",
            "position_id",
            "symbol",
            "fill_side",
            "actual_filled_lots",
            "StandardLotsV1",
            "fill_price",
            "ExecutionPriceV1",
            "filled_at_unix_ms",
            "execution_economics_ledger_sha256",
        ],
    );
    for forbidden in ["Option<f64>", "unwrap_or", "requested_volume"] {
        assert!(
            !report.contains(forbidden),
            "a filled V2 report cannot use `{forbidden}` for required fill evidence"
        );
    }
}

#[test]
fn partial_fills_charge_actual_lots_and_reduce_remaining_lots() {
    let mut oracle = FilledLotOracle::open(1.0);
    oracle.close(0.6, 7.0);
    assert_close(oracle.remaining_lots, 0.4);
    assert_close(oracle.exit_commission_account_currency, 4.2);
    oracle.close(0.4, 7.0);
    assert_close(oracle.remaining_lots, 0.0);
    assert_close(oracle.exit_commission_account_currency, 7.0);

    let source = production_source();
    let apply = item_body(&source, "pub fn apply_filled_execution_v2(");
    require_tokens(
        apply,
        &[
            "report.actual_filled_lots()",
            "remaining_lots",
            "execution_economics_ledger_sha256",
            "entry_commission_account_currency",
            "exit_commission_account_currency",
        ],
    );
    for forbidden in [
        "intent.volume",
        "volume.unwrap_or",
        ".min(pos_vol)",
        "open_legs",
    ] {
        assert!(
            !apply.contains(forbidden),
            "V2 must account the actual fill rather than legacy `{forbidden}`"
        );
    }
}

#[test]
fn stats_currency_and_quote_economics_ledger_hash_are_mandatory() {
    let source = production_source();
    let stats = item_body(&source, "pub struct EngineMoneyStatsV2");
    require_tokens(
        stats,
        &[
            "schema_version",
            "account_currency",
            "quote_execution_economics_ledger_sha256",
            "realized_pnl_account_currency",
            "unrealized_pnl_account_currency",
            "balance_account_currency",
            "equity_account_currency",
            "artifact_class",
            "promotion_eligibility",
        ],
    );
    require_tokens(
        &source,
        &[
            "ENGINE_MONEY_SCHEMA_VERSION_V2",
            "ExecutionEconomicsArtifactClassV1::ResearchOnly",
            "ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible",
            "#[serde(deny_unknown_fields)]",
        ],
    );
    assert!(
        !stats.contains("#[serde(default)]"),
        "currency and ledger identity must be mandatory on the V2 wire"
    );
}

#[test]
fn missing_mark_is_unavailable_not_zero() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "MoneyAvailabilityV2",
            "MoneyAvailabilityV2::Unavailable",
            "MissingMark",
            "MissingConversionEvidence",
        ],
    );
    for forbidden in ["filter_map(", "unwrap_or(0.0)", "unwrap_or_default()"] {
        assert!(
            !source.contains(forbidden),
            "missing monetary evidence must not collapse through `{forbidden}`"
        );
    }
}

#[test]
fn trader_consumes_typed_account_money_and_never_recomputes_price_pnl() {
    let source = production_source();
    let cargo = read("Cargo.toml");
    require_tokens(
        &source,
        &[
            "QuoteValidatedExecutionEconomicsLedgerV1",
            "economics_ledger.net_pnl_account_currency()",
            "economics_ledger.ledger_sha256()",
        ],
    );
    assert!(
        cargo.contains("neoethos-broker-truth"),
        "trader must consume the leaf economics contract directly"
    );
    for forbidden in [
        "exit_price - entry_price",
        "pip_size",
        "contract_size",
        "conversion_rate",
        "commission_per_lot",
        "spread_pips",
        "swap_pips",
    ] {
        assert!(
            !source.contains(forbidden),
            "trader must not rebuild broker-truth money using `{forbidden}`"
        );
    }
}

#[test]
fn legacy_exec_report_and_engine_stats_cannot_authorize_money_output() {
    let source = production_source();
    let lib = read("src/lib.rs");
    require_tokens(
        &source,
        &[
            "LegacyMonetaryAuthorityV2::Refused",
            "LegacyMoneyWireRefused",
            "try_from_legacy_exec_report",
            "try_from_legacy_engine_stats",
            "LoopDiagnosticsOnly",
            "UnsupportedSchemaVersion",
        ],
    );
    for forbidden in [
        "impl From<ExecReport> for FilledExecutionReportV2",
        "impl From<EngineStats> for EngineMoneyStatsV2",
    ] {
        assert!(
            !source.contains(forbidden),
            "legacy price-point wires must fail closed; found `{forbidden}`"
        );
    }
    for export in [
        "pub mod engine_money_v2;",
        "EngineMoneyStatsV2",
        "FilledExecutionReportV2",
    ] {
        assert!(
            lib.contains(export),
            "trader lib is missing V2 money export `{export}`"
        );
    }
}
