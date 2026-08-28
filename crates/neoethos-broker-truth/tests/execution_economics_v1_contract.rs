//! RED-only contract for the first versioned money-first execution ledger.
//!
//! This deliberately specifies a source boundary before production exists. The
//! ledger may consume only the opaque result of sealed historical quote replay;
//! a caller-built quote vector, receipt string, or detached ledger hash must not
//! be able to mint this boundary. Every monetary field below is account currency
//! unless its name explicitly says quote currency.

use std::fs;
use std::path::PathBuf;

const PRODUCTION_RELATIVE_PATH: &str = "src/execution_economics_v1.rs";

fn crate_root() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-broker-truth"))
}

fn production_source() -> String {
    let path = crate_root().join(PRODUCTION_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "RED: missing versioned execution-economics production boundary {}: {error}",
            path.display()
        )
    })
}

fn lib_source() -> String {
    let path = crate_root().join("src/lib.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read broker-truth lib {}: {error}", path.display()))
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing production function `{signature}`"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing body for production function `{signature}`"));
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
    panic!("unterminated body for production function `{signature}`")
}

fn require_tokens(source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(
            source.contains(token),
            "missing execution-economics contract token `{token}`"
        );
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-9,
        "expected {expected}, got {actual}"
    );
}

/// Independent dimensional oracle used only to state the contract's units.
/// Production must derive entry/exit prices from the sealed quote ledger.
struct MoneyOracle {
    base_units: f64,
    entry_notional_quote_currency: f64,
    gross_pnl_quote_currency: f64,
    gross_pnl_account_currency: f64,
    net_pnl_account_currency: f64,
}

#[allow(clippy::too_many_arguments)]
fn money_oracle(
    direction_sign: f64,
    entry_price_quote_per_base: f64,
    exit_price_quote_per_base: f64,
    contract_units_per_lot: f64,
    filled_lots: f64,
    conversion_rate_account_per_quote: f64,
    entry_commission_account_currency: f64,
    exit_commission_account_currency: f64,
    swap_account_currency_signed: f64,
    pnl_conversion_fee_account_currency: f64,
) -> MoneyOracle {
    let base_units = contract_units_per_lot * filled_lots;
    let entry_notional_quote_currency = entry_price_quote_per_base * base_units;
    let gross_pnl_quote_currency =
        direction_sign * (exit_price_quote_per_base - entry_price_quote_per_base) * base_units;
    let gross_pnl_account_currency = gross_pnl_quote_currency * conversion_rate_account_per_quote;
    let net_pnl_account_currency = gross_pnl_account_currency
        - entry_commission_account_currency
        - exit_commission_account_currency
        + swap_account_currency_signed
        - pnl_conversion_fee_account_currency;
    MoneyOracle {
        base_units,
        entry_notional_quote_currency,
        gross_pnl_quote_currency,
        gross_pnl_account_currency,
        net_pnl_account_currency,
    }
}

#[test]
fn ten_pip_eurusd_trade_is_100_usd_gross_and_86_usd_net() {
    let result = money_oracle(1.0, 1.1000, 1.1010, 100_000.0, 1.0, 1.0, 7.0, 7.0, 0.0, 0.0);
    assert_close(result.base_units, 100_000.0);
    assert_close(result.entry_notional_quote_currency, 110_000.0);
    assert_close(result.gross_pnl_quote_currency, 100.0);
    assert_close(result.gross_pnl_account_currency, 100.0);
    assert_close(result.net_pnl_account_currency, 86.0);

    let source = production_source();
    require_tokens(
        &source,
        &[
            "contract_units_per_lot",
            "filled_lots",
            "base_units",
            "entry_notional_quote_currency",
            "gross_pnl_quote_currency",
            "gross_pnl_account_currency",
            "commission_account_per_lot_per_fill",
            "entry_commission_account_currency",
            "exit_commission_account_currency",
            "net_pnl_account_currency",
        ],
    );
    let body = function_body(
        &source,
        "pub fn build_quote_validated_execution_economics_v1(",
    );
    require_tokens(
        body,
        &[
            "modeled_entry_price",
            "modeled_exit_price",
            "contract_units_per_lot",
            "filled_lots",
            "conversion_rate_account_per_quote",
            "entry_commission_account_currency",
            "exit_commission_account_currency",
            "swap_account_currency_signed",
            "pnl_conversion_fee_account_currency",
        ],
    );
}

#[test]
fn bid_ask_replay_prices_do_not_pay_scalar_spread_twice() {
    let source = production_source();
    let body = function_body(
        &source,
        "pub fn build_quote_validated_execution_economics_v1(",
    );
    require_tokens(
        body,
        &[
            "sealed_quote_ledger: &SealedHistoricalQuoteValidatedResearchLedgerV1",
            "modeled_entry_price",
            "modeled_exit_price",
            "additional_spread_account_currency",
            "AccountMoneyV1::zero",
        ],
    );
    for forbidden in [
        "spread_pips",
        "full_spread_pips_assumption",
        "screening_round_trip_cost_pips",
        "ReplayCostModel",
    ] {
        assert!(
            !body.contains(forbidden),
            "quote-side prices already contain spread; found forbidden scalar `{forbidden}`"
        );
    }
}

#[test]
fn cross_currency_trade_requires_causal_conversion_reference() {
    // 1,000 JPY quote-currency profit converted by a causal JPY->USD quote.
    let result = money_oracle(
        1.0, 150.00, 150.01, 100_000.0, 1.0, 0.0067, 0.0, 0.0, 0.0, 0.0,
    );
    assert_close(result.gross_pnl_quote_currency, 1_000.0);
    assert_close(result.gross_pnl_account_currency, 6.7);

    let source = production_source();
    require_tokens(
        &source,
        &[
            "CausalQuoteToAccountConversionV1",
            "source_currency",
            "target_account_currency",
            "conversion_rate_account_per_quote",
            "conversion_observed_at_unix_ms",
            "conversion_evidence_identity_sha256",
            "MissingConversionEvidence",
            "StaleConversionEvidence",
            "CurrencyMismatch",
        ],
    );
}

#[test]
fn missing_exit_fill_or_conversion_is_unavailable_not_zero() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "ExecutionEconomicsErrorCodeV1",
            "MissingExitFill",
            "MissingConversionEvidence",
            "InvalidFilledLots",
            "CurrencyMismatch",
            "Result<QuoteValidatedExecutionEconomicsLedgerV1, ExecutionEconomicsErrorV1>",
        ],
    );
    for forbidden in [
        "unwrap_or(0.0)",
        "unwrap_or_default()",
        ".filter_map(",
        ".max(0.0)",
    ] {
        assert!(
            !source.contains(forbidden),
            "missing financial evidence must refuse, not use `{forbidden}`"
        );
    }
}

#[test]
fn swap_is_a_signed_account_currency_cashflow_not_a_pip_scalar() {
    let result = money_oracle(
        1.0, 1.1000, 1.1010, 100_000.0, 1.0, 1.0, 7.0, 7.0, -2.0, 1.0,
    );
    assert_close(result.net_pnl_account_currency, 83.0);

    let source = production_source();
    require_tokens(
        &source,
        &[
            "swap_account_currency_signed",
            "swap_evidence_identity_sha256",
            "pnl_conversion_fee_account_currency",
        ],
    );
    assert!(
        !source.contains("swap_pips"),
        "the execution ledger records the broker cashflow in account currency"
    );
}

#[test]
fn ledger_identity_binds_volume_currency_conversion_and_each_cashflow() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "ExecutionEconomicsHashPayloadV1",
            "quote_ledger_sha256",
            "symbol_contract_identity_sha256",
            "account_currency",
            "conversion_evidence_identity_sha256",
            "entry_fill_identity_sha256",
            "exit_fill_identity_sha256",
            "commission_policy_identity_sha256",
            "swap_evidence_identity_sha256",
            "filled_lots",
            "entry_commission_account_currency",
            "exit_commission_account_currency",
            "swap_account_currency_signed",
            "pnl_conversion_fee_account_currency",
            "ledger_sha256",
        ],
    );
}

#[test]
fn legacy_price_only_wire_is_not_execution_economics_v1() {
    let source = production_source();
    let lib = lib_source();
    require_tokens(
        &source,
        &[
            "EXECUTION_ECONOMICS_SCHEMA_VERSION_V1",
            "schema_version",
            "#[serde(deny_unknown_fields)]",
            "UnsupportedSchemaVersion",
            "LegacyWireRefused",
            "ExecutionEconomicsArtifactClassV1::ResearchOnly",
            "ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible",
        ],
    );
    assert!(
        !source.contains("#[serde(default)]"),
        "legacy/missing economics fields must not be synthesized by serde defaults"
    );
    for export in [
        "mod execution_economics_v1;",
        "QuoteValidatedExecutionEconomicsLedgerV1",
        "build_quote_validated_execution_economics_v1",
    ] {
        assert!(
            lib.contains(export),
            "broker-truth lib is missing execution-economics export `{export}`"
        );
    }
}
