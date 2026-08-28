//! Source contract for two deliberately separate app boundaries:
//!
//! 1. live cTrader deal fields become typed, account-currency money evidence;
//! 2. an optional historical quote-ledger comparison emits parity-only research.
//!
//! A live deal cannot mint historical quote authority. cTrader does not report
//! an independent `netProfit` on `ProtoOAClosePositionDetail`, so live net money
//! is the checked sum of the broker's signed wire components.

use std::fs;
use std::path::PathBuf;

const PRODUCTION_RELATIVE_PATH: &str = "src/app_services/broker_deal_economics.rs";

fn crate_root() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-app"))
}

fn read(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read app source {}: {error}", path.display()))
}

fn production_source() -> String {
    let path = crate_root().join(PRODUCTION_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "RED: missing broker-deal money boundary {}: {error}",
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
            "missing broker-deal contract token `{token}`"
        );
    }
}

#[test]
fn live_wire_components_become_typed_money_evidence_without_broker_net() {
    let source = production_source();
    let body = item_body(&source, "pub fn build_broker_deal_money_evidence_v1(");
    require_tokens(
        body,
        &[
            "BrokerDealWireSnapshotV1",
            "BrokerDealMoneyEvidenceV1",
            "gross_profit_raw_scaled",
            "commission_raw_scaled_signed",
            "swap_raw_scaled_signed",
            "BrokerPnlConversionFeeV1::Charged",
            "BrokerPnlConversionFeeV1::NotApplied",
            "money_digits",
            "checked_add",
            "component_sum_account_currency",
        ],
    );
    for forbidden in [
        "deal.net_profit",
        "MissingNetProfit",
        "NetProfitMismatch",
        "unwrap_or_default()",
        "unwrap_or(0.0)",
    ] {
        assert!(
            !source.contains(forbidden),
            "live deal evidence must not invent broker net through `{forbidden}`"
        );
    }
}

#[test]
fn account_environment_fill_and_money_scale_are_mandatory_identities() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "BrokerDealMoneyErrorCodeV1",
            "MissingGrossProfit",
            "MissingCommission",
            "MissingSwap",
            "InvalidMoneyDigits",
            "CurrencyMismatch",
            "FillIdentityMismatch",
            "environment",
            "account_id",
            "deal_id",
            "order_id",
            "position_id",
            "symbol_id",
            "execution_timestamp_ms",
            "execution_price",
            "entry_price",
            "account_currency",
            "deal_identity_sha256",
        ],
    );
}

#[test]
fn broker_filled_centi_units_use_the_exact_symbol_lot_size() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "filled_volume_raw_centi_units",
            "lot_size_raw_centi_units",
            "contract_units_per_lot",
            "actual_filled_lots",
            "volume_scale_identity_sha256",
            "InvalidFilledVolume",
        ],
    );
    for forbidden in [
        "requested_lot",
        "symbol_metadata::resolve",
        "filled_volume.unwrap_or",
    ] {
        assert!(
            !source.contains(forbidden),
            "actual filled lots cannot use fallback `{forbidden}`"
        );
    }

    let broker_api = read("src/app_services/broker_api.rs");
    require_tokens(
        &broker_api,
        &[
            "BrokerSymbolVolumeScaleEvidenceV1::new(",
            "resolved.symbol.lot_size",
            "volume_scale_evidence",
        ],
    );
}

#[test]
fn deal_id_dedup_and_partial_close_finish_only_after_broker_is_flat() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "BrokerPositionMoneyAccumulatorV1",
            "seen_deal_ids",
            "observe_fill",
            "DuplicateDeal",
            "refuse_unverified_fill",
            "finalize_if_position_closed",
            "position_still_open",
            "BrokerClosedPositionMoneyV1",
        ],
    );

    let live = read("src/app_services/live_trading.rs");
    require_tokens(
        &live,
        &[
            "build_broker_deal_money_evidence_v1(",
            "observe_fill(",
            "verify_complete_filled_volume(",
            "finalize_if_position_closed(",
            "opened_entry_filled_volumes",
            "has_unresolved_broker_entry",
            "runtime.deposit_asset_name",
            "runtime.environment",
        ],
    );
    assert!(
        !live.contains("let Some(net) = deal.net_profit else { continue };"),
        "live trading still consumes the locally synthesized legacy scalar"
    );
    assert!(
        !live.contains("opened_ids.remove(&deal.position_id)"),
        "the first partial-close deal still removes the whole position"
    );
}

#[test]
fn optional_historical_parity_is_separate_and_cannot_mint_authority() {
    let source = production_source();
    let live_builder = item_body(&source, "pub fn build_broker_deal_money_evidence_v1(");
    let parity = item_body(&source, "pub fn reconcile_broker_deal_economics_v1(");

    assert!(
        !live_builder.contains("QuoteValidatedExecutionEconomicsLedgerV1"),
        "building live broker money must not require a historical quote ledger"
    );
    require_tokens(
        parity,
        &[
            "&BrokerDealMoneyEvidenceV1",
            "&QuoteValidatedExecutionEconomicsLedgerV1",
            "BrokerDealEconomicsParityV1",
            "BrokerDealEconomicsParityAuthorityV1::ParityOnly",
            "ExecutionEconomicsPromotionEligibilityV1::NotPromotionEligible",
            "execution_economics_ledger_sha256",
        ],
    );
    for forbidden in [
        "SealedHistoricalQuoteValidatedResearchLedgerV1",
        "open_sealed_historical",
        "replay_sealed_quote_validated",
        "HistoricalBidAskQuotesOnly",
        "BrokerFinancialTruthPermitV1",
        "build_quote_validated_execution_economics_v1",
    ] {
        assert!(
            !source.contains(forbidden),
            "app boundary must not mint quote authority through `{forbidden}`"
        );
    }
}

#[test]
fn app_wires_the_leaf_contract_without_using_legacy_engine_stats() {
    let source = production_source();
    let cargo = read("Cargo.toml");
    let module = read("src/app_services/mod.rs");
    assert!(cargo.contains("neoethos-broker-truth"));
    assert!(module.contains("pub mod broker_deal_economics;"));
    for forbidden in ["EngineStats", "ExecReport", "FilledExecutionReportV2"] {
        assert!(
            !source.contains(forbidden),
            "live broker money must not inherit legacy monetary authority from `{forbidden}`"
        );
    }
}
