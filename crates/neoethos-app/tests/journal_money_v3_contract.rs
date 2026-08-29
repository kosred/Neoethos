//! Source contract for the durable, money-first journal V3 boundary.
//!
//! This test intentionally uses only `std`, so it can be compiled directly
//! with `rustc --test` while another lane owns the Cargo graph.

use std::fs;
use std::path::PathBuf;

const PRODUCTION_RELATIVE_PATH: &str = "src/app_services/journal_money_v3.rs";

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
            "RED: missing strict Journal Money V3 boundary {}: {error}",
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
            "missing Journal Money V3 contract token `{token}`"
        );
    }
}

#[test]
fn v3_wire_is_versioned_strict_and_money_first() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "CLOSED_POSITION_JOURNAL_SCHEMA_VERSION_V3",
            "#[serde(deny_unknown_fields)]",
            "DurableBrokerDealMoneyV3",
            "ClosedPositionJournalReceiptV3",
            "gross_profit_raw_scaled",
            "commission_raw_scaled_signed",
            "swap_raw_scaled_signed",
            "pnl_conversion_fee",
            "component_sum_raw_scaled",
            "money_digits",
            "account_currency",
        ],
    );
    for forbidden in [
        "symbol_metadata::resolve",
        "component_sum_account_currency?",
        "unwrap_or(0.0)",
        "unwrap_or_default()",
        "net_profit: f64",
    ] {
        assert!(
            !source.contains(forbidden),
            "strict V3 cannot rebuild or silently default money through `{forbidden}`"
        );
    }
}

#[test]
fn every_fill_and_contract_identity_survives_the_durable_boundary() {
    let source = production_source();
    let fill = item_body(&source, "pub struct DurableBrokerDealMoneyV3");
    require_tokens(
        fill,
        &[
            "environment",
            "account_id",
            "deal_id",
            "order_id",
            "position_id",
            "symbol_id",
            "symbol_name",
            "trade_side",
            "filled_volume_raw_centi_units",
            "execution_timestamp_ms",
            "execution_price",
            "entry_price",
            "money_digits",
            "account_currency",
            "lot_size_raw_centi_units",
            "volume_scale_identity_sha256",
            "deal_identity_sha256",
        ],
    );
    let lifecycle = item_body(&source, "pub struct BrokerPositionLifecycleIdentityV3");
    require_tokens(
        lifecycle,
        &[
            "environment",
            "account_id",
            "position_id",
            "symbol_id",
            "symbol_name",
            "position_side",
            "account_currency",
            "money_digits",
            "expected_entry_filled_volume_raw_centi_units",
            "lot_size_raw_centi_units",
            "volume_scale_identity_sha256",
            "entry_timestamp_ms",
            "entry_price",
            "lifecycle_identity_sha256",
        ],
    );
}

#[test]
fn persistence_is_immutable_per_deal_and_fail_closed() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "JournalMoneyV3Store",
            "persist_immutable_json",
            "create_new(true)",
            "sync_all()",
            "DuplicateDealIdentityMismatch",
            "CorruptLedger",
            "UnsupportedSchemaVersion",
            "FilledVolumeMismatch",
            "PositionStillOpen",
        ],
    );
    for forbidden in [
        "HashSet<i64>",
        "seen_positions",
        "skipping malformed",
        "treating as empty",
    ] {
        assert!(
            !source.contains(forbidden),
            "durable V3 cannot use process-local or fail-open persistence `{forbidden}`"
        );
    }
}

#[test]
fn final_receipt_binds_flat_snapshot_fills_and_exact_component_totals() {
    let source = production_source();
    let receipt = item_body(&source, "pub struct ClosedPositionJournalReceiptV3");
    require_tokens(
        receipt,
        &[
            "artifact_class",
            "monetary_authority",
            "promotion_eligibility",
            "lifecycle",
            "fills",
            "flat_reconcile_evidence",
            "gross_profit_raw_scaled",
            "commission_raw_scaled_signed",
            "swap_raw_scaled_signed",
            "pnl_conversion_fee_raw_scaled_signed",
            "component_sum_raw_scaled",
            "closed_filled_volume_raw_centi_units",
            "receipt_identity_sha256",
        ],
    );
    require_tokens(
        &source,
        &[
            "BrokerFlatReconcileEvidenceV3::from_account_runtime",
            "runtime.reconcile.positions",
            "expected_entry_filled_volume_raw_centi_units",
        ],
    );
    assert!(
        !source.contains("position_still_open: bool"),
        "a caller-selected boolean is not broker-flat evidence"
    );
}

#[test]
fn legacy_v1_v2_are_display_only_and_cannot_be_upcast() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "LegacyJournalDispositionV3",
            "JournalMoneyArtifactClassV3::DisplayOnly",
            "JournalMonetaryAuthorityV3::Refused",
            "JournalMoneyPromotionEligibilityV3::NotPromotionEligible",
            "classify_legacy_journal_v1_v2",
        ],
    );
    for forbidden in [
        "impl From<ClosedTrade> for ClosedPositionJournalReceiptV3",
        "impl TryFrom<ClosedTrade> for ClosedPositionJournalReceiptV3",
        "query_closed_trades",
        "journal_store::ClosedTrade",
    ] {
        assert!(
            !source.contains(forbidden),
            "legacy scalar rows cannot become V3 monetary authority through `{forbidden}`"
        );
    }
}

#[test]
fn app_exports_the_new_boundary_without_replacing_the_legacy_display_store() {
    let app_services = read("src/app_services/mod.rs");
    assert!(
        app_services.contains("pub mod journal_money_v3;"),
        "app services must export the V3 boundary"
    );
    assert!(
        app_services.contains("pub mod journal_store;"),
        "legacy V1/V2 display history remains available to the UI"
    );
}
