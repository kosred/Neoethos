//! Runtime tests for monetary authority and legacy refusal at Journal V3.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use neoethos_app::app_services::broker_deal_economics::{
    BrokerDealWireSnapshotV1, BrokerPnlConversionFeeV1, BrokerSymbolVolumeScaleEvidenceV1,
    build_broker_deal_money_evidence_v1,
};
use neoethos_app::app_services::ctrader_account::{
    CTraderAccountRuntimeSnapshot, CTraderReconcileSnapshot, CTraderTraderSnapshot,
};
use neoethos_app::app_services::ctrader_live_auth::CTraderEnvironment;
use neoethos_app::app_services::journal_money_v3::{
    BrokerPositionLifecycleIdentityV3, BrokerPositionLifecycleWireV3, JournalAccountScopeV3,
    JournalMonetaryAuthorityV3, JournalMoneyArtifactClassV3, JournalMoneyErrorCodeV3,
    JournalMoneyPromotionEligibilityV3, JournalMoneyV3Store,
    build_broker_position_lifecycle_identity_v3, classify_legacy_journal_v1_v2,
};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "neoethos-journal-authority-v3-{tag}-{}-{nanos}",
            std::process::id()
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if self.0.parent() == Some(std::env::temp_dir().as_path()) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

fn scale() -> BrokerSymbolVolumeScaleEvidenceV1 {
    BrokerSymbolVolumeScaleEvidenceV1::new("demo", 42, 7, "EURUSD", 10_000_000)
        .expect("valid volume scale")
}

fn lifecycle() -> BrokerPositionLifecycleIdentityV3 {
    build_broker_position_lifecycle_identity_v3(
        &BrokerPositionLifecycleWireV3 {
            environment: "demo".to_string(),
            account_id: 42,
            position_id: 77,
            symbol_id: 7,
            symbol_name: "EURUSD".to_string(),
            position_side: "BUY".to_string(),
            account_currency: "USD".to_string(),
            money_digits: 2,
            expected_entry_filled_volume_raw_centi_units: 1_000_000,
            entry_timestamp_ms: 1_799_999_000_000,
            entry_price: 1.1000,
        },
        &scale(),
    )
    .expect("valid lifecycle")
}

fn losing_fill() -> neoethos_app::app_services::broker_deal_economics::BrokerDealMoneyEvidenceV1 {
    build_broker_deal_money_evidence_v1(
        &BrokerDealWireSnapshotV1 {
            environment: "demo".to_string(),
            account_id: 42,
            deal_id: 10,
            order_id: 910,
            position_id: 77,
            symbol_id: 7,
            symbol_name: "EURUSD".to_string(),
            deal_status: "FILLED".to_string(),
            trade_side: "SELL".to_string(),
            filled_volume_raw_centi_units: 1_000_000,
            execution_timestamp_ms: 1_800_000_000_010,
            execution_price: Some(1.0990),
            entry_price: Some(1.1000),
            money_digits: Some(2),
            gross_profit_raw_scaled: Some(-5_000),
            commission_raw_scaled_signed: Some(-1_400),
            swap_raw_scaled_signed: Some(-200),
            pnl_conversion_fee: BrokerPnlConversionFeeV1::NotApplied,
        },
        &scale(),
        "USD",
    )
    .expect("valid losing fill")
}

fn flat_runtime() -> CTraderAccountRuntimeSnapshot {
    CTraderAccountRuntimeSnapshot {
        environment: CTraderEnvironment::Demo,
        trader: CTraderTraderSnapshot {
            account_id: 42,
            balance: 9_934.0,
            leverage: Some(30.0),
            trader_login: Some(1),
            account_type: Some("HEDGED".to_string()),
            broker_name: Some("fixture".to_string()),
            money_digits: 2,
            deposit_asset_id: Some(8),
        },
        reconcile: CTraderReconcileSnapshot {
            account_id: 42,
            positions: Vec::new(),
            pending_orders: Vec::new(),
        },
        recent_deals: Vec::new(),
        unrealized_pnl: 0.0,
        unrealized_pnl_by_position: BTreeMap::new(),
        deposit_asset_name: "USD".to_string(),
    }
}

fn finalized_store(temp: &TempDir) -> (JournalMoneyV3Store, BrokerPositionLifecycleIdentityV3) {
    let store = JournalMoneyV3Store::new(temp.path());
    let identity = lifecycle();
    store
        .record_close_fill(&identity, &losing_fill())
        .expect("persist losing fill");
    store
        .finalize_from_account_runtime(&identity, &flat_runtime())
        .expect("finalize losing position");
    (store, identity)
}

fn receipt_path(root: &Path, identity: &BrokerPositionLifecycleIdentityV3) -> PathBuf {
    root.join("journal")
        .join("money-v3")
        .join("positions")
        .join(identity.lifecycle_identity_sha256())
        .join("receipt.v3.json")
}

#[test]
fn legacy_v1_v2_are_explicitly_display_only_non_promotable_and_non_monetary() {
    for schema_version in [1, 2] {
        let disposition = classify_legacy_journal_v1_v2(schema_version)
            .expect("known legacy versions have an explicit refusal disposition");
        assert_eq!(
            disposition.artifact_class(),
            JournalMoneyArtifactClassV3::DisplayOnly
        );
        assert_eq!(
            disposition.monetary_authority(),
            JournalMonetaryAuthorityV3::Refused
        );
        assert_eq!(
            disposition.promotion_eligibility(),
            JournalMoneyPromotionEligibilityV3::NotPromotionEligible
        );
    }
    for unsupported in [0, 3, 4] {
        let error = classify_legacy_journal_v1_v2(unsupported)
            .expect_err("only legacy V1/V2 can receive the display-only disposition");
        assert_eq!(
            error.code(),
            JournalMoneyErrorCodeV3::UnsupportedSchemaVersion
        );
    }
}

#[test]
fn strict_loader_exposes_only_final_receipts_and_typed_exact_period_losses() {
    let temp = TempDir::new("load");
    let (store, _) = finalized_store(&temp);
    let finalized = store
        .load_finalized_receipts_strict()
        .expect("strict finalized-receipt set");
    assert_eq!(finalized.len(), 1);
    assert!(!finalized.is_empty());
    assert_eq!(finalized.receipts()[0].component_sum_raw_scaled(), -6_600);

    let scope = JournalAccountScopeV3::new("demo", 42, "USD", 2).expect("exact scope");
    let losses = finalized
        .period_losses(&scope, 1_800_000_100_000)
        .expect("exact period-loss money");
    assert_eq!(losses.account_currency(), "USD");
    assert_eq!(losses.money_digits(), 2);
    assert_eq!(losses.day_loss_raw_scaled(), 6_600);
    assert_eq!(losses.week_loss_raw_scaled(), 6_600);
    assert_eq!(losses.month_loss_raw_scaled(), 6_600);
    assert_eq!(losses.day_loss_account_currency(), 66.0);
}

#[test]
fn pending_positions_are_not_monetary_authority() {
    let temp = TempDir::new("pending");
    let store = JournalMoneyV3Store::new(temp.path());
    let identity = lifecycle();
    store
        .record_close_fill(&identity, &losing_fill())
        .expect("persist pending fill");
    let finalized = store
        .load_finalized_receipts_strict()
        .expect("pending fills are not parse errors");
    assert!(finalized.is_empty());
    assert!(
        !store
            .position_state(&identity)
            .expect("state")
            .is_finalized()
    );
}

#[test]
fn malformed_or_unknown_receipts_fail_the_whole_authority_load() {
    let temp = TempDir::new("corrupt");
    let (store, identity) = finalized_store(&temp);
    let path = receipt_path(temp.path(), &identity);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).expect("read valid receipt before corruption"))
            .expect("valid receipt json");
    value
        .as_object_mut()
        .expect("receipt object")
        .insert("unknownAuthorityField".to_string(), serde_json::json!(true));
    fs::write(
        &path,
        serde_json::to_vec(&value).expect("encode corrupt row"),
    )
    .expect("corrupt fixture receipt");
    let error = store
        .load_finalized_receipts_strict()
        .expect_err("unknown fields cannot be skipped or accepted");
    assert_eq!(error.code(), JournalMoneyErrorCodeV3::CorruptLedger);

    value
        .as_object_mut()
        .expect("receipt object")
        .remove("unknownAuthorityField");
    value["schema_version"] = serde_json::json!(4);
    fs::write(
        &path,
        serde_json::to_vec(&value).expect("encode future row"),
    )
    .expect("future fixture receipt");
    let error = store
        .load_finalized_receipts_strict()
        .expect_err("future schemas fail closed");
    assert_eq!(
        error.code(),
        JournalMoneyErrorCodeV3::UnsupportedSchemaVersion
    );
}
