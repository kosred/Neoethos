//! Runtime contract for the durable per-deal Journal Money V3 ledger.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use neoethos_app::app_services::broker_deal_economics::{
    BrokerDealWireSnapshotV1, BrokerPnlConversionFeeV1, BrokerSymbolVolumeScaleEvidenceV1,
    build_broker_deal_money_evidence_v1,
};
use neoethos_app::app_services::ctrader_account::{
    CTraderAccountRuntimeSnapshot, CTraderPositionSnapshot, CTraderReconcileSnapshot,
    CTraderTraderSnapshot,
};
use neoethos_app::app_services::ctrader_live_auth::CTraderEnvironment;
use neoethos_app::app_services::journal_money_v3::{
    BrokerPositionLifecycleIdentityV3, BrokerPositionLifecycleWireV3, JournalDealObservationV3,
    JournalMonetaryAuthorityV3, JournalMoneyArtifactClassV3, JournalMoneyErrorCodeV3,
    JournalMoneyPromotionEligibilityV3, JournalMoneyV3Store, JournalPnlConversionFeeV3,
    build_broker_position_lifecycle_identity_v3,
};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "neoethos-journal-money-v3-{tag}-{}-{nanos}",
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
            let _ = fs::remove_file(&self.0);
        }
    }
}

fn contract(
    environment: &str,
    account_id: i64,
    lot_size_raw_centi_units: i64,
) -> BrokerSymbolVolumeScaleEvidenceV1 {
    BrokerSymbolVolumeScaleEvidenceV1::new(
        environment,
        account_id,
        7,
        "EURUSD",
        lot_size_raw_centi_units,
    )
    .expect("valid broker lot-size evidence")
}

fn lifecycle(
    environment: &str,
    account_id: i64,
    expected_entry_filled_volume_raw_centi_units: i64,
    money_digits: u32,
    volume_scale: &BrokerSymbolVolumeScaleEvidenceV1,
) -> BrokerPositionLifecycleIdentityV3 {
    build_broker_position_lifecycle_identity_v3(
        &BrokerPositionLifecycleWireV3 {
            environment: environment.to_string(),
            account_id,
            position_id: 77,
            symbol_id: 7,
            symbol_name: "EURUSD".to_string(),
            position_side: "BUY".to_string(),
            account_currency: "USD".to_string(),
            money_digits,
            expected_entry_filled_volume_raw_centi_units,
            entry_timestamp_ms: 1_799_999_000_000,
            entry_price: 1.1000,
        },
        volume_scale,
    )
    .expect("valid lifecycle identity")
}

#[allow(clippy::too_many_arguments)]
fn close_evidence(
    environment: &str,
    account_id: i64,
    deal_id: i64,
    filled_volume_raw_centi_units: i64,
    money_digits: u32,
    gross_profit_raw_scaled: i64,
    commission_raw_scaled_signed: i64,
    swap_raw_scaled_signed: i64,
    pnl_conversion_fee: BrokerPnlConversionFeeV1,
    volume_scale: &BrokerSymbolVolumeScaleEvidenceV1,
) -> neoethos_app::app_services::broker_deal_economics::BrokerDealMoneyEvidenceV1 {
    build_broker_deal_money_evidence_v1(
        &BrokerDealWireSnapshotV1 {
            environment: environment.to_string(),
            account_id,
            deal_id,
            order_id: 900 + deal_id,
            position_id: 77,
            symbol_id: 7,
            symbol_name: "EURUSD".to_string(),
            deal_status: "FILLED".to_string(),
            trade_side: "SELL".to_string(),
            filled_volume_raw_centi_units,
            execution_timestamp_ms: 1_800_000_000_000 + deal_id,
            execution_price: Some(1.1010),
            entry_price: Some(1.1000),
            money_digits: Some(money_digits),
            gross_profit_raw_scaled: Some(gross_profit_raw_scaled),
            commission_raw_scaled_signed: Some(commission_raw_scaled_signed),
            swap_raw_scaled_signed: Some(swap_raw_scaled_signed),
            pnl_conversion_fee,
        },
        volume_scale,
        "USD",
    )
    .expect("complete broker close-deal evidence")
}

fn runtime(
    environment: CTraderEnvironment,
    account_id: i64,
    money_digits: u32,
    position_still_open: bool,
) -> CTraderAccountRuntimeSnapshot {
    let positions = if position_still_open {
        vec![CTraderPositionSnapshot {
            position_id: 77,
            symbol_id: 7,
            trade_side: "BUY".to_string(),
            volume: 10_000.0,
            open_timestamp_ms: Some(1_799_999_000_000),
            price: Some(1.1000),
            stop_loss: None,
            take_profit: None,
            swap: Some(0.0),
            commission: Some(0.0),
            mirroring_commission: Some(0.0),
            used_margin: Some(100.0),
            label: None,
            comment: None,
            client_order_id: None,
        }]
    } else {
        Vec::new()
    };
    CTraderAccountRuntimeSnapshot {
        environment,
        trader: CTraderTraderSnapshot {
            account_id,
            balance: 10_000.0,
            leverage: Some(30.0),
            trader_login: Some(1),
            account_type: Some("HEDGED".to_string()),
            broker_name: Some("fixture".to_string()),
            money_digits,
            deposit_asset_id: Some(8),
        },
        reconcile: CTraderReconcileSnapshot {
            account_id,
            positions,
            pending_orders: Vec::new(),
        },
        recent_deals: Vec::new(),
        unrealized_pnl: 0.0,
        unrealized_pnl_by_position: BTreeMap::new(),
        deposit_asset_name: "USD".to_string(),
    }
}

#[test]
fn partial_closes_survive_reopen_and_finalize_only_on_exact_volume_plus_broker_flat() {
    let temp = TempDir::new("partial");
    let scale = contract("demo", 42, 10_000_000);
    let lifecycle = lifecycle("demo", 42, 1_000_000, 2, &scale);
    let first = close_evidence(
        "demo",
        42,
        10,
        400_000,
        2,
        5_000,
        -700,
        -100,
        BrokerPnlConversionFeeV1::Charged {
            raw_scaled_signed: -200,
        },
        &scale,
    );
    let second = close_evidence(
        "demo",
        42,
        11,
        600_000,
        2,
        5_000,
        -700,
        -100,
        BrokerPnlConversionFeeV1::Charged {
            raw_scaled_signed: -200,
        },
        &scale,
    );

    let store = JournalMoneyV3Store::new(temp.path());
    assert_eq!(
        store
            .record_close_fill(&lifecycle, &first)
            .expect("persist first partial fill"),
        JournalDealObservationV3::Added
    );

    let reopened = JournalMoneyV3Store::new(temp.path());
    let state = reopened
        .position_state(&lifecycle)
        .expect("reopen exact durable ledger");
    assert_eq!(state.deal_count(), 1);
    assert_eq!(state.closed_filled_volume_raw_centi_units(), 400_000);
    assert_eq!(state.component_sum_raw_scaled(), 4_000);
    assert!(!state.is_finalized());

    let error = reopened
        .finalize_from_account_runtime(&lifecycle, &runtime(CTraderEnvironment::Demo, 42, 2, true))
        .expect_err("an open broker position cannot finalize");
    assert_eq!(error.code(), JournalMoneyErrorCodeV3::PositionStillOpen);

    let error = reopened
        .finalize_from_account_runtime(&lifecycle, &runtime(CTraderEnvironment::Demo, 42, 2, false))
        .expect_err("flat state cannot hide an incomplete close volume");
    assert_eq!(error.code(), JournalMoneyErrorCodeV3::FilledVolumeMismatch);

    assert_eq!(
        reopened
            .record_close_fill(&lifecycle, &second)
            .expect("persist second partial fill"),
        JournalDealObservationV3::Added
    );
    let receipt = reopened
        .finalize_from_account_runtime(&lifecycle, &runtime(CTraderEnvironment::Demo, 42, 2, false))
        .expect("exact fills plus broker-flat snapshot finalize once");

    assert_eq!(
        receipt.artifact_class(),
        JournalMoneyArtifactClassV3::VerifiedBrokerDealMoney
    );
    assert_eq!(
        receipt.monetary_authority(),
        JournalMonetaryAuthorityV3::VerifiedBrokerDealComponents
    );
    assert_eq!(
        receipt.promotion_eligibility(),
        JournalMoneyPromotionEligibilityV3::EligibleForRiskAndPromotion
    );
    assert_eq!(receipt.fills().len(), 2);
    assert_eq!(receipt.closed_filled_volume_raw_centi_units(), 1_000_000);
    assert_eq!(receipt.gross_profit_raw_scaled(), 10_000);
    assert_eq!(receipt.commission_raw_scaled_signed(), -1_400);
    assert_eq!(receipt.swap_raw_scaled_signed(), -200);
    assert_eq!(receipt.pnl_conversion_fee_raw_scaled_signed(), -400);
    assert_eq!(receipt.component_sum_raw_scaled(), 8_000);
    assert_eq!(receipt.component_sum_account_currency(), 80.0);
    assert_eq!(receipt.receipt_identity_sha256().len(), 64);
    assert_eq!(
        receipt
            .flat_reconcile_evidence()
            .runtime_snapshot_identity_sha256()
            .len(),
        64
    );

    let reopened_again = JournalMoneyV3Store::new(temp.path());
    let same = reopened_again
        .finalize_from_account_runtime(&lifecycle, &runtime(CTraderEnvironment::Demo, 42, 2, false))
        .expect("finalization retry loads the immutable receipt");
    assert_eq!(
        same.receipt_identity_sha256(),
        receipt.receipt_identity_sha256()
    );
}

#[test]
fn same_deal_digest_is_idempotent_but_same_id_with_changed_money_is_refused() {
    let temp = TempDir::new("dedup");
    let scale = contract("demo", 42, 10_000_000);
    let lifecycle = lifecycle("demo", 42, 1_000_000, 2, &scale);
    let original = close_evidence(
        "demo",
        42,
        20,
        1_000_000,
        2,
        10_000,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::NotApplied,
        &scale,
    );
    let changed = close_evidence(
        "demo",
        42,
        20,
        1_000_000,
        2,
        10_001,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::NotApplied,
        &scale,
    );
    let store = JournalMoneyV3Store::new(temp.path());
    assert_eq!(
        store
            .record_close_fill(&lifecycle, &original)
            .expect("first observation"),
        JournalDealObservationV3::Added
    );
    assert_eq!(
        store
            .record_close_fill(&lifecycle, &original)
            .expect("identical replay"),
        JournalDealObservationV3::Duplicate
    );
    let error = store
        .record_close_fill(&lifecycle, &changed)
        .expect_err("same deal id with a different digest is tampering");
    assert_eq!(
        error.code(),
        JournalMoneyErrorCodeV3::DuplicateDealIdentityMismatch
    );
    let state = store
        .position_state(&lifecycle)
        .expect("valid original state");
    assert_eq!(state.deal_count(), 1);
    assert_eq!(state.component_sum_raw_scaled(), 8_400);
}

#[test]
fn failed_persistence_does_not_poison_retry_dedup_state() {
    let temp = TempDir::new("retry");
    fs::create_dir_all(temp.path()).expect("temp root");
    let blocked_root = temp.path().join("blocked-by-file");
    fs::write(&blocked_root, b"not a directory").expect("block journal root");

    let scale = contract("demo", 42, 10_000_000);
    let lifecycle = lifecycle("demo", 42, 1_000_000, 2, &scale);
    let fill = close_evidence(
        "demo",
        42,
        30,
        1_000_000,
        2,
        10_000,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::NotApplied,
        &scale,
    );
    let store = JournalMoneyV3Store::new(&blocked_root);
    let error = store
        .record_close_fill(&lifecycle, &fill)
        .expect_err("filesystem refusal must propagate");
    assert_eq!(error.code(), JournalMoneyErrorCodeV3::Io);

    fs::remove_file(&blocked_root).expect("repair fixture root");
    assert_eq!(
        store
            .record_close_fill(&lifecycle, &fill)
            .expect("retry persists after filesystem recovery"),
        JournalDealObservationV3::Added
    );
    assert_eq!(
        JournalMoneyV3Store::new(&blocked_root)
            .position_state(&lifecycle)
            .expect("reopened retry state")
            .deal_count(),
        1
    );
}

#[test]
fn currency_money_scale_contract_and_scope_mismatches_have_zero_ledger_side_effects() {
    let temp = TempDir::new("identity");
    let exact_scale = contract("demo", 42, 10_000_000);
    let lifecycle = lifecycle("demo", 42, 1_000_000, 2, &exact_scale);
    let scale_three = close_evidence(
        "demo",
        42,
        40,
        1_000_000,
        3,
        100_000,
        -14_000,
        -2_000,
        BrokerPnlConversionFeeV1::NotApplied,
        &exact_scale,
    );
    let different_lot_scale = contract("demo", 42, 20_000_000);
    let wrong_contract = close_evidence(
        "demo",
        42,
        41,
        1_000_000,
        2,
        10_000,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::NotApplied,
        &different_lot_scale,
    );
    let store = JournalMoneyV3Store::new(temp.path());
    for fill in [&scale_three, &wrong_contract] {
        let error = store
            .record_close_fill(&lifecycle, fill)
            .expect_err("identity drift must fail before persistence");
        assert_eq!(error.code(), JournalMoneyErrorCodeV3::IdentityMismatch);
    }
    assert!(
        store
            .position_state(&lifecycle)
            .expect("empty position state is valid")
            .is_empty()
    );
}

#[test]
fn charged_and_not_applied_conversion_states_survive_the_final_receipt() {
    let temp = TempDir::new("conversion");
    let scale = contract("demo", 42, 10_000_000);
    let lifecycle = lifecycle("demo", 42, 1_000_000, 2, &scale);
    let charged = close_evidence(
        "demo",
        42,
        50,
        400_000,
        2,
        5_000,
        -700,
        -100,
        BrokerPnlConversionFeeV1::Charged {
            raw_scaled_signed: -200,
        },
        &scale,
    );
    let not_applied = close_evidence(
        "demo",
        42,
        51,
        600_000,
        2,
        5_000,
        -700,
        -100,
        BrokerPnlConversionFeeV1::NotApplied,
        &scale,
    );
    let store = JournalMoneyV3Store::new(temp.path());
    store
        .record_close_fill(&lifecycle, &charged)
        .expect("charged fill");
    store
        .record_close_fill(&lifecycle, &not_applied)
        .expect("not-applied fill");
    let receipt = store
        .finalize_from_account_runtime(&lifecycle, &runtime(CTraderEnvironment::Demo, 42, 2, false))
        .expect("final receipt");
    assert_eq!(
        receipt.fills()[0].pnl_conversion_fee(),
        JournalPnlConversionFeeV3::Charged {
            raw_scaled_signed: -200
        }
    );
    assert_eq!(
        receipt.fills()[1].pnl_conversion_fee(),
        JournalPnlConversionFeeV3::NotApplied
    );
}

#[test]
fn identical_position_ids_are_isolated_by_account_and_environment() {
    let temp = TempDir::new("scope");
    let demo_scale = contract("demo", 42, 10_000_000);
    let live_scale = contract("live", 43, 10_000_000);
    let demo_lifecycle = lifecycle("demo", 42, 1_000_000, 2, &demo_scale);
    let live_lifecycle = lifecycle("live", 43, 1_000_000, 2, &live_scale);
    assert_ne!(
        demo_lifecycle.lifecycle_identity_sha256(),
        live_lifecycle.lifecycle_identity_sha256()
    );

    let store = JournalMoneyV3Store::new(temp.path());
    store
        .record_close_fill(
            &demo_lifecycle,
            &close_evidence(
                "demo",
                42,
                60,
                1_000_000,
                2,
                10_000,
                -1_400,
                -200,
                BrokerPnlConversionFeeV1::NotApplied,
                &demo_scale,
            ),
        )
        .expect("demo fill");
    store
        .record_close_fill(
            &live_lifecycle,
            &close_evidence(
                "live",
                43,
                61,
                1_000_000,
                2,
                -5_000,
                -1_400,
                -200,
                BrokerPnlConversionFeeV1::NotApplied,
                &live_scale,
            ),
        )
        .expect("live fill");
    assert_eq!(
        store
            .position_state(&demo_lifecycle)
            .expect("demo state")
            .component_sum_raw_scaled(),
        8_400
    );
    assert_eq!(
        store
            .position_state(&live_lifecycle)
            .expect("live state")
            .component_sum_raw_scaled(),
        -6_600
    );
}
