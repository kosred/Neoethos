//! Direct-runtime tests for the live broker-deal money leaf.
//!
//! This file deliberately path-includes the leaf so it can be compiled with
//! `rustc --test` while another lane owns Cargo. It exercises real production
//! code, not a duplicated oracle.

// Path-including a library leaf makes otherwise-public API look private to
// this one test binary. Cargo does not issue these dead-code warnings for the
// real public module, so suppress only that harness artifact.
#[allow(dead_code)]
#[path = "../src/app_services/broker_deal_economics.rs"]
mod broker_deal_economics;

use broker_deal_economics::{
    BrokerDealMoneyErrorCodeV1, BrokerDealObservationV1, BrokerDealWireSnapshotV1,
    BrokerPnlConversionFeeV1, BrokerPositionMoneyAccumulatorV1, BrokerSymbolVolumeScaleEvidenceV1,
    build_broker_deal_money_evidence_v1,
};

fn eurusd_contract() -> BrokerSymbolVolumeScaleEvidenceV1 {
    BrokerSymbolVolumeScaleEvidenceV1::new("demo", 42, 7, "EURUSD", 10_000_000)
        .expect("valid broker symbol contract")
}

fn close_wire(
    deal_id: i64,
    filled_volume_raw_centi_units: i64,
    gross_profit_raw_scaled: i64,
    commission_raw_scaled_signed: i64,
    swap_raw_scaled_signed: i64,
    pnl_conversion_fee: BrokerPnlConversionFeeV1,
) -> BrokerDealWireSnapshotV1 {
    BrokerDealWireSnapshotV1 {
        environment: "demo".to_string(),
        account_id: 42,
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
        money_digits: Some(2),
        gross_profit_raw_scaled: Some(gross_profit_raw_scaled),
        commission_raw_scaled_signed: Some(commission_raw_scaled_signed),
        swap_raw_scaled_signed: Some(swap_raw_scaled_signed),
        pnl_conversion_fee,
    }
}

#[test]
fn exact_wire_components_and_broker_lot_size_build_typed_money() {
    let wire = close_wire(
        1,
        1_000_000,
        10_000,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::Charged {
            raw_scaled_signed: -100,
        },
    );
    let evidence = build_broker_deal_money_evidence_v1(&wire, &eurusd_contract(), "USD")
        .expect("complete broker wire must become typed evidence");

    assert_eq!(evidence.account_currency(), "USD");
    assert_eq!(evidence.deal_id(), 1);
    assert_eq!(evidence.position_id(), 77);
    assert_eq!(evidence.money_digits(), 2);
    assert_eq!(evidence.gross_profit_raw_scaled(), 10_000);
    assert_eq!(evidence.commission_raw_scaled_signed(), -1_400);
    assert_eq!(evidence.swap_raw_scaled_signed(), -200);
    assert_eq!(evidence.component_sum_raw_scaled(), 8_300);
    assert_eq!(evidence.filled_volume_raw_centi_units(), 1_000_000);
    assert_eq!(evidence.contract_units_per_lot(), 100_000.0);
    assert_eq!(evidence.actual_filled_lots(), 0.1);
    assert_eq!(evidence.gross_profit_account_currency().amount(), 100.0);
    assert_eq!(
        evidence.commission_account_currency_signed().amount(),
        -14.0
    );
    assert_eq!(evidence.swap_account_currency_signed().amount(), -2.0);
    assert_eq!(
        evidence
            .pnl_conversion_fee_account_currency_signed()
            .amount(),
        -1.0
    );
    assert_eq!(evidence.component_sum_account_currency().amount(), 83.0);
    assert_eq!(evidence.deal_identity_sha256().len(), 64);
    assert_eq!(evidence.volume_scale_identity_sha256().len(), 64);
}

#[test]
fn missing_money_and_identity_mismatch_refuse_before_evidence_exists() {
    let mut missing = close_wire(
        2,
        1_000_000,
        10_000,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::NotApplied,
    );
    missing.gross_profit_raw_scaled = None;
    let error = build_broker_deal_money_evidence_v1(&missing, &eurusd_contract(), "USD")
        .expect_err("missing gross profit must fail closed");
    assert_eq!(error.code(), BrokerDealMoneyErrorCodeV1::MissingGrossProfit);

    let wrong_account = BrokerSymbolVolumeScaleEvidenceV1::new("demo", 43, 7, "EURUSD", 10_000_000)
        .expect("valid but different account contract");
    let complete = close_wire(
        3,
        1_000_000,
        10_000,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::NotApplied,
    );
    let error = build_broker_deal_money_evidence_v1(&complete, &wrong_account, "USD")
        .expect_err("deal and symbol-contract account identities must match");
    assert_eq!(
        error.code(),
        BrokerDealMoneyErrorCodeV1::FillIdentityMismatch
    );
}

#[test]
fn omitted_conversion_is_explicit_not_applied_evidence() {
    let wire = close_wire(
        4,
        1_000_000,
        10_000,
        -1_400,
        -200,
        BrokerPnlConversionFeeV1::NotApplied,
    );
    let evidence = build_broker_deal_money_evidence_v1(&wire, &eurusd_contract(), "USD")
        .expect("NotApplied is explicit evidence");

    assert_eq!(
        evidence.pnl_conversion_fee_state(),
        BrokerPnlConversionFeeV1::NotApplied
    );
    assert_eq!(
        evidence
            .pnl_conversion_fee_account_currency_signed()
            .amount(),
        0.0
    );
    assert_eq!(evidence.component_sum_account_currency().amount(), 84.0);
}

#[test]
fn duplicate_partial_fills_sum_once_and_finalize_only_when_broker_is_flat() {
    let first = build_broker_deal_money_evidence_v1(
        &close_wire(
            10,
            400_000,
            5_000,
            -700,
            -100,
            BrokerPnlConversionFeeV1::Charged {
                raw_scaled_signed: -200,
            },
        ),
        &eurusd_contract(),
        "USD",
    )
    .expect("first partial fill");
    let second = build_broker_deal_money_evidence_v1(
        &close_wire(
            11,
            600_000,
            5_000,
            -700,
            -100,
            BrokerPnlConversionFeeV1::Charged {
                raw_scaled_signed: -200,
            },
        ),
        &eurusd_contract(),
        "USD",
    )
    .expect("second partial fill");

    let mut accumulator = BrokerPositionMoneyAccumulatorV1::new(&first);
    assert_eq!(
        accumulator.observe_fill(&first).expect("first observation"),
        BrokerDealObservationV1::Added
    );
    assert!(
        accumulator
            .finalize_if_position_closed(true)
            .expect("an open broker position remains pending")
            .is_none()
    );
    assert_eq!(
        accumulator
            .observe_fill(&first)
            .expect("identical deal is idempotent"),
        BrokerDealObservationV1::Duplicate
    );
    assert_eq!(
        accumulator
            .observe_fill(&second)
            .expect("second partial observation"),
        BrokerDealObservationV1::Added
    );
    accumulator
        .verify_complete_filled_volume(1_000_000)
        .expect("partial-close fills exactly cover the entry fill");

    let closed = accumulator
        .finalize_if_position_closed(false)
        .expect("complete verified fills may finalize")
        .expect("flat broker position emits one closed-position result");
    assert_eq!(closed.position_id(), 77);
    assert_eq!(closed.deal_count(), 2);
    assert_eq!(closed.filled_volume_raw_centi_units(), 1_000_000);
    assert_eq!(closed.money_digits(), 2);
    assert_eq!(closed.component_sum_raw_scaled(), 8_000);
    assert_eq!(closed.actual_filled_lots(), 0.1);
    assert_eq!(closed.component_sum_account_currency().amount(), 80.0);
    assert!(accumulator.is_finalized());
}

#[test]
fn partial_close_money_scale_cannot_change_inside_one_position() {
    let first = build_broker_deal_money_evidence_v1(
        &close_wire(
            30,
            400_000,
            5_000,
            -700,
            -100,
            BrokerPnlConversionFeeV1::NotApplied,
        ),
        &eurusd_contract(),
        "USD",
    )
    .expect("first scale-2 fill");
    let mut scale_three_wire = close_wire(
        31,
        600_000,
        50_000,
        -7_000,
        -1_000,
        BrokerPnlConversionFeeV1::NotApplied,
    );
    scale_three_wire.money_digits = Some(3);
    let scale_three =
        build_broker_deal_money_evidence_v1(&scale_three_wire, &eurusd_contract(), "USD")
            .expect("individually valid scale-3 fill");

    let mut accumulator = BrokerPositionMoneyAccumulatorV1::new(&first);
    accumulator.observe_fill(&first).expect("first observation");
    let error = accumulator
        .observe_fill(&scale_three)
        .expect_err("one position cannot mix raw money scales");
    assert_eq!(
        error.code(),
        BrokerDealMoneyErrorCodeV1::FillIdentityMismatch
    );
}

#[test]
fn a_flat_position_cannot_finalize_money_from_an_incomplete_close_volume() {
    let partial = build_broker_deal_money_evidence_v1(
        &close_wire(
            12,
            400_000,
            5_000,
            -700,
            -100,
            BrokerPnlConversionFeeV1::NotApplied,
        ),
        &eurusd_contract(),
        "USD",
    )
    .expect("verified but incomplete partial close");
    let mut accumulator = BrokerPositionMoneyAccumulatorV1::new(&partial);
    accumulator
        .observe_fill(&partial)
        .expect("partial observation");

    let error = accumulator
        .verify_complete_filled_volume(1_000_000)
        .expect_err("missing 600,000 raw centi-units must fail closed");
    assert_eq!(
        error.code(),
        BrokerDealMoneyErrorCodeV1::FilledVolumeMismatch
    );
    accumulator.refuse_unverified_fill();
    assert!(
        accumulator.finalize_if_position_closed(false).is_err(),
        "flat broker state alone cannot authorize incomplete close money"
    );
}

#[test]
fn refused_fill_keeps_the_position_accumulator_pending_with_zero_money_change() {
    let first = build_broker_deal_money_evidence_v1(
        &close_wire(
            20,
            400_000,
            5_000,
            -700,
            -100,
            BrokerPnlConversionFeeV1::NotApplied,
        ),
        &eurusd_contract(),
        "USD",
    )
    .expect("verified first fill");
    let mut accumulator = BrokerPositionMoneyAccumulatorV1::new(&first);
    accumulator.observe_fill(&first).expect("first observation");
    let before = accumulator.component_sum_account_currency();

    accumulator.refuse_unverified_fill();
    let error = accumulator
        .finalize_if_position_closed(false)
        .expect_err("one unverified fill prevents monetary finalization");
    assert_eq!(error.code(), BrokerDealMoneyErrorCodeV1::UnverifiedFill);
    assert_eq!(accumulator.component_sum_account_currency(), before);
    assert!(!accumulator.is_finalized());
}
