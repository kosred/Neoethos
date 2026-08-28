use core::mem::{align_of, offset_of, size_of};
use neoethos_gpu_contracts::device::{
    NeoPopulationCounters, NeoPopulationEvent, NeoPopulationMetricRow, NeoPopulationOutcome,
    NeoPopulationSettings,
};
use neoethos_gpu_contracts::{
    ABI_VERSION, POPULATION_DIRECTION_LONG, POPULATION_DIRECTION_SHORT, POPULATION_EXIT_GAP,
    POPULATION_EXIT_MAX_HOLD, POPULATION_EXIT_NONE, POPULATION_EXIT_STOP, POPULATION_EXIT_TARGET,
    POPULATION_PRECEDENCE_STOP_FIRST, POPULATION_SETTINGS_FLAG_RISK_BASED_SIZING,
};

#[test]
fn population_abi_constants_are_numerically_pinned() {
    assert_eq!(ABI_VERSION, 4);
    assert_eq!(POPULATION_SETTINGS_FLAG_RISK_BASED_SIZING, 1);
    assert_eq!(POPULATION_DIRECTION_LONG, 1);
    assert_eq!(POPULATION_DIRECTION_SHORT, -1);
    assert_eq!(POPULATION_PRECEDENCE_STOP_FIRST, 0);
    assert_eq!(POPULATION_EXIT_NONE, 0);
    assert_eq!(POPULATION_EXIT_STOP, 1);
    assert_eq!(POPULATION_EXIT_TARGET, 2);
    assert_eq!(POPULATION_EXIT_MAX_HOLD, 3);
    assert_eq!(POPULATION_EXIT_GAP, 4);
}

#[test]
fn population_outcome_default_is_the_unresolved_sentinel() {
    let outcome = NeoPopulationOutcome::default();
    assert_eq!(outcome.exit_bar, -1);
    assert_eq!(outcome.entry_bar, -1);
    assert_eq!(outcome.mfe, 0.0);
    assert_eq!(outcome.mae, 0.0);
    assert_eq!(outcome.pnl, 0.0);
    assert_eq!(outcome.r_multiple, 0.0);
    assert_eq!(outcome.exit_reason, POPULATION_EXIT_NONE);
}

#[test]
fn population_settings_has_stable_c_layout() {
    assert_eq!(size_of::<NeoPopulationSettings>(), 184);
    assert_eq!(offset_of!(NeoPopulationSettings, spread_pips_asian), 160);
    assert_eq!(offset_of!(NeoPopulationSettings, spread_pips_late_ny), 176);
    assert_eq!(offset_of!(NeoPopulationSettings, trailing_enabled), 128);
    assert_eq!(
        offset_of!(NeoPopulationSettings, trailing_atr_multiplier),
        136
    );
    assert_eq!(
        offset_of!(NeoPopulationSettings, trailing_be_trigger_r),
        144
    );
    assert_eq!(
        offset_of!(NeoPopulationSettings, trailing_min_lock_pips),
        152
    );
    assert_eq!(align_of::<NeoPopulationSettings>(), 8);
    assert_eq!(offset_of!(NeoPopulationSettings, abi_version), 0);
    assert_eq!(offset_of!(NeoPopulationSettings, flags), 4);
    assert_eq!(offset_of!(NeoPopulationSettings, max_hold_bars), 8);
    assert_eq!(offset_of!(NeoPopulationSettings, min_hold_bars), 12);
    assert_eq!(offset_of!(NeoPopulationSettings, max_trades_per_day), 16);
    assert_eq!(offset_of!(NeoPopulationSettings, month_capacity), 20);
    assert_eq!(offset_of!(NeoPopulationSettings, gap_threshold_ms), 24);
    assert_eq!(offset_of!(NeoPopulationSettings, initial_equity), 32);
    assert_eq!(offset_of!(NeoPopulationSettings, pip_value), 40);
    assert_eq!(offset_of!(NeoPopulationSettings, spread_pips), 48);
    assert_eq!(offset_of!(NeoPopulationSettings, commission_per_trade), 56);
    assert_eq!(offset_of!(NeoPopulationSettings, pip_value_per_lot), 64);
    assert_eq!(
        offset_of!(NeoPopulationSettings, swap_long_pips_per_day),
        72
    );
    assert_eq!(
        offset_of!(NeoPopulationSettings, swap_short_pips_per_day),
        80
    );
    assert_eq!(
        offset_of!(NeoPopulationSettings, pnl_conversion_fee_rate),
        88
    );
    assert_eq!(offset_of!(NeoPopulationSettings, risk_per_trade_min), 96);
    assert_eq!(offset_of!(NeoPopulationSettings, risk_per_trade_max), 104);
    assert_eq!(
        offset_of!(NeoPopulationSettings, high_quality_confidence),
        112
    );
    assert_eq!(offset_of!(NeoPopulationSettings, adaptive_rr), 120);
}

#[test]
fn population_event_has_stable_c_layout() {
    assert_eq!(size_of::<NeoPopulationEvent>(), 56);
    assert_eq!(align_of::<NeoPopulationEvent>(), 8);
    assert_eq!(offset_of!(NeoPopulationEvent, candidate_id), 0);
    assert_eq!(offset_of!(NeoPopulationEvent, scenario_id), 8);
    assert_eq!(offset_of!(NeoPopulationEvent, entry_bar), 16);
    assert_eq!(offset_of!(NeoPopulationEvent, last_bar), 20);
    assert_eq!(offset_of!(NeoPopulationEvent, direction), 24);
    assert_eq!(offset_of!(NeoPopulationEvent, precedence), 28);
    assert_eq!(offset_of!(NeoPopulationEvent, stop_price), 32);
    assert_eq!(offset_of!(NeoPopulationEvent, target_price), 40);
    assert_eq!(offset_of!(NeoPopulationEvent, entry_price), 48);
}

#[test]
fn population_outcome_has_stable_c_layout() {
    assert_eq!(size_of::<NeoPopulationOutcome>(), 72);
    assert_eq!(align_of::<NeoPopulationOutcome>(), 8);
    assert_eq!(offset_of!(NeoPopulationOutcome, candidate_id), 0);
    assert_eq!(offset_of!(NeoPopulationOutcome, scenario_id), 8);
    assert_eq!(offset_of!(NeoPopulationOutcome, exit_bar), 16);
    assert_eq!(offset_of!(NeoPopulationOutcome, entry_bar), 24);
    assert_eq!(offset_of!(NeoPopulationOutcome, mfe), 32);
    assert_eq!(offset_of!(NeoPopulationOutcome, mae), 40);
    assert_eq!(offset_of!(NeoPopulationOutcome, exit_price), 48);
    assert_eq!(offset_of!(NeoPopulationOutcome, pnl), 56);
    assert_eq!(offset_of!(NeoPopulationOutcome, r_multiple), 64);
    assert_eq!(offset_of!(NeoPopulationOutcome, exit_reason), 20);
}

#[test]
fn population_metric_row_has_stable_c_layout() {
    assert_eq!(size_of::<NeoPopulationMetricRow>(), 104);
    assert_eq!(align_of::<NeoPopulationMetricRow>(), 8);
    assert_eq!(offset_of!(NeoPopulationMetricRow, candidate_id), 0);
    assert_eq!(offset_of!(NeoPopulationMetricRow, scenario_id), 8);
    assert_eq!(offset_of!(NeoPopulationMetricRow, values), 16);
}

#[test]
fn population_counters_have_stable_c_layout() {
    assert_eq!(size_of::<NeoPopulationCounters>(), 96);
    assert_eq!(align_of::<NeoPopulationCounters>(), 8);
    assert_eq!(offset_of!(NeoPopulationCounters, event_count), 0);
    assert_eq!(offset_of!(NeoPopulationCounters, accepted_trade_count), 8);
    assert_eq!(offset_of!(NeoPopulationCounters, kernel_submissions), 16);
    assert_eq!(
        offset_of!(NeoPopulationCounters, synchronization_events),
        24
    );
    assert_eq!(offset_of!(NeoPopulationCounters, dataset_upload_bytes), 32);
    assert_eq!(offset_of!(NeoPopulationCounters, gene_upload_bytes), 40);
    assert_eq!(offset_of!(NeoPopulationCounters, scenario_upload_bytes), 48);
    assert_eq!(
        offset_of!(NeoPopulationCounters, compact_readback_bytes),
        56
    );
    assert_eq!(offset_of!(NeoPopulationCounters, full_readback_bytes), 64);
    assert_eq!(offset_of!(NeoPopulationCounters, reserved), 72);
}
