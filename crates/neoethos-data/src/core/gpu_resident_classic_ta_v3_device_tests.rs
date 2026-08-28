use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, ensure};
use neoethos_gpu_cuda::resident_classic_ta_v3::{
    ResidentClassicTaDeviceFixtureRequestV3, ResidentClassicTaExpectedColumnV3,
    ResidentClassicTaRecipeV3, run_resident_classic_ta_v3_device_fixture,
};

use crate::Ohlcv;
use crate::core::features::{FeatureCellValidity, FeatureColumnF64};
use crate::core::gpu_resident_classic_ta_v3::preflight_resident_classic_ta_v3;

use super::{
    ALL_INDICATORS, ALT_PERIODS, ClassicTaAdmissionPlan, ClassicTaRunPlan, IndicatorComputePolicy,
    MULTI_PERIOD_IDS, build_classic_ta_admission_plan, classic_indicator_id_for_column,
    compute_classic_ta_feature_columns_f64_with_run_plan, planned_output_count,
};

// Stay below the global vocabulary-floor threshold: this fixture executes an
// explicit reviewed routeable ALL-order subset through Halftrend. The complete
// graph remains fail-closed in the census below; no production admission or
// route is narrowed. 4,096 rows still exceed every admitted 200-bar sweep.
const FIXTURE_ROWS_V3: usize = 4_096;
const SYNTHETIC_PACK_WIDTHS_V3: [usize; 6] = [1, 31, 32, 33, 63, 64];
const COMPACT_VALIDITY_AND_ROOT_D2H_BYTES_V3: u64 = 4 + 32;
const KNOWN_UNROUTEABLE_BASE_FAMILIES_THROUGH_HALFTREND_V3: [&str; 15] = [
    "dec_osc",
    "decycler",
    "demand_index",
    "donchian_channel_width",
    "dti",
    "dynamic_momentum_index",
    "ehlers_adaptive_cg",
    "ehlers_adaptive_cyber_cycle",
    "ehlers_detrending_filter",
    "ehlers_pma",
    "ehlers_simple_cycle_indicator",
    "fractal_dimension_index",
    "fvg_positioning_average",
    "gmma_oscillator",
    "goertzel_cycle_composite_wave",
];
const REVIEWED_HISTORICAL_IDS_THROUGH_HALFTREND_V3: [&str; 5] =
    ["ema", "atr", "adx", "cci", "bollinger_bands"];

fn all_indicator_ids_through_halftrend_v3() -> Result<Vec<&'static str>> {
    let prefix_end = ALL_INDICATORS
        .iter()
        .position(|id| *id == "halftrend")
        .context("canonical ALL order lost halftrend")?;
    Ok(ALL_INDICATORS[..=prefix_end].to_vec())
}

fn scope_classic_ta_device_fixture_admission_v3(
    rows: usize,
    admitted_indicator_ids: Vec<&'static str>,
    mode: &'static str,
) -> ClassicTaAdmissionPlan {
    let admitted_set = admitted_indicator_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut admission = build_classic_ta_admission_plan(rows, rows);
    admission.admitted_indicator_ids = admitted_indicator_ids;
    admission.capability_deferred_indicator_ids = Vec::new();
    admission.capability_deferred_output_count = 0;
    admission.gpu_route_mode = mode;
    admission.historical_indicator_ids = MULTI_PERIOD_IDS
        .iter()
        .copied()
        .filter(|id| admitted_set.contains(id))
        .collect();
    admission
        .budget_deferred_indicator_ids
        .retain(|id| !admitted_set.contains(id));
    admission.admitted_base_columns = admission
        .admitted_indicator_ids
        .iter()
        .map(|id| planned_output_count(id))
        .sum();
    admission.planned_base_columns = admission.admitted_base_columns;
    admission.working_set = None;
    admission
        .extended_groups
        .retain(|(id, _)| admitted_set.contains(id));
    admission.extended_budget_deferred_indicator_ids = admission
        .extended_budget_deferred_indicator_ids
        .into_iter()
        .filter(|id| admitted_set.contains(id))
        .collect();
    admission.extended_planned_columns = admission
        .extended_groups
        .iter()
        .map(|(id, periods)| planned_output_count(id) * periods.len())
        .sum();
    admission.extended_budget_columns = admission.extended_planned_columns;
    admission.extended_mode = mode;
    admission
}

fn prepare_classic_ta_device_fixture_plan_v3(rows: usize) -> Result<ClassicTaRunPlan> {
    ensure!(rows > 0, "Classic TA device fixture rows must be nonzero");
    let full_prefix = all_indicator_ids_through_halftrend_v3()?;
    let excluded_base_families = KNOWN_UNROUTEABLE_BASE_FAMILIES_THROUGH_HALFTREND_V3
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    ensure!(
        excluded_base_families.len() == KNOWN_UNROUTEABLE_BASE_FAMILIES_THROUGH_HALFTREND_V3.len()
            && excluded_base_families
                .iter()
                .all(|id| full_prefix.contains(id)),
        "the explicit through-Halftrend debt list drifted outside the canonical prefix"
    );
    let full_prefix_len = full_prefix.len();
    let admitted_indicator_ids = full_prefix
        .into_iter()
        .filter(|id| !excluded_base_families.contains(id))
        .collect::<Vec<_>>();
    ensure!(
        admitted_indicator_ids.len() + excluded_base_families.len() == full_prefix_len,
        "reviewed fixture did not remove exactly the explicit 15-family debt"
    );
    let admitted_set = admitted_indicator_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let reviewed_historical_ids = MULTI_PERIOD_IDS
        .iter()
        .copied()
        .filter(|id| admitted_set.contains(id))
        .collect::<Vec<_>>();
    ensure!(
        reviewed_historical_ids == REVIEWED_HISTORICAL_IDS_THROUGH_HALFTREND_V3,
        "reviewed through-Halftrend historical intersection drifted: {reviewed_historical_ids:?}"
    );
    let admission = scope_classic_ta_device_fixture_admission_v3(
        rows,
        admitted_indicator_ids,
        "device_fixture_reviewed_routeable_subset",
    );

    let plan = crate::core::classic_cuda_plan::build_exact_classic_cuda_plan(
        rows,
        &admission.admitted_indicator_ids,
        &reviewed_historical_ids,
        &admission.extended_groups,
    )?;
    let resident_cuda_launches =
        crate::core::classic_cuda_plan::resolve_gpu_only_classic_plan(&plan)?;
    Ok(ClassicTaRunPlan {
        policy: IndicatorComputePolicy::GpuOnly,
        admission,
        resident_cuda_launches: Some(resident_cuda_launches),
    })
}

fn compute_classic_ta_device_fixture_cpu_oracle_v3(
    ohlcv: &Ohlcv,
    run_plan: &ClassicTaRunPlan,
) -> Result<Vec<FeatureColumnF64>> {
    let mut cpu_plan = run_plan.clone();
    cpu_plan.policy = IndicatorComputePolicy::CpuOnly;
    cpu_plan.resident_cuda_launches = None;
    compute_classic_ta_feature_columns_f64_with_run_plan(ohlcv, &cpu_plan)
}

fn reviewed_routeable_subset_fixture_v3() -> Ohlcv {
    let mut open = Vec::with_capacity(FIXTURE_ROWS_V3);
    let mut high = Vec::with_capacity(FIXTURE_ROWS_V3);
    let mut low = Vec::with_capacity(FIXTURE_ROWS_V3);
    let mut close = Vec::with_capacity(FIXTURE_ROWS_V3);
    let mut volume = Vec::with_capacity(FIXTURE_ROWS_V3);
    let mut timestamp = Vec::with_capacity(FIXTURE_ROWS_V3);
    for row in 0..FIXTURE_ROWS_V3 {
        let drift = row as f64 * 0.000_000_7;
        let wave = match row % 11 {
            0 => 0.000_041,
            1 => -0.000_027,
            2 => 0.000_013,
            3 => -0.000_036,
            4 => 0.000_022,
            5 => -0.000_009,
            6 => 0.000_033,
            7 => -0.000_019,
            8 => 0.000_006,
            9 => -0.000_031,
            _ => 0.000_017,
        };
        let row_open = 1.075 + drift;
        let row_close = row_open + wave;
        open.push(row_open);
        high.push(row_open.max(row_close) + 0.000_08 + (row % 7) as f64 * 0.000_001);
        low.push(row_open.min(row_close) - 0.000_07 - (row % 5) as f64 * 0.000_001);
        close.push(row_close);
        volume.push(900.0 + (row % 97) as f64 * 3.25 + row as f64 * 0.001);
        timestamp.push(1_704_067_200_000 + row as i64 * 300_000);
    }
    let final_row = FIXTURE_ROWS_V3 - 1;
    close[final_row] = f64::from_bits(close[final_row].to_bits() ^ 1);
    high[final_row] = high[final_row].max(close[final_row] + 0.000_001);
    low[final_row] = low[final_row].min(close[final_row] - 0.000_001);
    Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
}

fn exact_expected_columns_v3(
    recipe: &ResidentClassicTaRecipeV3,
    cpu_columns: Vec<FeatureColumnF64>,
) -> Result<Vec<ResidentClassicTaExpectedColumnV3>> {
    let mut by_name = HashMap::with_capacity(cpu_columns.len());
    for column in cpu_columns {
        let name = column.name.clone();
        ensure!(
            by_name.insert(name.clone(), column).is_none(),
            "CPU fixture emitted duplicate feature `{name}`"
        );
    }
    let mut ordered = (0..recipe.output_count())
        .map(|_| None)
        .collect::<Vec<Option<ResidentClassicTaExpectedColumnV3>>>();
    for launch in recipe.launches() {
        for route in launch.outputs() {
            let column = by_name.remove(route.feature_name()).with_context(|| {
                format!(
                    "CPU authority omitted routeable GPU feature `{}`",
                    route.feature_name()
                )
            })?;
            ensure!(
                column.values.len() == FIXTURE_ROWS_V3 && column.validity.len() == FIXTURE_ROWS_V3,
                "CPU feature `{}` has a noncanonical fixture extent",
                route.feature_name()
            );
            let destination = route.destination_column();
            let slot = ordered
                .get_mut(destination)
                .with_context(|| format!("GPU destination {destination} exceeds recipe width"))?;
            ensure!(
                slot.is_none(),
                "GPU destination {destination} was duplicated"
            );
            *slot = Some(ResidentClassicTaExpectedColumnV3 {
                feature_name: route.feature_name().to_owned(),
                expected_value_bits: column.values.into_iter().map(f64::to_bits).collect(),
                expected_validity_codes: column
                    .validity
                    .into_iter()
                    .map(FeatureCellValidity::code)
                    .collect(),
            });
        }
    }
    let excluded_historical_ids = MULTI_PERIOD_IDS
        .iter()
        .copied()
        .filter(|id| !REVIEWED_HISTORICAL_IDS_THROUGH_HALFTREND_V3.contains(id))
        .collect::<HashSet<_>>();
    let expected_excluded_historical_columns = excluded_historical_ids
        .iter()
        .map(|id| planned_output_count(id) * ALT_PERIODS.len())
        .sum::<usize>();
    ensure!(
        by_name.len() == expected_excluded_historical_columns,
        "CPU authority emitted {} out-of-recipe columns; exact excluded historical sweep requires {expected_excluded_historical_columns}",
        by_name.len()
    );
    for name in by_name.keys() {
        let indicator_id = classic_indicator_id_for_column(name).with_context(|| {
            format!("out-of-recipe CPU column `{name}` lost canonical identity")
        })?;
        ensure!(
            excluded_historical_ids.contains(indicator_id),
            "CPU authority emitted non-historical out-of-recipe column `{name}`"
        );
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(destination, column)| {
            column.with_context(|| format!("GPU destination {destination} was not populated"))
        })
        .collect()
}

fn expected_full_graph_gap_columns_v3() -> [&'static str; 45] {
    [
        "dec_osc",
        "decycler",
        "demand_index_signal",
        "donchian_channel_width",
        "dti",
        "dynamic_momentum_index",
        "ehlers_adaptive_cg_trigger",
        "ehlers_adaptive_cyber_cycle_trigger",
        "ehlers_detrending_filter",
        "ehlers_pma_trigger",
        "ehlers_simple_cycle_indicator_trigger",
        "fractal_dimension_index",
        "fvg_positioning_average",
        "gmma_oscillator",
        "goertzel_cycle_composite_wave",
        "stoch_7_d",
        "stoch_21_d",
        "stoch_50_d",
        "stoch_100_d",
        "stoch_200_d",
        "macd_7_signal",
        "macd_7_hist",
        "macd_21_signal",
        "macd_21_hist",
        "macd_50_signal",
        "macd_50_hist",
        "macd_100_signal",
        "macd_100_hist",
        "macd_200_signal",
        "macd_200_hist",
        "keltner_7_middle",
        "keltner_7_lower",
        "keltner_21_middle",
        "keltner_21_lower",
        "keltner_50_middle",
        "keltner_50_lower",
        "keltner_100_middle",
        "keltner_100_lower",
        "keltner_200_middle",
        "keltner_200_lower",
        "supertrend_7_changed",
        "supertrend_21_changed",
        "supertrend_50_changed",
        "supertrend_100_changed",
        "supertrend_200_changed",
    ]
}

fn expected_full_graph_gap_reason_v3(
    column_name: &str,
) -> crate::core::classic_cuda_plan::ClassicCudaGapReason {
    use crate::core::classic_cuda_plan::ClassicCudaGapReason;

    match column_name {
        "dec_osc" | "decycler" => ClassicCudaGapReason::CanonicalCpuOutputUnavailable,
        "dti" => ClassicCudaGapReason::NoWindowKernelConsumesAnchor,
        "ehlers_adaptive_cg_trigger"
        | "ehlers_adaptive_cyber_cycle_trigger"
        | "ehlers_pma_trigger"
        | "ehlers_simple_cycle_indicator_trigger" => {
            ClassicCudaGapReason::MissingNamedProductionDispatcher
        }
        "donchian_channel_width"
        | "dynamic_momentum_index"
        | "ehlers_detrending_filter"
        | "fractal_dimension_index"
        | "fvg_positioning_average"
        | "gmma_oscillator"
        | "goertzel_cycle_composite_wave" => ClassicCudaGapReason::MissingParameterContract,
        _ => ClassicCudaGapReason::MissingNamedOutputRoute,
    }
}

fn expected_full_graph_gap_detail_v3(
    indicator_id: &str,
    reason: crate::core::classic_cuda_plan::ClassicCudaGapReason,
) -> String {
    use crate::core::classic_cuda_plan::ClassicCudaGapReason;

    match reason {
        ClassicCudaGapReason::CanonicalCpuOutputUnavailable => {
            "UnsupportedCapability 'cpu_batch': registered but no cpu_batch dispatch arm".to_owned()
        }
        ClassicCudaGapReason::MissingNamedOutputRoute => {
            "the named output has no resident f64 route".to_owned()
        }
        ClassicCudaGapReason::MissingParameterContract => format!(
            "{indicator_id}: base CPU dispatch is unregistered, so its `period` default cannot be proven from the canonical registry"
        ),
        ClassicCudaGapReason::NoWindowKernelConsumesAnchor => {
            "CPU declares no window, but the registered f64 kernel consumes the supplied anchor; passing an invented value would change the formula".to_owned()
        }
        ClassicCudaGapReason::MissingNamedProductionDispatcher => {
            "a low-level resident named-output kernel exists, but the canonical typed executor has no source-stable dispatcher for it yet".to_owned()
        }
        other => format!("unexpected through-Halftrend debt reason {other}"),
    }
}

#[test]
fn full_through_halftrend_graph_retains_exact_45_fail_closed_contracts() -> Result<()> {
    // Gate139 was the first real RTX 3090 attempt. It stopped on this complete
    // manifest before the first CUDA context/launch; preserving that RED is
    // what prevents the reviewed device subset below from masquerading as the
    // still-incomplete production GpuOnly graph.
    let full_prefix = all_indicator_ids_through_halftrend_v3()?;
    let admission = scope_classic_ta_device_fixture_admission_v3(
        FIXTURE_ROWS_V3,
        full_prefix,
        "device_fixture_full_graph_debt_census",
    );
    let plan = crate::core::classic_cuda_plan::build_exact_classic_cuda_plan(
        FIXTURE_ROWS_V3,
        &admission.admitted_indicator_ids,
        &MULTI_PERIOD_IDS,
        &admission.extended_groups,
    )?;
    let gaps = crate::core::classic_cuda_plan::preflight_exact_classic_cuda_plan(&plan)
        .expect_err("the unresolved full graph unexpectedly became routeable");
    ensure!(
        gaps.len() == 45,
        "through-Halftrend/full-history debt changed from 45 to {} contracts",
        gaps.len()
    );
    for (gap, expected_column) in gaps.iter().zip(expected_full_graph_gap_columns_v3()) {
        let expected_reason = expected_full_graph_gap_reason_v3(expected_column);
        let expected_indicator_id = classic_indicator_id_for_column(expected_column)
            .with_context(|| format!("expected gap `{expected_column}` lost canonical identity"))?;
        ensure!(
            gap.column_name == expected_column
                && gap.indicator_id == expected_indicator_id
                && gap.reason == expected_reason
                && gap.detail
                    == expected_full_graph_gap_detail_v3(expected_indicator_id, expected_reason),
            "through-Halftrend gap drifted: expected `{expected_column}`/{expected_reason}, found `{}`/{}/{}",
            gap.column_name,
            gap.reason,
            gap.detail
        );
    }
    let refusal = crate::core::classic_cuda_plan::resolve_gpu_only_classic_plan(&plan)
        .expect_err("GpuOnly must not admit the unresolved full graph")
        .to_string();
    ensure!(
        refusal.contains("45 unrouteable admitted output contract(s)")
            && refusal.contains("No output was excluded and no CPU/f32 substitute is permitted"),
        "GpuOnly full-graph refusal lost its atomic fail-closed contract: {refusal}"
    );
    Ok(())
}

#[test]
fn resident_classic_ta_v3_reviewed_routeable_subset_through_halftrend_is_exact_and_leak_free()
-> Result<()> {
    ensure!(
        std::env::var_os("NEOETHOS_REQUIRE_GPU").is_some(),
        "required-card Classic TA fixture refuses to skip without NEOETHOS_REQUIRE_GPU"
    );
    let ohlcv = reviewed_routeable_subset_fixture_v3();
    let run_plan = prepare_classic_ta_device_fixture_plan_v3(FIXTURE_ROWS_V3)?;
    let admission = run_plan.admission_report();
    ensure!(
        admission.admitted_indicator_ids.last().copied() == Some("halftrend"),
        "reviewed routeable ALL-order subset through Halftrend drifted"
    );
    let cpu_columns = compute_classic_ta_device_fixture_cpu_oracle_v3(&ohlcv, &run_plan)?;
    let resident_plan = preflight_resident_classic_ta_v3(&run_plan, FIXTURE_ROWS_V3)?;
    let recipe = resident_plan.recipe;
    let natural_launch_widths = recipe
        .launches()
        .iter()
        .map(|launch| launch.outputs().len())
        .collect::<Vec<_>>();
    ensure!(
        natural_launch_widths
            .iter()
            .all(|width| (1..=64).contains(width)),
        "natural Classic TA launch exceeded the 64-output resident bound"
    );
    let natural_launch_count = natural_launch_widths.len();
    let output_count = recipe.output_count();
    let expected_columns = exact_expected_columns_v3(&recipe, cpu_columns)?;
    let receipt =
        run_resident_classic_ta_v3_device_fixture(ResidentClassicTaDeviceFixtureRequestV3 {
            recipe,
            open: ohlcv.open,
            high: ohlcv.high,
            low: ohlcv.low,
            close: ohlcv.close,
            volume: ohlcv
                .volume
                .context("Classic TA fixture lost canonical volume")?,
            timestamps: ohlcv
                .timestamp
                .context("Classic TA fixture lost canonical timestamps")?,
            expected_columns,
        })
        .map_err(anyhow::Error::from_boxed)?;

    ensure!(
        receipt.reviewed_routeable_output_count == output_count
            && receipt.natural_launch_count == natural_launch_count
            && receipt.natural_launch_widths == natural_launch_widths,
        "the actual opaque executor did not traverse the frozen reviewed routeable graph"
    );
    ensure!(
        receipt.synthetic_pack_widths == SYNTHETIC_PACK_WIDTHS_V3,
        "synthetic resident pack boundaries drifted"
    );
    ensure!(
        receipt.parent_upload_count == 1
            && receipt.parent_reupload_count == 0
            && receipt.second_context_count == 0
            && receipt.second_stream_count == 0,
        "fixture reuploaded the parent or opened a second CUDA authority"
    );
    ensure!(
        receipt.changed_final_feature_bit_observed
            && receipt.launched_all_nan_compute_failure_observed
            && receipt.canonical_placeholder_warmup_observed
            && receipt.output_infinity_refused,
        "required Classic TA edge semantics were not observed"
    );

    let output_count_u64 = u64::try_from(output_count)?;
    let launch_count_u64 = u64::try_from(natural_launch_count)?;
    let rows_u64 = u64::try_from(FIXTURE_ROWS_V3)?;
    let expected_value_d2h = rows_u64
        .checked_mul(output_count_u64 + 3)
        .and_then(|cells| cells.checked_mul(8))
        .context("fixture expected value D2H overflow")?;
    let expected_validity_d2h = rows_u64
        .checked_mul(output_count_u64 + 4)
        .context("fixture expected validity D2H overflow")?;
    let expected_control_plane_d2h = launch_count_u64
        .checked_add(4)
        .and_then(|count| count.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(COMPACT_VALIDITY_AND_ROOT_D2H_BYTES_V3))
        .context("fixture expected control-plane D2H overflow")?;
    let expected_total_d2h = expected_value_d2h
        .checked_add(expected_validity_d2h)
        .and_then(|bytes| bytes.checked_add(expected_control_plane_d2h))
        .context("fixture expected total D2H overflow")?;
    ensure!(
        receipt.value_d2h_bytes == expected_value_d2h
            && receipt.validity_d2h_bytes == expected_validity_d2h
            && receipt.control_plane_d2h_bytes == expected_control_plane_d2h
            && receipt.bounded_test_parity_d2h_bytes == expected_total_d2h,
        "bounded test parity D2H accounting drifted"
    );
    Ok(())
}
