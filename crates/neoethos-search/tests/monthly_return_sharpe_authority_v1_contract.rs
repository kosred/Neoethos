use std::fs;
use std::path::{Path, PathBuf};

const MONTHS_PER_YEAR_SQRT_V1: f64 = 3.4641;
const INVALID_MONTHLY_RETURN_SHARPE_V1: f64 = f64::NEG_INFINITY;

fn search_manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_utf8(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn sample_mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

fn canonical_completed_month_sharpe(
    monthly_pnl_account_currency: &[f64],
    month_start_equity_account_currency: &[f64],
) -> Result<f64, &'static str> {
    if monthly_pnl_account_currency.len() != month_start_equity_account_currency.len() {
        return Err("shape mismatch");
    }
    let mut returns = Vec::with_capacity(monthly_pnl_account_currency.len());
    for (&pnl, &start_equity) in monthly_pnl_account_currency
        .iter()
        .zip(month_start_equity_account_currency)
    {
        if !pnl.is_finite() || !start_equity.is_finite() || start_equity <= 0.0 {
            return Err("invalid completed-month return input");
        }
        let period_return = pnl / start_equity;
        if !period_return.is_finite() {
            return Err("invalid completed-month return");
        }
        returns.push(period_return);
    }
    let (mean, stddev) = sample_mean_std(&returns);
    Ok(if stddev > 0.0 {
        (mean / stddev) * MONTHS_PER_YEAR_SQRT_V1
    } else {
        0.0
    })
}

fn canonical_completed_month_sharpe_or_reject(
    monthly_pnl_account_currency: &[f64],
    month_start_equity_account_currency: &[f64],
) -> f64 {
    canonical_completed_month_sharpe(
        monthly_pnl_account_currency,
        month_start_equity_account_currency,
    )
    .unwrap_or(INVALID_MONTHLY_RETURN_SHARPE_V1)
}

#[test]
fn oracle_proves_equal_money_pnls_are_not_equal_period_returns() {
    let sharpe = canonical_completed_month_sharpe(&[1_000.0, 1_000.0], &[100_000.0, 200_000.0])
        .expect("valid completed-month fixture");
    assert!(
        sharpe > 0.0,
        "money-PnL Sharpe would incorrectly be zero for this fixture"
    );
}

#[test]
fn oracle_refuses_zero_or_nonfinite_month_start_equity() {
    assert!(canonical_completed_month_sharpe(&[100.0], &[0.0]).is_err());
    assert!(canonical_completed_month_sharpe(&[100.0], &[f64::NAN]).is_err());
    assert!(canonical_completed_month_sharpe(&[f64::INFINITY], &[10_000.0]).is_err());
}

#[test]
fn invalid_period_return_inputs_reach_the_existing_fitness_rejection() {
    assert_eq!(
        canonical_completed_month_sharpe_or_reject(&[100.0], &[0.0]),
        f64::NEG_INFINITY
    );

    let scoring = read_utf8(&search_manifest_dir().join("src/scoring/named.rs"));
    assert!(
        scoring.matches("if !sharpe.is_finite()").count() >= 2
            && scoring.contains("return f64::NEG_INFINITY;"),
        "both strict and growth fitness must keep non-finite Sharpe as a hard rejection"
    );
    let evolution = read_utf8(&search_manifest_dir().join("src/genetic/evolution_math.rs"));
    assert!(
        evolution.contains("crate::scoring::ga_fitness_growth(m)")
            && evolution.contains("crate::scoring::ga_fitness(m)"),
        "population metrics must flow directly to guarded fitness without a zero scrub"
    );
}

#[test]
fn cpu_backtest_uses_completed_month_returns_for_sharpe_only() {
    let source = read_utf8(&search_manifest_dir().join("src/eval.rs"));
    assert!(
        source.contains("completed_month_return_sharpe_v1(")
            && source.contains("&monthly_pnls[..=limit]")
            && source.contains("&month_start_equities[..=limit]"),
        "CPU Sharpe must consume completed-month PnL and that month's start equity"
    );
    assert!(
        source.contains("let (avg_m, std_m) = mean_std(&monthly_pnls[..=limit]);"),
        "custom consistency must remain on its existing raw-money semantics in this batch"
    );
    assert!(
        source.contains("const INVALID_MONTHLY_RETURN_SHARPE_V1: f64 = f64::NEG_INFINITY;")
            && source.contains("sanitize_sharpe_v1(sharpe)"),
        "CPU invalid month-return inputs must survive metric assembly as NEG_INFINITY"
    );
    assert!(
        !source.contains("sanitize(sharpe)"),
        "the generic non-finite scrub must never convert invalid Sharpe back to zero"
    );
    assert!(
        !source.contains("month_returns.extend_from_slice(&monthly_pnls[..=limit]);"),
        "raw account-money PnLs must not be called returns or feed Sharpe"
    );
}

#[test]
fn cubecl_host_assembly_uses_completed_month_returns_for_both_readback_paths() {
    let source = read_utf8(&search_manifest_dir().join("src/cubecl_eval.rs"));
    assert_eq!(
        source
            .matches("completed_month_return_sharpe_v1(monthly_pnls, month_start_equities)")
            .count(),
        2,
        "both CubeCL compact-metric assembly paths must normalize PnL by month-start equity"
    );
    assert!(
        source.contains("let (average_month_pnl, month_pnl_stddev) = mean_std(monthly_pnls);"),
        "CubeCL custom consistency must retain raw-money semantics"
    );
    assert!(
        source.contains("let (avg_month_pnl, std_month_pnl) = mean_std(monthly_pnls);"),
        "the second CubeCL assembly path must also retain raw-money consistency semantics"
    );
    assert!(
        source.contains("completed_month_return_sharpe_v1")
            && source.contains("use crate::eval::")
            && !source.contains("sanitize(sharpe)"),
        "CubeCL must use the shared fail-closed return authority without a local zero scrub"
    );
}

#[test]
fn native_cuda_b_normalizes_each_completed_month_before_sharpe() {
    let source = read_utf8(
        &search_manifest_dir().join("../neoethos-gpu-cuda/native/prototype_b_population.cu"),
    );
    assert!(
        source.contains("const double period_return = monthly[index] / start_equity;"),
        "native CUDA B must divide each completed-month PnL by its own start equity"
    );
    assert!(
        source.contains("monthly_return_mean") && source.contains("monthly_return_std"),
        "native CUDA B needs distinct return statistics for Sharpe"
    );
    assert!(
        source.contains("monthly_mean / monthly_std") && source.contains("consistency"),
        "native CUDA B must preserve the existing raw-money consistency calculation"
    );
    assert!(
        source.contains("invalid_monthly_return_sharpe_v1()")
            && source.contains("row.values[1] = sharpe;"),
        "native CUDA B must preserve the invalid sentinel instead of sanitizing it"
    );
    assert!(
        !source.contains("row.values[1] = sanitize(sharpe);"),
        "native CUDA B must never convert invalid Sharpe to zero"
    );
}

#[test]
fn prototype_c_normalizes_each_completed_month_before_sharpe() {
    let source =
        read_utf8(&search_manifest_dir().join("src/gpu_native/prototype_c_engine/device.rs"));
    assert!(
        source.contains("let period_return = month_workspace[month_base + index] / start_equity;"),
        "Prototype C must divide each completed-month PnL by its own start equity"
    );
    assert!(
        source.contains("monthly_return_mean") && source.contains("monthly_return_std"),
        "Prototype C needs distinct return statistics for Sharpe"
    );
    assert!(
        source.contains("monthly_mean.read() / monthly_std.read()"),
        "Prototype C must preserve the existing raw-money consistency calculation"
    );
    assert!(
        source.contains("RuntimeCell::<f64>::new(f64::NEG_INFINITY)")
            && source.contains("metrics_out[metric_base + 1] = sharpe.read();"),
        "Prototype C must carry the invalid Sharpe sentinel to its resident metric row"
    );
}
