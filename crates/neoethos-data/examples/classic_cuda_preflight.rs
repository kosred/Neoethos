use anyhow::{Context, Result, ensure};
use neoethos_data::core::hpc_ta::{IndicatorComputePolicy, prepare_classic_ta_run_plan};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    ensure!(
        args.len() == 2,
        "usage: classic_cuda_preflight <widest-direct-timeframe-row-count>"
    );
    let rows = args[1]
        .parse::<usize>()
        .with_context(|| format!("invalid row count {:?}", args[1]))?;
    ensure!(rows > 0, "row count must be positive");

    let plan = prepare_classic_ta_run_plan(rows, IndicatorComputePolicy::GpuOnly)?;
    let report = plan.admission_report();
    println!("classic_cuda_preflight_status=routeable");
    println!("rows={}", report.budget_rows);
    println!("available_bytes={}", report.available_bytes_at_admission);
    println!("max_columns={}", report.max_columns);
    println!(
        "admitted_base_ids={}",
        report.admitted_indicator_ids.join(",")
    );
    println!(
        "extended_ids={}",
        report.extended_admitted_indicator_ids.join(",")
    );
    Ok(())
}
