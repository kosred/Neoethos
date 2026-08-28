use anyhow::Result;
use neoethos_broker_history::cli::{HistoricalFetchCli, execute, render_receipt_stdout};
use neoethos_execution_budget::{
    detected_request_with_parent, install_process_budget, parse_parent_cpu_assignment,
};
use std::io::Write;

fn main() -> Result<()> {
    neoethos_data::initialize_source_seal_before_runtime()?;
    let args = std::env::args().collect::<Vec<_>>();
    let parent = parse_parent_cpu_assignment(&args)?;
    let budget = install_process_budget(detected_request_with_parent(parent))?;
    let cli = HistoricalFetchCli::try_parse_from(&args)?;
    let receipt = execute(cli, budget)?;
    let bytes = render_receipt_stdout(&receipt)?;
    std::io::stdout().write_all(&bytes)?;
    Ok(())
}
