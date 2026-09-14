use anyhow::Result;
use clap::Parser;
use neoethos_broker_history::tick_archive::{TickArchiveCli, execute};
use neoethos_execution_budget::{
    detected_request_with_parent, install_process_budget, parse_parent_cpu_assignment,
};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let cli = TickArchiveCli::parse_from(&args);
    neoethos_data::initialize_source_seal_before_runtime()?;
    let budget = install_process_budget(detected_request_with_parent(
        parse_parent_cpu_assignment(&args)?,
    ))?;
    let summary = execute(cli, budget)?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}
