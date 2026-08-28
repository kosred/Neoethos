use anyhow::Result;
use neoethos_broker_history::bulk_cli::{
    CanonicalTrendbarBulkCli, execute_canonical_trendbar_bulk_v1,
    render_canonical_trendbar_bulk_stdout_v1,
};
use neoethos_execution_budget::{
    detected_request_with_parent, install_process_budget, parse_parent_cpu_assignment,
};
use std::io::Write;
use std::process::ExitCode;

fn run() -> Result<Vec<u8>> {
    neoethos_data::initialize_source_seal_before_runtime()?;
    let args = std::env::args().collect::<Vec<_>>();
    let parent = parse_parent_cpu_assignment(&args)?;
    let budget = install_process_budget(detected_request_with_parent(parent))?;
    let prepared = CanonicalTrendbarBulkCli::try_parse_from(&args)?.prepare()?;
    let outcome = execute_canonical_trendbar_bulk_v1(prepared, budget)?;
    render_canonical_trendbar_bulk_stdout_v1(&outcome)
}

fn main() -> ExitCode {
    match run() {
        Ok(bytes) => match std::io::stdout().write_all(&bytes) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(std::io::stderr(), "bulk stdout write failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            let _ = writeln!(std::io::stderr(), "{error:#}");
            ExitCode::FAILURE
        }
    }
}
