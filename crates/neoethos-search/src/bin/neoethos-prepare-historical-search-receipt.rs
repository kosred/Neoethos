use anyhow::Result;

fn main() -> Result<()> {
    // SourceSeal ownership and the immutable process CPU authority must exist
    // before selected Vortex input or feature code can initialize a runtime.
    neoethos_data::initialize_source_seal_before_runtime()?;
    let args = std::env::args().collect::<Vec<_>>();
    neoethos_search::historical_search_cli::install_historical_search_process_budget(&args)?;
    neoethos_search::historical_search_receipt_prep::run(&args[1..])
}
