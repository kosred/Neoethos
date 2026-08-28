use anyhow::Result;

fn main() -> Result<()> {
    // SourceSeal signal ownership and the immutable process CPU authority must
    // exist before feature code can initialize a worker pool.
    neoethos_data::initialize_source_seal_before_runtime()?;
    let args = std::env::args().collect::<Vec<_>>();
    neoethos_search::historical_search_cli::install_historical_search_process_budget(&args)?;
    neoethos_search::historical_search_cli::run(&args[1..])
}
