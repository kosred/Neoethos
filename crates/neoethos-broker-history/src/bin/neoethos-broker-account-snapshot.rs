use anyhow::Result;
use clap::Parser;
use neoethos_broker_history::account_snapshot_cli::{AccountSnapshotCli, capture};

fn main() -> Result<()> {
    let cli = AccountSnapshotCli::parse();
    neoethos_data::initialize_source_seal_before_runtime()?;
    println!("{}", serde_json::to_string(&capture(cli)?)?);
    Ok(())
}
