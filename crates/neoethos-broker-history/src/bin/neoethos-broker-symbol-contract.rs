use anyhow::Result;
use neoethos_broker_history::symbol_contract_cli::{
    ExactBrokerSymbolContractCaptureCliV1, capture_exact_production_broker_symbol_contract_v1,
    render_exact_broker_symbol_contract_receipt_v1,
};
use std::io::Write;

fn main() -> Result<()> {
    neoethos_data::initialize_source_seal_before_runtime()?;
    let args = std::env::args().collect::<Vec<_>>();
    let prepared = ExactBrokerSymbolContractCaptureCliV1::try_parse_from(&args)?.prepare()?;
    let receipt = capture_exact_production_broker_symbol_contract_v1(&prepared)?;
    std::io::stdout().write_all(&render_exact_broker_symbol_contract_receipt_v1(&receipt)?)?;
    Ok(())
}
