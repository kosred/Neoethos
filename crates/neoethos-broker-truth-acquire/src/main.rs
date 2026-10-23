//! One-shot fail-closed broker-truth acquisition entrypoint.
//!
//! Parses the reviewed argument set (`BrokerTruthAcquisitionArgsV1`),
//! prepares the exact content-addressed inputs, and executes the prepared
//! acquisition with the production cancellation token. It never places an
//! order and never mints a permit: the outcome is evidence-only.

use std::process::ExitCode;

use neoethos_broker_history::ProductionBrokerTruthCancellationV2;
use neoethos_broker_truth_acquire::{
    BrokerTruthAcquisitionArgsV1, execute_prepared_acquisition_v1, prepare_acquisition_v1,
};

fn main() -> ExitCode {
    let args = match BrokerTruthAcquisitionArgsV1::try_parse_from(std::env::args_os().skip(1)) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("broker-truth-acquire: {error}");
            return ExitCode::FAILURE;
        }
    };
    let prepared = match prepare_acquisition_v1(args) {
        Ok(prepared) => prepared,
        Err(error) => {
            eprintln!("broker-truth-acquire: preparation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    match execute_prepared_acquisition_v1(
        prepared,
        &ProductionBrokerTruthCancellationV2::new(),
    ) {
        Ok(outcome) => {
            println!("authority_receipt={:?}", outcome.authority_receipt());
            println!("broker_truth_receipt={:?}", outcome.broker_truth_receipt());
            println!("link_receipt={:?}", outcome.link_receipt());
            println!("semantic_status={:?}", outcome.semantic_status());
            println!("promotion_eligibility={:?}", outcome.promotion_eligibility());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("broker-truth-acquire: execution failed: {error}");
            ExitCode::FAILURE
        }
    }
}
