//! One-shot fail-closed broker-truth acquisition entrypoint.
//!
//! Parses the reviewed argument set (BrokerTruthAcquisitionArgsV1),
//! prepares the exact content-addressed inputs, and executes the prepared
//! acquisition with the production cancellation token. It never places an
//! order and never mints a permit: the outcome is evidence-only.
//!
//! With --generate-reviewed-inputs <out-dir> it instead emits the
//! agent-reviewed input bundle (capture plan, review record, protocol
//! evidence, trust root) from real canonical input bytes. The agent acts as
//! the reviewer; no permit or capability is produced in either mode.

use std::process::ExitCode;

use neoethos_broker_history::ProductionBrokerTruthCancellationV2;
use neoethos_broker_truth_acquire::{
    AGENT_REVIEWER_IDENTITY_V1, BrokerTruthAcquisitionArgsV1, ReviewedInputGenerationArgsV1,
    execute_prepared_acquisition_v1, generate_reviewed_inputs_v1, prepare_acquisition_v1,
};

fn run_reviewed_input_generation_v1() -> ExitCode {
    let args = match ReviewedInputGenerationArgsV1::try_parse_from(std::env::args_os()) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("broker-truth-acquire: {error}");
            return ExitCode::FAILURE;
        }
    };
    match generate_reviewed_inputs_v1(args) {
        Ok(generated) => {
            println!("reviewed_input_generation=ok");
            println!("reviewer={AGENT_REVIEWER_IDENTITY_V1}");
            for (label, artifact) in [
                ("capture_plan", generated.capture_plan()),
                ("review_record", generated.review_record()),
                ("protocol_evidence", generated.protocol_evidence()),
                ("trust_root", generated.trust_root()),
            ] {
                println!(
                    "{label}={} {label}_sha256={}",
                    artifact.path().display(),
                    artifact.sha256()
                );
            }
            if let Some(artifact) = generated.canonical_root_verification() {
                println!(
                    "canonical_root_verification={} canonical_root_verification_sha256={}",
                    artifact.path().display(),
                    artifact.sha256()
                );
            }
            if let Some(artifact) = generated.canonical_scope_window_binding() {
                println!(
                    "canonical_scope_window_binding={} canonical_scope_window_binding_sha256={}",
                    artifact.path().display(),
                    artifact.sha256()
                );
            }
            println!("permit_issued=false");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("broker-truth-acquire: reviewed-input generation failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments
        .iter()
        .any(|argument| argument == "--generate-reviewed-inputs")
    {
        return run_reviewed_input_generation_v1();
    }
    let args = match BrokerTruthAcquisitionArgsV1::try_parse_from(std::env::args_os()) {
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
    match execute_prepared_acquisition_v1(prepared, &ProductionBrokerTruthCancellationV2::new()) {
        Ok(outcome) => {
            println!("authority_receipt={:?}", outcome.authority_receipt());
            println!("broker_truth_receipt={:?}", outcome.broker_truth_receipt());
            println!("link_receipt={:?}", outcome.link_receipt());
            println!("semantic_status={:?}", outcome.semantic_status());
            println!(
                "promotion_eligibility={:?}",
                outcome.promotion_eligibility()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("broker-truth-acquire: execution failed: {error}");
            ExitCode::FAILURE
        }
    }
}
