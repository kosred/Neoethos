use std::path::PathBuf;

use neoethos_app::app_services::entrypoints::{
    HeadlessExecutionPipelineIntentV1, HeadlessExecutionStartErrorV1,
    run_headless_execution_pipeline_v1,
};
use neoethos_app::server::state::{AppApiState, install_config_path};
use neoethos_search::{ProcessExecutionKindV1, try_acquire_process_execution_lease_v1};

#[tokio::test(flavor = "current_thread")]
async fn busy_simultaneous_headless_pipeline_does_no_config_or_data_work() {
    install_config_path(PathBuf::from(
        "/definitely/missing/chunk4c-busy-loser/config.yaml",
    ));
    let state = AppApiState::new();
    let held = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::NativeResearch)
        .expect("test owns the process-global execution lease");
    let intent = HeadlessExecutionPipelineIntentV1::checked_new(
        "EURUSD".to_owned(),
        "M5".to_owned(),
        true,
        true,
    )
    .expect("valid lightweight headless intent");

    let error = match run_headless_execution_pipeline_v1(state, intent) {
        Ok(_) => panic!("Busy loser must not start a worker"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        HeadlessExecutionStartErrorV1::Busy {
            requested: "Discovery".to_owned(),
            active: "NativeResearch".to_owned(),
        }
    );

    drop(held);
    let released = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Training)
        .expect("dropping the held lease releases exact process authority");
    drop(released);
}
