const DATA_CONTROL: &str = include_str!("../src/server/data_control.rs");
const EXECUTION_ADMISSION: &str = include_str!("../src/app_services/execution_admission.rs");
const HISTORICAL_ADMISSION: &str =
    include_str!("../src/app_services/ctrader_historical_admission.rs");

fn broker_fetch_handler() -> &'static str {
    let start = DATA_CONTROL
        .find("pub async fn fetch(")
        .expect("POST /data/fetch handler exists");
    let end = DATA_CONTROL[start..]
        .find("// ─── POST /data/import")
        .map(|offset| start + offset)
        .expect("POST /data/import follows the fetch handler");
    &DATA_CONTROL[start..end]
}

#[test]
fn broker_fetch_registers_the_shared_run_before_cpu_queue_or_broker_work() {
    let handler = broker_fetch_handler();
    let registration = handler
        .find("begin_process_historical_capture()")
        .expect("shared process fetch registration");
    let execution_state = handler
        .find("execution_state()")
        .expect("broker fetch must fail closed without process execution state");
    let admission = handler
        .find(".submit(broker_fetch_cpu_demand())")
        .expect("broker fetch must submit a cancellable typed CPU request");
    let wait = handler
        .find("pending_admission.wait()")
        .expect("queued admission wait");
    let spawn = handler
        .find("tokio::task::spawn_blocking")
        .expect("broker fetch moves synchronous work off the async runtime");
    let execute = handler
        .find("admitted.execute")
        .expect("the admitted lease must be transferred into blocking work");
    let settings = handler
        .find("Settings::from_yaml")
        .expect("settings are read only after admission");
    let broker = handler
        .find("download_history_blocking(")
        .expect("shared broker download adapter remains connected");

    assert!(
        registration < execution_state
            && execution_state < admission
            && admission < wait
            && wait < spawn
            && spawn < execute
            && execute < settings
            && settings < broker
    );
    assert!(!handler[..admission].contains("spawn_blocking"));
    assert!(handler.contains("tokio::select!"));
    assert!(handler.contains("active_fetch.is_cancelled()"));
    assert!(handler.contains("HistoricalFetchStartFailure::AlreadyActive"));
    assert!(!handler.contains("begin_process_historical_fetch_queued"));
    assert!(!handler.contains("active_fetch.execute_if_not_cancelled"));
}

#[test]
fn broker_fetch_current_demand_is_exactly_one_worker() {
    let demand_start = DATA_CONTROL
        .find("fn broker_fetch_cpu_demand()")
        .expect("typed broker fetch demand hook exists");
    let demand = &DATA_CONTROL[demand_start..];
    let demand_end = demand
        .find("\n}")
        .expect("typed broker fetch demand hook ends");
    let demand = &demand[..=demand_end + 1];

    assert!(demand.contains("WorkerLimit::new(1)"));
    assert!(demand.contains("CpuPermitRequest::local"));
    assert!(!demand.contains("installed_limit"));
}

#[test]
fn queued_cancel_and_conflict_have_behavioral_raii_tests_and_live_permits_return() {
    let handler = broker_fetch_handler();
    assert!(handler.contains("let active_fetch = match begin_process_historical_capture()"));
    assert!(handler.contains("let pending_admission = match execution"));
    assert!(handler.contains("let admitted = loop"));
    assert!(handler.contains("admitted.execute"));
    assert!(!handler.contains("mem::forget"));

    assert!(
        HISTORICAL_ADMISSION
            .contains("fn active_queued_fetch_rejects_a_second_caller_before_cpu_submission()")
    );
    assert!(
        HISTORICAL_ADMISSION
            .contains("fn cancellation_while_cpu_queued_drops_pending_and_never_executes_work()")
    );

    let pending_drop = EXECUTION_ADMISSION
        .find("impl Drop for PendingAdmission")
        .expect("queued CPU admission has cancellation-on-drop");
    let admitted_drop = EXECUTION_ADMISSION
        .find("impl Drop for AdmittedCpuLease")
        .expect("granted CPU admission has lease-return-on-drop");
    assert!(EXECUTION_ADMISSION[pending_drop..].contains("CoordinatorCommand::Cancel"));
    assert!(EXECUTION_ADMISSION[admitted_drop..].contains("drop(self.lease.take())"));
    assert!(EXECUTION_ADMISSION[admitted_drop..].contains("CoordinatorCommand::LeaseReturned"));
}
