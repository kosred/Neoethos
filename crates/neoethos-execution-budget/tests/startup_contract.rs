use neoethos_execution_budget::{
    BudgetCap, CapacityDetection, CoordinationScope, ExecutionBudgetRequest, LogicalThreadCount,
    ParentCpuAssignmentError, StartupEvent, StartupRuntimeKind, StartupTrace, WorkerLimit,
    format_startup_diagnostics, install_process_budget, parse_parent_cpu_assignment,
};

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[test]
fn parent_assignment_parser_is_strict_and_accepts_both_cli_forms() {
    assert_eq!(
        parse_parent_cpu_assignment(&args(&["neoethos", "--cpu-threads", "7"]))
            .unwrap()
            .unwrap()
            .get(),
        7
    );
    assert_eq!(
        parse_parent_cpu_assignment(&args(&["neoethos", "--cpu-threads=5"]))
            .unwrap()
            .unwrap()
            .get(),
        5
    );
    assert_eq!(
        parse_parent_cpu_assignment(&args(&["neoethos", "--cpu-threads", "0"])),
        Err(ParentCpuAssignmentError::InvalidValue("0".to_string()))
    );
    assert_eq!(
        parse_parent_cpu_assignment(&args(&["neoethos", "--cpu-threads"])),
        Err(ParentCpuAssignmentError::MissingValue)
    );
    assert_eq!(
        parse_parent_cpu_assignment(&args(&[
            "neoethos",
            "--cpu-threads=2",
            "--cpu-threads",
            "3",
        ])),
        Err(ParentCpuAssignmentError::Duplicate)
    );
}

#[test]
fn trace_rejects_duplicate_or_backward_startup_events() {
    let mut trace = StartupTrace::default();
    trace.record(StartupEvent::ConfigurationLoaded).unwrap();
    trace.record(StartupEvent::ParentCpuCapParsed).unwrap();
    assert!(
        trace
            .record(StartupEvent::ConfigurationSeededOrLocated)
            .is_err()
    );
    assert!(trace.record(StartupEvent::ParentCpuCapParsed).is_err());
}

#[test]
fn diagnostics_report_the_reclamped_parent_budget_without_double_subtraction() {
    let parent = WorkerLimit::new(8).unwrap();
    let request = ExecutionBudgetRequest {
        host_logical_threads: None,
        detection: CapacityDetection::supplied(LogicalThreadCount::new(6).unwrap()),
        persistent_limit: None,
        legacy_persistent_limit: None,
        parent_limit: Some(BudgetCap::parent(parent)),
        coordination_scope: CoordinationScope::ManagedProcessTree,
    };
    let installed = install_process_budget(request).unwrap();
    let mut trace = StartupTrace::default();
    trace.record(StartupEvent::ParentCpuCapParsed).unwrap();
    trace.record(StartupEvent::CpuBudgetResolved).unwrap();
    trace.record(StartupEvent::CpuBudgetInstalled).unwrap();
    trace.record(StartupEvent::TokioRuntimeBuilt).unwrap();

    let line = format_startup_diagnostics(
        "fixture",
        installed,
        StartupRuntimeKind::Tokio,
        Some(installed.resolved().effective_worker_limit.get()),
        &trace,
    );
    assert!(line.starts_with("NEOETHOS_STARTUP_V1 "));
    assert!(line.contains("executable=fixture"));
    assert!(line.contains("effective_logical_threads=6"));
    assert!(line.contains("reserved_logical_threads=2"));
    assert!(line.contains("automatic_worker_limit=4"));
    assert!(line.contains("effective_worker_limit=4"));
    assert!(line.contains("capacity_source=supplied_for_resolution"));
    assert!(line.contains("coordination_scope=managed_process_tree"));
    assert!(line.contains("runtime_kind=tokio"));
    assert!(line.contains("runtime_worker_threads=4"));
    assert!(line.contains(
        "events=parent_cpu_cap_parsed,cpu_budget_resolved,cpu_budget_installed,tokio_runtime_built"
    ));
}
