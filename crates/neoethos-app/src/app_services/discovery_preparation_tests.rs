use super::*;

#[test]
fn preparation_failure_retains_the_actual_io_cause_and_requested_scope() {
    let mut before = JobSnapshot::new(JobKind::Discovery);
    before.report.counters.push(("population".into(), 200));
    before
        .report
        .highlights
        .push(("base_tf".into(), "M5".into()));
    let error = anyhow::anyhow!("disk guard: reserved free space would be consumed")
        .context("encode Vortex chunk")
        .context("write features.vortex");
    let after = failed_snapshot_from(before, error);
    assert_eq!(after.state, JobState::Failed);
    assert!(
        after
            .report
            .summary
            .contains("write features.vortex: encode Vortex chunk: disk guard:")
    );
    assert!(after.report.counters.contains(&("population".into(), 200)));
    assert!(
        after
            .report
            .highlights
            .contains(&("base_tf".into(), "M5".into()))
    );
}

#[test]
fn preparation_progress_is_local_work_not_search_percent_or_candidates() {
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    snapshot
        .report
        .counters
        .push(("target_candidates".into(), 200_000));
    apply_feature_preparation_progress(
        &mut snapshot,
        &FeatureBuildProgress {
            timeframe: "M5".into(),
            stage: "vortex_write_rows",
            item: "1000 columns".into(),
            completed: 8192,
            total: 791263,
        },
    );
    assert_eq!(snapshot.progress.percent, None);
    assert_eq!(snapshot.progress.stage, "preparing_vortex_write_rows");
    assert!(snapshot.progress.message.contains("8192/791263"));
    assert!(
        snapshot
            .report
            .counters
            .contains(&("target_candidates".into(), 200_000))
    );
    assert!(
        !snapshot
            .report
            .counters
            .iter()
            .any(|(key, _)| key == "candidates")
    );
}

#[test]
fn cancelled_preparation_retains_diagnostics_but_does_not_invent_success() {
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    snapshot
        .report
        .counters
        .push(("preparation_stage_completed".into(), 12));
    let cancelled = cancelled_snapshot_from(snapshot, "operator stopped preparation");
    assert_eq!(cancelled.state, JobState::Cancelled);
    assert!(cancelled.report.errors.is_empty());
    assert_eq!(
        cancelled.report.counters,
        [("preparation_stage_completed".into(), 12)]
    );
}
