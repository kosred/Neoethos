use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    Discovery,
    Training,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Degraded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(u64);

impl JobId {
    pub fn next() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct JobProgress {
    pub percent: Option<f32>,
    pub stage: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JobReport {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub counters: Vec<(String, u64)>,
    pub summary: String,
    pub log_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JobSnapshot {
    pub id: JobId,
    pub kind: JobKind,
    pub state: JobState,
    pub progress: JobProgress,
    pub report: JobReport,
}

impl JobSnapshot {
    pub fn new(kind: JobKind) -> Self {
        Self {
            id: JobId::next(),
            kind,
            state: JobState::Queued,
            progress: JobProgress::default(),
            report: JobReport::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CancellationFlag {
    requested: Arc<AtomicBool>,
}

impl CancellationFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request(&self) {
        self.requested.store(true, Ordering::SeqCst);
    }

    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_snapshot_starts_queued_without_fake_progress() {
        let snapshot = JobSnapshot::new(JobKind::Discovery);

        assert_eq!(snapshot.kind, JobKind::Discovery);
        assert_eq!(snapshot.state, JobState::Queued);
        assert_eq!(snapshot.progress.percent, None);
        assert!(snapshot.report.errors.is_empty());
        assert!(snapshot.report.warnings.is_empty());
    }

    #[test]
    fn job_snapshot_can_transition_from_running_to_cancelled() {
        let mut snapshot = JobSnapshot::new(JobKind::Training);

        snapshot.state = JobState::Running;
        snapshot.state = JobState::Cancelled;

        assert_eq!(snapshot.state, JobState::Cancelled);
    }

    #[test]
    fn job_snapshot_can_store_progress_and_report_updates() {
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);

        snapshot.progress = JobProgress {
            percent: Some(0.42),
            stage: "validation".to_string(),
            message: "evaluating accepted strategies".to_string(),
        };
        snapshot.report.counters.push(("accepted".to_string(), 7));
        snapshot.report.summary = "7 accepted strategies".to_string();

        assert_eq!(snapshot.progress.percent, Some(0.42));
        assert_eq!(snapshot.progress.stage, "validation");
        assert_eq!(snapshot.report.counters, vec![("accepted".to_string(), 7)]);
        assert_eq!(snapshot.report.summary, "7 accepted strategies");
    }

    #[test]
    fn cancellation_flag_starts_clear_and_can_be_requested() {
        let cancel = CancellationFlag::new();

        assert!(!cancel.is_requested());
        cancel.request();
        assert!(cancel.is_requested());
    }
}
