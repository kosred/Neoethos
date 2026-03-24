use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    Discovery,
    Training,
    Bootstrap,
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
    pub highlights: Vec<(String, String)>,
    pub entries: Vec<String>,
    pub events: Vec<JobEvent>,
    pub summary: String,
    pub log_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobEventLevel {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobEvent {
    pub level: JobEventLevel,
    pub message: String,
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

pub fn push_recent_event(
    events: &[JobEvent],
    level: JobEventLevel,
    message: impl Into<String>,
) -> Vec<JobEvent> {
    let mut next = events.to_vec();
    next.push(JobEvent {
        level,
        message: message.into(),
    });
    if next.len() > 8 {
        next.drain(0..(next.len() - 8));
    }
    next
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
        snapshot
            .report
            .highlights
            .push(("best_strategy".to_string(), "alpha-7".to_string()));
        snapshot
            .report
            .entries
            .push("alpha-7 | sharpe=1.92 | win_rate=0.61".to_string());
        snapshot
            .report
            .events
            .push(JobEvent {
                level: JobEventLevel::Info,
                message: "loaded dataset for EURUSD".to_string(),
            });
        snapshot.report.summary = "7 accepted strategies".to_string();

        assert_eq!(snapshot.progress.percent, Some(0.42));
        assert_eq!(snapshot.progress.stage, "validation");
        assert_eq!(snapshot.report.counters, vec![("accepted".to_string(), 7)]);
        assert_eq!(
            snapshot.report.highlights,
            vec![("best_strategy".to_string(), "alpha-7".to_string())]
        );
        assert_eq!(
            snapshot.report.entries,
            vec!["alpha-7 | sharpe=1.92 | win_rate=0.61".to_string()]
        );
        assert_eq!(
            snapshot.report.events,
            vec![JobEvent {
                level: JobEventLevel::Info,
                message: "loaded dataset for EURUSD".to_string(),
            }]
        );
        assert_eq!(snapshot.report.summary, "7 accepted strategies");
    }

    #[test]
    fn push_recent_event_keeps_only_latest_eight_items() {
        let mut events = Vec::new();
        for idx in 1..=10 {
            events = push_recent_event(&events, JobEventLevel::Info, format!("event-{idx}"));
        }

        assert_eq!(events.len(), 8);
        assert_eq!(events.first().map(|event| event.message.as_str()), Some("event-3"));
        assert_eq!(events.last().map(|event| event.message.as_str()), Some("event-10"));
        assert!(events.iter().all(|event| event.level == JobEventLevel::Info));
    }

    #[test]
    fn cancellation_flag_starts_clear_and_can_be_requested() {
        let cancel = CancellationFlag::new();

        assert!(!cancel.is_requested());
        cancel.request();
        assert!(cancel.is_requested());
    }
}
