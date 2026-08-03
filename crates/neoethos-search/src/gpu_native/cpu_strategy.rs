//! Per-work-unit audit boundary for CPU strategy computation.
//!
//! Every candidate-dependent CPU evaluator must enter through [`run`].  This
//! makes strict GPU execution fail before the CPU closure executes and gives
//! each work unit independent attempted/executed counters.

use crate::backend::{EvaluationBackend, FallbackPolicy};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuStrategyCategory {
    PopulationEvaluation = 0,
    SignalSynthesis = 1,
    CandidateBacktest = 2,
    ValidationSimulation = 3,
    RiskDiagnostics = 4,
    CorrelationRanking = 5,
    RobustnessReplay = 6,
}

impl CpuStrategyCategory {
    pub const ALL: [Self; 7] = [
        Self::PopulationEvaluation,
        Self::SignalSynthesis,
        Self::CandidateBacktest,
        Self::ValidationSimulation,
        Self::RiskDiagnostics,
        Self::CorrelationRanking,
        Self::RobustnessReplay,
    ];

    const COUNT: usize = Self::ALL.len();

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuStrategyExecutionMode {
    /// Production execution. `ForbidCpu` is enforced.
    Production,
    /// Explicit CPU oracle used by parity tests and validation tooling. This
    /// mode must never be mixed into clean GPU timing runs.
    ValidationReference,
}

#[derive(Debug, Clone)]
pub struct CpuStrategyAuditContext {
    inner: Arc<AuditInner>,
}

#[derive(Debug)]
struct AuditInner {
    work_unit_id: u64,
    mode: CpuStrategyExecutionMode,
    attempted: [AtomicU64; CpuStrategyCategory::COUNT],
    executed: [AtomicU64; CpuStrategyCategory::COUNT],
}

impl CpuStrategyAuditContext {
    pub fn new(work_unit_id: u64, mode: CpuStrategyExecutionMode) -> Self {
        Self {
            inner: Arc::new(AuditInner {
                work_unit_id,
                mode,
                attempted: std::array::from_fn(|_| AtomicU64::new(0)),
                executed: std::array::from_fn(|_| AtomicU64::new(0)),
            }),
        }
    }

    pub fn production(work_unit_id: u64) -> Self {
        Self::new(work_unit_id, CpuStrategyExecutionMode::Production)
    }

    pub fn validation_reference(work_unit_id: u64) -> Self {
        Self::new(work_unit_id, CpuStrategyExecutionMode::ValidationReference)
    }

    pub fn work_unit_id(&self) -> u64 {
        self.inner.work_unit_id
    }

    pub fn mode(&self) -> CpuStrategyExecutionMode {
        self.inner.mode
    }

    pub fn snapshot(&self) -> CpuStrategyAuditSnapshot {
        CpuStrategyAuditSnapshot {
            work_unit_id: self.inner.work_unit_id,
            mode: self.inner.mode,
            attempted: std::array::from_fn(|index| {
                self.inner.attempted[index].load(Ordering::Relaxed)
            }),
            executed: std::array::from_fn(|index| {
                self.inner.executed[index].load(Ordering::Relaxed)
            }),
        }
    }

    fn record_attempt(&self, category: CpuStrategyCategory) {
        self.inner.attempted[category.index()].fetch_add(1, Ordering::Relaxed);
    }

    fn record_execution(&self, category: CpuStrategyCategory) {
        self.inner.executed[category.index()].fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuStrategyAuditSnapshot {
    pub work_unit_id: u64,
    pub mode: CpuStrategyExecutionMode,
    attempted: [u64; CpuStrategyCategory::COUNT],
    executed: [u64; CpuStrategyCategory::COUNT],
}

impl CpuStrategyAuditSnapshot {
    pub fn attempted(&self, category: CpuStrategyCategory) -> u64 {
        self.attempted[category.index()]
    }

    pub fn executed(&self, category: CpuStrategyCategory) -> u64 {
        self.executed[category.index()]
    }

    pub fn total_attempted(&self) -> u64 {
        self.attempted.iter().sum()
    }

    pub fn total_executed(&self) -> u64 {
        self.executed.iter().sum()
    }

    pub fn assert_zero_executed(&self) -> Result<(), CpuStrategyAuditError> {
        if self.total_executed() == 0 {
            Ok(())
        } else {
            Err(CpuStrategyAuditError::InvariantViolation {
                work_unit_id: self.work_unit_id,
                executed: self.total_executed(),
            })
        }
    }
}

pub fn run<T, F>(
    backend: EvaluationBackend,
    audit: &CpuStrategyAuditContext,
    category: CpuStrategyCategory,
    call_site: &'static str,
    operation: F,
) -> Result<T, CpuStrategyAuditError>
where
    F: FnOnce() -> T,
{
    audit.record_attempt(category);

    let strict_production = audit.mode() == CpuStrategyExecutionMode::Production
        && backend.fallback == FallbackPolicy::ForbidCpu;
    if strict_production {
        return Err(CpuStrategyAuditError::CpuForbidden {
            work_unit_id: audit.work_unit_id(),
            category,
            call_site,
        });
    }

    audit.record_execution(category);
    Ok(operation())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuStrategyAuditError {
    CpuForbidden {
        work_unit_id: u64,
        category: CpuStrategyCategory,
        call_site: &'static str,
    },
    InvariantViolation {
        work_unit_id: u64,
        executed: u64,
    },
}

impl fmt::Display for CpuStrategyAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuForbidden {
                work_unit_id,
                category,
                call_site,
            } => write!(
                f,
                "CPU strategy computation forbidden for work unit {work_unit_id}: category={category:?}, call_site={call_site}"
            ),
            Self::InvariantViolation {
                work_unit_id,
                executed,
            } => write!(
                f,
                "GPU-required work unit {work_unit_id} executed {executed} CPU strategy operations"
            ),
        }
    }
}

impl Error for CpuStrategyAuditError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_production_records_attempt_but_does_not_execute() {
        let audit = CpuStrategyAuditContext::production(42);
        let mut closure_ran = false;
        let result = run(
            EvaluationBackend::GPU_REQUIRED,
            &audit,
            CpuStrategyCategory::CandidateBacktest,
            "test::strict",
            || {
                closure_ran = true;
                7
            },
        );

        assert!(matches!(
            result,
            Err(CpuStrategyAuditError::CpuForbidden { .. })
        ));
        assert!(!closure_ran);
        let snapshot = audit.snapshot();
        assert_eq!(snapshot.total_attempted(), 1);
        assert_eq!(snapshot.total_executed(), 0);
        snapshot.assert_zero_executed().unwrap();
    }

    #[test]
    fn optional_cpu_path_records_attempt_and_execution() {
        let audit = CpuStrategyAuditContext::production(43);
        let value = run(
            EvaluationBackend::AUTO,
            &audit,
            CpuStrategyCategory::PopulationEvaluation,
            "test::optional",
            || 11,
        )
        .unwrap();

        assert_eq!(value, 11);
        let snapshot = audit.snapshot();
        assert_eq!(snapshot.total_attempted(), 1);
        assert_eq!(snapshot.total_executed(), 1);
    }

    #[test]
    fn validation_reference_is_explicitly_separate() {
        let audit = CpuStrategyAuditContext::validation_reference(44);
        let value = run(
            EvaluationBackend::GPU_REQUIRED,
            &audit,
            CpuStrategyCategory::ValidationSimulation,
            "test::oracle",
            || 13,
        )
        .unwrap();

        assert_eq!(value, 13);
        assert_eq!(audit.snapshot().total_executed(), 1);
        assert_eq!(audit.mode(), CpuStrategyExecutionMode::ValidationReference);
    }

    #[test]
    fn cloned_contexts_share_only_their_own_work_unit() {
        let first = CpuStrategyAuditContext::production(100);
        let first_clone = first.clone();
        let second = CpuStrategyAuditContext::production(200);

        run(
            EvaluationBackend::AUTO,
            &first_clone,
            CpuStrategyCategory::SignalSynthesis,
            "test::first",
            || (),
        )
        .unwrap();

        assert_eq!(first.snapshot().total_executed(), 1);
        assert_eq!(second.snapshot().total_executed(), 0);
        assert_eq!(first.snapshot().work_unit_id, 100);
        assert_eq!(second.snapshot().work_unit_id, 200);
    }
}
