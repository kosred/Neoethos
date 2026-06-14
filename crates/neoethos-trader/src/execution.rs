//! Phase-1 `MockExecutionAdapter` — simulated fills, ZERO broker calls.
//!
//! Fills every `Open`/`Close` at the `mark_price` the engine observed at
//! decision time and hands back synthetic position ids. It records a full fill
//! log so the replay harness + tests can assert what happened. This is the
//! offline-dry-run adapter behind the `ExecutionAdapter` trait; the real cTrader
//! `broker_api` adapter (Phase 5) implements the SAME trait — demo vs live is
//! the connected account, not separate code.

use crate::contracts::{ExecReport, ExecStatus, ExecutionAdapter, TradeIntent};

/// One recorded simulated fill (for assertions + a dry-run trade log).
#[derive(Debug, Clone)]
pub struct MockFill {
    pub kind: &'static str,
    pub mark_price: f64,
    pub report: ExecReport,
}

/// Simulates execution in-memory. Optionally rejects a fraction of intents to
/// let tests exercise the rejection path (default: fill everything).
#[derive(Debug, Default)]
pub struct MockExecutionAdapter {
    next_id: u64,
    fills: Vec<MockFill>,
}

impl MockExecutionAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fills(&self) -> &[MockFill] {
        &self.fills
    }

    pub fn fill_count(&self) -> usize {
        self.fills.len()
    }

    fn alloc_position_id(&mut self) -> String {
        self.next_id += 1;
        format!("mock-pos-{}", self.next_id)
    }
}

impl ExecutionAdapter for MockExecutionAdapter {
    fn execute(&mut self, intent: &TradeIntent, mark_price: f64) -> anyhow::Result<ExecReport> {
        let report = match intent {
            TradeIntent::Open { .. } => ExecReport {
                status: ExecStatus::Filled,
                fill_price: Some(mark_price),
                position_id: Some(self.alloc_position_id()),
                detail: "mock open filled".to_string(),
            },
            TradeIntent::Close { position_id, .. } => ExecReport {
                status: ExecStatus::Filled,
                fill_price: Some(mark_price),
                position_id: Some(position_id.clone()),
                detail: "mock close filled".to_string(),
            },
            TradeIntent::Amend { position_id, .. } => ExecReport {
                status: ExecStatus::Filled,
                fill_price: None,
                position_id: Some(position_id.clone()),
                detail: "mock amend applied".to_string(),
            },
            TradeIntent::Cancel { order_id } => ExecReport {
                status: ExecStatus::Filled,
                fill_price: None,
                position_id: None,
                detail: format!("mock cancel {order_id}"),
            },
        };
        self.fills.push(MockFill {
            kind: intent.kind(),
            mark_price,
            report: report.clone(),
        });
        Ok(report)
    }
}
