//! Append-only JSONL journal for actual broker-side execution outcomes.
//!
//! The discovery / forward-test / prop-firm validation artifacts produced
//! by `neoethos-search` describe what *should* happen on simulated data.
//! This journal captures what *did* happen on live broker fills, one
//! JSON object per line, so the deployment side has a verifiable
//! evidence chain matching the canonical artifact contract.
//!
//! The journal is intentionally append-only and rotation-free: one file
//! per process lifetime, opened lazily on the first record. Operators
//! can rotate manually between sessions; downstream tooling can parse a
//! line at a time without holding the whole file in memory.

use crate::app_services::ctrader_execution::{
    CTraderExecutionOutcome, CTraderExecutionRuntimeRequest,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// One row of the live trade journal. Mirrors the broker outcome but
/// stripped of types that don't serialise cleanly and enriched with
/// caller intent (operator action, requested vs filled, etc.) so a
/// reviewer can reason about the trade without rejoining external
/// state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTradeJournalEntry {
    pub schema_version: u32,
    pub recorded_at_unix_ms: i64,
    pub operator_action: String,
    pub account_id: i64,
    pub symbol_id: Option<i64>,
    pub order_id: Option<i64>,
    pub position_id: Option<i64>,
    pub deal_id: Option<i64>,
    pub trade_side: Option<String>,
    pub order_type: Option<String>,
    pub status: String,
    pub requested_lot_size: Option<f64>,
    pub filled_lot_size: Option<f64>,
    pub execution_price: Option<f64>,
    pub gross_profit: Option<f64>,
    pub fee: Option<f64>,
    pub swap: Option<f64>,
    pub net_profit: Option<f64>,
    pub broker_timestamp_ms: Option<i64>,
    pub error_code: Option<String>,
    pub description: Option<String>,
}

const SCHEMA_VERSION: u32 = 1;

impl LiveTradeJournalEntry {
    pub fn from_outcome(
        operator_action: impl Into<String>,
        request: &CTraderExecutionRuntimeRequest,
        outcome: &CTraderExecutionOutcome,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            recorded_at_unix_ms: now_unix_ms(),
            operator_action: operator_action.into(),
            account_id: outcome.account_id,
            symbol_id: outcome.symbol_id,
            order_id: outcome.order_id,
            position_id: outcome.position_id,
            deal_id: outcome.deal_id,
            trade_side: outcome.trade_side.clone(),
            order_type: outcome.order_type.clone(),
            status: format!("{:?}", outcome.status),
            requested_lot_size: outcome.requested_lot_size,
            filled_lot_size: outcome.filled_lot_size,
            execution_price: outcome.execution_price,
            gross_profit: outcome.gross_profit,
            fee: outcome.fee,
            swap: outcome.swap,
            net_profit: outcome.net_profit,
            broker_timestamp_ms: outcome.timestamp_ms,
            error_code: outcome.error_code.clone(),
            description: outcome.description.clone(),
        }
        .with_environment_hint(request)
    }

    fn with_environment_hint(mut self, request: &CTraderExecutionRuntimeRequest) -> Self {
        // Environment is intentionally captured in the operator_action
        // string rather than as a typed field so the schema stays stable
        // when new environments (paper accounts, sandbox, etc.) appear.
        if !self.operator_action.contains('|') {
            self.operator_action = format!(
                "{}|env={}",
                self.operator_action,
                request.environment.endpoint_host()
            );
        }
        self
    }
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn journal_path() -> Option<PathBuf> {
    // F-CORE3 closure (2026-05-25): routed through the canonical
    // `env_overrides::live_journal_path_override` typed getter so the
    // env var is grep-able from one place.
    crate::app_services::env_overrides::live_journal_path_override()
        .map(PathBuf::from)
}

fn writer_lock() -> &'static Mutex<()> {
    static WRITER: OnceLock<Mutex<()>> = OnceLock::new();
    WRITER.get_or_init(|| Mutex::new(()))
}

/// Persist `entry` to the configured journal path. Returns `Ok(false)`
/// when the journal is disabled (env var unset); errors only on real
/// I/O failures so the caller never aborts a successful trade because
/// of a journal hiccup.
pub fn record_live_outcome(entry: &LiveTradeJournalEntry) -> Result<bool> {
    let Some(path) = journal_path() else {
        return Ok(false);
    };
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to ensure live journal parent directory {}",
                parent.display()
            )
        })?;
    }
    let serialized = serde_json::to_string(entry).context("failed to serialise journal entry")?;
    let _guard = writer_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("live journal lock poisoned"))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open live journal file {}", path.display()))?;
    writeln!(file, "{serialized}")
        .with_context(|| format!("failed to append journal entry to {}", path.display()))?;
    Ok(true)
}

/// Best-effort variant: logs a warning instead of returning an error,
/// suitable for fire-and-forget call sites in the trading hot path.
pub fn record_live_outcome_best_effort(entry: &LiveTradeJournalEntry) {
    match record_live_outcome(entry) {
        Ok(true) => {}
        Ok(false) => {}
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::live_journal",
                error = %err,
                "failed to record live trade journal entry"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::ctrader_execution::CTraderExecutionStatus;
    use crate::app_services::ctrader_live_auth::CTraderEnvironment;
    use std::env;

    fn sample_outcome() -> CTraderExecutionOutcome {
        CTraderExecutionOutcome {
            status: CTraderExecutionStatus::Filled,
            account_id: 712345,
            symbol_id: Some(14),
            order_id: Some(1),
            position_id: Some(2),
            deal_id: Some(3),
            trade_side: Some("BUY".into()),
            order_type: Some("MARKET".into()),
            lot_size: Some(0.10),
            requested_lot_size: Some(0.10),
            filled_lot_size: Some(0.10),
            execution_price: Some(1.0987),
            gross_profit: Some(0.0),
            fee: Some(0.0),
            swap: Some(0.0),
            net_profit: Some(0.0),
            timestamp_ms: Some(1_710_000_000_000),
            error_code: None,
            description: None,
        }
    }

    fn sample_request() -> CTraderExecutionRuntimeRequest {
        use crate::app_services::ctrader_execution::CTraderExecutionRequest;
        use crate::app_services::ctrader_messages::{
            CTraderNewOrderRequest, CTraderOrderType, CTraderTradeSide,
        };
        CTraderExecutionRuntimeRequest {
            client_id: "id".into(),
            client_secret: "secret".into(),
            access_token: "tok".into(),
            environment: CTraderEnvironment::Demo,
            account_id: "712345".into(),
            request: CTraderExecutionRequest::NewOrder(Box::new(CTraderNewOrderRequest {
                account_id: 712345,
                symbol_id: 14,
                order_type: CTraderOrderType::Market,
                trade_side: CTraderTradeSide::Buy,
                volume: 10_000,
                limit_price: None,
                stop_price: None,
                time_in_force: None,
                expiration_timestamp_ms: None,
                stop_loss: None,
                take_profit: None,
                comment: None,
                base_slippage_price: None,
                slippage_in_points: None,
                label: None,
                position_id: None,
                client_order_id: Some("buy-eurusd-1".into()),
                relative_stop_loss: None,
                relative_take_profit: None,
                guaranteed_stop_loss: None,
                trailing_stop_loss: None,
                stop_trigger_method: None,
            })),
        }
    }

    #[test]
    fn journal_entry_captures_outcome_fields() {
        let entry =
            LiveTradeJournalEntry::from_outcome("submit", &sample_request(), &sample_outcome());
        assert_eq!(entry.schema_version, SCHEMA_VERSION);
        assert_eq!(entry.account_id, 712345);
        assert_eq!(entry.requested_lot_size, Some(0.10));
        assert_eq!(entry.filled_lot_size, Some(0.10));
        assert!(entry.operator_action.starts_with("submit"));
        assert!(entry.operator_action.contains("env="));
    }

    #[test]
    fn record_returns_false_when_journal_disabled() {
        // Make sure the env var isn't set for this test by overriding
        // to empty. Any other test setting the path will not interfere.
        unsafe {
            env::remove_var("NEOETHOS_BOT_LIVE_JOURNAL_PATH");
        }
        let entry =
            LiveTradeJournalEntry::from_outcome("noop", &sample_request(), &sample_outcome());
        let recorded = record_live_outcome(&entry).expect("disabled journal should not error");
        assert!(!recorded);
    }
}
