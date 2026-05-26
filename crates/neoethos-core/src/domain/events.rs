use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResult {
    pub signal: i32,
    pub confidence: f64,
    pub model_votes: HashMap<String, f64>,
    pub regime: String,
    pub meta_features: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    // Store probabilities as simple Vec for serialization simplicity, or use ndarray with serde feature if enabled
    pub probs: Option<Vec<f64>>,
    // Pandas Series are typically just arrays/Vecs in this context.
    pub trade_probability: Option<f64>, // Porting Series as scalar for single result or Vec if history?
    // Python code implies these are Series, possibly single row or full history?
    // "SignalResult" implies a single point in time, likely scalars.
    // If it's a history, it should be Vec<f64>.
    // Looking at usage: signal history can also be carried as a full numeric series for backtests.
    // So some are scalars, some are series.
    pub stacking_confidence: Option<f64>,
    pub recommended_rr: Option<f64>,
    pub recommended_sl: Option<f64>,
    pub win_probability: Option<f64>,
    // For backtesting, we might store fuller data, but let's keep it simple for core struct.
    // In Rust, for high performance, we might separate "LiveSignal" from "BacktestResult".
}

impl Default for SignalResult {
    fn default() -> Self {
        Self {
            signal: 0,
            confidence: 0.0,
            model_votes: HashMap::new(),
            regime: "neutral".to_string(),
            meta_features: HashMap::new(),
            timestamp: Utc::now(),
            probs: None,
            trade_probability: None,
            stacking_confidence: None,
            recommended_rr: None,
            recommended_sl: None,
            win_probability: None,
        }
    }
}

// PreparedDataset corresponds to python's dataclass with X/y
// In Rust, we might just pass references to ndarrays or Polars DataFrames.
// For a DTO/event, we might store simplified paths or small batches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedDatasetLite {
    pub feature_names: Vec<String>,
    // Actual data might be too heavy for a simple event struct unless we use shared pointers or paths.
    // Keeping it minimal for now.
}

/// **F-099 fix (2026-05-25)** — typed trade side. Replaces the
/// previous untyped `String` field with a serde-tagged enum so that
/// stringly-typed "buy " (with trailing space) / "Buy" / "BUY"
/// inconsistencies are caught at deserialisation rather than silently
/// mis-routing fills downstream.
///
/// Serialised as lowercase strings (`"buy"`, `"sell"`) for backward
/// compatibility with existing on-disk event ledgers — old data still
/// loads, new data is normalised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Buy,
    Sell,
}

impl TradeSide {
    /// Lowercase string form ("buy" / "sell") — matches serde
    /// representation. Used by callers that still need a `String` for
    /// legacy interop.
    pub fn as_lowercase(self) -> &'static str {
        match self {
            Self::Buy => "buy",
            Self::Sell => "sell",
        }
    }

    /// Lenient parse used by the on-disk-ledger migration: accepts
    /// any case + trims whitespace. Returns `None` for unknown
    /// inputs so the caller can decide whether to bail or skip.
    pub fn from_lenient(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "buy" | "long" => Some(Self::Buy),
            "sell" | "short" => Some(Self::Sell),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvent {
    pub symbol: String,
    /// Trade direction. Typed enum per F-099 fix (2026-05-25).
    pub side: TradeSide,
    pub volume: f64,
    pub open_price: f64,
    pub open_time: DateTime<Utc>,
    pub close_price: Option<f64>,
    pub close_time: Option<DateTime<Utc>>,
    pub pnl: Option<f64>,
    pub commission: f64,
    pub swap: f64,
    pub comment: String,
    pub magic: i64,
}

/// **F-099 fix (2026-05-25)** — typed risk severity. Same pattern as
/// [`TradeSide`]: replaces an untyped `String` so that
/// "critical"/"CRITICAL"/"Critical" cannot drift across producers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskSeverity {
    Info,
    Warn,
    Error,
    Critical,
}

impl RiskSeverity {
    pub fn from_lenient(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "info" | "notice" => Some(Self::Info),
            "warn" | "warning" => Some(Self::Warn),
            "error" | "err" => Some(Self::Error),
            "critical" | "crit" | "fatal" => Some(Self::Critical),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub category: String,
    pub message: String,
    /// Typed severity per F-099 fix (2026-05-25). Was `String` before.
    pub severity: RiskSeverity,
    pub context: serde_json::Value,
}

impl RiskEvent {
    pub fn new(
        category: impl Into<String>,
        message: impl Into<String>,
        severity: RiskSeverity,
        context: Option<serde_json::Value>,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().simple().to_string()[..16].to_string(),
            timestamp: Utc::now(),
            category: category.into(),
            message: message.into(),
            severity,
            context: context.unwrap_or(serde_json::json!({})),
        }
    }
}
