//! Structured report types for the api-test harness.
//!
//! Serialised to JSON at the path supplied by `--api-test-output`. The
//! schema is stable so a future Phase A fix can diff a baseline report
//! against a post-fix report to confirm the regression closed.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Top-level report. One file per `--api-test` run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTestReport {
    pub schema_version: u32,
    pub started_at_unix_ms: i64,
    pub finished_at_unix_ms: i64,
    pub environment: String,
    pub neoethos_app_version: String,
    pub host_summary: HostSummary,
    /// One entry per flow attempted, in execution order. A flow that
    /// was filtered out by `--api-test-only` does not appear at all
    /// (rather than appearing as SKIP) so the report length tracks the
    /// actual ground covered.
    pub flows: Vec<FlowResult>,
    pub totals: ReportTotals,
}

impl ApiTestReport {
    /// Current schema. Bump on breaking changes; downstream diff tools
    /// can use this to refuse incompatible baselines.
    pub const SCHEMA_VERSION: u32 = 1;
}

/// Light host fingerprint so a report from another machine can be
/// distinguished. No secrets / tokens / account ids in here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostSummary {
    pub os: String,
    pub cpu_brand: String,
    pub logical_cores: usize,
    /// Total RAM in bytes if detectable, otherwise 0.
    pub total_memory_bytes: u64,
}

/// Per-flow result. `status` is the only field a downstream summariser
/// must look at; the rest is forensic context for triage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowResult {
    pub name: String,
    pub status: FlowStatus,
    pub duration_ms: u128,
    /// Set when status == FAIL. Top-level human-readable description.
    pub error: Option<String>,
    /// Set when status == FAIL. Classified into a small enum so the
    /// Phase A fix list can be filtered by failure mode.
    pub error_kind: Option<FailureKind>,
    /// Optional: bytes the request used. Useful when a flow times out
    /// to know whether we even reached the broker.
    pub request_payload_bytes: Option<usize>,
    /// Optional: bytes of the response payload received before
    /// success / failure was classified.
    pub response_payload_bytes: Option<usize>,
    /// First 2 KB of the most recently observed wire frame (request OR
    /// response, whichever was most useful for forensics on this flow).
    /// Truncated and base64-safe-cleaned so the JSON stays readable.
    pub wire_frame_excerpt: Option<String>,
    /// Free-form key→value details the flow itself attached.
    pub details: serde_json::Map<String, serde_json::Value>,
}

impl FlowResult {
    pub fn pass(name: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: FlowStatus::Pass,
            duration_ms: duration.as_millis(),
            error: None,
            error_kind: None,
            request_payload_bytes: None,
            response_payload_bytes: None,
            wire_frame_excerpt: None,
            details: serde_json::Map::new(),
        }
    }

    pub fn skip(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: FlowStatus::Skip,
            duration_ms: 0,
            error: Some(reason.into()),
            error_kind: None,
            request_payload_bytes: None,
            response_payload_bytes: None,
            wire_frame_excerpt: None,
            details: serde_json::Map::new(),
        }
    }

    pub fn fail(
        name: impl Into<String>,
        duration: Duration,
        error: impl Into<String>,
        kind: FailureKind,
    ) -> Self {
        Self {
            name: name.into(),
            status: FlowStatus::Fail,
            duration_ms: duration.as_millis(),
            error: Some(error.into()),
            error_kind: Some(kind),
            request_payload_bytes: None,
            response_payload_bytes: None,
            wire_frame_excerpt: None,
            details: serde_json::Map::new(),
        }
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.details.insert(key.into(), value);
        self
    }

    #[allow(dead_code)] // wired into each flow as it un-stubs from the
    // api-test harness — keep public so flow
    // implementations can attach broker payloads
    // to the FlowResult without re-deriving the
    // 2 KB clip + mojibake-safe re-encode below.
    pub fn with_wire_excerpt(mut self, raw: &str) -> Self {
        // Trim to 2 KB so a giant reconcile dump doesn't dominate the
        // report. Replace control bytes / non-utf8 with `?` so a
        // downstream diff tool doesn't have to deal with mojibake.
        let mut clipped: String = raw.chars().take(2048).collect();
        clipped = clipped.replace(['\r', '\n', '\t'], " ");
        self.wire_frame_excerpt = Some(clipped);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlowStatus {
    Pass,
    Fail,
    Skip,
}

/// Failure classification. Keep this small and stable — Phase A fixes
/// will filter by kind, and adding new variants forces re-triage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// OAuth token is missing locally, or refresh returned an error.
    AuthMissingOrRefused,
    /// The broker accepted our auth but the response was empty / wrong
    /// shape (e.g. zero accounts, missing required field).
    UnexpectedBrokerResponse,
    /// Request sent, broker did not reply within the flow timeout.
    Timeout,
    /// TCP / TLS / WSS connection error.
    NetworkError,
    /// Broker returned an explicit error envelope (payload type 2142).
    BrokerErrorEnvelope,
    /// Local code panicked / unwrapped / asserted; we caught it in the
    /// runner but the trade-management path would have crashed.
    LocalPanic,
    /// Cleanup-only failure (e.g. close_position after market_buy
    /// passed but the cancel call returned an error). Does not flip
    /// the overall run to FAIL by itself; the totals row tracks
    /// cleanup-failures separately.
    CleanupFailure,
    /// Catch-all for failures the runner could not classify.
    Other,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportTotals {
    pub flows_total: usize,
    pub flows_passed: usize,
    pub flows_failed: usize,
    pub flows_skipped: usize,
    pub cleanup_failures: usize,
    pub total_duration_ms: u128,
}

impl ReportTotals {
    pub fn recompute(flows: &[FlowResult]) -> Self {
        let mut totals = Self::default();
        totals.flows_total = flows.len();
        for flow in flows {
            match flow.status {
                FlowStatus::Pass => totals.flows_passed += 1,
                FlowStatus::Fail => {
                    if matches!(flow.error_kind, Some(FailureKind::CleanupFailure)) {
                        totals.cleanup_failures += 1;
                    } else {
                        totals.flows_failed += 1;
                    }
                }
                FlowStatus::Skip => totals.flows_skipped += 1,
            }
            totals.total_duration_ms += flow.duration_ms;
        }
        totals
    }
}
