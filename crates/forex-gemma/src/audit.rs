//! Audit log writer for every Gemma interaction.
//!
//! Phase G0 — schema definition for `AuditRow` (versioned via
//! `forex_core::SchemaVersion` + `HasSchemaVersion` so the audit
//! file format is forward-compatible from day one) plus an
//! `AuditLog` trait with an in-memory test backend.
//!
//! The disk-backed JSONL writer lands in G7. Doing the
//! schema first means the producers (the runtime + tools) can
//! emit `AuditRow` values now without waiting on G7.
//!
//! ## What gets logged
//!
//! Every Gemma turn writes one row:
//!
//! - **What the user said** — by default just `sha256(prompt)`
//!   (PII-safe). The wizard opt-in `audit.store_full_text`
//!   flag adds the verbatim text.
//! - **What the model returned** — same hash-by-default policy.
//! - **Topic-gate verdict** — Allow / SoftWarning / Refuse + the
//!   reason string.
//! - **Tool calls** — list of `(tool_name, args_hash, result_hash,
//!   verdict)`. Trading-tool calls additionally carry the
//!   `TradeOrigin::Gemma` flag so audit-side analytics can
//!   slice "Gemma-initiated trades" cleanly.
//! - **Model id** — the `GemmaRuntime.model_id()` (e.g.
//!   `"gemma-3-e4b-uncensored-q5_k_m"`) so a row is tied to the
//!   exact build / quantization.
//! - **Latency** — milliseconds from prompt arrival to final
//!   token. Useful for diagnosing slow CPU paths.

use crate::error::GemmaError;
use crate::gate::TopicCheck;
use forex_core::{HasSchemaVersion, SchemaVersion, default_v1};
use serde::{Deserialize, Serialize};

/// Schema version of the on-disk `gemma_audit.jsonl` rows.
///
/// v1 (current): the G0 layout. New optional fields go behind
/// `#[serde(default)]`; renames / breaking shape changes bump
/// the version + ship a migration in the reader.
pub const GEMMA_AUDIT_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1);

/// Verdict the gate gave for this turn. Lifts `TopicCheck` into
/// a serializable shape (the original `TopicCheck` carries an
/// `Allow` unit variant and `Refuse { canned_response, .. }`
/// strings we DON'T want in the audit log when full-text is
/// off).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuditGateVerdict {
    Allow,
    SoftWarning { reason: String },
    Refuse { reason: String },
}

impl From<&TopicCheck> for AuditGateVerdict {
    fn from(c: &TopicCheck) -> Self {
        match c {
            TopicCheck::Allow => Self::Allow,
            TopicCheck::SoftWarning { reason } => Self::SoftWarning {
                reason: reason.clone(),
            },
            TopicCheck::Refuse { reason, .. } => Self::Refuse {
                reason: reason.clone(),
            },
        }
    }
}

/// One tool invocation inside a turn. The runtime records the
/// args + result as hashes by default; the wizard opt-in
/// `audit.store_full_text` adds the verbatim JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditToolCall {
    pub tool_name: String,
    /// SHA-256 of `serde_json::to_string(&args)`.
    pub args_sha256: String,
    /// SHA-256 of the result body, or `None` if the tool failed
    /// before producing a result.
    #[serde(default)]
    pub result_sha256: Option<String>,
    /// Outcome short-string: `"ok"`, `"denied"`, `"not_found"`,
    /// `"errored"`. Cheap to filter on for audit-side analytics.
    pub outcome: String,
    /// Tool category — captured at log time so the audit row is
    /// self-contained even if the registry changes later.
    pub category: String,
    /// Full args JSON when `audit.store_full_text` is on; `None`
    /// otherwise.
    #[serde(default)]
    pub args_full: Option<serde_json::Value>,
    /// Full result JSON when `audit.store_full_text` is on; `None`
    /// otherwise.
    #[serde(default)]
    pub result_full: Option<serde_json::Value>,
}

/// One audit row. Each Gemma turn emits exactly one row, written
/// atomically. The full-text policy is per-row, not per-file —
/// the same audit file can mix hashes-only and full-text rows
/// across sessions if the wizard flag toggled in between.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditRow {
    /// On-disk schema version. Backward-compatible v1 default.
    #[serde(default = "default_v1")]
    pub schema_version: SchemaVersion,
    /// Wall-clock UTC timestamp the turn started.
    pub started_at_unix_ms: i64,
    /// Wall-clock UTC timestamp the turn finished. Useful for
    /// latency analysis without a separate field.
    pub finished_at_unix_ms: i64,
    /// Stable session id (the helper's chat session, not the
    /// DXtrade / cTrader auth session).
    pub session_id: String,
    /// Model id from `GemmaRuntime.model_id()` (e.g.
    /// `"gemma-3-e4b-uncensored-q5_k_m"`).
    pub model_id: String,
    /// SHA-256 of the user's prompt.
    pub user_prompt_sha256: String,
    /// Full prompt text when `audit.store_full_text` is on.
    #[serde(default)]
    pub user_prompt_full: Option<String>,
    /// SHA-256 of the response Gemma produced (after post-filter).
    pub response_sha256: String,
    /// Full response text when `audit.store_full_text` is on.
    #[serde(default)]
    pub response_full: Option<String>,
    /// Gate verdict — primary slot for audit-side analytics
    /// ("how often did we refuse?").
    pub gate_verdict: AuditGateVerdict,
    /// Verdict from the POST-filter specifically. `Some(Refuse)`
    /// means Gemma's reply was swapped for the canned refusal.
    /// `None` ⇒ post-filter not run (refused at input).
    #[serde(default)]
    pub post_filter_verdict: Option<AuditGateVerdict>,
    /// Tool calls Gemma made during the turn (zero or more).
    #[serde(default)]
    pub tool_calls: Vec<AuditToolCall>,
}

impl HasSchemaVersion for AuditRow {
    const CURRENT: SchemaVersion = GEMMA_AUDIT_SCHEMA_VERSION;
    fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }
}

impl AuditRow {
    /// Convenience constructor that stamps the current schema
    /// version and sane empty-tool defaults. Most fields are
    /// required so the caller still has to fill them in.
    pub fn new(
        session_id: impl Into<String>,
        model_id: impl Into<String>,
        started_at_unix_ms: i64,
        finished_at_unix_ms: i64,
        user_prompt_sha256: impl Into<String>,
        response_sha256: impl Into<String>,
        gate_verdict: AuditGateVerdict,
    ) -> Self {
        Self {
            schema_version: GEMMA_AUDIT_SCHEMA_VERSION,
            started_at_unix_ms,
            finished_at_unix_ms,
            session_id: session_id.into(),
            model_id: model_id.into(),
            user_prompt_sha256: user_prompt_sha256.into(),
            user_prompt_full: None,
            response_sha256: response_sha256.into(),
            response_full: None,
            gate_verdict,
            post_filter_verdict: None,
            tool_calls: Vec::new(),
        }
    }
}

/// Writer trait. G7 ships a `JsonlAuditLog` backed by a rolling
/// file at `~/.forex-ai/gemma_audit.jsonl`; G0 ships only the
/// in-memory backend below + the trait.
pub trait AuditLog: Send + Sync {
    /// Append one row. Implementations MUST be atomic at the
    /// row level — a partial row on disk would poison the
    /// JSONL parser.
    fn append(&self, row: AuditRow) -> Result<(), GemmaError>;

    /// Read all rows currently held (in-memory backend) or
    /// recent N rows (file backend, future). G0 keeps it
    /// simple — return everything we have.
    fn snapshot(&self) -> Result<Vec<AuditRow>, GemmaError>;
}

/// In-memory audit log — useful for tests and for the chrome
/// to display a "this session's audit trail" panel without
/// touching disk. Thread-safe via `Mutex`.
pub struct InMemoryAuditLog {
    rows: std::sync::Mutex<Vec<AuditRow>>,
}

impl InMemoryAuditLog {
    pub fn new() -> Self {
        Self {
            rows: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl Default for InMemoryAuditLog {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditLog for InMemoryAuditLog {
    fn append(&self, row: AuditRow) -> Result<(), GemmaError> {
        self.rows
            .lock()
            .map_err(|_| GemmaError::AuditWriteFailed {
                reason: "in-memory audit log mutex poisoned".to_string(),
            })?
            .push(row);
        Ok(())
    }

    fn snapshot(&self) -> Result<Vec<AuditRow>, GemmaError> {
        Ok(self
            .rows
            .lock()
            .map_err(|_| GemmaError::AuditWriteFailed {
                reason: "in-memory audit log mutex poisoned".to_string(),
            })?
            .clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> AuditRow {
        AuditRow::new(
            "session-1",
            "gemma-stub",
            1_700_000_000_000,
            1_700_000_001_000,
            "deadbeef",
            "feedface",
            AuditGateVerdict::Allow,
        )
    }

    #[test]
    fn new_stamps_current_schema_version() {
        let r = sample_row();
        assert_eq!(r.schema_version, GEMMA_AUDIT_SCHEMA_VERSION);
    }

    #[test]
    fn has_schema_version_trait_returns_field() {
        let r = sample_row();
        assert_eq!(r.schema_version(), GEMMA_AUDIT_SCHEMA_VERSION);
    }

    #[test]
    fn pre_versioning_rows_default_to_v1() {
        // A row written by a pre-versioning build won't carry
        // `schema_version`; serde must fall back to v1.
        let raw = r#"{
            "started_at_unix_ms": 0,
            "finished_at_unix_ms": 0,
            "session_id": "x",
            "model_id": "y",
            "user_prompt_sha256": "a",
            "response_sha256": "b",
            "gate_verdict": { "kind": "allow" }
        }"#;
        let parsed: AuditRow = serde_json::from_str(raw).expect("de");
        assert_eq!(parsed.schema_version, SchemaVersion::new(1));
    }

    #[test]
    fn topic_check_lifts_into_audit_verdict_without_canned_text() {
        let check = TopicCheck::Refuse {
            reason: "jailbreak".to_string(),
            canned_response: "I can only help with forex-ai…".to_string(),
        };
        let verdict: AuditGateVerdict = (&check).into();
        let serialized = serde_json::to_string(&verdict).unwrap();
        // The canned response text MUST NOT leak into the audit
        // — that's user-facing UX, not audit signal.
        assert!(!serialized.contains("I can only help"));
        assert!(serialized.contains("jailbreak"));
    }

    #[test]
    fn in_memory_audit_log_round_trips_a_row() {
        let log = InMemoryAuditLog::new();
        let row = sample_row();
        log.append(row.clone()).expect("append");
        let snap = log.snapshot().expect("snapshot");
        assert_eq!(snap, vec![row]);
    }

    #[test]
    fn in_memory_audit_log_preserves_insertion_order() {
        let log = InMemoryAuditLog::new();
        for i in 0..5i64 {
            let mut r = sample_row();
            r.started_at_unix_ms = i;
            log.append(r).unwrap();
        }
        let snap = log.snapshot().unwrap();
        let stamps: Vec<i64> = snap.iter().map(|r| r.started_at_unix_ms).collect();
        assert_eq!(stamps, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn audit_row_serializes_to_jsonl_friendly_shape() {
        // JSONL = one JSON document per line, no inter-line
        // delimiters. Pin that `serde_json::to_string` emits a
        // single-line document.
        let row = sample_row();
        let s = serde_json::to_string(&row).unwrap();
        assert!(!s.contains('\n'), "row must serialize on a single line");
    }
}

// ---------------------------------------------------------------------------
// G7 — JSONL audit log file writer
// ---------------------------------------------------------------------------

/// JSONL-on-disk audit-log backend. Each `append` call writes one
/// row + newline to the configured path; rows never share lines.
/// Thread-safe — internal `Mutex` serialises writes so two
/// concurrent threads can call `append` without corrupting the
/// file.
///
/// ## File rotation
///
/// When the file exceeds `max_size_bytes` (set from
/// `AuditLogConfig.max_size_mb`), the next append rotates it:
/// `gemma_audit.jsonl` is renamed `gemma_audit.jsonl.1` (after
/// removing any older `.1` file) and a fresh `gemma_audit.jsonl`
/// starts collecting rows. Lossless within the rotation pair;
/// older history beyond `.1` is discarded by design (the audit
/// trail is meant to be *recent* by default — operators who want
/// long-term retention should ship `gemma_audit.jsonl.1` off-box
/// periodically).
///
/// ## Crash safety
///
/// We `flush()` after every row. A crash mid-write would only
/// truncate the last in-progress line; the `snapshot()` reader
/// tolerates a single trailing malformed line by stopping at
/// the last clean newline.
pub struct JsonlAuditLog {
    inner: std::sync::Mutex<JsonlInner>,
}

struct JsonlInner {
    path: std::path::PathBuf,
    max_size_bytes: u64,
}

impl JsonlAuditLog {
    /// Construct a writer pointing at `path`. Parent directories
    /// are created on demand on first `append`. The path itself
    /// doesn't need to exist yet — it'll be created on the first
    /// write.
    pub fn new(path: impl Into<std::path::PathBuf>, max_size_bytes: u64) -> Self {
        Self {
            inner: std::sync::Mutex::new(JsonlInner {
                path: path.into(),
                max_size_bytes: max_size_bytes.max(1024),
            }),
        }
    }

    /// Convenience — translate `AuditLogConfig` to a path +
    /// limit. Used by the chrome at startup.
    pub fn from_config(
        path: impl Into<std::path::PathBuf>,
        cfg: &crate::config::AuditLogConfig,
    ) -> Self {
        Self::new(path, (cfg.max_size_mb as u64) * 1_024 * 1_024)
    }

    fn rotate_if_needed(inner: &JsonlInner) -> std::io::Result<()> {
        if let Ok(meta) = std::fs::metadata(&inner.path) {
            if meta.len() >= inner.max_size_bytes {
                let rotated = inner.path.with_extension({
                    let ext = inner
                        .path
                        .extension()
                        .map(|e| e.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    if ext.is_empty() {
                        "1".to_string()
                    } else {
                        format!("{ext}.1")
                    }
                });
                // Drop the older .1 to keep exactly 2 generations.
                let _ = std::fs::remove_file(&rotated);
                std::fs::rename(&inner.path, &rotated)?;
            }
        }
        Ok(())
    }
}

impl AuditLog for JsonlAuditLog {
    fn append(&self, row: AuditRow) -> Result<(), GemmaError> {
        use std::io::Write;
        let inner = self
            .inner
            .lock()
            .map_err(|_| GemmaError::AuditWriteFailed {
                reason: "JSONL audit log mutex poisoned".to_string(),
            })?;

        // Ensure parent dir exists.
        if let Some(parent) = inner.path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| GemmaError::AuditWriteFailed {
                    reason: format!("create_dir_all({}): {e}", parent.display()),
                })?;
            }
        }

        // Rotate if oversize.
        Self::rotate_if_needed(&inner).map_err(|e| GemmaError::AuditWriteFailed {
            reason: format!("rotate at {}: {e}", inner.path.display()),
        })?;

        let line = serde_json::to_string(&row).map_err(|e| GemmaError::AuditWriteFailed {
            reason: format!("serialize: {e}"),
        })?;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&inner.path)
            .map_err(|e| GemmaError::AuditWriteFailed {
                reason: format!("open({}): {e}", inner.path.display()),
            })?;
        writeln!(file, "{line}").map_err(|e| GemmaError::AuditWriteFailed {
            reason: format!("write: {e}"),
        })?;
        file.flush().map_err(|e| GemmaError::AuditWriteFailed {
            reason: format!("flush: {e}"),
        })?;
        Ok(())
    }

    fn snapshot(&self) -> Result<Vec<AuditRow>, GemmaError> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| GemmaError::AuditWriteFailed {
                reason: "JSONL audit log mutex poisoned".to_string(),
            })?;
        if !inner.path.exists() {
            return Ok(vec![]);
        }
        let text =
            std::fs::read_to_string(&inner.path).map_err(|e| GemmaError::AuditWriteFailed {
                reason: format!("read({}): {e}", inner.path.display()),
            })?;
        let mut rows = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<AuditRow>(line) {
                Ok(r) => rows.push(r),
                // Trailing in-progress write would corrupt the
                // last line on crash. Tolerate it by stopping at
                // the last clean newline rather than failing the
                // whole snapshot.
                Err(_e) if idx == text.lines().count() - 1 => break,
                Err(e) => {
                    return Err(GemmaError::AuditWriteFailed {
                        reason: format!("parse line {}: {e}", idx + 1),
                    });
                }
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod jsonl_tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("forex-gemma-audit-{name}-{nonce}.jsonl"));
        p
    }

    fn sample(session: &str) -> AuditRow {
        AuditRow::new(
            session,
            "gemma-stub",
            1_700_000_000_000,
            1_700_000_001_000,
            "deadbeef",
            "feedface",
            AuditGateVerdict::Allow,
        )
    }

    #[test]
    fn jsonl_writes_one_row_and_reads_it_back() {
        let path = temp_path("write-one");
        let log = JsonlAuditLog::new(&path, 1024 * 1024);
        let row = sample("s-1");
        log.append(row.clone()).expect("append");
        let snap = log.snapshot().expect("snapshot");
        assert_eq!(snap, vec![row]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn jsonl_appends_multiple_rows_preserving_order() {
        let path = temp_path("write-many");
        let log = JsonlAuditLog::new(&path, 1024 * 1024);
        for i in 0..5_i64 {
            let mut r = sample("s-1");
            r.started_at_unix_ms = i;
            log.append(r).unwrap();
        }
        let snap = log.snapshot().unwrap();
        assert_eq!(snap.len(), 5);
        for (i, r) in snap.iter().enumerate() {
            assert_eq!(r.started_at_unix_ms, i as i64);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn jsonl_rotates_at_size_threshold() {
        let path = temp_path("rotate");
        // Tiny rotation threshold so a handful of rows triggers it.
        let log = JsonlAuditLog::new(&path, 1024);
        for _ in 0..50 {
            log.append(sample("s-rot")).unwrap();
        }
        // Main file exists and the .1 backup also exists.
        let rotated = path.with_extension("jsonl.1");
        assert!(path.exists(), "main jsonl should exist post-rotation");
        assert!(rotated.exists(), ".1 backup should exist post-rotation");
        // Main file is smaller than the rotation threshold + one row.
        let main_size = std::fs::metadata(&path).unwrap().len();
        assert!(main_size < 2048);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&rotated);
    }

    #[test]
    fn jsonl_snapshot_returns_empty_when_file_missing() {
        let path = temp_path("missing");
        let log = JsonlAuditLog::new(&path, 1024 * 1024);
        let snap = log.snapshot().unwrap();
        assert!(snap.is_empty());
    }

    #[test]
    fn jsonl_creates_parent_directories_on_demand() {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("forex-gemma-audit-mkdirp-{nonce}"));
        p.push("subdir");
        p.push("audit.jsonl");
        let log = JsonlAuditLog::new(&p, 1024 * 1024);
        log.append(sample("s-mkdir")).expect("append");
        assert!(p.exists());
        // Clean up
        let _ = std::fs::remove_file(&p);
        if let Some(parent) = p.parent() {
            let _ = std::fs::remove_dir(parent);
            if let Some(grand) = parent.parent() {
                let _ = std::fs::remove_dir(grand);
            }
        }
    }

    #[test]
    fn jsonl_from_config_uses_max_size_mb_field() {
        let path = temp_path("from-config");
        let cfg = crate::config::AuditLogConfig {
            store_full_text: false,
            max_size_mb: 1,
        };
        let log = JsonlAuditLog::from_config(&path, &cfg);
        log.append(sample("s-cfg")).unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn jsonl_tolerates_trailing_corrupt_line_on_last_position() {
        use std::io::Write;
        let path = temp_path("trailing-corrupt");
        let log = JsonlAuditLog::new(&path, 1024 * 1024);
        log.append(sample("ok-1")).unwrap();
        log.append(sample("ok-2")).unwrap();
        // Simulate a crash mid-write: append a half-baked line.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        write!(f, "{{ \"started_at_unix_ms\": ").unwrap();
        let snap = log.snapshot().expect("snapshot tolerates trailing");
        assert_eq!(snap.len(), 2);
        let _ = std::fs::remove_file(&path);
    }
}
