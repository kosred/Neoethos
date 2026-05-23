//! Pending-action queue for LLM-proposed trade-management actions (#136).
//!
//! When the Gemma layer wants to take an action that affects real money
//! (close a position, cancel a pending order, …) it CANNOT execute the
//! action directly. Instead it calls the `propose_*` tool which adds a
//! `PendingAction` to this queue. The Flutter UI surfaces the proposal
//! with a Confirm/Reject prompt; the user's explicit click is what
//! actually fires the broker call.
//!
//! ## Why this layer
//!
//! 1. **Safety**. The LLM hallucinates. Letting it close real positions
//!    autonomously is a category of risk we don't take. Every
//!    money-moving action requires a human click.
//! 2. **Audit**. Every proposal + decision is journalled to
//!    `<data_dir>/neoethos/pending_actions.jsonl` so the operator can
//!    later answer "why did the model want to close this trade?".
//! 3. **Bounded staleness**. Each action carries an expiry timestamp
//!    (default 60 s). Confirms after expiry are rejected. Prevents an
//!    action proposed during an ECB conference from getting confirmed
//!    minutes later when the market has changed.
//!
//! ## Strict whitelist
//!
//! Currently the only allowed action kind is `ClosePosition`. NO order
//! placement, NO modify-SL, NO cancel-pending. Each new action kind
//! requires a deliberate code change here AND a matching tool
//! definition in `gemma_tools::propose_*` — there is no generic
//! "execute arbitrary command" backdoor.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// How long an action stays "pending" before it auto-expires.
/// 60 s is enough for a user looking at the screen to react; a
/// pending-action notification sat unread for longer than that has
/// almost certainly stopped being correct (market moved, position
/// state changed via another path, …).
pub const PENDING_ACTION_TTL_SECS: i64 = 60;

/// Cap on simultaneously pending actions. If the LLM is somehow
/// stuck in a loop proposing actions every chat turn, we don't want
/// the queue to grow unbounded.
pub const MAX_PENDING_ACTIONS: usize = 16;

/// Kind of action being proposed. Strict enum — adding a variant
/// here is a code change that must be reviewed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionKind {
    /// Close an existing open position. Both arguments come from the
    /// proposing LLM; `volume_units` 0 means "close entire position"
    /// (we'll look up the current volume at execute time).
    ClosePosition {
        position_id: i64,
        volume_units: i64,
        symbol_hint: Option<String>,
    },
}

impl ActionKind {
    /// Human-readable one-liner used by the Gemma `list_pending_actions`
    /// tool — feature-gated, so this method is dead code in the
    /// default build. The cfg_attr keeps the warning quiet without
    /// hiding it behind a blanket `#[allow]`.
    #[cfg_attr(not(feature = "gemma-backend"), allow(dead_code))]
    pub fn summary(&self) -> String {
        match self {
            Self::ClosePosition {
                position_id,
                volume_units,
                symbol_hint,
            } => {
                let vol = if *volume_units == 0 {
                    "entire".to_string()
                } else {
                    format!("{volume_units} units")
                };
                let sym = symbol_hint.as_deref().unwrap_or("?");
                format!("Close {vol} of position #{position_id} ({sym})")
            }
        }
    }
}

/// Current state of a queued action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    Pending,
    Confirmed,
    Rejected,
    Expired,
    /// Executed successfully on the broker. Carried separately from
    /// `Confirmed` so the UI can distinguish "user clicked yes" from
    /// "yes click → broker accepted → fill complete".
    Executed,
    /// The execute path returned an error. Audit trail keeps the
    /// reason in `result_note`.
    Failed,
}

/// One row on the queue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingAction {
    /// Stable UUID-style identifier the UI references when the user
    /// clicks Confirm / Reject.
    pub id: String,
    pub kind: ActionKind,
    /// Plain-language explanation from the LLM. Surfaced verbatim in
    /// the UI so the user knows why the model wants to act.
    pub reason: String,
    pub proposed_at_unix_ms: i64,
    pub expires_at_unix_ms: i64,
    pub status: ActionStatus,
    /// Free-form post-execution note. Populated after the user
    /// clicks Confirm / Reject and the action runs to completion.
    /// Empty while `status == Pending`.
    pub result_note: String,
}

impl PendingAction {
    fn is_expired(&self, now_ms: i64) -> bool {
        now_ms > self.expires_at_unix_ms
    }
}

fn current_unix_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Used only by `propose()` which is itself feature-gated to the
/// Gemma tool path — no other code path mints action IDs today.
#[cfg_attr(not(feature = "gemma-backend"), allow(dead_code))]
fn next_id() -> String {
    // Cheap unique id — Unix-ms + a small counter so two proposals
    // in the same millisecond don't collide. We don't need UUID's
    // collision-resistance properties for an in-process queue capped
    // at 16 entries.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("act-{}-{}", current_unix_ms(), seq)
}

static QUEUE: OnceLock<Mutex<VecDeque<PendingAction>>> = OnceLock::new();

fn queue() -> &'static Mutex<VecDeque<PendingAction>> {
    QUEUE.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_PENDING_ACTIONS)))
}

/// Add a new proposed action. Returns the new action's id. Drops
/// expired entries during the lock window so the queue self-prunes
/// without a background sweeper. When MAX_PENDING_ACTIONS would be
/// exceeded, the OLDEST `Pending` action is evicted (operator's
/// most recent thought takes priority over a stale unanswered
/// prompt from 10 minutes ago).
/// Called by the `propose_close_position` Gemma tool — feature-gated,
/// so unused in the default build. The endpoint stack
/// (`server/pending_actions::list/confirm/reject`) still works
/// without the LLM; it just always sees an empty queue.
#[cfg_attr(not(feature = "gemma-backend"), allow(dead_code))]
pub fn propose(kind: ActionKind, reason: String) -> Result<String> {
    let mut q = queue().lock().map_err(|_| anyhow!("queue mutex poisoned"))?;
    sweep_expired(&mut q);
    if q.iter().filter(|a| a.status == ActionStatus::Pending).count() >= MAX_PENDING_ACTIONS {
        // Evict the oldest Pending entry. Confirmed / Executed
        // entries we keep around for audit + UI history.
        if let Some(idx) = q.iter().position(|a| a.status == ActionStatus::Pending) {
            let mut evicted = q.remove(idx).expect("position from iter");
            evicted.status = ActionStatus::Expired;
            evicted.result_note =
                "Evicted by queue-cap pressure to make room for newer proposal".to_string();
            journal_to_disk(&evicted);
        }
    }
    let now = current_unix_ms();
    let action = PendingAction {
        id: next_id(),
        kind,
        reason: reason.trim().to_string(),
        proposed_at_unix_ms: now,
        expires_at_unix_ms: now + PENDING_ACTION_TTL_SECS * 1_000,
        status: ActionStatus::Pending,
        result_note: String::new(),
    };
    journal_to_disk(&action);
    let id = action.id.clone();
    q.push_back(action);
    tracing::info!(
        target: "neoethos_app::pending_actions",
        id = %id,
        "new action proposed by LLM"
    );
    Ok(id)
}

/// Return all currently-known actions, newest first. Used by both
/// the HTTP GET endpoint and the LLM `list_pending_actions` tool.
/// Includes finalised actions (Confirmed/Executed/Rejected/Expired)
/// so the operator sees recent history alongside live prompts.
pub fn list_all() -> Vec<PendingAction> {
    let Ok(mut q) = queue().lock() else {
        return Vec::new();
    };
    sweep_expired(&mut q);
    let mut out: Vec<PendingAction> = q.iter().cloned().collect();
    out.sort_by(|a, b| b.proposed_at_unix_ms.cmp(&a.proposed_at_unix_ms));
    out
}

/// Look up + mark as Confirmed. Returns the action so the caller
/// can dispatch to the broker. Errors if the id doesn't exist, has
/// already been confirmed/rejected, or has expired.
pub fn mark_confirmed(id: &str) -> Result<PendingAction> {
    let mut q = queue().lock().map_err(|_| anyhow!("queue mutex poisoned"))?;
    let now = current_unix_ms();
    let entry = q
        .iter_mut()
        .find(|a| a.id == id)
        .ok_or_else(|| anyhow!("no pending action with id `{id}`"))?;
    if entry.status != ActionStatus::Pending {
        anyhow::bail!(
            "action `{id}` is already in state `{:?}` — cannot confirm",
            entry.status
        );
    }
    if entry.is_expired(now) {
        entry.status = ActionStatus::Expired;
        entry.result_note = format!(
            "Auto-expired after {PENDING_ACTION_TTL_SECS} s (user click arrived too late)"
        );
        let snapshot = entry.clone();
        journal_to_disk(&snapshot);
        anyhow::bail!("action `{id}` has expired");
    }
    entry.status = ActionStatus::Confirmed;
    let snapshot = entry.clone();
    journal_to_disk(&snapshot);
    Ok(snapshot)
}

/// Mark as Rejected. Same lookup / expiry semantics as
/// `mark_confirmed` minus the actionable side-effect — rejected
/// actions just sit in the queue for audit.
pub fn mark_rejected(id: &str, reason: Option<&str>) -> Result<PendingAction> {
    let mut q = queue().lock().map_err(|_| anyhow!("queue mutex poisoned"))?;
    let now = current_unix_ms();
    let entry = q
        .iter_mut()
        .find(|a| a.id == id)
        .ok_or_else(|| anyhow!("no pending action with id `{id}`"))?;
    if entry.status != ActionStatus::Pending {
        anyhow::bail!(
            "action `{id}` is already in state `{:?}` — cannot reject",
            entry.status
        );
    }
    entry.status = ActionStatus::Rejected;
    entry.result_note = reason
        .map(|r| format!("Rejected by operator: {r}"))
        .unwrap_or_else(|| "Rejected by operator (no reason given)".to_string());
    if entry.is_expired(now) {
        // Edge case — already expired AND just got rejected. Note
        // both but keep status = Rejected so the audit trail makes
        // sense ("user clicked Reject on a stale entry").
        entry.result_note.push_str("; was already past expiry");
    }
    let snapshot = entry.clone();
    journal_to_disk(&snapshot);
    Ok(snapshot)
}

/// Set the post-execution status (Executed / Failed) + the broker's
/// response text. Called by the HTTP handler after the actual
/// broker call returns. Does not error if the action isn't in
/// `Confirmed` state — log + carry on, since the broker call
/// already happened.
pub fn mark_completed(id: &str, status: ActionStatus, note: String) {
    let Ok(mut q) = queue().lock() else { return };
    let Some(entry) = q.iter_mut().find(|a| a.id == id) else {
        return;
    };
    entry.status = status;
    entry.result_note = note;
    let snapshot = entry.clone();
    journal_to_disk(&snapshot);
}

/// Drop pending entries that have crossed `expires_at_unix_ms`.
/// Mutates them to `Expired` + journal, then leaves them in the
/// queue so the operator's UI sees the timeout history.
fn sweep_expired(q: &mut VecDeque<PendingAction>) {
    let now = current_unix_ms();
    for action in q.iter_mut() {
        if action.status == ActionStatus::Pending && action.is_expired(now) {
            action.status = ActionStatus::Expired;
            action.result_note =
                format!("Auto-expired after {PENDING_ACTION_TTL_SECS} s of no operator response");
            journal_to_disk(action);
        }
    }
    // Prune anything older than 24 h regardless of status — long
    // enough for the operator to find audit history; short enough
    // that the in-memory deque doesn't grow forever.
    let cutoff = now - 24 * 3600 * 1000;
    q.retain(|a| a.proposed_at_unix_ms >= cutoff);
}

/// Canonical on-disk audit path. Honours
/// `NEOETHOS_PENDING_ACTIONS_PATH` for tests / CI.
pub fn default_journal_path() -> PathBuf {
    if let Ok(custom) = std::env::var("NEOETHOS_PENDING_ACTIONS_PATH") {
        if !custom.trim().is_empty() {
            return PathBuf::from(custom);
        }
    }
    let base = dirs::data_dir().unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".local")
    });
    base.join("neoethos").join("pending_actions.jsonl")
}

/// Append one JSON line per state transition to the audit file.
/// Best-effort: failure logs warn but never propagates.
fn journal_to_disk(action: &PendingAction) {
    if let Err(err) = write_audit_line(action) {
        tracing::warn!(
            target: "neoethos_app::pending_actions",
            id = %action.id,
            error = %err,
            "failed to append pending-action audit row"
        );
    }
}

fn write_audit_line(action: &PendingAction) -> Result<()> {
    let path = default_journal_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("failed to create directory for pending-actions audit at {}", parent.display())
        })?;
    }
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open pending-actions audit at {}", path.display()))?;
    let line = serde_json::to_string(action).context("serialize pending action to JSON")?;
    writeln!(f, "{line}").context("write pending-action audit line")?;
    Ok(())
}

#[cfg(test)]
pub fn clear_for_tests() {
    if let Ok(mut q) = queue().lock() {
        q.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn temp_audit_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "neoethos-pending-actions-{name}-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        // SAFETY: TEST_LOCK serialises env mutation across these
        // tests.
        unsafe {
            std::env::set_var("NEOETHOS_PENDING_ACTIONS_PATH", &p);
        }
        p
    }

    fn cleanup_env() {
        unsafe {
            std::env::remove_var("NEOETHOS_PENDING_ACTIONS_PATH");
        }
    }

    #[test]
    fn propose_returns_id_and_appears_in_list() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _p = temp_audit_path("propose-list");
        clear_for_tests();
        let id = propose(
            ActionKind::ClosePosition {
                position_id: 42,
                volume_units: 0,
                symbol_hint: Some("EURUSD".to_string()),
            },
            "User asked to flatten; position bleeding 30 pips.".to_string(),
        )
        .expect("propose");
        let listed = list_all();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);
        assert_eq!(listed[0].status, ActionStatus::Pending);
        cleanup_env();
    }

    #[test]
    fn confirm_marks_status_and_returns_snapshot() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _p = temp_audit_path("confirm");
        clear_for_tests();
        let id = propose(
            ActionKind::ClosePosition {
                position_id: 1,
                volume_units: 100_000,
                symbol_hint: None,
            },
            "test".to_string(),
        )
        .unwrap();
        let snap = mark_confirmed(&id).expect("confirm");
        assert_eq!(snap.status, ActionStatus::Confirmed);
        // Re-confirming is rejected (idempotency-via-explicit-error).
        assert!(mark_confirmed(&id).is_err());
        cleanup_env();
    }

    #[test]
    fn reject_records_reason() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _p = temp_audit_path("reject");
        clear_for_tests();
        let id = propose(
            ActionKind::ClosePosition {
                position_id: 1,
                volume_units: 0,
                symbol_hint: None,
            },
            "test".to_string(),
        )
        .unwrap();
        let snap = mark_rejected(&id, Some("I changed my mind")).expect("reject");
        assert_eq!(snap.status, ActionStatus::Rejected);
        assert!(snap.result_note.contains("I changed my mind"));
        cleanup_env();
    }

    #[test]
    fn unknown_id_errors_cleanly() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _p = temp_audit_path("unknown-id");
        clear_for_tests();
        assert!(mark_confirmed("nope").is_err());
        assert!(mark_rejected("nope", None).is_err());
        cleanup_env();
    }

    #[test]
    fn audit_journal_is_appended() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let path = temp_audit_path("audit");
        clear_for_tests();
        let id = propose(
            ActionKind::ClosePosition {
                position_id: 9,
                volume_units: 0,
                symbol_hint: None,
            },
            "test".to_string(),
        )
        .unwrap();
        let _ = mark_confirmed(&id);
        let body = std::fs::read_to_string(&path).expect("audit file");
        let lines: Vec<&str> = body.lines().collect();
        // Two transitions: Pending (propose) + Confirmed (mark_confirmed)
        assert_eq!(lines.len(), 2);
        let first: PendingAction = serde_json::from_str(lines[0]).expect("row 0");
        let second: PendingAction = serde_json::from_str(lines[1]).expect("row 1");
        assert_eq!(first.status, ActionStatus::Pending);
        assert_eq!(second.status, ActionStatus::Confirmed);
        cleanup_env();
    }

    #[test]
    fn summary_describes_close_position() {
        let s = ActionKind::ClosePosition {
            position_id: 12345,
            volume_units: 0,
            symbol_hint: Some("EURUSD".to_string()),
        }
        .summary();
        assert!(s.contains("12345"));
        assert!(s.contains("entire"));
        assert!(s.contains("EURUSD"));
    }
}
