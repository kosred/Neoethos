//! Persistence layer for the Risky Mode kill-switch cooldown.
//!
//! A deleted setup flow originally wrote arm/acknowledgement fields into
//! this file. Production never consumed those fields; live entry decisions use
//! resolved Settings and the RiskyModeManager. Schema v3 removes that inert
//! state and keeps the one value that is actively enforced across restarts:
//! `last_killed_at_utc_ms`.
//!
//! # Lookup order (highest priority first)
//!
//! 1. `$NEOETHOS_RISKY_MODE_STATE_PATH` runtime env var (tests / CI).
//! 2. `<dirs::config_dir>/neoethos/risky_mode_state.json` — `%APPDATA%`
//!    on Windows, `$XDG_CONFIG_HOME` on Linux, `~/Library/Application
//!    Support` on macOS.
//! 3. `<cwd>/.local/neoethos/risky_mode_state.json` — dev machine
//!    fallback.
//!
//! # Why a sibling file instead of `Settings`
//!
//! - Settings describe operator policy; this file records a runtime safety
//!   event and its cooldown clock.
//! - The pattern mirrors `broker_persistence.rs`, which is already the
//!   established neoethos-app convention for "one persisted contract per
//!   sibling file".
//!
//! # Schema versioning
//!
//! Carries `schema_version: SchemaVersion` exactly like every other
//! Phase-D4 contract. Future shape changes bump the version; readers
//! older than the file's version log an error and treat the cooldown state as
//! unreadable/fail-closed at the live-entry boundary.

use anyhow::{Context, Result};
use neoethos_core::{HasSchemaVersion, SchemaVersion, default_v1};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::{env, fs};

/// Current schema version this build writes when saving Risky Mode
/// state. Bumped on breaking changes to [`RiskyModeStateFile`].
///
/// **v2 (2026-05-25)** — added `last_killed_at_utc_ms` for the 24h
/// cooldown. **v3 (2026-08-31)** — removed the deleted setup flow's inert
/// arm/acknowledgement fields. Serde ignores those fields in v1/v2 JSON, so old
/// files still load and the next safety-event write naturally compacts them.
pub const RISKY_MODE_STATE_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(3);

/// Cooldown duration before Risky Mode auto re-arms after a
/// kill-switch trip. Operator-chosen 24h via the architectural
/// AskUserQuestion answer (2026-05-25).
pub const RISKY_MODE_KILL_SWITCH_COOLDOWN_MS: i64 = 24 * 60 * 60 * 1000;

const APP_CONFIG_SUBDIR: &str = "neoethos";
const STATE_FILENAME: &str = "risky_mode_state.json";

// **F-CORE3 closure (2026-05-25)**: the canonical env-var name lives
// in `app_services::env_overrides::ENV_RISKY_MODE_STATE_PATH`; the
// test-only alias below keeps the in-file `#[cfg(test)]` set_var /
// remove_var calls readable (and `cargo check` clean) without
// duplicating the string literal.
#[cfg(test)]
const ENV_OVERRIDE_VAR: &str = crate::app_services::env_overrides::ENV_RISKY_MODE_STATE_PATH;

/// On-disk representation of the enforced Risky Mode cooldown state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskyModeStateFile {
    /// Phase-D4 schema version. Defaults to v1 (pre-versioning shape)
    /// when missing so files written by older builds load without
    /// breaking — see [`RISKY_MODE_STATE_SCHEMA_VERSION`].
    #[serde(default = "default_v1")]
    pub schema_version: SchemaVersion,
    /// Last write time as Unix-milliseconds, UTC. Lets a concurrent
    /// app instance detect a stale file.
    #[serde(default)]
    pub last_updated_utc_ms: i64,
    /// **F-231/F-501/F-630 closure (2026-05-25 — schema v2)**: Unix-
    /// millisecond timestamp of the last time the Risky Mode kill-
    /// switch tripped. When non-`None` AND within the
    /// [`RISKY_MODE_KILL_SWITCH_COOLDOWN_MS`] window from `now`, the
    /// kill-switch stays active for 24 h. After the cooldown,
    /// the background task in `server::bridge::run` calls
    /// `clear_expired_kill_switch`, which clears this field.
    ///
    /// `None` = no kill on record (initial state, or post-re-arm).
    #[serde(default)]
    pub last_killed_at_utc_ms: Option<i64>,
}

impl RiskyModeStateFile {
    /// How many seconds remain on the kill-switch cooldown. `None` =
    /// either no kill on record (`last_killed_at_utc_ms is None`) or
    /// the 24 h has already elapsed. UI surfaces the remainder as
    /// "Auto re-arm in 17h 23m" so the operator knows when Risky
    /// Mode will come back online without an explicit action.
    pub fn cooldown_remaining_secs(&self, now_utc_ms: i64) -> Option<u64> {
        let killed_at = self.last_killed_at_utc_ms?;
        let elapsed = now_utc_ms.saturating_sub(killed_at);
        if elapsed >= RISKY_MODE_KILL_SWITCH_COOLDOWN_MS {
            None
        } else {
            Some(((RISKY_MODE_KILL_SWITCH_COOLDOWN_MS - elapsed) / 1000) as u64)
        }
    }

    /// Whether the kill-switch cooldown has elapsed since the last
    /// kill. `true` when (a) a kill is on record AND (b) more than
    /// 24 h have passed. The background task uses this gate before
    /// clearing `last_killed_at_utc_ms`.
    pub fn cooldown_expired(&self, now_utc_ms: i64) -> bool {
        match self.last_killed_at_utc_ms {
            Some(killed_at) => {
                let elapsed = now_utc_ms.saturating_sub(killed_at);
                elapsed >= RISKY_MODE_KILL_SWITCH_COOLDOWN_MS
            }
            None => false,
        }
    }
}

impl Default for RiskyModeStateFile {
    fn default() -> Self {
        Self {
            schema_version: RISKY_MODE_STATE_SCHEMA_VERSION,
            last_updated_utc_ms: 0,
            last_killed_at_utc_ms: None,
        }
    }
}

impl HasSchemaVersion for RiskyModeStateFile {
    const CURRENT: SchemaVersion = RISKY_MODE_STATE_SCHEMA_VERSION;
    fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }
}

/// Resolves the path to the Risky Mode state JSON file.
///
/// Order of resolution:
/// 1. `$NEOETHOS_RISKY_MODE_STATE_PATH` if non-empty (authoritative)
/// 2. `<dirs::config_dir>/neoethos/risky_mode_state.json`
/// 3. `<cwd>/.local/neoethos/risky_mode_state.json`
///
/// Returns the first candidate that EXISTS. If none exists, returns
/// the preferred candidate so the caller can create it.
pub fn state_file_path() -> Result<PathBuf> {
    // The env override matches the broker_persistence pattern — when
    // set it bypasses both existing-file lookup AND the fallback
    // chain so tests target an isolated temp path without silently
    // falling through to the operator's real config dir.
    //
    // **F-CORE3 closure (2026-05-25)**: routed through the canonical
    // `env_overrides::risky_mode_state_path_override` typed getter.
    if let Some(custom) = crate::app_services::env_overrides::risky_mode_state_path_override() {
        return Ok(PathBuf::from(custom));
    }

    let candidates = candidate_paths()?;

    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }

    candidates
        .into_iter()
        .next()
        .context("no candidate path could be resolved for risky mode state")
}

fn candidate_paths() -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(2);

    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join(APP_CONFIG_SUBDIR).join(STATE_FILENAME));
    }

    if let Ok(cwd) = env::current_dir() {
        paths.push(
            cwd.join(".local")
                .join(APP_CONFIG_SUBDIR)
                .join(STATE_FILENAME),
        );
    }

    if paths.is_empty() {
        anyhow::bail!("unable to determine risky mode state path on this platform");
    }
    Ok(paths)
}

/// Persist the Risky Mode kill-switch state to disk. Stamps
/// `last_updated_utc_ms` to the current wall clock before writing so
/// staleness checks work.
///
/// Idempotent: a re-save with the same field values rewrites the
/// file in place. Uses the M07 atomic writer (temp + fsync + rename)
/// so a crash mid-save can never leave a torn JSON on disk.
// **F-231/F-501/F-630 closure (2026-05-25)**: `save_risky_mode_state`
// is now USED by `record_kill_switch_trip` and `clear_expired_kill_switch`
// below, so the old `#[allow(dead_code)]` is gone. Production write path is live.
pub fn save_risky_mode_state(state: &RiskyModeStateFile) -> Result<()> {
    let path = state_file_path().context("resolve risky mode state path")?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent dir {}", parent.display()))?;
    }

    let mut to_write = state.clone();
    to_write.schema_version = RISKY_MODE_STATE_SCHEMA_VERSION;
    to_write.last_updated_utc_ms = current_unix_ms();

    let serialised =
        serde_json::to_string_pretty(&to_write).context("serialise risky mode state to JSON")?;
    neoethos_core::storage::json::write_bytes_atomic(&path, serialised.as_bytes())
        .with_context(|| format!("write risky mode state to {}", path.display()))?;

    tracing::info!(
        target: "neoethos_app::risky_mode_persistence",
        path = %path.display(),
        schema_version = %to_write.schema_version,
        cooldown_active = to_write.last_killed_at_utc_ms.is_some(),
        "persisted Risky Mode kill-switch state"
    );

    Ok(())
}

/// **F-231/F-501/F-630 closure (2026-05-25)** — record a kill-switch
/// trip on the persistent state file.
///
/// **Two producers as of 2026-08-09** (this list was "one" until #238 landed;
/// keep it accurate — a reader who believes there is only one will reason
/// wrongly about why a cooldown appeared):
/// 1. `live_trading::run` (`:1967-1985`) — an account-level tier of
///    `RiskyModeManager::check_trade_allowed` returned `Err`
///    (`PerDay` / `PerStage` / `PerMonth`), i.e. the bankroll is in trouble.
/// 2. `app_services::margin_call::arm` — the broker reports a MARGIN CALL, or
///    has been unreachable long enough that the account's margin level is
///    unknown. This is the `HardwareConnLoss`-class halt that
///    `live_trading.rs:263` recorded as having "no producer".
///
/// Note the asymmetry, deliberately stated: the cooldown this writes is read
/// by `live_trading` only under `trading_mode_risky` (`:1597`). Producer (2)
/// therefore ALSO holds a mode-independent, process-local halt at the broker
/// boundary (`broker_api::prepare_new_order`), because a margin call must stop
/// prop-firm-mode entries too.
///
/// Effect:
/// - `last_killed_at_utc_ms = Some(now)` (24 h cooldown clock starts)
///
/// The background task in `server::bridge::run` checks
/// `cooldown_expired` every 5 s; once the 24 h cooldown elapses it
/// calls `clear_expired_kill_switch` (below) to clear
/// `last_killed_at_utc_ms` automatically.
///
/// **2026-08-09 (W3) — the `#[allow(dead_code)]` is GONE, and that is the
/// point.** The attribute used to read "called only by the autonomous risk
/// gate ... which is Phase 2-5 pending", and it was the compiler-verified
/// proof that nothing could start the cooldown clock. Meanwhile
/// `bridge.rs:240` polled every 5 s and `server/risk.rs`
/// rendered a cooldown the operator could never see move. The writer is now
/// wired: `live_trading::run` calls this on the account-level `Err` branch of
/// `RiskyModeManager::check_trade_allowed`. Do NOT re-add the attribute to
/// silence a warning — if this ever goes callerless again, that is a defect
/// to fix, not a lint to suppress.
pub fn record_kill_switch_trip() -> Result<()> {
    let mut state = load_risky_mode_state()
        .context("load risky mode state before kill-switch trip")?
        .unwrap_or_default();
    state.last_killed_at_utc_ms = Some(current_unix_ms());
    save_risky_mode_state(&state).context("persist kill-switch trip timestamp")?;
    tracing::warn!(
        target: "neoethos_app::risky_mode_persistence",
        killed_at_utc_ms = state.last_killed_at_utc_ms,
        cooldown_hours = RISKY_MODE_KILL_SWITCH_COOLDOWN_MS / (60 * 60 * 1000),
        "Risky Mode kill-switch tripped; 24h auto re-arm cooldown started"
    );
    Ok(())
}

/// **F-231/F-501/F-630 closure (2026-05-25)** — auto-rearm helper
/// called periodically from the bridge polling loop. Returns
/// `Ok(true)` when the cooldown has elapsed AND the state was
/// flipped; `Ok(false)` when no action is needed (no kill on
/// record, or cooldown still in progress).
///
/// Effect (when ready):
/// - `last_killed_at_utc_ms = None` (cooldown cleared)
///
/// Idempotent: calling repeatedly after re-arm returns `Ok(false)`.
/// The bridge task can poll every 5 s without worrying about
/// double-flips.
pub fn clear_expired_kill_switch() -> Result<bool> {
    let Some(mut state) =
        load_risky_mode_state().context("load risky mode state for auto re-arm check")?
    else {
        // No state file → no kill on record → nothing to re-arm.
        return Ok(false);
    };
    let now = current_unix_ms();
    if !state.cooldown_expired(now) {
        return Ok(false);
    }
    state.last_killed_at_utc_ms = None;
    save_risky_mode_state(&state).context("persist expired kill-switch cleanup")?;
    tracing::warn!(
        target: "neoethos_app::risky_mode_persistence",
        cooldown_cleared_at_utc_ms = now,
        "Risky Mode kill-switch cooldown expired and was cleared"
    );
    Ok(true)
}

/// Seconds still to run on the Risky Mode kill-switch cooldown, or `None` when
/// there is no kill on record / the 24 h has already elapsed.
///
/// **2026-08-09 (W3)**: this is the read side of the halt that
/// [`record_kill_switch_trip`] writes, and it is what makes the trip survive an
/// app restart. The in-process [`neoethos_core::domain::risky_mode::RiskyModeManager`]
/// loses its accumulators and its sticky halt when the process dies; the
/// persisted timestamp does not. `live_trading::run` consults this before every
/// Risky-Mode entry, so a tripped kill switch keeps refusing entries across
/// restarts until `clear_expired_kill_switch` (bridge poll, 5 s) clears it 24 h later.
///
/// **FAIL-CLOSED (2026-08-09 correction).** This used to be
/// `load_risky_mode_state().ok().flatten()`, and `.ok()` decided in favour of
/// trading: a corrupt, truncated, permission-denied or unparseable
/// `risky_mode_state.json` turned an ACTIVE 24 h halt into `None`, which
/// `live_trading` reads as "not halted" and lets the entry through. It was the
/// only place in this wave that resolved an error toward SEND.
///
/// "We cannot tell whether the kill switch is tripped" now means REFUSE: an
/// unreadable state file reports a full cooldown remaining, loudly. `Ok(None)`
/// — genuinely no file, the first-run case — still permits trading, which is
/// the only path that should.
pub fn kill_switch_cooldown_remaining_secs() -> Option<u64> {
    match load_risky_mode_state() {
        Ok(state) => state.and_then(|s| s.cooldown_remaining_secs(current_unix_ms())),
        Err(e) => {
            tracing::error!(
                target: "neoethos_app::risky_mode_persistence",
                error = %e,
                "the Risky Mode state file could not be read, so it is UNKNOWN whether \
                 the kill switch is tripped — treating it as TRIPPED and refusing \
                 Risky-Mode entries. Fix or delete the file to clear this."
            );
            Some(RISKY_MODE_KILL_SWITCH_COOLDOWN_MS as u64 / 1000)
        }
    }
}

/// Load the Risky Mode arm state from disk.
///
/// Returns `Ok(None)` when no file exists yet — the safe default is
/// "Risky Mode disabled". Returns `Ok(Some(_))` when a file exists
/// and its schema version is readable. Returns `Err` only on actual
/// IO / parse errors so the caller can decide whether to fall back
/// to disabled or surface the error to the operator.
pub fn load_risky_mode_state() -> Result<Option<RiskyModeStateFile>> {
    let path = state_file_path().context("resolve risky mode state path")?;

    if !path.is_file() {
        tracing::debug!(
            target: "neoethos_app::risky_mode_persistence",
            path = %path.display(),
            "no risky mode state file found; treating as disabled"
        );
        return Ok(None);
    }

    let contents = fs::read_to_string(&path)
        .with_context(|| format!("read risky mode state from {}", path.display()))?;

    let state: RiskyModeStateFile = serde_json::from_str(&contents)
        .with_context(|| format!("parse risky mode state at {}", path.display()))?;

    if let Err(err) = neoethos_core::check_schema_version_readable(&state, "risky_mode_state.json")
    {
        tracing::error!(
            target: "neoethos_app::risky_mode_persistence",
            path = %path.display(),
            error = %err,
            "risky_mode_state.json schema version mismatch; treating as disabled"
        );
        return Ok(None);
    }

    tracing::info!(
        target: "neoethos_app::risky_mode_persistence",
        path = %path.display(),
        schema_version = %state.schema_version,
        cooldown_active = state.last_killed_at_utc_ms.is_some(),
        "loaded Risky Mode kill-switch state from disk"
    );

    Ok(Some(state))
}

// Used by save_risky_mode_state to stamp last_updated_utc_ms, by
// record_kill_switch_trip / clear_expired_kill_switch, and by
// kill_switch_cooldown_remaining_secs.
fn current_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests are serialised via this lock because `state_file_path`
    /// reads a process-wide env var. Without the lock, parallel test
    /// runs would race on `$NEOETHOS_RISKY_MODE_STATE_PATH`.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn unique_temp_state_path(label: &str) -> PathBuf {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("neoethos-rms-{label}-{pid}-{ts}.json"))
    }

    #[test]
    fn save_and_load_roundtrip() {
        let _g = ENV_LOCK.lock().unwrap();
        let path = unique_temp_state_path("roundtrip");
        // SAFETY: tests are serialised through ENV_LOCK; this is the
        // only block that touches `$NEOETHOS_RISKY_MODE_STATE_PATH`
        // during the test's lifetime.
        unsafe {
            std::env::set_var(ENV_OVERRIDE_VAR, &path);
        }

        let mut state = RiskyModeStateFile {
            last_killed_at_utc_ms: Some(123_456),
            ..Default::default()
        };
        // last_updated_utc_ms gets stamped on save; assert the round
        // trip preserves the other fields regardless.
        save_risky_mode_state(&state).expect("save");
        let loaded = load_risky_mode_state()
            .expect("load")
            .expect("file present");

        // Save stamps the schema version + timestamp, so update the
        // expected fields before comparing.
        state.schema_version = RISKY_MODE_STATE_SCHEMA_VERSION;
        state.last_updated_utc_ms = loaded.last_updated_utc_ms;
        assert_eq!(loaded, state);

        unsafe {
            std::env::remove_var(ENV_OVERRIDE_VAR);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_file_loads_as_none() {
        let _g = ENV_LOCK.lock().unwrap();
        let path = unique_temp_state_path("missing");
        unsafe {
            std::env::set_var(ENV_OVERRIDE_VAR, &path);
        }
        // No file at `path` — the loader must return Ok(None), not
        // Err. The safe default ("Risky Mode disabled") flows from
        // None at the call site.
        let loaded = load_risky_mode_state().expect("load");
        assert!(loaded.is_none());

        unsafe {
            std::env::remove_var(ENV_OVERRIDE_VAR);
        }
    }

    #[test]
    fn missing_schema_version_field_defaults_to_v1() {
        // A file written by a pre-versioning build (no
        // `schema_version` field at all) must still load — the
        // `#[serde(default = "default_v1")]` attribute and the
        // matching #[serde(default)] elsewhere together preserve
        // backward compatibility.
        let _g = ENV_LOCK.lock().unwrap();
        let path = unique_temp_state_path("pre-versioning");
        unsafe {
            std::env::set_var(ENV_OVERRIDE_VAR, &path);
        }
        fs::write(
            &path,
            br#"{"armed": true, "ruin_ceiling_acknowledged": 0.99}"#,
        )
        .unwrap();

        let loaded = load_risky_mode_state()
            .expect("load")
            .expect("file present");
        assert_eq!(loaded.schema_version, SchemaVersion::new(1));
        assert_eq!(loaded.last_killed_at_utc_ms, None);

        save_risky_mode_state(&loaded).expect("rewrite legacy state");
        let rewritten = fs::read_to_string(&path).expect("read compacted state");
        assert!(!rewritten.contains("armed"));
        assert!(!rewritten.contains("ruin_ceiling_acknowledged"));
        assert!(rewritten.contains("\"schema_version\": 3"));

        unsafe {
            std::env::remove_var(ENV_OVERRIDE_VAR);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn malformed_json_surfaces_error() {
        let _g = ENV_LOCK.lock().unwrap();
        let path = unique_temp_state_path("malformed");
        unsafe {
            std::env::set_var(ENV_OVERRIDE_VAR, &path);
        }
        fs::write(&path, b"{ this is not json").unwrap();

        let result = load_risky_mode_state();
        assert!(result.is_err(), "malformed JSON must surface an error");

        unsafe {
            std::env::remove_var(ENV_OVERRIDE_VAR);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn future_schema_version_loads_as_none_with_log() {
        // A file with a schema version newer than this build's
        // MAX_READABLE (= CURRENT) must NOT crash the app. It
        // logs an error and treats Risky Mode as disabled until the
        // operator updates the app.
        let _g = ENV_LOCK.lock().unwrap();
        let path = unique_temp_state_path("future-version");
        unsafe {
            std::env::set_var(ENV_OVERRIDE_VAR, &path);
        }
        fs::write(&path, br#"{"schema_version": 9999, "armed": true}"#).unwrap();

        let loaded = load_risky_mode_state().expect("load");
        assert!(
            loaded.is_none(),
            "schema version from the future must fall back to None"
        );

        unsafe {
            std::env::remove_var(ENV_OVERRIDE_VAR);
        }
        let _ = fs::remove_file(&path);
    }
}
