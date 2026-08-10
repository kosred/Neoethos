//! Centralised env-var overrides for `neoethos-app`.
//!
//! Mirror of `neoethos_core::env_overrides` for the app crate's
//! own knobs. Single grep-able file for every `NEOETHOS_BOT_CTRADER_*`,
//! `NEOETHOS_BOT_PNL_*`, and `NEOETHOS_*` runtime override that the
//! HTTP server / trading layer honours.
//!
//! ## Why this exists (F-CORE3 cluster consolidation, 2026-05-25)
//!
//! Before this module, the app crate spread `std::env::var(...)` reads
//! across at least 10 files: `ctrader_execution.rs`, `ctrader_streaming.rs`,
//! `ctrader_messages.rs`, `pnl.rs`, `server/mod.rs`, `live_journal.rs`,
//! `pending_actions.rs`, `risky_mode_persistence.rs`, etc. Each had its
//! own local clamping helper, which:
//!
//! - **Made auditing painful** — no single place to see what knobs exist.
//! - **Duplicated parse logic** — same `parse + clamp + fallback` pattern repeated.
//! - **Diverged on tolerances** — one site clamps `[1, 5]`, another `[1, 10]`
//!   for the same conceptual knob.
//!
//! This module is the canonical registry. Each entry has:
//!
//! - A `pub const NAME: &str` for the env-var name (grep-able from one place).
//! - A typed getter `fn() -> Option<T>` (or `fn() -> T` when a clamped
//!   default is the right semantics) that parses + validates.
//! - A doc-comment explaining what the var controls and what unset means.
//!
//! Call sites elsewhere in the crate import these getters / constants
//! rather than calling `std::env::var(...)` directly.
//!
//! ## Migration COMPLETE — 2026-08-10
//!
//! The file name is now historical. Every BEHAVIOURAL knob here reads
//! `neoethos_core::config::AppRuntimeConfig`, installed once at startup from
//! the single `Settings` — there is no environment involved. What remains on
//! the environment is exactly one thing, in exactly one function
//! (`fixture_path_override`): the three test/CI path redirections and the
//! one-shot broker-payload capture directory, each of which now announces
//! itself at WARN the first time it is used.
//!
//! Retired names are NOT read here and NOT silently ignored either — they are
//! listed in `app_services::retired_env`, which prints an ERROR naming the
//! variable and its replacement config key when one is still exported.

use std::env;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

// ---------------------------------------------------------------------------
// Env-var names — canonical string constants
// ---------------------------------------------------------------------------
//
// 2026-08-08 dead-code purge: the 10 catalog-only ENV_* name constants
// (SERVER_BIND, the six CTRADER_* retry/stream knobs, CHART_MERGE_SIDE and
// the two PNL_* thresholds) were deleted together with their 5 DEFAULT_*
// twins. The runtime values are read through the AppRuntimeConfig OnceLock
// (config-consolidation S3-app), and the only remaining references were this
// file's own test assertions — the claimed "drift-guard" did not exist.
//
// 2026-08-10 update: `knob_catalog.rs` used to repeat those names as its own
// string literals in an `env_var` field. That field is deleted (W2-7); the
// names now live in `app_services::retired_env` beside the config key that
// replaced each one, and are printed only when one is still exported.
//
// The four PATH-override constants below stay live: their getters are the
// test/CI call sites, and each announces itself at WARN when used.

/// Override path for the live trading journal. Test/CI use; production
/// reads from the platform user-data-dir.
pub const ENV_LIVE_JOURNAL_PATH: &str = "NEOETHOS_BOT_LIVE_JOURNAL_PATH";

/// Override path for the pending-actions store. Test/CI use.
pub const ENV_PENDING_ACTIONS_PATH: &str = "NEOETHOS_PENDING_ACTIONS_PATH";

/// Override path for the Risky Mode persistence file. Test/CI use.
pub const ENV_RISKY_MODE_STATE_PATH: &str = "NEOETHOS_RISKY_MODE_STATE_PATH";

/// **2026-05-25 — real-data fixture capture** (operator directive
/// "ότι άλλο υπάρχει ανοιχτό μην αφήσουμε τίποτα"). When set to a
/// directory path, the cTrader message-parsing layer writes every
/// parsed `ProtoOA*` response payload to that directory as
/// `<message_type>_<unix_ms>.bin`. The operator runs the app once
/// with this env var set, performs the operations that the
/// `TODO(real-data)` tests need a fixture for (place a market order,
/// fetch positions, etc.), and the captured payloads appear on
/// disk. The previously-`#[ignore]`'d tests then load them via
/// `ctrader_test_fixtures::load_captured`.
///
/// Unset = capture is OFF (the default in production).
pub const ENV_CAPTURE_FIXTURES_DIR: &str = "NEOETHOS_CAPTURE_FIXTURES_DIR";

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

pub const DEFAULT_BIND_ADDR: SocketAddr =
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 7423);

// ---------------------------------------------------------------------------
// Config-driven cache (config-consolidation S3-app)
// ---------------------------------------------------------------------------
//
// The behavior getters below now read a
// `neoethos_core::config::AppRuntimeConfig` installed once at startup from the
// single `Settings` — NOT `std::env`. The `ENV_*` consts above are retained
// only for documentation + the knob catalog's `env_var` field. Clamping stays
// in the getters (same bounds the env readers used) so an out-of-range config
// value can't wedge the trading loop. The PATH overrides further down stay on
// env (test/CI fixtures, like the broker-credentials path).

static APP_RUNTIME: std::sync::OnceLock<neoethos_core::config::AppRuntimeConfig> =
    std::sync::OnceLock::new();

/// Install the app/server/trading runtime config once at startup. The binary
/// passes `settings.app_runtime`. Idempotent — the first install wins.
pub fn install_app_runtime_overrides(cfg: neoethos_core::config::AppRuntimeConfig) {
    let _ = APP_RUNTIME.set(cfg);
}

/// The installed app-runtime config, or the deterministic defaults when no
/// install has happened (tests / very early startup).
fn app_runtime() -> neoethos_core::config::AppRuntimeConfig {
    APP_RUNTIME.get().cloned().unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Typed getters
// ---------------------------------------------------------------------------

/// Resolve the HTTP server bind address from config. Logs a `tracing::warn!`
/// when the configured value is non-empty but unparseable, then falls back.
pub fn server_bind_addr() -> SocketAddr {
    let raw = app_runtime().server_bind;
    if let Ok(parsed) = raw.parse::<SocketAddr>() {
        return parsed;
    }
    if !raw.trim().is_empty() {
        tracing::warn!(
            target: "neoethos_app::env_overrides",
            raw = %raw,
            fallback = %DEFAULT_BIND_ADDR,
            "app_runtime.server_bind unparseable; falling back to default"
        );
    }
    DEFAULT_BIND_ADDR
}

/// cTrader read-timeout (seconds). 0 disables the timeout. Clamped to
/// `[0, 3600]` so a typo can't wedge the trading loop indefinitely.
pub fn ctrader_read_timeout_secs() -> u64 {
    app_runtime().ctrader_read_timeout_secs.min(3600)
}

/// Maximum cTrader execution attempts. Clamped `[1, 5]`.
pub fn ctrader_max_attempts() -> u32 {
    app_runtime().ctrader_max_attempts.clamp(1, 5)
}

/// cTrader retry backoff base (ms). Clamped `[10, 2000]`.
pub fn ctrader_backoff_base_ms() -> u64 {
    app_runtime().ctrader_backoff_base_ms.clamp(10, 2_000)
}

/// Whether partial fills are accepted. Default `false`.
pub fn ctrader_allow_partial_fill() -> bool {
    app_runtime().ctrader_allow_partial_fill
}

/// Maximum bar-fetch attempts in the live trading loop. Clamped `[1, 5]`.
pub fn ctrader_stream_max_attempts() -> u32 {
    app_runtime().ctrader_stream_max_attempts.clamp(1, 5)
}

/// Backoff base (ms) for the live trading bar-fetch retries and spot-stream
/// reconnect. Clamped `[10, 2000]`.
pub fn ctrader_stream_backoff_base_ms() -> u64 {
    app_runtime().ctrader_stream_backoff_base_ms.clamp(10, 2_000)
}

/// Chart-merge quote side (`mid`/`bid`/`ask`); `None` when the configured
/// value is empty (the caller then uses its own default).
/// `dead_code` until the live chart-merge display path is wired.
#[allow(dead_code)]
pub fn chart_merge_side_raw() -> Option<String> {
    let v = app_runtime().chart_merge_side.trim().to_ascii_lowercase();
    if v.is_empty() { None } else { Some(v) }
}

/// PnL audit drift threshold. Clamped `[1e-5, 0.05]`.
pub fn pnl_audit_drift_fraction() -> f64 {
    app_runtime().pnl_audit_drift_fraction.clamp(1e-5, 0.05)
}

/// PnL circuit-breaker threshold. Clamped `[1e-4, 0.20]`.
pub fn pnl_circuit_breaker_fraction() -> f64 {
    app_runtime().pnl_circuit_breaker_fraction.clamp(1e-4, 0.20)
}

// ---------------------------------------------------------------------------
// Test/CI fixture path overrides — the ONE env read left in this module
// ---------------------------------------------------------------------------
//
// 2026-08-10, config consolidation. The four getters below used to make four
// separate `env::var` calls, each silently returning a path. That is the
// pattern non-negotiable #3 names: a substitution wearing the costume of a
// choice. If one of these were set in a production shell — say a leftover
// `NEOETHOS_BOT_LIVE_JOURNAL_PATH` from a debugging session — the live
// trading journal would be written somewhere the operator was not looking,
// and NOTHING would say so. The journal is the record every P&L number is
// reconstructed from.
//
// They are not promoted to config keys, because they are not operator
// configuration: they exist so a test can point a store at a temp directory
// without racing the real one, and `NEOETHOS_CAPTURE_FIXTURES_DIR` exists so
// the operator can run the app ONCE to capture real broker payloads. A
// config key would invite permanent use of a facility that is meant to be
// momentary.
//
// So they stay on the environment, and instead they SHOUT. One call site,
// one WARN per variable per process, naming the variable and the path it
// substituted. A relocated journal is now impossible to miss in the log.

/// Read a fixture path override and announce the substitution once.
///
/// `purpose` completes the sentence "…is redirecting <purpose> to". Returns
/// the trimmed value, or `None` when unset or blank (blank is treated as
/// unset — an exported-but-empty variable is an accident, not an instruction
/// to use the empty path).
fn fixture_path_override(var: &'static str, purpose: &'static str) -> Option<String> {
    // The single `env::var` in this module. Everything above reads
    // `AppRuntimeConfig`, which comes from the one config file.
    let value = env::var(var).ok()?;
    let value = value.trim().to_string();
    if value.is_empty() {
        return None;
    }
    say_once(var, || {
        tracing::warn!(
            target: "neoethos_app::env_overrides",
            env_var = var,
            path = %value,
            "TEST/CI FIXTURE OVERRIDE ACTIVE: `{var}` is set, so {purpose} is \
             being read from and written to `{value}` instead of the platform \
             user-data-dir. This is a test facility. If you did not mean to set \
             it, unset it and restart — production state written under this path \
             will not be found by a normal launch."
        );
    });
    Some(value)
}

/// Emit `f` the first time this process sees `key`. Keeps the warning honest
/// (it names a real substitution) without turning a per-call getter into
/// log wallpaper — several of these are consulted on every store access.
fn say_once(key: &'static str, f: impl FnOnce()) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    // A poisoned mutex must never silence a substitution notice.
    let mut guard = match seen.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.insert(key) {
        drop(guard);
        f();
    }
}

/// Live-journal path override. `None` when unset → callers fall back to
/// the platform user-data-dir default.
pub fn live_journal_path_override() -> Option<String> {
    fixture_path_override(ENV_LIVE_JOURNAL_PATH, "the LIVE TRADING JOURNAL")
}

/// Pending-actions path override. `None` when unset.
pub fn pending_actions_path_override() -> Option<String> {
    fixture_path_override(ENV_PENDING_ACTIONS_PATH, "the pending-actions store")
}

/// Risky-Mode persistence path override. `None` when unset.
pub fn risky_mode_state_path_override() -> Option<String> {
    fixture_path_override(
        ENV_RISKY_MODE_STATE_PATH,
        "the Risky Mode kill-switch state",
    )
}

/// Real-data fixture capture directory. `None` = capture disabled
/// (production default). When `Some(dir)`, the cTrader parser layer
/// writes each parsed response to
/// `<dir>/<message_type>_<unix_ms>.bin` so the
/// `ctrader_test_fixtures` loader can replay them in future tests.
pub fn capture_fixtures_dir() -> Option<String> {
    fixture_path_override(
        ENV_CAPTURE_FIXTURES_DIR,
        "cTrader payload capture (every parsed broker response written to disk)",
    )
}

/// Write a captured proto-payload to the configured fixture
/// directory. No-op when `NEOETHOS_CAPTURE_FIXTURES_DIR` is unset
/// (production default — zero overhead on the hot path).
///
/// **Usage** — call from any cTrader message-parser site after
/// successfully decoding a `ProtoOA*` response:
/// ```no_run
/// use neoethos_app::app_services::env_overrides::capture_fixture;
///
/// let raw_bytes: &[u8] = b"\x08\x01";
/// capture_fixture("ProtoOADealListRes", raw_bytes);
/// ```
///
/// This example COMPILES, and it did not before 2026-08-10: it said
/// `use crate::app_services::…` (a doctest is a separate crate, so `crate::`
/// is the doctest itself — E0433) and called an undeclared `raw_bytes`
/// (E0425). It is `no_run` rather than plain: type-checking the call is the
/// whole value, while executing it would go looking for the configured
/// capture directory from inside a doctest.
///
/// Errors are logged at `warn` level but never propagated — capture
/// is best-effort diagnostic, not a correctness contract. A failed
/// fixture write must NEVER block trading.
pub fn capture_fixture(message_type: &str, payload: &[u8]) {
    let Some(dir) = capture_fixtures_dir() else {
        return;
    };
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let safe_type: String = message_type
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let path = std::path::Path::new(&dir).join(format!("{safe_type}_{now_ms}.bin"));
    if let Some(parent) = path.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            tracing::warn!(
                target: "neoethos_app::env_overrides::capture_fixture",
                path = %parent.display(),
                error = %err,
                "capture-fixtures dir not creatable; skipping this payload"
            );
            return;
        }
    }
    if let Err(err) = std::fs::write(&path, payload) {
        tracing::warn!(
            target: "neoethos_app::env_overrides::capture_fixture",
            path = %path.display(),
            error = %err,
            "capture-fixtures write failed; skipping"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin every env-var constant so renames break loudly here. The
    /// operator's docs + Flutter Settings UI + the audit reference all
    /// rely on these names — silently renaming any one of them is a
    /// breaking change.
    #[test]
    fn env_var_names_are_stable() {
        assert_eq!(ENV_LIVE_JOURNAL_PATH, "NEOETHOS_BOT_LIVE_JOURNAL_PATH");
        assert_eq!(ENV_PENDING_ACTIONS_PATH, "NEOETHOS_PENDING_ACTIONS_PATH");
        assert_eq!(
            ENV_RISKY_MODE_STATE_PATH,
            "NEOETHOS_RISKY_MODE_STATE_PATH"
        );
        assert_eq!(
            ENV_CAPTURE_FIXTURES_DIR,
            "NEOETHOS_CAPTURE_FIXTURES_DIR"
        );
    }

    #[test]
    fn defaults_are_sensible() {
        // Sanity-check the defaults against operator-documented values.
        assert_eq!(
            DEFAULT_BIND_ADDR.to_string(),
            "127.0.0.1:7423"
        );
    }

    #[test]
    fn getters_with_default_config_match_documented_defaults() {
        // Config-consolidation S3-app behavior-preservation: with no install
        // (default config), the getters reproduce the legacy env-unset
        // defaults exactly. No test installs a non-default config, so the
        // process-wide OnceLock stays at Default here.
        assert_eq!(ctrader_read_timeout_secs(), 30);
        assert_eq!(ctrader_max_attempts(), 3);
        assert_eq!(ctrader_backoff_base_ms(), 200);
        assert_eq!(ctrader_stream_max_attempts(), 3);
        assert_eq!(ctrader_stream_backoff_base_ms(), 200);
        assert!(!ctrader_allow_partial_fill());
        assert!(chart_merge_side_raw().is_none());
        assert!((pnl_audit_drift_fraction() - 0.001).abs() < 1e-9);
        assert!((pnl_circuit_breaker_fraction() - 0.01).abs() < 1e-9);
        assert_eq!(server_bind_addr().to_string(), "127.0.0.1:7423");
    }
}
