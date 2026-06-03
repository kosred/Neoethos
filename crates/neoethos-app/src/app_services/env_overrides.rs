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
//! ## Migration plan
//!
//! Phase A (this commit) — registry created with the highest-impact knobs.
//! Phase B — remaining call sites swap their inline reads for the
//!   typed getters. Each migration is mechanical + behaviour-preserving.

use std::env;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

// ---------------------------------------------------------------------------
// Env-var names — canonical string constants
// ---------------------------------------------------------------------------

/// HTTP server bind address (`host:port`). When unset, falls back to
/// `127.0.0.1:7423`. Read by `server::serve` at startup.
pub const ENV_SERVER_BIND: &str = "NEOETHOS_SERVER_BIND";

/// Maximum TCP read time (seconds) for `execute_via_session`. 0 disables
/// the timeout. Clamped to `[0, 3600]`; default 30s.
pub const ENV_CTRADER_READ_TIMEOUT_SECS: &str = "NEOETHOS_BOT_CTRADER_READ_TIMEOUT_SECS";

/// Maximum attempts (initial + retries) per cTrader execution call.
/// Clamped to `[1, 5]`; default 3. Retry safety relies on the broker
/// deduping by `clientOrderId`.
pub const ENV_CTRADER_MAX_ATTEMPTS: &str = "NEOETHOS_BOT_CTRADER_MAX_ATTEMPTS";

/// Base backoff (ms) for cTrader retries; doubles per attempt with
/// 0-99ms jitter, capped at 5s total. Clamped to `[10, 2000]`; default 200.
pub const ENV_CTRADER_BACKOFF_BASE_MS: &str = "NEOETHOS_BOT_CTRADER_BACKOFF_BASE_MS";

/// Whether partial fills are accepted as final (`1`/`true`/`yes` → on).
/// Default off — partial fills error out so the risk-per-trade math
/// stays consistent.
pub const ENV_CTRADER_ALLOW_PARTIAL_FILL: &str = "NEOETHOS_BOT_CTRADER_ALLOW_PARTIAL_FILL";

/// Maximum attempts for the streaming chart-update poll. Clamped
/// `[1, 5]`; default 3. Stateless polls are safe to retry.
pub const ENV_CTRADER_STREAM_MAX_ATTEMPTS: &str = "NEOETHOS_BOT_CTRADER_STREAM_MAX_ATTEMPTS";

/// Base backoff (ms) for the streaming layer. Clamped `[10, 2000]`;
/// default 200.
pub const ENV_CTRADER_STREAM_BACKOFF_BASE_MS: &str = "NEOETHOS_BOT_CTRADER_STREAM_BACKOFF_BASE_MS";

/// Quote side (`mid` / `bid` / `ask`) used for chart-merge when a
/// single price is required (e.g. latest-close display). Default `mid`.
pub const ENV_CHART_MERGE_SIDE: &str = "NEOETHOS_BOT_CHART_MERGE_SIDE";

/// PnL drift threshold (fraction of notional) above which an audit
/// warning is logged. Clamped `[1e-5, 0.05]`; default 0.001 (10bp).
pub const ENV_PNL_AUDIT_DRIFT_FRACTION: &str = "NEOETHOS_BOT_PNL_AUDIT_DRIFT_FRACTION";

/// PnL drift threshold (fraction of notional) that halts the auto-trader.
/// Clamped `[1e-4, 0.20]` so the breaker cannot be silenced by a typo.
/// Default 0.01 (1%).
pub const ENV_PNL_CIRCUIT_BREAKER_FRACTION: &str = "NEOETHOS_BOT_PNL_CIRCUIT_BREAKER_FRACTION";

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
// Defaults — exported as `pub const` so the catalog + tests don't drift
// ---------------------------------------------------------------------------

pub const DEFAULT_BIND_ADDR: SocketAddr =
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 7423);
pub const DEFAULT_CTRADER_READ_TIMEOUT_SECS: u64 = 30;
pub const DEFAULT_CTRADER_MAX_ATTEMPTS: u32 = 3;
pub const DEFAULT_CTRADER_BACKOFF_BASE_MS: u64 = 200;
pub const DEFAULT_PNL_AUDIT_DRIFT_FRACTION: f64 = 0.001;
pub const DEFAULT_PNL_CIRCUIT_BREAKER_FRACTION: f64 = 0.01;

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

/// Maximum streaming poll attempts. Clamped `[1, 5]`.
pub fn ctrader_stream_max_attempts() -> u32 {
    app_runtime().ctrader_stream_max_attempts.clamp(1, 5)
}

/// Streaming retry backoff base (ms). Clamped `[10, 2000]`.
pub fn ctrader_stream_backoff_base_ms() -> u64 {
    app_runtime().ctrader_stream_backoff_base_ms.clamp(10, 2_000)
}

/// Chart-merge quote side (`mid`/`bid`/`ask`); `None` when the configured
/// value is empty (the caller then uses its own default).
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

/// Live-journal path override. `None` when unset → callers fall back to
/// the platform user-data-dir default.
pub fn live_journal_path_override() -> Option<String> {
    env::var(ENV_LIVE_JOURNAL_PATH)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Pending-actions path override. `None` when unset.
pub fn pending_actions_path_override() -> Option<String> {
    env::var(ENV_PENDING_ACTIONS_PATH)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Risky-Mode persistence path override. `None` when unset.
pub fn risky_mode_state_path_override() -> Option<String> {
    env::var(ENV_RISKY_MODE_STATE_PATH)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Real-data fixture capture directory. `None` = capture disabled
/// (production default). When `Some(dir)`, the cTrader parser layer
/// writes each parsed response to
/// `<dir>/<message_type>_<unix_ms>.bin` so the
/// `ctrader_test_fixtures` loader can replay them in future tests.
pub fn capture_fixtures_dir() -> Option<String> {
    env::var(ENV_CAPTURE_FIXTURES_DIR)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Write a captured proto-payload to the configured fixture
/// directory. No-op when `NEOETHOS_CAPTURE_FIXTURES_DIR` is unset
/// (production default — zero overhead on the hot path).
///
/// **Usage** — call from any cTrader message-parser site after
/// successfully decoding a `ProtoOA*` response:
/// ```rust
/// use crate::app_services::env_overrides::capture_fixture;
/// capture_fixture("ProtoOADealListRes", raw_bytes);
/// ```
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
        assert_eq!(ENV_SERVER_BIND, "NEOETHOS_SERVER_BIND");
        assert_eq!(
            ENV_CTRADER_READ_TIMEOUT_SECS,
            "NEOETHOS_BOT_CTRADER_READ_TIMEOUT_SECS"
        );
        assert_eq!(ENV_CTRADER_MAX_ATTEMPTS, "NEOETHOS_BOT_CTRADER_MAX_ATTEMPTS");
        assert_eq!(
            ENV_CTRADER_BACKOFF_BASE_MS,
            "NEOETHOS_BOT_CTRADER_BACKOFF_BASE_MS"
        );
        assert_eq!(
            ENV_CTRADER_ALLOW_PARTIAL_FILL,
            "NEOETHOS_BOT_CTRADER_ALLOW_PARTIAL_FILL"
        );
        assert_eq!(
            ENV_CTRADER_STREAM_MAX_ATTEMPTS,
            "NEOETHOS_BOT_CTRADER_STREAM_MAX_ATTEMPTS"
        );
        assert_eq!(
            ENV_CTRADER_STREAM_BACKOFF_BASE_MS,
            "NEOETHOS_BOT_CTRADER_STREAM_BACKOFF_BASE_MS"
        );
        assert_eq!(ENV_CHART_MERGE_SIDE, "NEOETHOS_BOT_CHART_MERGE_SIDE");
        assert_eq!(
            ENV_PNL_AUDIT_DRIFT_FRACTION,
            "NEOETHOS_BOT_PNL_AUDIT_DRIFT_FRACTION"
        );
        assert_eq!(
            ENV_PNL_CIRCUIT_BREAKER_FRACTION,
            "NEOETHOS_BOT_PNL_CIRCUIT_BREAKER_FRACTION"
        );
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
        assert_eq!(DEFAULT_CTRADER_READ_TIMEOUT_SECS, 30);
        assert_eq!(DEFAULT_CTRADER_MAX_ATTEMPTS, 3);
        assert_eq!(DEFAULT_CTRADER_BACKOFF_BASE_MS, 200);
        assert_eq!(DEFAULT_PNL_AUDIT_DRIFT_FRACTION, 0.001);
        assert_eq!(DEFAULT_PNL_CIRCUIT_BREAKER_FRACTION, 0.01);
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
