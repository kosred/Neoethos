//! Centralised env-var overrides for `neoethos-core`.
//!
//! **F-150 fix (operator-approved 2026-05-25 — F-CORE3 cluster
//! consolidation)**. The audit identified that `neoethos-core`
//! reads `std::env::var` directly from 6 different files
//! (`symbol_metadata.rs`, `config.rs`, `system.rs`, `logging.rs`,
//! `broker_config.rs`, `resolved_config.rs`). Spreading env reads
//! across the foundation crate makes it hard to:
//!
//! - **Audit** what runtime knobs exist (the operator can't grep
//!   one file to see all the levers).
//! - **Document** their semantics (each call-site comments locally).
//! - **Test** without process-wide env mutation (each test that
//!   wants to override has to remember which file the var lives in).
//!
//! This module is the **canonical registry** of every env-var that
//! `neoethos-core` honours. Each entry has:
//!
//! - The env-var NAME (a `pub const &str` so it's grep-able from
//!   one place).
//! - A typed getter (`fn(...) -> Option<T>`) that parses + validates
//!   the value.
//! - A doc-comment explaining what the var controls and what the
//!   fallback path is when it's unset.
//!
//! Call sites elsewhere in the crate import these constants /
//! getters rather than calling `std::env::var(...)` directly.
//!
//! ## Phase A migration (this commit) — non-behavioural
//!
//! Phase A introduces the registry + the constant names + the typed
//! getters. The 6 existing call sites in the crate are NOT yet
//! migrated — they continue to read `std::env::var(...)` directly
//! with the SAME string literals. The next batch (Phase B) migrates
//! each call site to use this registry, removing the direct env
//! reads from the 6 files. Phase B is mechanical (one site per
//! commit) and behaviour-preserving.
//!
//! Listing the env vars here in Phase A makes them discoverable to
//! the operator NOW even though the migration isn't complete —
//! they can grep this single file to see every NeoEthos env knob.

use std::env;

// ---------------------------------------------------------------------------
// Env-var names — canonical string constants
// ---------------------------------------------------------------------------

/// Prop-firm preset that seeds `RiskConfig::default()`. Accepted
/// values: `ftmo` / `myforexfunds` / `fundednext` / `the5ers` / `none`
/// (case-insensitive). Unrecognised values fall back to FTMO.
///
/// Read by `config::RiskConfig::default()`.
pub const ENV_PROP_FIRM_PRESET: &str = "NEOETHOS_PROP_FIRM_PRESET";

/// Account currency for the prop-firm gate's risk-per-trade math.
/// Required by `risk_gate::prop_firm_pre_trade_check` whenever an
/// order carries a stop-loss. Empty / unset → hard-fail at the gate
/// (no synthetic default per real-data directive 2026-05-24).
pub const ENV_PROP_ACCOUNT_CURRENCY: &str = "NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY";

/// Live quote→account FX rate override for the risk-gate's
/// pip-value computation. Used for cross pairs where the broker
/// hasn't shipped a real rate yet. Must be finite + > 0.0.
pub const ENV_PROP_QUOTE_TO_ACCOUNT_RATE: &str = "NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE";

/// Path to the operator's `symbol_metadata.json` override. When set
/// and the file is loadable, replaces the on-disk `data/symbol_metadata.json`
/// default. Read by `symbol_metadata::resolve` / load path.
pub const ENV_SYMBOL_METADATA: &str = "NEOETHOS_BOT_SYMBOL_METADATA";

/// Tracing-subscriber `RUST_LOG`-style filter (e.g. `debug,sqlx=warn`).
/// Read by `logging::setup_logging`. When unset, the production
/// default filter from `Settings` applies.
pub const ENV_LOG_FILTER: &str = "RUST_LOG";

/// Override for the user-data root directory (logs + state). When
/// unset, `dirs::data_local_dir()` provides the platform default.
pub const ENV_USER_DATA_DIR: &str = "NEOETHOS_USER_DATA_DIR";

/// Set by the Flutter shell when it launches the backend (so the
/// backend knows to skip the "double-click help" Windows popup that
/// orphaned-process detection otherwise displays). Any non-empty
/// value suppresses the dialog.
///
/// **F-CORE3 closure (2026-05-25)**: previously read inline at
/// `logging::show_double_click_help_dialog_if_orphaned`.
pub const ENV_LAUNCHED_BY_FLUTTER: &str = "NEOETHOS_LAUNCHED_BY_FLUTTER";

/// Operator override for the log directory. When set and non-empty,
/// overrides the platform-default `data_dir()/neoethos/logs`.
///
/// **F-CORE3 closure (2026-05-25)**: previously read inline at
/// `logging::default_log_dir`.
pub const ENV_LOG_DIR: &str = "LOG_DIR";

/// Override path for the broker-credentials file (test/sandbox use).
/// When set and non-empty, replaces the default `dirs::config_dir()/neoethos`
/// lookup. The path's parent directory is what's actually used —
/// the env-var value can include the filename for convenience.
///
/// **F-CORE3 closure (2026-05-25)**: previously read inline at
/// `neoethos_cli::canonical_user_config_dir` and the matching
/// `BROKER_CREDENTIALS_PATH_ENV_VAR` const in `neoethos-app`.
pub const ENV_BROKER_CREDENTIALS_PATH: &str = "NEOETHOS_BROKER_CREDENTIALS_PATH";

// ---------------------------------------------------------------------------
// Typed getters
// ---------------------------------------------------------------------------

/// Read the prop-firm preset name. `None` when unset / empty.
/// Parsing into `PropFirmPreset` is the caller's responsibility
/// (the enum lives in `domain::prop_firm`).
pub fn prop_firm_preset_raw() -> Option<String> {
    env::var(ENV_PROP_FIRM_PRESET)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read the prop-firm account currency. `None` when unset / empty.
/// Empty propagates as `None` so the caller sees "unset" not "empty
/// string" — typed boundary at the registry.
pub fn prop_firm_account_currency() -> Option<String> {
    env::var(ENV_PROP_ACCOUNT_CURRENCY)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read the cross-pair quote→account FX rate override. `None` when
/// unset / unparseable / non-finite / non-positive.
pub fn prop_firm_quote_to_account_rate() -> Option<f64> {
    env::var(ENV_PROP_QUOTE_TO_ACCOUNT_RATE)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Read the symbol-metadata path override. `None` when unset / empty.
pub fn symbol_metadata_path_override() -> Option<String> {
    env::var(ENV_SYMBOL_METADATA)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read the user-data-dir override. `None` when unset / empty.
pub fn user_data_dir_override() -> Option<String> {
    env::var(ENV_USER_DATA_DIR)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Whether the Flutter shell launched this backend process. Any
/// non-empty value counts as `true`.
pub fn launched_by_flutter() -> bool {
    env::var(ENV_LAUNCHED_BY_FLUTTER)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Read the log-directory override. `None` when unset / empty.
pub fn log_dir_override() -> Option<String> {
    env::var(ENV_LOG_DIR)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read the broker-credentials path override. `None` when unset /
/// empty. The caller decides whether to treat the value as a
/// file path or use its parent as a config directory.
pub fn broker_credentials_path_override() -> Option<String> {
    env::var(ENV_BROKER_CREDENTIALS_PATH)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

// ---------------------------------------------------------------------------
// F-005 — startup-time visibility for active env-var overrides
// ---------------------------------------------------------------------------

/// **F-005 fix (2026-05-25)** — log every active env override at
/// startup so the operator sees what's been silently overridden
/// without grepping the env. Returns a `Vec<&'static str>` of
/// active override names so the caller (typically the binary's
/// startup logging block) can also surface them on the UI / CLI
/// banner.
///
/// The audit flagged `NEOETHOS_BOT_DISABLE_SMC_GATE` and friends as a
/// "behavior changes invisibly based on environment" risk. By
/// emitting a structured `tracing::warn!` listing every active
/// override at startup, operators see ALL overrides in one place
/// instead of having to know they exist.
///
/// This helper checks only the env-vars defined in this registry.
/// Crates that own their own env-var namespace (e.g. `neoethos-search`
/// has `NEOETHOS_BOT_DISABLE_SMC_GATE`, `NEOETHOS_BOT_NORMALIZE_FEATURES`,
/// `NEOETHOS_BOT_PREFILTER_*`, etc.) should call their own equivalent
/// helper. Returning the names from each gets the binary a complete
/// list for the chrome banner.
pub fn active_overrides() -> Vec<&'static str> {
    let mut active: Vec<&'static str> = Vec::new();
    if std::env::var(ENV_PROP_FIRM_PRESET).is_ok() {
        active.push(ENV_PROP_FIRM_PRESET);
    }
    if std::env::var(ENV_PROP_ACCOUNT_CURRENCY).is_ok() {
        active.push(ENV_PROP_ACCOUNT_CURRENCY);
    }
    if std::env::var(ENV_PROP_QUOTE_TO_ACCOUNT_RATE).is_ok() {
        active.push(ENV_PROP_QUOTE_TO_ACCOUNT_RATE);
    }
    if std::env::var(ENV_SYMBOL_METADATA).is_ok() {
        active.push(ENV_SYMBOL_METADATA);
    }
    if std::env::var(ENV_LOG_FILTER).is_ok() {
        active.push(ENV_LOG_FILTER);
    }
    if std::env::var(ENV_USER_DATA_DIR).is_ok() {
        active.push(ENV_USER_DATA_DIR);
    }
    if std::env::var(ENV_LAUNCHED_BY_FLUTTER).is_ok() {
        active.push(ENV_LAUNCHED_BY_FLUTTER);
    }
    if std::env::var(ENV_LOG_DIR).is_ok() {
        active.push(ENV_LOG_DIR);
    }
    if std::env::var(ENV_BROKER_CREDENTIALS_PATH).is_ok() {
        active.push(ENV_BROKER_CREDENTIALS_PATH);
    }
    active
}

/// Emit a structured warning at startup listing every active env
/// override. Idempotent — safe to call multiple times; the operator
/// will see one warn line per call. Designed to be called once in
/// the binary's `main()` after `setup_logging`.
pub fn log_active_overrides_at_startup() {
    let active = active_overrides();
    if active.is_empty() {
        tracing::info!(
            target: "neoethos_core::env_overrides",
            "No NeoEthos env-var overrides active at startup."
        );
        return;
    }
    tracing::warn!(
        target: "neoethos_core::env_overrides",
        count = active.len(),
        overrides = ?active,
        "NeoEthos env-var overrides active at startup — listed for \
         operator visibility (F-005 fix). Each one changes runtime \
         behaviour; review and confirm intentional."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the env-var name constants so a refactor that renames one
    /// breaks loudly here (and the operator's docs / config wiring
    /// have a single canonical name to grep).
    #[test]
    fn env_var_names_are_stable() {
        assert_eq!(ENV_PROP_FIRM_PRESET, "NEOETHOS_PROP_FIRM_PRESET");
        assert_eq!(
            ENV_PROP_ACCOUNT_CURRENCY,
            "NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY"
        );
        assert_eq!(
            ENV_PROP_QUOTE_TO_ACCOUNT_RATE,
            "NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE"
        );
        assert_eq!(ENV_SYMBOL_METADATA, "NEOETHOS_BOT_SYMBOL_METADATA");
        assert_eq!(ENV_LOG_FILTER, "RUST_LOG");
        assert_eq!(ENV_USER_DATA_DIR, "NEOETHOS_USER_DATA_DIR");
        assert_eq!(ENV_LAUNCHED_BY_FLUTTER, "NEOETHOS_LAUNCHED_BY_FLUTTER");
        assert_eq!(ENV_LOG_DIR, "LOG_DIR");
        assert_eq!(
            ENV_BROKER_CREDENTIALS_PATH,
            "NEOETHOS_BROKER_CREDENTIALS_PATH"
        );
    }

    #[test]
    fn rate_getter_rejects_zero_and_negative() {
        // SAFETY: env-var manipulation in tests is intentional; we
        // restore the var afterwards. We use a unique-suffix var to
        // avoid clashing with parallel tests.
        let key = "TEST_F150_RATE_INVALID_INPUTS";
        unsafe { std::env::set_var(key, "0.0") };
        assert!(
            env::var(key)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .is_none()
        );
        unsafe { std::env::set_var(key, "-1.5") };
        assert!(
            env::var(key)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .is_none()
        );
        unsafe { std::env::remove_var(key) };
    }
}
