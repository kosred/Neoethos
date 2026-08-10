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
//! ## Migration status — CORRECTED 2026-08-09 (audit #144)
//!
//! The header that stood here described "Phase A" as if NO call site
//! had been migrated. That stopped being true long ago and the stale
//! text made the registry look decorative. The real state, verified by
//! grep on 2026-08-09:
//!
//! | Var | Getter | Migrated call site |
//! |---|---|---|
//! | `NEOETHOS_USER_DATA_DIR` | `user_data_dir_override` | `config.rs:2521` — LIVE |
//! | `LOG_DIR` | `log_dir_override` | `logging.rs:508` — LIVE |
//! | `NEOETHOS_BROKER_CREDENTIALS_PATH` | `broker_credentials_path_override` | `neoethos-cli/src/main.rs:2169` — LIVE |
//! | `NEOETHOS_BOT_SYMBOL_METADATA` | `symbol_metadata_path_override` | `symbol_metadata.rs:724` — MIGRATED 2026-08-09 |
//! | `RUST_LOG` | (none; `EnvFilter` reads it itself) | tracing-subscriber |
//! | `NEOETHOS_PROP_FIRM_PRESET` | `prop_firm_preset_raw` | **NONE — retired, see below** |
//! | `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` | `prop_firm_account_currency` | **NONE — see below** |
//! | `NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE` | `prop_firm_quote_to_account_rate` | **NONE — see below** |
//!
//! One direct read remains OUTSIDE this crate's migration: a local
//! const for the broker-credentials path at
//! `neoethos-core/src/broker_config.rs:164`. It is not in this wave's
//! territory; it is named here so the next reader does not have to
//! rediscover it.
//!
//! ## Three names in this registry are INERT
//!
//! `NEOETHOS_PROP_FIRM_PRESET`, `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY`
//! and `NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE` have **no reader
//! anywhere** in `crates/`, `desktop/`, `mesh/` or `mcp/`. Setting
//! them changes nothing. They are kept — not deleted — because two of
//! them describe a capability the operator still needs (see
//! [`ENV_PROP_QUOTE_TO_ACCOUNT_RATE`]), and deleting the name would
//! delete the record that the capability is missing. They are
//! reported separately by [`log_active_overrides_at_startup`] so they
//! can never again be presented as active levers.
//!
//! ## NOT in this registry, and do not delete it either
//!
//! `RAYON_NUM_THREADS` (audit #279) is dead at
//! `neoethos-search/src/genetic/eval.rs:535` (an orphaned `from_env`)
//! but **LIVE at `neoethos-models/src/tree_models/config.rs:119` on
//! every tree train**. A purge that deletes the variable on the
//! `eval.rs` finding alone removes the operator's only thread control
//! over tree training. Check BOTH sites before touching it.

use std::env;

// ---------------------------------------------------------------------------
// Env-var names — canonical string constants
// ---------------------------------------------------------------------------

/// **RETIRED in v0.4.36 — INERT. Setting this changes nothing.**
///
/// It once seeded `RiskConfig::default()`. `config.rs:512-521` now
/// says in its own words that the override was retired and that
/// headless deployments set `risk.preset` in `config.yaml` instead.
///
/// The name is kept so an operator who still has it exported gets a
/// startup line telling them it is dead, instead of a startup line
/// telling them it is an active override (audit #143 — the previous
/// text did the latter, which reassured the operator that tighter
/// preset limits were in force while the raw ones ran).
pub const ENV_PROP_FIRM_PRESET: &str = "NEOETHOS_PROP_FIRM_PRESET";

/// **INERT. Setting this changes nothing.**
///
/// The doc that stood here said this was *"required by
/// `risk_gate::prop_firm_pre_trade_check` whenever an order carries a
/// stop-loss"*. **That function does not exist** — grep across
/// `crates/`, `desktop/`, `mesh/` and `mcp/` on 2026-08-09 returns
/// only the three comments that mention it (audit #142). No caller
/// reads [`prop_firm_account_currency`] either.
///
/// The account currency that ACTUALLY drives sizing is
/// `Settings.risk.account_currency` / the `account_currency` key in
/// `config.yaml`, consumed via
/// `neoethos_search::EvaluationConfig::for_symbol`.
pub const ENV_PROP_ACCOUNT_CURRENCY: &str = "NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY";

/// **INERT. Setting this changes nothing — and that is a gap, not a
/// tidy-up.**
///
/// Intended as the live quote→account FX rate override for cross
/// pairs whose pip value the broker has not shipped. That is exactly
/// the lever the operator would reach for against the measured
/// **192× EURJPY pip-value inflation in the backtest** — and it is
/// connected to nothing (audit #142). [`prop_firm_quote_to_account_rate`]
/// has zero callers.
///
/// Kept, not deleted, so the missing capability stays visible. If a
/// rate override is wired in future it must go through the same
/// `MarketCostProfile` boundary the backtest uses, or live and
/// backtest will disagree about pip value all over again.
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

// REMOVED 2026-08-09 (dead-code purge, batch D2): `ENV_LAUNCHED_BY_FLUTTER`
// ("NEOETHOS_LAUNCHED_BY_FLUTTER") and its `launched_by_flutter()` getter.
// It suppressed the Windows "double-click help" dialog when the Flutter shell
// spawned the backend. Flutter died in the 2026-06-22 Tauri migration and the
// desktop shell runs the backend in-process, so nothing set it.
// BEHAVIOUR NOTE: this was the last remaining suppression switch for that
// modal. A release `neoethos-app.exe` started standalone on Windows now always
// shows it — which is what already happened, since nothing set the var.

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
///
/// **ZERO CALLERS** — see [`ENV_PROP_FIRM_PRESET`]. Retained so the
/// startup report can name the variable as inert.
pub fn prop_firm_preset_raw() -> Option<String> {
    env::var(ENV_PROP_FIRM_PRESET)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read the prop-firm account currency. `None` when unset / empty.
/// Empty propagates as `None` so the caller sees "unset" not "empty
/// string" — typed boundary at the registry.
///
/// **ZERO CALLERS** — see [`ENV_PROP_ACCOUNT_CURRENCY`].
pub fn prop_firm_account_currency() -> Option<String> {
    env::var(ENV_PROP_ACCOUNT_CURRENCY)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read the cross-pair quote→account FX rate override. `None` when
/// unset / unparseable / non-finite / non-positive.
///
/// **ZERO CALLERS** — see [`ENV_PROP_QUOTE_TO_ACCOUNT_RATE`]. The
/// validation below is correct and unused.
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
/// **CORRECTED 2026-08-09 (audit #143).** This list now contains ONLY
/// variables that have a verified live reader. Three names that are
/// set but read by nothing —
/// [`ENV_PROP_FIRM_PRESET`], [`ENV_PROP_ACCOUNT_CURRENCY`] and
/// [`ENV_PROP_QUOTE_TO_ACCOUNT_RATE`] — moved to
/// [`inert_overrides_present`]. They were previously reported here
/// under the sentence *"Each one changes runtime behaviour"*, which
/// is how a retired preset variable could reassure the operator that
/// tighter limits were in force while the raw ones ran.
pub fn active_overrides() -> Vec<&'static str> {
    let mut active: Vec<&'static str> = Vec::new();
    if std::env::var(ENV_SYMBOL_METADATA).is_ok() {
        active.push(ENV_SYMBOL_METADATA);
    }
    if std::env::var(ENV_LOG_FILTER).is_ok() {
        active.push(ENV_LOG_FILTER);
    }
    if std::env::var(ENV_USER_DATA_DIR).is_ok() {
        active.push(ENV_USER_DATA_DIR);
    }
    if std::env::var(ENV_LOG_DIR).is_ok() {
        active.push(ENV_LOG_DIR);
    }
    if std::env::var(ENV_BROKER_CREDENTIALS_PATH).is_ok() {
        active.push(ENV_BROKER_CREDENTIALS_PATH);
    }
    active
}

/// Variables that are SET in the environment but that nothing reads.
///
/// Reported separately and loudly, because a dead lever the operator
/// believes is live is more dangerous than no lever at all: it makes
/// him stop looking for the real control.
pub fn inert_overrides_present() -> Vec<&'static str> {
    let mut inert: Vec<&'static str> = Vec::new();
    if std::env::var(ENV_PROP_FIRM_PRESET).is_ok() {
        inert.push(ENV_PROP_FIRM_PRESET);
    }
    if std::env::var(ENV_PROP_ACCOUNT_CURRENCY).is_ok() {
        inert.push(ENV_PROP_ACCOUNT_CURRENCY);
    }
    if std::env::var(ENV_PROP_QUOTE_TO_ACCOUNT_RATE).is_ok() {
        inert.push(ENV_PROP_QUOTE_TO_ACCOUNT_RATE);
    }
    inert
}

/// Emit a structured warning at startup listing every active env
/// override, and a SEPARATE warning naming any variable that is set
/// but dead. Idempotent — safe to call multiple times; the operator
/// will see one line per call. Designed to be called once in the
/// binary's `main()` after `setup_logging`.
pub fn log_active_overrides_at_startup() {
    let active = active_overrides();
    if active.is_empty() {
        tracing::info!(
            target: "neoethos_core::env_overrides",
            "No NeoEthos env-var overrides active at startup."
        );
    } else {
        tracing::warn!(
            target: "neoethos_core::env_overrides",
            count = active.len(),
            overrides = ?active,
            "NeoEthos env-var overrides active at startup — listed for \
             operator visibility (F-005 fix). Each one changes runtime \
             behaviour; review and confirm intentional."
        );
    }

    let inert = inert_overrides_present();
    if !inert.is_empty() {
        tracing::warn!(
            target: "neoethos_core::env_overrides",
            count = inert.len(),
            inert_overrides = ?inert,
            "These NeoEthos env vars are SET but NOTHING READS THEM — they change \
             nothing (audit #142/#143). NEOETHOS_PROP_FIRM_PRESET was retired in \
             v0.4.36 (set risk.preset in config.yaml). The two NEOETHOS_BOT_PROP_* \
             vars were documented as required by risk_gate::prop_firm_pre_trade_check, \
             a function that does not exist in this repo. Do not rely on any of them."
        );
    }
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
