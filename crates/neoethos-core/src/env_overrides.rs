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
//! ## 2026-08-10 — what is left, and why each one is left
//!
//! The directive is ONE config file: no environment variable may decide
//! anything an operator could have written down. Everything that survives in
//! this module is a **bootstrap locator** — an input that decides WHICH FILE
//! or WHICH DIRECTORY is read, and therefore cannot live inside that file
//! without asking the config where the config is. That is the entire
//! exemption, and it is deliberately small enough to list:
//!
//! | Var | Decides | Read at | Why it cannot be a config field |
//! |---|---|---|---|
//! | `CONFIG_FILE` | which config file `Settings::load` opens | `config.rs::load` | It selects the file. Every branch is tagged in `ConfigProvenance` and logged. |
//! | `NEOETHOS_USER_DATA_DIR` | the root under which `config.yaml` lives | `config.rs::user_config_path` | Same — it is upstream of the file. |
//! | `LOCALAPPDATA` / `HOME` / `XDG_DATA_HOME` | the platform data dir | `config.rs::user_config_path` | OS-standard, set by the OS, not by us. |
//! | `LOG_DIR` | where logs are written | `logging.rs::default_log_dir` | Logging is initialised BEFORE `Settings` is loaded — it is what reports a load failure. A config field would be unreadable exactly when it matters. |
//! | `RUST_LOG` | the tracing filter | `tracing_subscriber::EnvFilter` | Ecosystem standard, parsed by the subscriber itself. |
//! | `NEOETHOS_BROKER_CREDENTIALS_PATH` | which secrets file | `broker_config.rs::credentials_file_path` | Credentials deliberately do NOT live in the config file. |
//! | `NEOETHOS_BOT_SYMBOL_METADATA` | which symbol-metadata file | `symbol_metadata.rs::metadata_path` | A data-file locator; the table it points at is rewritten by the cTrader reconcile, not by the operator. |
//!
//! Nothing above changes an arithmetic result, a limit, or a money value. If a
//! future variable in this module ever would, it belongs in `Settings` and in
//! the run profile, not here.
//!
//! ## Retired names are KEPT, on purpose
//!
//! [`RETIRED_ENV_VARS`] holds names whose readers are gone. They stay so that
//! an operator who still exports one gets an ERROR line naming the variable,
//! the value found and what decides that quantity now. Deleting the name would
//! restore the silence. Their typed getters were deleted in the same change —
//! a getter with zero callers is not a record, it is a function that looks
//! like a lever.
//!
//! ## NOT in this registry, and do not delete it either
//!
//! `RAYON_NUM_THREADS` is retired in `neoethos-search`
//! (`models.backtest_runtime.rayon_threads`) but **LIVE at
//! `neoethos-models/src/tree_models/config.rs:119` on every tree train**. One
//! variable, two crates, two answers. Deleting it on the search finding alone
//! removes the operator's only thread control over tree training. Routed to
//! `docs/pending-edits-forbidden-territory.md` as D-F4.

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

// DELETED 2026-08-10 (env→config wave 2): `prop_firm_preset_raw`,
// `prop_firm_account_currency`, `prop_firm_quote_to_account_rate`.
//
// All three had ZERO callers in `crates/`, `desktop/`, `mesh/` and `mcp/` —
// verified by workspace grep in the same change that deleted them. They were
// kept "so the startup report can name the variable as inert", but the report
// does not need a typed getter to do that: it needs the NAME, which
// [`RETIRED_ENV_VARS`] below still holds, together with what decides the
// quantity now. A parsed-and-validated value that nothing consumes is not a
// record of a missing capability, it is a function that looks like a lever.
//
// The capability note survives, because it is the part that mattered: there is
// still no quote→account FX-rate override, and that is the lever the measured
// 192× EURJPY pip-value inflation in the backtest would need. When one is
// wired it must go through the `MarketCostProfile` boundary the backtest uses,
// or live and backtest will disagree about pip value all over again.

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
/// **CORRECTED 2026-08-09 (audit #143).** This list contains ONLY
/// variables that have a verified live reader. Names that are set but
/// read by nothing live in [`RETIRED_ENV_VARS`] and are reported by
/// [`inert_overrides_present`]. They were previously reported here
/// under the sentence *"Each one changes runtime behaviour"*, which
/// is how a retired preset variable could reassure the operator that
/// tighter limits were in force while the raw ones ran.
///
/// **NARROWED 2026-08-10.** Every name still returned here is a bootstrap
/// locator — it selects a FILE or a DIRECTORY, never a number, a limit or a
/// policy. The module header carries the full table with the reason each one
/// cannot be a config field. The paragraph that used to stand here told the
/// reader to consult `neoethos-search`'s equivalent helper for
/// `NEOETHOS_BOT_DISABLE_SMC_GATE`, `NEOETHOS_BOT_NORMALIZE_FEATURES` and
/// `NEOETHOS_BOT_PREFILTER_*`; all of those are retired config fields now, and
/// `neoethos-search` / `neoethos-data` announce their own retired names from
/// their own tables at startup.
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

/// Environment variables this crate used to honour and no longer does —
/// `(name, what decides that quantity now)`.
///
/// Same shape and same contract as `neoethos_search::execution_profile::
/// RETIRED_ENV_VARS` and `neoethos_data::RETIRED_ENV_VARS`: a name lands here
/// when its reader is deleted, and it stays here so that an operator who still
/// exports it is TOLD, by name, that the value did not reach the run. Silence
/// is the disease; being ignored loudly is the cure.
pub const RETIRED_ENV_VARS: &[(&str, &str)] = &[
    (
        ENV_PROP_FIRM_PRESET,
        "risk.preset in the config file (the env seed was retired in v0.4.36, and the \
         preset now re-derives the six seeded money fields at load)",
    ),
    (
        ENV_PROP_ACCOUNT_CURRENCY,
        "risk.account_currency, consumed via neoethos_search::EvaluationConfig::for_symbol",
    ),
    (
        ENV_PROP_QUOTE_TO_ACCOUNT_RATE,
        "NOTHING — there is still no quote→account FX-rate override anywhere in the \
         workspace. This is a MISSING CAPABILITY, not a relocated one, and it is the lever \
         the 192x EURJPY pip-value inflation in the backtest would need",
    ),
    // The four hardware/accelerator names that died with
    // `HardwareRuntimeOverrides::from_env` (deleted 2026-08-03 — it had zero
    // callers, so these had already stopped doing anything). Until now their
    // deletion was recorded only in a doc comment, which an exported variable
    // does not read. Verified absent from every crate's source before listing.
    //
    // NOT listed, deliberately: `NEOETHOS_BOT_CPU_BUDGET`, which
    // `neoethos-cli/src/main.rs` still reads on a live path. Announcing it as
    // ignored while one binary honours it would be a worse lie than the
    // silence — one name, two readers, routed to
    // docs/pending-edits-forbidden-territory.md. Also not listed:
    // `NEOETHOS_BOT_TRAIN_PRECISION` / `FOREX_TRAIN_PRECISION`, which
    // `neoethos_search::execution_profile::RETIRED_ENV_VARS` already announces
    // — one variable must produce one line, not two.
    (
        "NEOETHOS_BOT_CUDA_PRECISIONS",
        "system.hardware.cuda_precisions in the config file",
    ),
    (
        "NEOETHOS_BOT_ROCM_PRECISIONS",
        "system.hardware.rocm_precisions in the config file",
    ),
    (
        "NEOETHOS_BOT_WGPU_PRECISIONS",
        "system.hardware.wgpu_precisions in the config file",
    ),
    (
        "NEOETHOS_BOT_WGPU_DEVICES",
        "system.hardware.wgpu_device_names in the config file",
    ),
];

/// Retired variables that are SET in this process.
///
/// Reported separately and loudly, because a dead lever the operator
/// believes is live is more dangerous than no lever at all: it makes
/// him stop looking for the real control.
///
/// Name kept (2026-08-10) because `neoethos-cli`, `neoethos-app` and the
/// desktop shell all call [`log_active_overrides_at_startup`], which calls
/// this; what changed is that each name now carries its replacement.
pub fn inert_overrides_present() -> Vec<&'static str> {
    RETIRED_ENV_VARS
        .iter()
        .filter(|(name, _)| std::env::var(name).is_ok_and(|value| !value.trim().is_empty()))
        .map(|(name, _)| *name)
        .collect()
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
            "NeoEthos bootstrap locators are set — each one selects WHICH FILE or \
             WHICH DIRECTORY this process reads (config file, user-data root, log \
             directory, credentials path, symbol-metadata path, tracing filter). None \
             of them changes a number, a limit or a policy; those live in the config \
             file only. Review and confirm intentional."
        );
    }

    // Retired names, one ERROR line each, naming the variable, the value found
    // and what decides that quantity now. One line per variable rather than one
    // summary line: a summary tells the operator that SOMETHING was ignored,
    // which is not actionable; this tells him which key to set instead.
    for (name, replacement) in RETIRED_ENV_VARS {
        let Ok(value) = std::env::var(name) else {
            continue;
        };
        if value.trim().is_empty() {
            continue;
        }
        tracing::error!(
            target: "neoethos_core::env_overrides",
            env_var = %name,
            value_found = %value,
            decided_by = %replacement,
            "RETIRED ENVIRONMENT VARIABLE IS SET AND WAS IGNORED — this value did NOT \
             reach the run."
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

    // DELETED 2026-08-10: `rate_getter_rejects_zero_and_negative`. It was named
    // after `prop_firm_quote_to_account_rate` but never called it — it inlined
    // the parse against a throwaway variable, so it exercised `str::parse::<f64>`
    // and nothing of ours. With the getter gone it tested the standard library.

    /// Every retired name must state what decides that quantity now. A row
    /// with an empty replacement produces a startup line that says a value was
    /// ignored without saying what to set instead, which sends the operator
    /// back to grepping — the condition this module exists to end.
    #[test]
    fn every_retired_name_names_its_replacement() {
        for (name, replacement) in RETIRED_ENV_VARS {
            assert!(
                name.starts_with("NEOETHOS_"),
                "retired row {name} is not one of ours"
            );
            assert!(
                replacement.len() > 20,
                "retired name {name} has no usable replacement sentence: {replacement:?}"
            );
        }
    }

    /// The retired table and the live getters must not overlap. An overlap
    /// would mean one variable being announced as ignored while a getter in
    /// this same file still honours it.
    #[test]
    fn retired_and_active_names_are_disjoint() {
        let active = [
            ENV_SYMBOL_METADATA,
            ENV_LOG_FILTER,
            ENV_USER_DATA_DIR,
            ENV_LOG_DIR,
            ENV_BROKER_CREDENTIALS_PATH,
        ];
        for (retired, _) in RETIRED_ENV_VARS {
            assert!(
                !active.contains(retired),
                "{retired} is both retired and live — one of the two is a lie"
            );
        }
    }
}
