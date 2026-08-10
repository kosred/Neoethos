//! **Retired environment variables — the app crate's tombstone list.**
//!
//! # Why this file exists
//!
//! The 2026-08-10 config consolidation removed every environment variable
//! that could change what this binary DOES. One config file, one load path,
//! no second lever. Deleting the reader is only half the job: an operator's
//! shell profile, a `.bat` launcher, a systemd unit or a CI workflow may
//! still export the old name. Before this module, that export simply stopped
//! working — silently. The operator would set `NEOETHOS_CORS_ALLOW_ANY=1`,
//! see no error, and believe the API was open to his phone.
//!
//! Non-negotiable #4 of the consolidation directive: **when a deleted env var
//! is still set, say so** — name the variable, name the replacement field,
//! state that the value was ignored. Silence is the disease.
//!
//! # What this is NOT
//!
//! It is not a compatibility layer. Nothing here reads a value and applies
//! it. [`report_retired_env_vars`] only looks at whether the name is present
//! in the process environment and, if so, prints an ERROR naming what to do
//! instead. The value itself is never parsed and never used.
//!
//! # Adding to this list
//!
//! When you delete an env-var reader in this crate, add its name here in the
//! same change. An entry costs one line and one `getenv`; an omission costs
//! an operator an afternoon.

use std::env;

/// One retired environment variable and where its behaviour went.
struct RetiredEnvVar {
    /// The exact variable name an operator may still be exporting.
    name: &'static str,
    /// The config key (or mechanism) that replaced it. `""` when the
    /// behaviour was removed outright rather than moved.
    replacement: &'static str,
    /// What it used to do, in one clause, so the message is actionable
    /// without opening the source.
    used_to: &'static str,
}

/// Every environment variable this crate USED to read and no longer does.
///
/// Ordered by the wave that retired them so the history stays legible.
const RETIRED: &[RetiredEnvVar] = &[
    // ── 2026-08-10, this wave ────────────────────────────────────────────
    RetiredEnvVar {
        name: "NEOETHOS_CORS_ALLOW_ANY",
        replacement: "",
        used_to: "disabled the CORS origin allowlist, letting ANY website open \
                  in a browser call the unauthenticated trading endpoints. The \
                  escape hatch is GONE — there is no config key for it. To reach \
                  the API from another machine, bind a non-loopback address \
                  (`app_runtime.server_bind`), which already REQUIRES an API \
                  token, and put your own proxy in front",
    },
    RetiredEnvVar {
        name: "NEOETHOS_MCP_URL",
        replacement: "the `port` field of `mcp_servers.json` (Settings → MCP)",
        used_to: "relocated the local MCP sidecar's base URL. The sidecar's own \
                  config file already carries its port; two places to state one \
                  port is how they drift",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_MIN_HISTORY_YEARS",
        replacement: "models.discovery_runtime.min_history_years",
        used_to: "set the minimum years of history discovery demands, and the \
                  auto-fetch threshold in the UI discovery job read it \
                  SEPARATELY from the value the search enforced — so the two \
                  could disagree",
    },
    // ── 2026-08-08, the app-crate dead-code purge ────────────────────────
    // These ten lost their readers when the app runtime moved onto
    // `AppRuntimeConfig` (config-consolidation S3-app). Their names survived
    // only as knob-catalog string literals, which this wave also removed.
    RetiredEnvVar {
        name: "NEOETHOS_SERVER_BIND",
        replacement: "app_runtime.server_bind",
        used_to: "set the HTTP server's host:port",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_CTRADER_READ_TIMEOUT_SECS",
        replacement: "app_runtime.ctrader_read_timeout_secs",
        used_to: "capped the cTrader execution read timeout",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_CTRADER_MAX_ATTEMPTS",
        replacement: "app_runtime.ctrader_max_attempts",
        used_to: "set how many times an order submit is retried",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_CTRADER_BACKOFF_BASE_MS",
        replacement: "app_runtime.ctrader_backoff_base_ms",
        used_to: "set the retry backoff base for order submits",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_CTRADER_ALLOW_PARTIAL_FILL",
        replacement: "app_runtime.ctrader_allow_partial_fill",
        used_to: "decided whether a partial fill counts as final",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_CTRADER_STREAM_MAX_ATTEMPTS",
        replacement: "app_runtime.ctrader_stream_max_attempts",
        used_to: "set the live loop's bar-fetch retry count",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_CTRADER_STREAM_BACKOFF_BASE_MS",
        replacement: "app_runtime.ctrader_stream_backoff_base_ms",
        used_to: "set the live loop's bar-fetch backoff base",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_CHART_MERGE_SIDE",
        replacement: "app_runtime.chart_merge_side",
        used_to: "chose mid/bid/ask for the chart-merge quote side",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_PNL_AUDIT_DRIFT_FRACTION",
        replacement: "app_runtime.pnl_audit_drift_fraction",
        used_to: "set the PnL drift threshold that logs an audit warning",
    },
    RetiredEnvVar {
        name: "NEOETHOS_BOT_PNL_CIRCUIT_BREAKER_FRACTION",
        replacement: "app_runtime.pnl_circuit_breaker_fraction",
        used_to: "set the PnL drift threshold that HALTS the auto-trader",
    },
];

/// Scan the process environment for retired variable names and print one
/// ERROR per name that is still set.
///
/// Called from `neoethos_app::install_runtime_overrides_from_settings` — the
/// single path both front-ends (headless `main.rs` and the in-process Tauri
/// shell) take — so neither can skip the report.
///
/// The variable's VALUE is deliberately not logged: some of these names have
/// been used to carry paths, and a diagnostics bundle should not acquire an
/// operator's directory layout as a side effect of a deprecation notice.
/// What matters is that the name is set and is being ignored.
///
/// Returns the number of retired names found, so a test can assert on it
/// without capturing the log.
pub fn report_retired_env_vars() -> usize {
    let mut found = 0usize;
    for entry in RETIRED {
        // `var_os` rather than `var`: a name set to non-UTF-8 bytes is still
        // set, and an operator whose value we cannot decode deserves the
        // notice more than one whose value we can.
        if env::var_os(entry.name).is_none() {
            continue;
        }
        found += 1;
        if entry.replacement.is_empty() {
            tracing::error!(
                target: "neoethos_app::retired_env",
                env_var = entry.name,
                "RETIRED ENVIRONMENT VARIABLE IS STILL SET AND WAS IGNORED. \
                 `{}` {}. There is NO replacement setting — the behaviour was \
                 removed, not moved. Unset it so the next reader of your \
                 launcher is not misled.",
                entry.name,
                entry.used_to
            );
        } else {
            tracing::error!(
                target: "neoethos_app::retired_env",
                env_var = entry.name,
                replacement = entry.replacement,
                "RETIRED ENVIRONMENT VARIABLE IS STILL SET AND WAS IGNORED. \
                 `{}` {}. Its value had NO effect on this run. Set `{}` in the \
                 config file instead, then unset the variable.",
                entry.name,
                entry.used_to,
                entry.replacement
            );
        }
    }
    if found > 0 {
        tracing::error!(
            target: "neoethos_app::retired_env",
            retired_env_vars_still_set = found,
            "{} retired environment variable(s) are set in this process and \
             every one of them was IGNORED. One config file is the only lever; \
             see the lines above for each replacement key.",
            found
        );
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry must name itself and say what it used to do. An entry with
    /// an empty `used_to` produces a notice an operator cannot act on, which
    /// is the failure mode this module exists to prevent.
    #[test]
    fn every_entry_is_actionable() {
        assert!(!RETIRED.is_empty(), "the tombstone list must not be empty");
        for e in RETIRED {
            assert!(e.name.starts_with("NEOETHOS_"), "unexpected name {}", e.name);
            assert!(
                !e.used_to.trim().is_empty(),
                "{} has no `used_to` clause — the notice would be unactionable",
                e.name
            );
        }
    }

    /// A duplicated name would print the same notice twice and suggests a
    /// merge conflict resolved by keeping both sides.
    #[test]
    fn names_are_unique() {
        let mut names: Vec<&str> = RETIRED.iter().map(|e| e.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate retired env-var name");
    }

    /// The names this wave retired must actually be in the list — this is the
    /// pinning test non-negotiable #5 asks for, on the app-crate side.
    #[test]
    fn this_waves_deletions_are_tombstoned() {
        for expected in [
            "NEOETHOS_CORS_ALLOW_ANY",
            "NEOETHOS_MCP_URL",
            "NEOETHOS_BOT_MIN_HISTORY_YEARS",
        ] {
            assert!(
                RETIRED.iter().any(|e| e.name == expected),
                "{expected} was deleted from the code but not tombstoned here"
            );
        }
    }

    /// With none of the names set, the scan must be silent. (The test process
    /// does not export them; if a future test does, it must clean up.)
    #[test]
    fn clean_environment_reports_nothing() {
        let unset: Vec<&str> = RETIRED
            .iter()
            .map(|e| e.name)
            .filter(|n| std::env::var_os(n).is_none())
            .collect();
        assert_eq!(
            unset.len(),
            RETIRED.len(),
            "a retired env var is set in the test process: {:?}",
            RETIRED
                .iter()
                .map(|e| e.name)
                .filter(|n| std::env::var_os(n).is_some())
                .collect::<Vec<_>>()
        );
    }
}
