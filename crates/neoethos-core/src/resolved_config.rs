//! `ResolvedConfig` — the typed, single-source-of-truth view of
//! [`crate::Settings`] that every consumer (CLI, TUI, app, search,
//! training, cTrader connector) should resolve through.
//!
//! Why: the codebase has two ergonomic problems with `Settings`:
//! 1. **Sentinel `0` semantics are implicit.** `prop_search_max_indicators: 0`
//!    silently became `5` in some code paths and `usize::MAX` in others.
//!    `prop_search_val_candidates: 0` silently became `population` in one
//!    place and `population * generations` in another.
//! 2. **Side-channel env vars override behavior.** A user reading
//!    `config.yaml` cannot tell that `NEOETHOS_BOT_NORMALIZE_FEATURES=1`
//!    or `NEOETHOS_BOT_DISABLE_SMC_GATE=1` flips the search regime.
//!
//! `ResolvedConfig` makes both visible. Every field is **resolved** —
//! sentinels are converted to real numbers, env overrides are applied
//! and recorded, and a `display_table()` helper emits a `(field, raw,
//! resolved, source)` table the TUI can render verbatim.
//!
//! This is **additive**: existing code keeps reading `Settings` and
//! `model_settings.prop_search_*` fields directly. Wherever a consumer
//! wants the resolved view, it calls `ResolvedConfig::from_settings(&s)`
//! and reads the typed sections.

use serde::{Deserialize, Serialize};

use crate::Settings;
use crate::contracts::CANONICAL_TIMEFRAMES;

/// One resolved field — captures both the operator-supplied value and
/// the value the system will actually use, plus where the resolution
/// came from. The TUI's Config page renders these as a single table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedField {
    pub section: String,
    pub field: String,
    pub raw: String,
    pub resolved: String,
    pub source: ResolvedSource,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolvedSource {
    /// Read directly from the operator's config.yaml.
    Config,
    /// Resolved from a sentinel (e.g. `0` → "use all" / "population×generations").
    SentinelExpanded,
    /// An environment variable overrode or augmented the config value.
    EnvOverride,
    /// Built-in default (operator did not set anything).
    Default,
}

impl ResolvedSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::SentinelExpanded => "sentinel→resolved",
            Self::EnvOverride => "env",
            Self::Default => "default",
        }
    }
}

/// Resolved data section — matches the spec's `data` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedDataConfig {
    pub root: String,
    pub recursive_scan: bool,
    pub vortex_force_rebuild: bool,
    pub canonical_layout: String,
}

/// Resolved search section — covers all the GA/discovery knobs the
/// previous implementation hid behind silent fallbacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedSearchConfig {
    /// `0` in raw → "use all available features"; resolved is
    /// `usize::MAX` (downstream clamps to actual feature count).
    pub max_indicators_raw: usize,
    pub max_indicators_resolved: usize,
    /// `0` in raw → "no artificial cap, use population×generations".
    pub candidate_count_raw: usize,
    pub candidate_count_resolved: usize,
    pub population: usize,
    /// The SEARCH-MORE knob. `true` + CUDA card: discovery raises the GA
    /// population to the card's fits ceiling (≤ 16 384, never below
    /// `population`) at run start and logs the resolved value. The ceiling
    /// depends on the device and dataset, so it cannot be shown here — the
    /// run's own log line is the record of what actually searched.
    pub population_auto: bool,
    pub generations: usize,
    pub portfolio_size: usize,
    pub min_trades_per_day_raw: f64,
    pub min_trades_per_day_resolved: f64,
    pub corr_threshold: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub mode: String,
    /// `NEOETHOS_BOT_NORMALIZE_FEATURES` resolved to bool.
    pub normalize_features_env: bool,
    /// `NEOETHOS_BOT_DISABLE_SMC_GATE` resolved to bool.
    pub disable_smc_gate_env: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedFiltersConfig {
    /// Renamed from `min_profit` per spec — the field was being
    /// compared against the composite `fitness` score, not net profit;
    /// this label makes that intent explicit.
    pub min_fitness_score: f64,
    pub min_trades: f64,
    pub max_drawdown: f64,
    pub min_sharpe: f64,
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTimeframesConfig {
    pub base: String,
    pub higher: Vec<String>,
    /// Default canonical timeframe set used by `batch-discover` when
    /// `--timeframes` is omitted. Per the spec includes M3 and M30.
    pub canonical_default: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedConfig {
    pub data: ResolvedDataConfig,
    pub timeframes: ResolvedTimeframesConfig,
    pub search: ResolvedSearchConfig,
    pub filters: ResolvedFiltersConfig,
    /// Field-level table for UI display — every entry surfaces
    /// (section, field, raw, resolved, source).
    pub display_fields: Vec<ResolvedField>,
}

impl ResolvedConfig {
    pub fn from_settings(s: &Settings) -> Self {
        // Search section ---------------------------------------------------
        let max_indicators_raw = s.models.prop_search_max_indicators;
        let max_indicators_resolved = if max_indicators_raw == 0 {
            usize::MAX
        } else {
            max_indicators_raw.max(1)
        };
        let candidate_count_raw = s.models.prop_search_val_candidates;
        let candidate_count_resolved = if candidate_count_raw == 0 {
            s.models
                .prop_search_population
                .saturating_mul(s.models.prop_search_generations.max(1))
                .max(s.models.prop_search_population.max(50))
        } else {
            candidate_count_raw.max(1)
        };
        let min_trades_per_day_raw = s.models.prop_search_val_min_trades_per_day;
        let mode = resolve_discovery_mode(s);
        let min_trades_per_day_resolved = if mode == "prop_firm" && min_trades_per_day_raw == 0.0 {
            0.001
        } else {
            min_trades_per_day_raw.max(0.0)
        };
        let normalize_features_env = env_truthy("NEOETHOS_BOT_NORMALIZE_FEATURES");
        let disable_smc_gate_env = env_truthy("NEOETHOS_BOT_DISABLE_SMC_GATE");

        // Filters section --------------------------------------------------
        // **F-148 documentation (2026-05-25)** — these literals
        // intentionally MIRROR (not import) the values in
        // `crate::genetic::FilteringConfig::default()` (over in
        // neoethos-search). Importing across the search→core crate
        // boundary would create a cycle (search depends on core
        // already for `Settings`). The current pattern is:
        //   - core resolved_config.rs holds the DISPLAY copy (this)
        //   - neoethos-search FilteringConfig holds the ENFORCED copy
        //   - both are documented as having to stay in sync; the
        //     two test sites `resolved_config_tests::*_floors_match_*`
        //     would assert byte-equality if added (Phase-C task).
        //
        // The PROP_FIRM-mode permissive floors `(0.0, 1.0, 0.50,
        // -10.0, 0.0, 0.0)` are intentional — discovery in challenge
        // mode runs with weaker quality filters so the gauntlet does
        // the heavy lifting (operator directive 2026-05-15).
        // The engine has THREE floor sets, not two — `apply_mode_overrides`
        // (discovery.rs:643) rewrites the filter floors for PropFirm AND for
        // Risky, and leaves Strict on `FilteringConfig::default()`.
        //
        // CORRECTED 2026-08-04. This was a two-branch `if mode == "prop_firm"
        // { .. } else { .. }` whose else-branch comment read:
        //
        //     "Strict/Risky: no mode override fires (apply_mode_overrides only
        //      rewrites the floors for PropFirm), so what the engine enforces
        //      is `FilteringConfig::default()`"
        //
        // That sentence is false about Risky: discovery.rs:746-760 rewrites all
        // six floors for Risky (max_dd 0.60, sharpe -5.0, the rest wide open).
        // It went unnoticed because `resolve_discovery_mode` above could not
        // return "risky" either, so Risky reached the PropFirm display branch
        // and the false sentence described a branch nothing took. Two mistakes
        // that hid each other: the 2026-08-03 pass then carefully corrected
        // three literals inside the unreachable branch.
        //
        // Net effect for the operator, who runs Risky: the report announced
        // PropFirm's floors (maxDD 0.50, sharpe -10.0) for a search running
        // Risky's (maxDD 0.60, sharpe -5.0), and labelled the mode "prop_firm".
        //
        // These literals still MIRROR rather than import — core cannot depend
        // on neoethos-search without a cycle (F-148, 2026-05-25). What is new
        // is that the mirror is now checked from the search side, where both
        // crates are visible, for every mode:
        // `display_floors_match_the_enforced_ones`.
        let (min_fitness_score, min_trades, max_drawdown, min_sharpe, min_win_rate, min_pf) =
            match mode {
                // discovery.rs:682-690 — permissive; the FTMO window-pass gate
                // downstream does the real filtering (operator directive
                // 2026-05-15).
                "prop_firm" => (0.0_f64, 1.0_f64, 0.50_f64, -10.0_f64, 0.0_f64, 0.0_f64),
                // discovery.rs:755-760 — loose-but-sane; growth-tilted ranking
                // picks the fastest compounders, deep drawdown is accepted.
                "risky" => (0.0_f64, 1.0_f64, 0.60_f64, -5.0_f64, 0.0_f64, 0.0_f64),
                // Strict is the one mode `apply_mode_overrides` leaves alone, so
                // the engine enforces `FilteringConfig::default()` in
                // neoethos-search/src/genetic/strategy_gene.rs:101-117.
                //
                // CORRECTED 2026-08-03. Three of these six had drifted from the
                // values actually enforced, so `neoethos-cli config` and the UI
                // were reporting a system that does not exist:
                //     max_drawdown  0.20 -> 0.15   (engine is STRICTER by 5 pts)
                //     min_sharpe    0.5  -> 0.3    (engine is LOOSER)
                //     min_win_rate  0.45 -> 0.50   (engine is STRICTER)
                // Only min_profit_factor (1.2) was right.
                _ => (
                    0.0_f64,
                    s.models.prop_min_trades.max(1) as f64,
                    0.15_f64,
                    0.3_f64,
                    0.50_f64,
                    1.2_f64,
                ),
            };

        // Timeframes section -----------------------------------------------
        // Canonical default reused by `batch-discover` when `--timeframes`
        // is omitted. Sourced from `CANONICAL_TIMEFRAMES` so adding /
        // removing a supported timeframe needs a single edit.
        let canonical_default: Vec<String> = CANONICAL_TIMEFRAMES
            .iter()
            .map(|tf| (*tf).to_string())
            .collect();

        // Data section -----------------------------------------------------
        let data_root = s.system.data_dir.to_string_lossy().to_string();

        let mut display_fields = Vec::new();
        push_field(
            &mut display_fields,
            "search",
            "max_indicators",
            max_indicators_raw.to_string(),
            if max_indicators_resolved == usize::MAX {
                "ALL features".to_string()
            } else {
                max_indicators_resolved.to_string()
            },
            if max_indicators_raw == 0 {
                ResolvedSource::SentinelExpanded
            } else {
                ResolvedSource::Config
            },
            Some("0 = use every available feature column"),
        );
        push_field(
            &mut display_fields,
            "search",
            "candidate_count",
            candidate_count_raw.to_string(),
            candidate_count_resolved.to_string(),
            if candidate_count_raw == 0 {
                ResolvedSource::SentinelExpanded
            } else {
                ResolvedSource::Config
            },
            Some("0 = population × generations"),
        );
        push_field(
            &mut display_fields,
            "search",
            "population",
            s.models.prop_search_population.to_string(),
            s.models.prop_search_population.max(10).to_string(),
            ResolvedSource::Config,
            Some("floor 10; population_auto=true raises it at run start"),
        );
        push_field(
            &mut display_fields,
            "search",
            "population_auto",
            s.models.prop_search_population_auto.to_string(),
            s.models.prop_search_population_auto.to_string(),
            ResolvedSource::Config,
            Some("true + CUDA card: GA population raised to the card's fits ceiling (≤16384) — SEARCHES MORE, results differ; run log records the resolved value"),
        );
        push_field(
            &mut display_fields,
            "search",
            "generations",
            s.models.prop_search_generations.to_string(),
            s.models.prop_search_generations.max(1).to_string(),
            ResolvedSource::Config,
            None,
        );
        push_field(
            &mut display_fields,
            "search",
            "min_trades_per_day",
            min_trades_per_day_raw.to_string(),
            min_trades_per_day_resolved.to_string(),
            if mode == "prop_firm" && min_trades_per_day_raw == 0.0 {
                ResolvedSource::SentinelExpanded
            } else {
                ResolvedSource::Config
            },
            Some("PROP_FIRM mode floors at 0.001 if 0"),
        );
        push_field(
            &mut display_fields,
            "search",
            "discovery_mode",
            s.models.discovery_mode.clone(),
            mode.to_string(),
            ResolvedSource::Config,
            Some("config models.discovery_mode: prop_firm (default) | strict"),
        );
        push_field(
            &mut display_fields,
            "search",
            "normalize_features",
            std::env::var("NEOETHOS_BOT_NORMALIZE_FEATURES").unwrap_or_default(),
            normalize_features_env.to_string(),
            ResolvedSource::EnvOverride,
            Some("NEOETHOS_BOT_NORMALIZE_FEATURES=1; default off"),
        );
        push_field(
            &mut display_fields,
            "search",
            "disable_smc_gate",
            std::env::var("NEOETHOS_BOT_DISABLE_SMC_GATE").unwrap_or_default(),
            disable_smc_gate_env.to_string(),
            ResolvedSource::EnvOverride,
            Some("NEOETHOS_BOT_DISABLE_SMC_GATE=1; diagnostic"),
        );
        push_field(
            &mut display_fields,
            "filters",
            "min_fitness_score",
            "0".to_string(),
            min_fitness_score.to_string(),
            if mode == "prop_firm" {
                ResolvedSource::EnvOverride
            } else {
                ResolvedSource::Default
            },
            Some("renamed from `min_profit`; compared against gene fitness, not net profit"),
        );
        push_field(
            &mut display_fields,
            "filters",
            "min_trades",
            s.models.prop_min_trades.to_string(),
            min_trades.to_string(),
            ResolvedSource::Config,
            None,
        );
        push_field(
            &mut display_fields,
            "filters",
            "max_drawdown",
            "0".to_string(),
            max_drawdown.to_string(),
            if mode == "prop_firm" {
                ResolvedSource::EnvOverride
            } else {
                ResolvedSource::Default
            },
            None,
        );
        push_field(
            &mut display_fields,
            "data",
            "root",
            s.system.data_dir.display().to_string(),
            data_root.clone(),
            ResolvedSource::Config,
            None,
        );
        push_field(
            &mut display_fields,
            "timeframes",
            "base",
            s.system.base_timeframe.clone(),
            s.system.base_timeframe.clone(),
            ResolvedSource::Config,
            None,
        );
        push_field(
            &mut display_fields,
            "timeframes",
            "canonical_default",
            String::new(),
            canonical_default.join(","),
            ResolvedSource::Default,
            Some("includes M3 + M30 per spec"),
        );

        Self {
            data: ResolvedDataConfig {
                root: data_root,
                recursive_scan: true,
                vortex_force_rebuild: false,
                canonical_layout: "data/symbol={SYM}/timeframe={TF}/data.vortex".to_string(),
            },
            timeframes: ResolvedTimeframesConfig {
                base: s.system.base_timeframe.clone(),
                higher: s.system.higher_timeframes.clone(),
                canonical_default,
            },
            search: ResolvedSearchConfig {
                max_indicators_raw,
                max_indicators_resolved,
                candidate_count_raw,
                candidate_count_resolved,
                population: s.models.prop_search_population.max(10),
                population_auto: s.models.prop_search_population_auto,
                generations: s.models.prop_search_generations.max(1),
                portfolio_size: s.models.prop_search_portfolio_size.max(1),
                min_trades_per_day_raw,
                min_trades_per_day_resolved,
                corr_threshold: 0.85,
                walkforward_splits: s.models.walkforward_splits.max(2),
                embargo_minutes: s.models.embargo_minutes,
                mode: mode.to_string(),
                normalize_features_env,
                disable_smc_gate_env,
            },
            filters: ResolvedFiltersConfig {
                min_fitness_score,
                min_trades,
                max_drawdown,
                min_sharpe,
                min_win_rate,
                min_profit_factor: min_pf,
            },
            display_fields,
        }
    }

    /// Render every display field as `(section, field, raw, resolved,
    /// source)` rows for the TUI Config page.
    pub fn display_table(&self) -> Vec<[String; 5]> {
        self.display_fields
            .iter()
            .map(|f| {
                [
                    f.section.to_string(),
                    f.field.to_string(),
                    f.raw.clone(),
                    f.resolved.clone(),
                    f.source.label().to_string(),
                ]
            })
            .collect()
    }
}

fn push_field(
    out: &mut Vec<ResolvedField>,
    section: &'static str,
    field: &'static str,
    raw: String,
    resolved: String,
    source: ResolvedSource,
    note: Option<&'static str>,
) {
    out.push(ResolvedField {
        section: section.to_string(),
        field: field.to_string(),
        raw,
        resolved,
        source,
        note: note.map(|s| s.to_string()),
    });
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Resolve the mode this report DESCRIBES. Must agree, for every `Settings`,
/// with the mode the engine RUNS — `neoethos_search::discovery::
/// resolve_discovery_mode` (discovery.rs:3706).
///
/// CORRECTED 2026-08-04. What this function used to be:
///
/// ```ignore
/// match s.models.discovery_mode.trim().to_ascii_lowercase().as_str() {
///     "strict" => "strict",
///     _ => "prop_firm",
/// }
/// ```
///
/// Two crates, one function name, different inputs. The engine reads BOTH
/// `system.trading_mode` (the operator's master switch, per config.rs:91-102)
/// and `models.discovery_mode` (the power-user escape hatch); this copy read
/// only the escape hatch. So it could never return "risky" at all, and every
/// Risky run was reported as `prop_firm`:
///
///   - `system.trading_mode = "risky"` → engine runs `DiscoveryMode::Risky`,
///     where `apply_mode_overrides` does NOT rewrite the filter floors
///     (discovery.rs:682 gates that rewrite on PropFirm alone), so the engine
///     enforces `FilteringConfig::default()` — maxDD 0.15, sharpe 0.3, win
///     rate 0.50, PF 1.2. The report took the PropFirm display branch and
///     announced maxDD 0.50, sharpe -10.0, win rate 0.0, PF 0.0: a system
///     with essentially no quality filter, which is not the one running.
///   - `models.discovery_mode = "legacy"` → engine `Strict`
///     (`discovery_mode_from_config` accepts "strict" | "legacy"); this copy
///     matched "strict" only and reported `prop_firm`.
///   - The `min_trades_per_day` 0.001 sentinel below is likewise PropFirm-only
///     in the engine, and was being applied to Risky runs in the report.
///
/// The irony is on the record: the 2026-08-03 pass corrected three drifted
/// floor literals in the non-PropFirm branch and wrote "Strict/Risky: no mode
/// override fires" above them — correct values, in a branch Risky could not
/// reach. Fixing the numbers in an unreachable branch is why the mode resolver
/// itself, not just the literals, now has a test:
/// `display_mode_matches_the_engine_mode` in neoethos-search, where both
/// crates are visible.
fn resolve_discovery_mode(s: &Settings) -> &'static str {
    // Precedence MIRRORS the engine: the "strict"/"legacy" escape hatch wins,
    // otherwise the top-level trading mode decides.
    if matches!(
        s.models.discovery_mode.trim().to_ascii_lowercase().as_str(),
        "strict" | "legacy"
    ) {
        return "strict";
    }
    match s.system.trading_mode.trim().to_ascii_lowercase().as_str() {
        "risky" | "growth" => "risky",
        _ => "prop_firm",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_zero_max_indicators_expands_to_all() {
        let mut s = Settings::default();
        s.models.prop_search_max_indicators = 0;
        let r = ResolvedConfig::from_settings(&s);
        assert_eq!(r.search.max_indicators_resolved, usize::MAX);
        let row = r
            .display_fields
            .iter()
            .find(|f| f.field == "max_indicators")
            .expect("max_indicators row");
        assert_eq!(row.source, ResolvedSource::SentinelExpanded);
        assert_eq!(row.raw, "0");
        assert_eq!(row.resolved, "ALL features");
    }

    #[test]
    fn sentinel_zero_candidate_count_expands_to_pop_x_gens() {
        let mut s = Settings::default();
        s.models.prop_search_population = 200;
        s.models.prop_search_generations = 5;
        s.models.prop_search_val_candidates = 0;
        let r = ResolvedConfig::from_settings(&s);
        assert_eq!(r.search.candidate_count_resolved, 1000);
    }

    #[test]
    fn canonical_default_timeframes_include_m3_and_m30() {
        let s = Settings::default();
        let r = ResolvedConfig::from_settings(&s);
        assert!(r.timeframes.canonical_default.contains(&"M3".to_string()));
        assert!(r.timeframes.canonical_default.contains(&"M30".to_string()));
    }

    #[test]
    fn canonical_default_matches_global_canonical_timeframes() {
        let s = Settings::default();
        let r = ResolvedConfig::from_settings(&s);
        let expected: Vec<String> = CANONICAL_TIMEFRAMES
            .iter()
            .map(|tf| (*tf).to_string())
            .collect();
        assert_eq!(r.timeframes.canonical_default, expected);
    }

    #[test]
    fn min_fitness_score_field_label_renamed() {
        let s = Settings::default();
        let r = ResolvedConfig::from_settings(&s);
        assert!(
            r.display_fields
                .iter()
                .any(|f| f.field == "min_fitness_score"),
            "expected min_fitness_score row in display_fields"
        );
    }

    #[test]
    fn display_table_has_canonical_columns() {
        let s = Settings::default();
        let r = ResolvedConfig::from_settings(&s);
        let table = r.display_table();
        assert!(!table.is_empty());
        // Each row is exactly [section, field, raw, resolved, source].
        for row in &table {
            assert_eq!(row.len(), 5);
        }
    }
}
