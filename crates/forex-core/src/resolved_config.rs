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
//!    `config.yaml` cannot tell that `FOREX_BOT_NORMALIZE_FEATURES=1`
//!    or `FOREX_BOT_DISABLE_SMC_GATE=1` flips the search regime.
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
    pub generations: usize,
    pub portfolio_size: usize,
    pub min_trades_per_day_raw: f64,
    pub min_trades_per_day_resolved: f64,
    pub corr_threshold: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub mode: String,
    /// `FOREX_BOT_NORMALIZE_FEATURES` resolved to bool.
    pub normalize_features_env: bool,
    /// `FOREX_BOT_DISABLE_SMC_GATE` resolved to bool.
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
        let mode = resolve_discovery_mode_str();
        let min_trades_per_day_resolved = if mode == "prop_firm" && min_trades_per_day_raw == 0.0 {
            0.001
        } else {
            min_trades_per_day_raw.max(0.0)
        };
        let normalize_features_env = env_truthy("FOREX_BOT_NORMALIZE_FEATURES");
        let disable_smc_gate_env = env_truthy("FOREX_BOT_DISABLE_SMC_GATE");

        // Filters section --------------------------------------------------
        // In PROP_FIRM mode the search engine overrides `filtering`
        // with permissive floors — we report those, not the YAML
        // value, because that's what the GA actually applies.
        let (min_fitness_score, min_trades, max_drawdown, min_sharpe, min_win_rate, min_pf) =
            if mode == "prop_firm" {
                (0.0_f64, 1.0_f64, 0.50_f64, -10.0_f64, 0.0_f64, 0.0_f64)
            } else {
                // Strict mode uses crate::genetic::FilteringConfig::default()
                // values; we mirror them here for display (the actual values
                // are still applied in forex-search).
                (
                    0.0_f64,
                    s.models.prop_min_trades.max(1) as f64,
                    0.20_f64,
                    0.5_f64,
                    0.45_f64,
                    1.2_f64,
                )
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
            None,
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
            std::env::var("FOREX_BOT_DISCOVERY_MODE").unwrap_or_default(),
            mode.to_string(),
            if std::env::var("FOREX_BOT_DISCOVERY_MODE").is_ok() {
                ResolvedSource::EnvOverride
            } else {
                ResolvedSource::Default
            },
            Some("default = prop_firm; FOREX_BOT_DISCOVERY_MODE=strict opts out"),
        );
        push_field(
            &mut display_fields,
            "search",
            "normalize_features",
            std::env::var("FOREX_BOT_NORMALIZE_FEATURES").unwrap_or_default(),
            normalize_features_env.to_string(),
            ResolvedSource::EnvOverride,
            Some("FOREX_BOT_NORMALIZE_FEATURES=1; default off"),
        );
        push_field(
            &mut display_fields,
            "search",
            "disable_smc_gate",
            std::env::var("FOREX_BOT_DISABLE_SMC_GATE").unwrap_or_default(),
            disable_smc_gate_env.to_string(),
            ResolvedSource::EnvOverride,
            Some("FOREX_BOT_DISABLE_SMC_GATE=1; diagnostic"),
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

fn resolve_discovery_mode_str() -> &'static str {
    match std::env::var("FOREX_BOT_DISCOVERY_MODE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "strict" => "strict",
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
