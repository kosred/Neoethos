//! Typed boundary for the legacy `NEOETHOS_BOT_*` env vars that previously
//! reached deep into the genetic search engine on every call. These knobs
//! change *production* search semantics — RNG seed, novelty weighting,
//! tournament size, stagnation patience, archive capacity, SMC gate
//! shaping, archive scoring thresholds, and selection-policy weighting —
//! so the audit (P0-8) requires them to live in typed config rather than
//! ambient env state.
//!
//! 2026-08-10 — the env half is GONE. This struct is the single owner of
//! those values and its only reader is
//! [`GeneticSearchRuntimeOverrides::from_settings`]; production binaries
//! install it once at startup via
//! [`install_genetic_search_runtime_overrides_from_settings`]. The historical
//! variable names are kept in the field docs below because that is what
//! someone hunting a knob will search for.

use super::evolution_math::{ParentSelectionPolicy, SurvivorSelectionPolicy};
use neoethos_core::contracts::DeterminismPolicy;
use serde::Serialize;
use std::sync::OnceLock;

/// SMC gate curve knobs. The gate threshold starts at `start`, eases to
/// `end` along a power curve of exponent `curve`, and relaxes by
/// `stagnation_step` per stagnant generation once the patience window has
/// been exceeded.
///
/// `disable_gate` is the operator's hard-bypass escape hatch (legacy
/// `NEOETHOS_BOT_DISABLE_SMC_GATE=1` env var, now read once at startup
/// through this typed boundary): when set, the gate collapses (active
/// SMC sum forced to 0) so the raw signal passes through. Lets operators
/// isolate "SMC indicators don't trigger on this symbol" from genuine
/// signal-generation issues without recompiling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SmcGateOverrides {
    pub start: f64,
    pub end: f64,
    pub curve: f64,
    pub stagnation_step: f64,
    pub disable_gate: bool,
}

impl Default for SmcGateOverrides {
    fn default() -> Self {
        Self {
            start: 0.75,
            end: 0.35,
            curve: 1.0,
            stagnation_step: 0.03,
            disable_gate: false,
        }
    }
}

impl SmcGateOverrides {
    fn resolved_curve(&self) -> f64 {
        if self.curve.is_finite() && self.curve >= 0.1 {
            self.curve
        } else {
            0.1
        }
    }

    fn resolved_stagnation_step(&self) -> f64 {
        if self.stagnation_step.is_finite() && self.stagnation_step >= 0.0 {
            self.stagnation_step
        } else {
            0.0
        }
    }
}

/// Archive scoring thresholds. `mode` selects which metric is used to gate
/// archive admission ("net", "pf", "sharpe"); the corresponding `min_*`
/// floors must be cleared before a candidate is archived.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArchiveScoringOverrides {
    pub mode: String,
    pub min_net: f64,
    pub min_pf: f64,
    pub min_sharpe: f64,
}

impl Default for ArchiveScoringOverrides {
    fn default() -> Self {
        Self {
            mode: "net".to_string(),
            min_net: 0.0,
            min_pf: 1.0,
            min_sharpe: 0.0,
        }
    }
}

/// Selection-policy knobs that previously lived in
/// `NEOETHOS_BOT_PROP_PARENT_SELECTION` / `SURVIVOR_SELECTION` /
/// `RANDOM_IMMIGRANTS` / `SURVIVOR_FRACTION` (or `ELITE_FRACTION`) /
/// `SELECTION_TEMPERATURE`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SelectionPolicyOverrides {
    pub parent: ParentSelectionPolicy,
    pub survivor: SurvivorSelectionPolicy,
    pub immigrant_ratio: f64,
    pub survivor_fraction: f64,
    pub temperature: f64,
}

impl Default for SelectionPolicyOverrides {
    fn default() -> Self {
        Self {
            parent: ParentSelectionPolicy::RankWeighted,
            survivor: SurvivorSelectionPolicy::RankWeighted,
            immigrant_ratio: 0.25,
            survivor_fraction: 0.10,
            temperature: 0.75,
        }
    }
}

impl SelectionPolicyOverrides {
    fn resolved_immigrant_ratio(&self) -> f64 {
        if self.immigrant_ratio.is_finite() {
            self.immigrant_ratio.clamp(0.0, 0.95)
        } else {
            0.25
        }
    }

    fn resolved_survivor_fraction(&self) -> f64 {
        if self.survivor_fraction.is_finite() {
            self.survivor_fraction.clamp(0.0, 0.95)
        } else {
            0.10
        }
    }

    fn resolved_temperature(&self) -> f64 {
        if self.temperature.is_finite() {
            self.temperature.max(1e-3)
        } else {
            0.75
        }
    }
}

/// Typed replacement for the search-engine's most production-affecting
/// `NEOETHOS_BOT_*` env vars.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeneticSearchRuntimeOverrides {
    /// Optional deterministic RNG seed for the genetic search. `None`
    /// means "seed from the OS RNG" (non-deterministic).
    pub seed: Option<u64>,
    /// Novelty bonus weight applied during candidate ranking. `0.0`
    /// disables novelty scoring (default).
    pub novelty_weight: f64,
    /// Exact `k` for mean k-nearest-neighbor novelty over the current
    /// population plus the permanent archive. It is explicit because changing
    /// `k` changes selection and therefore the run identity.
    pub novelty_neighbors: usize,
    /// Number of stagnant generations the search tolerates before
    /// triggering the SOFT diversity kick / gate-relaxation. Always at
    /// least `1`. (The HARD early-stop is the separate, larger
    /// `convergence_patience`.)
    pub stagnation_patience: usize,
    /// Generations of no meaningful improvement before the GA HARD
    /// early-stops the combo (returns the archive so the auto-loop advances
    /// to the next symbol×timeframe). `0` disables. Distinct from — and
    /// larger than — `stagnation_patience`, which only triggers the soft
    /// diversity kick.
    pub convergence_patience: usize,
    /// Minimum top-fitness increase counted as an improvement when tracking
    /// stagnation (replaces the legacy hard-coded `1e-12`).
    pub min_improvement: f64,
    /// Wall-clock floor for the convergence early-stop, as a fraction of the
    /// per-combo time budget. The early-stop fires only after this fraction
    /// of `max_runtime` has elapsed — makes it throughput-robust so fast
    /// timeframes (where 250 gens ≈ 1 s) are not killed before they search.
    pub convergence_min_elapsed_fraction: f64,
    /// Optional explicit tournament size for tournament-based selection.
    /// `None` means "derive from population" (`max(population/12, 3)`).
    pub tournament_size_override: Option<usize>,
    /// Optional explicit archive capacity. `None` means
    /// "derive from population × generations" with audit-aligned bounds.
    pub archive_cap_override: Option<usize>,
    /// Number of times the seen-signature memory retries to draw a unique
    /// gene before giving up.
    pub seen_retry_attempts: usize,
    pub smc_gate: SmcGateOverrides,
    pub archive_scoring: ArchiveScoringOverrides,
    pub selection: SelectionPolicyOverrides,
}

impl Default for GeneticSearchRuntimeOverrides {
    fn default() -> Self {
        Self {
            seed: None,
            novelty_weight: 0.0,
            novelty_neighbors: 15,
            stagnation_patience: 2,
            convergence_patience: 250,
            min_improvement: 1e-12,
            convergence_min_elapsed_fraction: 0.5,
            tournament_size_override: None,
            archive_cap_override: None,
            seen_retry_attempts: 16,
            smc_gate: SmcGateOverrides::default(),
            archive_scoring: ArchiveScoringOverrides::default(),
            selection: SelectionPolicyOverrides::default(),
        }
    }
}

impl GeneticSearchRuntimeOverrides {
    // `from_env()` DELETED 2026-08-10. It carried 22 `NEOETHOS_BOT_*` names —
    // the RNG SEED, the novelty weight, early-stop patience, tournament size,
    // archive cap and mode, the SMC gate curve, the SMC-gate hard bypass, and
    // the whole parent/survivor selection policy. Every one of them changes
    // which genes are created and which survive, so a run's meaning depended
    // on a shell nothing recorded. All of them are typed on
    // `models.search_runtime` and installed by
    // `install_genetic_search_runtime_overrides_from_settings`.

    /// Config-driven constructor — the operator sets these knobs in the
    /// single `Settings` (config / UI / TUI), never the environment.
    /// Mirrors the deleted `from_env` reader field-for-field; an empty policy /
    /// archive-mode string means "keep the engine default" so the config
    /// default need not duplicate the parser vocabulary. The
    /// `from_settings_default_matches_env_default` test guarantees a
    /// fresh `Settings` reproduces [`Self::default`] exactly (no behavior
    /// change vs the pre-config build).
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.search_runtime;
        let defaults = Self::default();
        Self {
            seed: c.seed,
            novelty_weight: c.novelty_weight,
            novelty_neighbors: c.novelty_neighbors,
            stagnation_patience: c.stagnation_patience,
            convergence_patience: c.convergence_patience,
            min_improvement: c.min_improvement,
            convergence_min_elapsed_fraction: c.convergence_min_elapsed_fraction,
            tournament_size_override: c.tournament_size_override,
            archive_cap_override: c.archive_cap_override,
            seen_retry_attempts: c.seen_retry_attempts,
            smc_gate: SmcGateOverrides {
                start: c.smc_gate_start,
                end: c.smc_gate_end,
                curve: c.smc_gate_curve,
                stagnation_step: c.smc_gate_stagnation_step,
                disable_gate: c.disable_smc_gate,
            },
            archive_scoring: ArchiveScoringOverrides {
                mode: if c.archive_mode.trim().is_empty() {
                    defaults.archive_scoring.mode.clone()
                } else {
                    c.archive_mode.trim().to_ascii_lowercase()
                },
                min_net: c.archive_min_net,
                min_pf: c.archive_min_pf,
                min_sharpe: c.archive_min_sharpe,
            },
            selection: SelectionPolicyOverrides {
                parent: if c.parent_selection.trim().is_empty() {
                    defaults.selection.parent
                } else {
                    ParentSelectionPolicy::parse(&c.parent_selection.trim().to_ascii_lowercase())
                },
                survivor: if c.survivor_selection.trim().is_empty() {
                    defaults.selection.survivor
                } else {
                    SurvivorSelectionPolicy::parse(
                        &c.survivor_selection.trim().to_ascii_lowercase(),
                    )
                },
                immigrant_ratio: c.immigrant_ratio,
                survivor_fraction: c.survivor_fraction,
                temperature: c.selection_temperature,
            },
        }
    }

    /// Resolved SMC gate fields with audit-aligned clamping applied.
    pub fn resolved_smc_gate(&self) -> SmcGateOverrides {
        SmcGateOverrides {
            start: self.smc_gate.start,
            end: self.smc_gate.end,
            curve: self.smc_gate.resolved_curve(),
            stagnation_step: self.smc_gate.resolved_stagnation_step(),
            disable_gate: self.smc_gate.disable_gate,
        }
    }

    /// Resolved selection-policy fields with audit-aligned clamping
    /// applied (immigrant ratio + survivor fraction in `[0, 0.95]`,
    /// temperature ≥ 1e-3).
    pub fn resolved_selection(&self) -> SelectionPolicyOverrides {
        SelectionPolicyOverrides {
            parent: self.selection.parent,
            survivor: self.selection.survivor,
            immigrant_ratio: self.selection.resolved_immigrant_ratio(),
            survivor_fraction: self.selection.resolved_survivor_fraction(),
            temperature: self.selection.resolved_temperature(),
        }
    }

    /// Number of unique-candidate retry attempts. Always at least `1`.
    pub fn effective_seen_retry_attempts(&self) -> usize {
        self.seen_retry_attempts.max(1)
    }

    /// Resolve the effective tournament size for the given population. The
    /// minimum is always `2` regardless of override values, which matches
    /// the tournament-selection pre-condition.
    pub fn effective_tournament_size(&self, population: usize) -> usize {
        self.tournament_size_override
            .unwrap_or_else(|| (population / 12).max(3))
            .max(2)
    }

    /// Resolve the effective archive cap for a population × generations
    /// product. The cap is always at least `population` and capped at
    /// `200_000` to prevent memory blow-ups on very long HPC runs.
    pub fn effective_archive_cap(&self, population: usize, generations: usize) -> usize {
        let derived = (population * generations.max(1)).min(50_000);
        let raw = self.archive_cap_override.unwrap_or(derived);
        raw.max(population).min(200_000)
    }

    /// Checked form used by pre-allocation resident admission. It refuses an
    /// unrepresentable population x generation product rather than silently
    /// changing the Search identity.
    pub fn checked_effective_archive_cap(
        &self,
        population: usize,
        generations: usize,
    ) -> Option<usize> {
        let derived = population.checked_mul(generations.max(1))?.min(50_000);
        let raw = self.archive_cap_override.unwrap_or(derived);
        Some(raw.max(population).min(200_000))
    }

    /// Resolve the effective stagnation patience, guaranteeing a minimum
    /// of `1` so callers do not need to clamp themselves.
    pub fn effective_stagnation_patience(&self) -> usize {
        self.stagnation_patience.max(1)
    }

    /// Resolve the convergence early-stop patience. `0` means "disabled"
    /// (the GA runs to the time / generation cap as before); any positive
    /// value is the number of flat generations after which the combo is
    /// hard-stopped.
    pub fn effective_convergence_patience(&self) -> usize {
        self.convergence_patience
    }

    /// Resolve the stagnation improvement epsilon, guarding against
    /// non-finite / negative configured values (falls back to the legacy
    /// `1e-12`).
    pub fn effective_min_improvement(&self) -> f64 {
        if self.min_improvement.is_finite() && self.min_improvement >= 0.0 {
            self.min_improvement
        } else {
            1e-12
        }
    }

    /// Resolve the convergence wall-clock floor fraction, clamped to
    /// `[0.0, 1.0]`. Non-finite / out-of-range values fall back to the safe
    /// default of `0.5` (every combo gets at least half its time budget
    /// before the early-stop can fire).
    pub fn effective_convergence_min_elapsed_fraction(&self) -> f64 {
        let f = self.convergence_min_elapsed_fraction;
        if f.is_finite() && (0.0..=1.0).contains(&f) {
            f
        } else {
            0.5
        }
    }

    /// Resolve the legacy `seed: Option<u64>` field into the canonical
    /// [`DeterminismPolicy`] enum from `neoethos-core::contracts`. `Some(seed)`
    /// maps to `Deterministic { seed }`; `None` maps to
    /// `NonDeterministicAllowed` (the existing behavior is to seed from
    /// the OS RNG when no seed is configured). Callers that want the
    /// `BestEffort` mode should install it directly via
    /// [`GeneticSearchRuntimeOverrides`] and then provide an explicit
    /// `BestEffort` decision through the typed accessor.
    pub fn determinism_policy(&self) -> DeterminismPolicy {
        match self.seed {
            Some(seed) => DeterminismPolicy::Deterministic { seed },
            None => DeterminismPolicy::NonDeterministicAllowed,
        }
    }
}

/// Optional symbol/currency/cost knobs that the legacy
/// `NEOETHOS_BOT_PROP_SYMBOL` / `ACCOUNT_CURRENCY` / `PIP_VALUE` /
/// `QUOTE_TO_ACCOUNT_RATE` / `PIP_VALUE_PER_LOT` / `SPREAD_PIPS` /
/// `COMMISSION` env vars used to populate. Each field is `None` when no
/// override has been installed; production callers that pass explicit
/// values continue to bypass these fallbacks.
///
/// `reject_pip_fallback` mirrors the legacy `NEOETHOS_BOT_REJECT_PIP_FALLBACK=1`
/// env var (F-CORE3 closure, 2026-05-25): when set, the cross-pair
/// pip-value fallback `bail!()`s instead of silently returning the
/// quote-currency pip value. Previously read inline inside
/// `reject_cross_pair_fallback()` (strategy_gene.rs); now consolidated
/// at this typed boundary so the env is hit at most once per process.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CostProfileRuntimeOverrides {
    pub symbol: Option<String>,
    pub account_currency: Option<String>,
    pub pip_value: Option<f64>,
    pub quote_to_account_rate: Option<f64>,
    pub pip_value_per_lot: Option<f64>,
    pub spread_pips: Option<f64>,
    pub commission_per_trade: Option<f64>,
    /// Is a BROKER-SOURCED per-lot commission quoted per side?
    ///
    /// Mirrors `risk.commission_per_lot_is_per_side`. It governs only the
    /// broker-metadata and synthetic-fallback arms of
    /// `infer_market_cost_profile` — `commission_per_trade` above is a caller's
    /// value and is already a round trip by the name of the field, so it is
    /// never doubled. See
    /// [`crate::genetic::strategy_gene::round_trip_commission_per_lot`] for
    /// why one subtraction per closed trade means this conversion has to happen
    /// somewhere, and why it happens at exactly two places.
    pub commission_is_per_side: bool,
    pub reject_pip_fallback: bool,
}

impl Default for CostProfileRuntimeOverrides {
    /// Hand-written rather than derived for one field: `reject_pip_fallback`
    /// defaults to `true`, so an unconfigured process refuses a pip value it
    /// cannot express in the account currency instead of booking a foreign
    /// amount as if it were account currency. The derived `false` matched the
    /// old lenient behaviour, and the equality gate against
    /// `from_settings(&Settings::default())` is what keeps the two boundaries
    /// from drifting apart again.
    fn default() -> Self {
        Self {
            symbol: None,
            account_currency: None,
            pip_value: None,
            quote_to_account_rate: None,
            pip_value_per_lot: None,
            spread_pips: None,
            commission_per_trade: None,
            // A broker quotes commission per side. Matches
            // `RiskConfig::commission_per_lot_is_per_side`; the
            // `from_settings(&Settings::default()) == default()` gate keeps the
            // two from drifting.
            commission_is_per_side: true,
            reject_pip_fallback: true,
        }
    }
}

impl CostProfileRuntimeOverrides {
    // `populate_from_env` DELETED 2026-08-10 — see the note on
    // `StrategyEvaluationRuntimeOverrides`.
}

/// SMC weight knobs that previously lived in the
/// `NEOETHOS_BOT_PROP_SMC_W_*` env vars and the `NEOETHOS_BOT_PROP_SMC_GATE`
/// fallback used by `EvaluationConfig::default`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SmcWeightRuntimeOverrides {
    pub gate_threshold: f64,
    pub w_ob: f64,
    pub w_fvg: f64,
    pub w_liq: f64,
    pub w_mtf: f64,
    pub w_premium: f64,
    pub w_inducement: f64,
    pub w_bos: f64,
    pub w_choch: f64,
    pub w_eqh: f64,
    pub w_eql: f64,
    pub w_displacement: f64,
}

impl Default for SmcWeightRuntimeOverrides {
    fn default() -> Self {
        Self {
            gate_threshold: 0.75,
            w_ob: 1.0,
            w_fvg: 1.0,
            w_liq: 1.0,
            w_mtf: 1.0,
            w_premium: 1.0,
            w_inducement: 1.0,
            w_bos: 1.0,
            w_choch: 1.0,
            w_eqh: 1.0,
            w_eql: 1.0,
            w_displacement: 1.0,
        }
    }
}

impl SmcWeightRuntimeOverrides {
    // `populate_from_env` DELETED 2026-08-10 — see the note on
    // `StrategyEvaluationRuntimeOverrides`.
}

/// Typed runtime overrides for `EvaluationConfig::default` and
/// `infer_market_cost_profile`. Cost knobs replace the
/// `NEOETHOS_BOT_PROP_SYMBOL` / `ACCOUNT_CURRENCY` / `PIP_VALUE` /
/// `QUOTE_TO_ACCOUNT_RATE` / `PIP_VALUE_PER_LOT` / `SPREAD_PIPS` /
/// `COMMISSION` env vars; SMC weight knobs replace the
/// `NEOETHOS_BOT_PROP_SMC_W_*` and `NEOETHOS_BOT_PROP_SMC_GATE` env vars used
/// at evaluation-config construction time.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct StrategyEvaluationRuntimeOverrides {
    pub cost_profile: CostProfileRuntimeOverrides,
    pub smc_weights: SmcWeightRuntimeOverrides,
    /// Exit geometry. Config recipient for what `EvaluationConfig::for_symbol`
    /// hardcoded from 2026-06-06 until 2026-08-09. See [`ExitPolicyOverrides`].
    pub exit_policy: ExitPolicyOverrides,
}

/// The trailing-stop geometry every discovery evaluation runs under — the typed
/// mirror of `neoethos_core::config::ExitPolicyConfig`.
///
/// BEHAVIOUR CHANGE, stated explicitly (2026-08-09). `trailing_enabled` defaults
/// to `false`; it was an unreachable `true` literal in
/// `strategy_gene.rs:851`. What this PERMITS: the take-profit is now reachable,
/// so a realised payoff above ~1.1 is expressible at all — measured on real
/// EURUSD bars, the old geometry produced payoff 0.87 at both tp 45 and tp 300
/// (average win 6.10 vs 6.11 pips), i.e. the take-profit was dead code and the
/// configured 2.0 payoff floor was unreachable by construction. What it REFUSES:
/// the automatic move-to-break-even at +1R that the old comment credited with
/// lowering drawdown enough to clear the prop-firm gate. That effect was real and
/// is now gone unless the operator asks for it back.
///
/// It buys NO expected profit. Across every trailing configuration measured,
/// expectancy stayed at -4.15 pips per trade while payoff moved 0.91 → 2.53.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ExitPolicyOverrides {
    pub trailing_enabled: bool,
    pub trailing_be_trigger_r: f64,
    /// A multiple of the position's own stop distance, NOT of ATR, despite the
    /// `trailing_atr_multiplier` name it is copied into downstream
    /// (`eval.rs:1030-1035`, and both GPU kernels bind that name).
    pub trailing_stop_multiplier: f64,
    pub trailing_min_lock_pips: f64,
}

impl Default for ExitPolicyOverrides {
    fn default() -> Self {
        Self {
            trailing_enabled: false,
            trailing_be_trigger_r: 1.0,
            trailing_stop_multiplier: 1.0,
            trailing_min_lock_pips: 2.0,
        }
    }
}

impl ExitPolicyOverrides {
    /// Config-driven constructor. Non-finite or negative values are refused in
    /// favour of the default rather than propagated — a NaN trigger would arm
    /// the trail never, silently, on every gene.
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.exit_policy;
        let d = Self::default();
        let sane = |value: f64, fallback: f64| {
            if value.is_finite() && value >= 0.0 {
                value
            } else {
                fallback
            }
        };
        Self {
            trailing_enabled: c.trailing_enabled,
            trailing_be_trigger_r: sane(c.trailing_be_trigger_r, d.trailing_be_trigger_r),
            trailing_stop_multiplier: sane(c.trailing_stop_multiplier, d.trailing_stop_multiplier),
            trailing_min_lock_pips: sane(c.trailing_min_lock_pips, d.trailing_min_lock_pips),
        }
    }
}

impl StrategyEvaluationRuntimeOverrides {
    // `from_env()` DELETED 2026-08-10 together with the two
    // `populate_from_env` helpers it called. Between them they carried the
    // COST PROFILE (symbol, account currency, pip value, spread, commission,
    // slippage) and the twelve SMC scoring weights plus the gate threshold —
    // i.e. the numbers a candidate's P&L is computed from. All are typed on
    // `models.eval_runtime` and installed by
    // `install_strategy_evaluation_runtime_overrides_from_settings`.

    /// Config-driven constructor — reads the cost-profile + SMC-weight
    /// knobs from the single `Settings` (config / UI / TUI) instead of the
    /// environment. Mirrors the deleted `from_env` reader; `None` cost fields stay
    /// `None`. Numeric cost overrides are validated the same way the env
    /// reader validated them (positive / non-negative finite). A
    /// `from_settings(&Settings::default()) == default()` test guarantees
    /// no behavior change vs the pre-config build.
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.eval_runtime;
        Self {
            cost_profile: CostProfileRuntimeOverrides {
                symbol: c
                    .symbol
                    .clone()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty()),
                account_currency: c
                    .account_currency
                    .clone()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty()),
                pip_value: c.pip_value.filter(|v| v.is_finite() && *v > 0.0),
                quote_to_account_rate: c
                    .quote_to_account_rate
                    .filter(|v| v.is_finite() && *v > 0.0),
                pip_value_per_lot: c.pip_value_per_lot.filter(|v| v.is_finite() && *v > 0.0),
                spread_pips: c.spread_pips.filter(|v| v.is_finite() && *v >= 0.0),
                commission_per_trade: c
                    .commission_per_trade
                    .filter(|v| v.is_finite() && *v >= 0.0),
                // Lives on `risk`, not on `models.eval_runtime`: it describes
                // the BROKER's quoting convention, which is a property of the
                // account, and `risk.commission_per_lot` is the number it
                // qualifies. Read from there so there is one answer.
                commission_is_per_side: s.risk.commission_per_lot_is_per_side,
                reject_pip_fallback: c.reject_pip_fallback,
            },
            smc_weights: SmcWeightRuntimeOverrides {
                gate_threshold: c.smc_gate_threshold,
                w_ob: c.smc_w_ob,
                w_fvg: c.smc_w_fvg,
                w_liq: c.smc_w_liq,
                w_mtf: c.smc_w_mtf,
                w_premium: c.smc_w_premium,
                w_inducement: c.smc_w_inducement,
                w_bos: c.smc_w_bos,
                w_choch: c.smc_w_choch,
                w_eqh: c.smc_w_eqh,
                w_eql: c.smc_w_eql,
                w_displacement: c.smc_w_displacement,
            },
            exit_policy: ExitPolicyOverrides::from_settings(s),
        }
    }
}

/// The stop/target band gene generation and mutation may draw within — the typed
/// mirror of `neoethos_core::config::GeneStopBoundsConfig`.
///
/// The multiples live here (process-wide, installed once from `Settings`); the
/// per-run ATR that turns them into pips is a separate replaceable cell in
/// `evolution_math`, because it is a property of the DATASET, not of the config,
/// and the batch orchestrator runs many (symbol, timeframe) combos in one
/// process. Keeping them apart is what stops an M5 scale leaking into an H4 run —
/// the exact defect audit D06 found in the adaptive threshold ladder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GeneStopBoundsOverrides {
    pub atr_scaled: bool,
    pub sl_min_atr: f64,
    pub sl_max_atr: f64,
    pub rr_min: f64,
    pub rr_max: f64,
    pub sl_min_pips: f64,
    pub sl_max_pips: f64,
    pub tp_min_pips: f64,
    pub tp_max_pips: f64,
}

impl Default for GeneStopBoundsOverrides {
    fn default() -> Self {
        Self {
            atr_scaled: true,
            sl_min_atr: 1.0,
            sl_max_atr: 4.0,
            rr_min: 1.5,
            rr_max: 4.0,
            sl_min_pips: 6.0,
            sl_max_pips: 20.0,
            tp_min_pips: 12.0,
            tp_max_pips: 45.0,
        }
    }
}

impl GeneStopBoundsOverrides {
    /// Config-driven constructor. Every band is repaired to a usable ordering
    /// rather than accepted as written: an inverted or non-finite band would
    /// make `clamp` panic deep inside gene mutation, which is a crash in the
    /// middle of a multi-hour run over a typo in a YAML file.
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.gene_stop_bounds;
        let d = Self::default();
        let pos = |value: f64, fallback: f64| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                fallback
            }
        };
        let sl_min_atr = pos(c.sl_min_atr, d.sl_min_atr);
        let sl_min_pips = pos(c.sl_min_pips, d.sl_min_pips);
        let tp_min_pips = pos(c.tp_min_pips, d.tp_min_pips);
        let rr_min = pos(c.rr_min, d.rr_min);
        Self {
            atr_scaled: c.atr_scaled,
            sl_min_atr,
            sl_max_atr: pos(c.sl_max_atr, d.sl_max_atr).max(sl_min_atr),
            rr_min,
            rr_max: pos(c.rr_max, d.rr_max).max(rr_min),
            sl_min_pips,
            sl_max_pips: pos(c.sl_max_pips, d.sl_max_pips).max(sl_min_pips),
            tp_min_pips,
            tp_max_pips: pos(c.tp_max_pips, d.tp_max_pips).max(tp_min_pips),
        }
    }
}

static GENE_STOP_BOUNDS_OVERRIDES: OnceLock<GeneStopBoundsOverrides> = OnceLock::new();

/// Config-driven install of the stop/target band multiples. Idempotent — called
/// once at startup from `install_search_runtime_overrides_from_settings`.
pub fn install_gene_stop_bounds_overrides_from_settings(s: &neoethos_core::Settings) {
    let _ = GENE_STOP_BOUNDS_OVERRIDES.set(GeneStopBoundsOverrides::from_settings(s));
}

/// The installed band multiples, or the deterministic defaults when nothing was
/// installed (the `neoethos-models` GA and every test fixture land here).
pub fn current_gene_stop_bounds_overrides() -> GeneStopBoundsOverrides {
    GENE_STOP_BOUNDS_OVERRIDES
        .get()
        .copied()
        .unwrap_or_default()
}

static STRATEGY_EVALUATION_RUNTIME_OVERRIDES: OnceLock<StrategyEvaluationRuntimeOverrides> =
    OnceLock::new();

/// Install process-wide strategy-evaluation runtime overrides. Returns
/// `Err(existing)` when overrides were already installed earlier (the
/// first install wins).
pub fn install_strategy_evaluation_runtime_overrides(
    overrides: StrategyEvaluationRuntimeOverrides,
) -> Result<(), StrategyEvaluationRuntimeOverrides> {
    STRATEGY_EVALUATION_RUNTIME_OVERRIDES.set(overrides)
}

/// Config-driven install — reads the strategy-evaluation knobs from the
/// single `Settings` instead of the environment. Idempotent.
pub fn install_strategy_evaluation_runtime_overrides_from_settings(s: &neoethos_core::Settings) {
    let _ = STRATEGY_EVALUATION_RUNTIME_OVERRIDES
        .set(StrategyEvaluationRuntimeOverrides::from_settings(s));
}

/// Returns the currently installed strategy-evaluation runtime
/// overrides, or the deterministic defaults when no install has happened.
pub fn current_strategy_evaluation_runtime_overrides() -> StrategyEvaluationRuntimeOverrides {
    STRATEGY_EVALUATION_RUNTIME_OVERRIDES
        .get()
        .cloned()
        .unwrap_or_default()
}

// The nine `env_*` parsers that lived here (env_u64, env_string_nonempty,
// env_f64_positive_finite, env_f64_non_negative_finite, env_usize_positive,
// env_f64_finite, env_f32_finite, env_string_lowercase, env_truthy) were
// DELETED 2026-08-10 with their last caller. Nothing in this crate reads the
// environment for a knob any more; `crates/neoethos-search/tests
// /env_surface_is_empty.rs` is the ratchet that keeps it that way.

static GENETIC_SEARCH_RUNTIME_OVERRIDES: OnceLock<GeneticSearchRuntimeOverrides> = OnceLock::new();

/// Install process-wide genetic search runtime overrides. Returns
/// `Err(existing)` when overrides were already installed earlier (the
/// first install wins).
pub fn install_genetic_search_runtime_overrides(
    overrides: GeneticSearchRuntimeOverrides,
) -> Result<(), GeneticSearchRuntimeOverrides> {
    GENETIC_SEARCH_RUNTIME_OVERRIDES.set(overrides)
}

/// Config-driven install — reads the genetic-search knobs from the single
/// `Settings` instead of the environment. Idempotent (first install wins).
pub fn install_genetic_search_runtime_overrides_from_settings(s: &neoethos_core::Settings) {
    let _ = GENETIC_SEARCH_RUNTIME_OVERRIDES.set(GeneticSearchRuntimeOverrides::from_settings(s));
}

/// Returns the currently installed genetic search runtime overrides, or
/// the deterministic defaults when no install has happened.
pub fn current_genetic_search_runtime_overrides() -> GeneticSearchRuntimeOverrides {
    GENETIC_SEARCH_RUNTIME_OVERRIDES
        .get()
        .cloned()
        .unwrap_or_default()
}

/// Whether the SMC gate is bypassed — the ONE field the per-gene signal-synth
/// hot path needs.
///
/// Perf (2026-07-22): the per-gene synthesis read this via
/// `current_genetic_search_runtime_overrides()`, which CLONES the whole
/// overrides struct — and that struct owns a `String` (`archive_scoring.mode`),
/// so every gene evaluation heap-allocated a String just to read one bool. In a
/// GA run that is population × generations allocations per combo (e.g.
/// 200 × 450 = 90k), pure churn on the hottest CPU path. This borrows the
/// installed value and copies out only the bool — no allocation.
pub fn smc_gate_disabled() -> bool {
    GENETIC_SEARCH_RUNTIME_OVERRIDES
        .get()
        .map(|o| o.smc_gate.disable_gate)
        .unwrap_or_else(|| SmcGateOverrides::default().disable_gate)
}

/// Convenience accessor returning the canonical
/// [`neoethos_core::contracts::DeterminismPolicy`] derived from the
/// installed genetic-search runtime overrides. Production callers can
/// route this through `ArtifactProvenance` so persisted artifacts
/// document the determinism mode used to produce them.
pub fn current_determinism_policy() -> DeterminismPolicy {
    current_genetic_search_runtime_overrides().determinism_policy()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smc_gate_disabled_matches_the_full_accessor() {
        // The lean per-gene accessor must return exactly what reading the field
        // off the full (cloned) struct would — it only skips the allocation.
        assert_eq!(
            smc_gate_disabled(),
            current_genetic_search_runtime_overrides()
                .smc_gate
                .disable_gate,
        );
        // And with nothing installed it falls back to the default, same as the
        // full path's `unwrap_or_default`.
        assert_eq!(
            smc_gate_disabled(),
            GeneticSearchRuntimeOverrides::default()
                .smc_gate
                .disable_gate,
        );
    }

    #[test]
    fn defaults_match_legacy_env_defaults() {
        let defaults = GeneticSearchRuntimeOverrides::default();
        assert_eq!(defaults.seed, None);
        assert!((defaults.novelty_weight - 0.0).abs() < 1e-9);
        assert_eq!(defaults.novelty_neighbors, 15);
        assert_eq!(defaults.stagnation_patience, 2);
        assert_eq!(defaults.convergence_patience, 250);
        assert!((defaults.min_improvement - 1e-12).abs() < 1e-18);
        assert!((defaults.convergence_min_elapsed_fraction - 0.5).abs() < 1e-9);
        assert_eq!(defaults.tournament_size_override, None);
        assert_eq!(defaults.archive_cap_override, None);
        assert_eq!(defaults.seen_retry_attempts, 16);
        assert!((defaults.smc_gate.start - 0.75).abs() < 1e-6);
        assert!((defaults.smc_gate.end - 0.35).abs() < 1e-6);
        assert!((defaults.smc_gate.curve - 1.0).abs() < 1e-6);
        assert!((defaults.smc_gate.stagnation_step - 0.03).abs() < 1e-6);
        assert_eq!(defaults.archive_scoring.mode, "net");
        assert!((defaults.archive_scoring.min_net - 0.0).abs() < 1e-9);
        assert!((defaults.archive_scoring.min_pf - 1.0).abs() < 1e-9);
        assert!((defaults.archive_scoring.min_sharpe - 0.0).abs() < 1e-9);
        assert!((defaults.selection.immigrant_ratio - 0.25).abs() < 1e-9);
        assert!((defaults.selection.survivor_fraction - 0.10).abs() < 1e-9);
        assert!((defaults.selection.temperature - 0.75).abs() < 1e-9);
    }

    #[test]
    fn from_settings_default_matches_env_default() {
        // Behavior-preservation gate: an operator who sets nothing must
        // get byte-identical overrides to the pre-config (env-default)
        // build. Guards the duplicated defaults in
        // `neoethos_core::config::SearchRuntimeConfig::default()`.
        let s = neoethos_core::Settings::default();
        assert_eq!(
            GeneticSearchRuntimeOverrides::from_settings(&s),
            GeneticSearchRuntimeOverrides::default()
        );
    }

    #[test]
    fn strategy_eval_from_settings_default_matches_env_default() {
        // Behavior-preservation gate for the eval (cost-profile + SMC
        // weight) knobs — a fresh `Settings` must reproduce the engine
        // defaults exactly.
        let s = neoethos_core::Settings::default();
        assert_eq!(
            StrategyEvaluationRuntimeOverrides::from_settings(&s),
            StrategyEvaluationRuntimeOverrides::default()
        );
    }

    #[test]
    fn gene_stop_bounds_from_settings_default_matches_default() {
        // Same behaviour-preservation gate as the two above: the duplicated
        // defaults in `neoethos_core::config::GeneStopBoundsConfig` and here must
        // not drift apart, or the band the GA draws within depends on whether a
        // config file was loaded.
        let s = neoethos_core::Settings::default();
        assert_eq!(
            GeneStopBoundsOverrides::from_settings(&s),
            GeneStopBoundsOverrides::default()
        );
    }

    #[test]
    fn the_discovery_trail_is_off_unless_the_operator_asks_for_it() {
        // The point of the whole change. A fresh `Settings` must produce a
        // DISABLED trail: from 2026-06-06 to 2026-08-09 this was `true` and
        // unreachable, and it capped the realised payoff near 1.0 against a
        // configured floor of 2.0 — 0 of 174 candidates could survive.
        let s = neoethos_core::Settings::default();
        let resolved = ExitPolicyOverrides::from_settings(&s);
        assert!(
            !resolved.trailing_enabled,
            "discovery must default to NO trailing stop"
        );
        assert_eq!(resolved, ExitPolicyOverrides::default());

        // And it must still be reachable — this is a knob, not a deletion.
        let mut on = neoethos_core::Settings::default();
        on.models.exit_policy.trailing_enabled = true;
        assert!(ExitPolicyOverrides::from_settings(&on).trailing_enabled);

        // A non-finite trigger arms the trail never, on every gene, silently.
        // Refuse it at the boundary instead.
        let mut broken = neoethos_core::Settings::default();
        broken.models.exit_policy.trailing_be_trigger_r = f64::NAN;
        assert_eq!(
            ExitPolicyOverrides::from_settings(&broken).trailing_be_trigger_r,
            1.0
        );
    }

    #[test]
    fn effective_tournament_size_matches_legacy_formula() {
        let defaults = GeneticSearchRuntimeOverrides::default();
        assert_eq!(defaults.effective_tournament_size(120), 10);
        // Population-derived minimum tournament size never drops below 3.
        assert_eq!(defaults.effective_tournament_size(12), 3);
        // Explicit override wins, but never goes below the algorithmic
        // minimum of 2.
        let overridden = GeneticSearchRuntimeOverrides {
            tournament_size_override: Some(1),
            ..GeneticSearchRuntimeOverrides::default()
        };
        assert_eq!(overridden.effective_tournament_size(1000), 2);
    }

    #[test]
    fn effective_archive_cap_clamps_to_population_and_max() {
        let defaults = GeneticSearchRuntimeOverrides::default();
        // Derived from pop * generations, capped at 50_000 by default.
        assert_eq!(defaults.effective_archive_cap(1000, 10), 10_000);
        // Floor is the population so we always keep at least one elite per slot.
        assert_eq!(defaults.effective_archive_cap(60_000, 1), 60_000);
        // Hard ceiling at 200_000 guards against env-driven memory blow-ups.
        let huge = GeneticSearchRuntimeOverrides {
            archive_cap_override: Some(10_000_000),
            ..GeneticSearchRuntimeOverrides::default()
        };
        assert_eq!(huge.effective_archive_cap(1000, 10), 200_000);
    }

    #[test]
    fn effective_stagnation_patience_is_at_least_one() {
        let zero = GeneticSearchRuntimeOverrides {
            stagnation_patience: 0,
            ..GeneticSearchRuntimeOverrides::default()
        };
        assert_eq!(zero.effective_stagnation_patience(), 1);
    }

    #[test]
    fn effective_convergence_patience_and_min_improvement_resolve() {
        let d = GeneticSearchRuntimeOverrides::default();
        // Default-ON early-stop; legacy improvement epsilon preserved.
        assert_eq!(d.effective_convergence_patience(), 250);
        assert!((d.effective_min_improvement() - 1e-12).abs() < 1e-18);
        // `0` disables the early-stop (unlike stagnation_patience, NOT floored to 1).
        let off = GeneticSearchRuntimeOverrides {
            convergence_patience: 0,
            ..GeneticSearchRuntimeOverrides::default()
        };
        assert_eq!(off.effective_convergence_patience(), 0);
        // Non-finite / negative min_improvement falls back to 1e-12 (fail-safe).
        for bad in [-1.0_f64, f64::NAN, f64::INFINITY] {
            let o = GeneticSearchRuntimeOverrides {
                min_improvement: bad,
                ..GeneticSearchRuntimeOverrides::default()
            };
            assert!((o.effective_min_improvement() - 1e-12).abs() < 1e-18);
        }
        // A valid positive epsilon is honored.
        let custom = GeneticSearchRuntimeOverrides {
            min_improvement: 1e-6,
            ..GeneticSearchRuntimeOverrides::default()
        };
        assert!((custom.effective_min_improvement() - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn effective_convergence_min_elapsed_fraction_clamps() {
        let d = GeneticSearchRuntimeOverrides::default();
        assert!((d.effective_convergence_min_elapsed_fraction() - 0.5).abs() < 1e-9);
        // In-range values honored.
        for (val, exp) in [(0.0, 0.0), (0.25, 0.25), (1.0, 1.0)] {
            let o = GeneticSearchRuntimeOverrides {
                convergence_min_elapsed_fraction: val,
                ..GeneticSearchRuntimeOverrides::default()
            };
            assert!((o.effective_convergence_min_elapsed_fraction() - exp).abs() < 1e-9);
        }
        // Out-of-range / non-finite fall back to 0.5 (safe).
        for bad in [-0.1_f64, 1.5, f64::NAN, f64::INFINITY] {
            let o = GeneticSearchRuntimeOverrides {
                convergence_min_elapsed_fraction: bad,
                ..GeneticSearchRuntimeOverrides::default()
            };
            assert!((o.effective_convergence_min_elapsed_fraction() - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn current_overrides_returns_legal_values() {
        let observed = current_genetic_search_runtime_overrides();
        assert!(observed.novelty_weight.is_finite());
    }

    #[test]
    fn smc_gate_clamps_invalid_curve_and_stagnation_step() {
        let bad = GeneticSearchRuntimeOverrides {
            smc_gate: SmcGateOverrides {
                start: 0.8,
                end: 0.2,
                curve: 0.0,
                stagnation_step: f64::NAN,
                disable_gate: false,
            },
            ..GeneticSearchRuntimeOverrides::default()
        };
        let resolved = bad.resolved_smc_gate();
        assert!((resolved.curve - 0.1).abs() < 1e-6);
        assert!((resolved.stagnation_step - 0.0).abs() < 1e-6);

        let valid = GeneticSearchRuntimeOverrides {
            smc_gate: SmcGateOverrides {
                start: 0.7,
                end: 0.3,
                curve: 2.5,
                stagnation_step: 0.05,
                disable_gate: false,
            },
            ..GeneticSearchRuntimeOverrides::default()
        };
        let resolved = valid.resolved_smc_gate();
        assert!((resolved.curve - 2.5).abs() < 1e-6);
        assert!((resolved.stagnation_step - 0.05).abs() < 1e-6);
    }

    #[test]
    fn selection_policy_clamps_immigrant_and_survivor_fractions_and_temperature() {
        let bad = GeneticSearchRuntimeOverrides {
            selection: SelectionPolicyOverrides {
                immigrant_ratio: 5.0,
                survivor_fraction: -1.0,
                temperature: 0.0,
                ..SelectionPolicyOverrides::default()
            },
            ..GeneticSearchRuntimeOverrides::default()
        };
        let resolved = bad.resolved_selection();
        assert!((resolved.immigrant_ratio - 0.95).abs() < 1e-9);
        assert!((resolved.survivor_fraction - 0.0).abs() < 1e-9);
        assert!((resolved.temperature - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn effective_seen_retry_is_at_least_one() {
        let zero = GeneticSearchRuntimeOverrides {
            seen_retry_attempts: 0,
            ..GeneticSearchRuntimeOverrides::default()
        };
        assert_eq!(zero.effective_seen_retry_attempts(), 1);
    }

    #[test]
    fn cost_profile_overrides_default_to_none() {
        let cost = CostProfileRuntimeOverrides::default();
        assert!(cost.symbol.is_none());
        assert!(cost.account_currency.is_none());
        assert!(cost.pip_value.is_none());
        assert!(cost.quote_to_account_rate.is_none());
        assert!(cost.pip_value_per_lot.is_none());
        assert!(cost.spread_pips.is_none());
        assert!(cost.commission_per_trade.is_none());
    }

    #[test]
    fn smc_weight_overrides_default_to_neutral_unit_weights() {
        let smc = SmcWeightRuntimeOverrides::default();
        assert!((smc.gate_threshold - 0.75).abs() < 1e-6);
        for w in [
            smc.w_ob,
            smc.w_fvg,
            smc.w_liq,
            smc.w_mtf,
            smc.w_premium,
            smc.w_inducement,
            smc.w_bos,
            smc.w_choch,
            smc.w_eqh,
            smc.w_eql,
            smc.w_displacement,
        ] {
            assert!((w - 1.0).abs() < 1e-6, "expected unit weight, got {w}");
        }
    }

    #[test]
    fn strategy_evaluation_overrides_default_to_neutral_state() {
        let overrides = StrategyEvaluationRuntimeOverrides::default();
        assert_eq!(
            overrides.cost_profile,
            CostProfileRuntimeOverrides::default()
        );
        assert_eq!(overrides.smc_weights, SmcWeightRuntimeOverrides::default());
    }

    #[test]
    fn current_strategy_evaluation_overrides_returns_legal_values() {
        let observed = current_strategy_evaluation_runtime_overrides();
        assert!(observed.smc_weights.gate_threshold.is_finite());
    }

    #[test]
    fn determinism_policy_maps_seed_some_to_deterministic_and_none_to_nondeterministic() {
        let with_seed = GeneticSearchRuntimeOverrides {
            seed: Some(42),
            ..GeneticSearchRuntimeOverrides::default()
        };
        match with_seed.determinism_policy() {
            DeterminismPolicy::Deterministic { seed } => assert_eq!(seed, 42),
            other => panic!("expected Deterministic, got {other:?}"),
        }

        let without_seed = GeneticSearchRuntimeOverrides::default();
        match without_seed.determinism_policy() {
            DeterminismPolicy::NonDeterministicAllowed => {}
            other => panic!("expected NonDeterministicAllowed, got {other:?}"),
        }
    }

    #[test]
    fn determinism_policy_seed_round_trip_through_neoethos_core_helper() {
        let policy = DeterminismPolicy::Deterministic { seed: 7 };
        assert_eq!(policy.seed(), Some(7));
        assert_eq!(DeterminismPolicy::BestEffort.seed(), None);
        assert_eq!(DeterminismPolicy::NonDeterministicAllowed.seed(), None);
    }

    #[test]
    fn current_determinism_policy_returns_default_non_deterministic() {
        // The OnceLock-installed overrides may carry whatever any earlier
        // test in this process installed, but the default-derived policy
        // is `NonDeterministicAllowed` and the legality check below holds
        // for all three legal variants.
        let observed = current_determinism_policy();
        match observed {
            DeterminismPolicy::Deterministic { seed: _ }
            | DeterminismPolicy::BestEffort
            | DeterminismPolicy::NonDeterministicAllowed => {}
        }
    }
}
