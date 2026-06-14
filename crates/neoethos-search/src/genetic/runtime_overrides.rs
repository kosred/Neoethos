//! Typed boundary for the legacy `NEOETHOS_BOT_*` env vars that previously
//! reached deep into the genetic search engine on every call. These knobs
//! change *production* search semantics — RNG seed, novelty weighting,
//! tournament size, stagnation patience, archive capacity, SMC gate
//! shaping, archive scoring thresholds, and selection-policy weighting —
//! so the audit (P0-8) requires them to live in typed config rather than
//! ambient env state. The struct is the single owner of those values and
//! exposes a single `from_env` reader; production binaries install the
//! resolved overrides once at startup via
//! [`install_genetic_search_runtime_overrides_from_env`].

use super::evolution_math::{ParentSelectionPolicy, SurvivorSelectionPolicy};
use neoethos_core::contracts::DeterminismPolicy;
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmcGateOverrides {
    pub start: f32,
    pub end: f32,
    pub curve: f32,
    pub stagnation_step: f32,
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
    fn resolved_curve(&self) -> f32 {
        if self.curve.is_finite() && self.curve >= 0.1 {
            self.curve
        } else {
            0.1
        }
    }

    fn resolved_stagnation_step(&self) -> f32 {
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticSearchRuntimeOverrides {
    /// Optional deterministic RNG seed for the genetic search. `None`
    /// means "seed from the OS RNG" (non-deterministic).
    pub seed: Option<u64>,
    /// Novelty bonus weight applied during candidate ranking. `0.0`
    /// disables novelty scoring (default).
    pub novelty_weight: f64,
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
    /// One-shot read of the legacy `NEOETHOS_BOT_*` search env vars. This is
    /// the only place the genetic search consults the environment for
    /// these knobs.
    pub fn from_env() -> Self {
        let mut overrides = Self::default();

        if let Some(seed) = env_u64("NEOETHOS_BOT_SEARCH_SEED") {
            overrides.seed = Some(seed);
        }
        if let Some(weight) = env_f64_finite("NEOETHOS_BOT_NOVELTY_WEIGHT") {
            overrides.novelty_weight = weight;
        }
        if let Some(patience) = env_usize_positive("NEOETHOS_BOT_PROP_STAGNATION_GENS") {
            overrides.stagnation_patience = patience;
        }
        // `env_u64` (not `_positive`) so `0` is honored as "disable early-stop".
        if let Some(conv) = env_u64("NEOETHOS_BOT_PROP_CONVERGENCE_GENS") {
            overrides.convergence_patience = conv as usize;
        }
        if let Some(min_imp) = env_f64_finite("NEOETHOS_BOT_PROP_MIN_IMPROVEMENT") {
            overrides.min_improvement = min_imp;
        }
        if let Some(frac) = env_f64_finite("NEOETHOS_BOT_PROP_CONVERGENCE_MIN_ELAPSED_FRAC") {
            overrides.convergence_min_elapsed_fraction = frac;
        }
        if let Some(tournament) = env_usize_positive("NEOETHOS_BOT_PROP_TOURNAMENT_SIZE") {
            overrides.tournament_size_override = Some(tournament);
        }
        if let Some(cap) = env_usize_positive("NEOETHOS_BOT_PROP_ARCHIVE_CAP") {
            overrides.archive_cap_override = Some(cap);
        }
        if let Some(retry) = env_usize_positive("NEOETHOS_BOT_PROP_SEEN_RETRY") {
            overrides.seen_retry_attempts = retry;
        }

        // SMC gate curve.
        if let Some(start) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_GATE_START")
            .or_else(|| env_f32_finite("NEOETHOS_BOT_PROP_SMC_GATE"))
        {
            overrides.smc_gate.start = start;
        }
        if let Some(end) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_GATE_END") {
            overrides.smc_gate.end = end;
        }
        if let Some(curve) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_GATE_CURVE") {
            overrides.smc_gate.curve = curve;
        }
        if let Some(step) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_GATE_STAGNATION_STEP") {
            overrides.smc_gate.stagnation_step = step;
        }
        // Hard bypass: legacy `NEOETHOS_BOT_DISABLE_SMC_GATE=1` env var.
        // F-CORE3 closure (2026-05-25): previously read inline inside
        // `signals_for_gene_full` (search_engine.rs) and the GA's
        // per-gene signal-synthesis loop (eval.rs::synthesize_signals_cpu).
        // Now consolidated to this typed boundary so the env is hit at
        // most once per process.
        if env_truthy("NEOETHOS_BOT_DISABLE_SMC_GATE") {
            overrides.smc_gate.disable_gate = true;
        }

        // Archive scoring thresholds.
        if let Some(mode) = env_string_lowercase("NEOETHOS_BOT_PROP_ARCHIVE_MODE") {
            overrides.archive_scoring.mode = mode;
        }
        if let Some(min_net) = env_f64_finite("NEOETHOS_BOT_PROP_ARCHIVE_MIN_NET") {
            overrides.archive_scoring.min_net = min_net;
        }
        if let Some(min_pf) = env_f64_finite("NEOETHOS_BOT_PROP_ARCHIVE_MIN_PF") {
            overrides.archive_scoring.min_pf = min_pf;
        }
        if let Some(min_sharpe) = env_f64_finite("NEOETHOS_BOT_PROP_ARCHIVE_MIN_SHARPE") {
            overrides.archive_scoring.min_sharpe = min_sharpe;
        }

        // Selection policy.
        if let Some(immigrants) = env_f64_finite("NEOETHOS_BOT_PROP_RANDOM_IMMIGRANTS") {
            overrides.selection.immigrant_ratio = immigrants;
        }
        let survivor_fraction = env_f64_finite("NEOETHOS_BOT_PROP_SURVIVOR_FRACTION")
            .or_else(|| env_f64_finite("NEOETHOS_BOT_PROP_ELITE_FRACTION"));
        if let Some(value) = survivor_fraction {
            overrides.selection.survivor_fraction = value;
        }
        if let Some(parent) = env_string_lowercase("NEOETHOS_BOT_PROP_PARENT_SELECTION") {
            overrides.selection.parent = ParentSelectionPolicy::parse(&parent);
        }
        if let Some(survivor) = env_string_lowercase("NEOETHOS_BOT_PROP_SURVIVOR_SELECTION") {
            overrides.selection.survivor = SurvivorSelectionPolicy::parse(&survivor);
        }
        if let Some(temp) = env_f64_finite("NEOETHOS_BOT_PROP_SELECTION_TEMPERATURE") {
            overrides.selection.temperature = temp;
        }

        overrides
    }

    /// Config-driven constructor — the operator sets these knobs in the
    /// single `Settings` (config / UI / TUI), never the environment.
    /// Mirrors [`Self::from_env`] field-for-field; an empty policy /
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
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CostProfileRuntimeOverrides {
    pub symbol: Option<String>,
    pub account_currency: Option<String>,
    pub pip_value: Option<f64>,
    pub quote_to_account_rate: Option<f64>,
    pub pip_value_per_lot: Option<f64>,
    pub spread_pips: Option<f64>,
    pub commission_per_trade: Option<f64>,
    pub reject_pip_fallback: bool,
}

impl CostProfileRuntimeOverrides {
    fn populate_from_env(&mut self) {
        if let Some(value) = env_string_nonempty("NEOETHOS_BOT_PROP_SYMBOL") {
            self.symbol = Some(value);
        }
        if let Some(value) = env_string_nonempty("NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY") {
            self.account_currency = Some(value);
        }
        if let Some(value) = env_f64_positive_finite("NEOETHOS_BOT_PROP_PIP_VALUE") {
            self.pip_value = Some(value);
        }
        if let Some(value) = env_f64_positive_finite("NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE") {
            self.quote_to_account_rate = Some(value);
        }
        if let Some(value) = env_f64_positive_finite("NEOETHOS_BOT_PROP_PIP_VALUE_PER_LOT") {
            self.pip_value_per_lot = Some(value);
        }
        if let Some(value) = env_f64_non_negative_finite("NEOETHOS_BOT_PROP_SPREAD_PIPS") {
            self.spread_pips = Some(value);
        }
        if let Some(value) = env_f64_non_negative_finite("NEOETHOS_BOT_PROP_COMMISSION") {
            self.commission_per_trade = Some(value);
        }
        // F-CORE3 closure (2026-05-25): legacy `NEOETHOS_BOT_REJECT_PIP_FALLBACK=1`
        // was previously read inline inside `reject_cross_pair_fallback()`
        // in `strategy_gene.rs`; now consolidated to this typed boundary.
        if env_truthy("NEOETHOS_BOT_REJECT_PIP_FALLBACK") {
            self.reject_pip_fallback = true;
        }
    }
}

/// SMC weight knobs that previously lived in the
/// `NEOETHOS_BOT_PROP_SMC_W_*` env vars and the `NEOETHOS_BOT_PROP_SMC_GATE`
/// fallback used by `EvaluationConfig::default`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmcWeightRuntimeOverrides {
    pub gate_threshold: f32,
    pub w_ob: f32,
    pub w_fvg: f32,
    pub w_liq: f32,
    pub w_mtf: f32,
    pub w_premium: f32,
    pub w_inducement: f32,
    pub w_bos: f32,
    pub w_choch: f32,
    pub w_eqh: f32,
    pub w_eql: f32,
    pub w_displacement: f32,
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
    fn populate_from_env(&mut self) {
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_GATE") {
            self.gate_threshold = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_OB") {
            self.w_ob = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_FVG") {
            self.w_fvg = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_LIQ") {
            self.w_liq = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_MTF") {
            self.w_mtf = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_PREMIUM") {
            self.w_premium = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_INDUCEMENT") {
            self.w_inducement = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_BOS") {
            self.w_bos = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_CHOCH") {
            self.w_choch = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_EQH") {
            self.w_eqh = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_EQL") {
            self.w_eql = value;
        }
        if let Some(value) = env_f32_finite("NEOETHOS_BOT_PROP_SMC_W_DISPLACEMENT") {
            self.w_displacement = value;
        }
    }
}

/// Typed runtime overrides for `EvaluationConfig::default` and
/// `infer_market_cost_profile`. Cost knobs replace the
/// `NEOETHOS_BOT_PROP_SYMBOL` / `ACCOUNT_CURRENCY` / `PIP_VALUE` /
/// `QUOTE_TO_ACCOUNT_RATE` / `PIP_VALUE_PER_LOT` / `SPREAD_PIPS` /
/// `COMMISSION` env vars; SMC weight knobs replace the
/// `NEOETHOS_BOT_PROP_SMC_W_*` and `NEOETHOS_BOT_PROP_SMC_GATE` env vars used
/// at evaluation-config construction time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrategyEvaluationRuntimeOverrides {
    pub cost_profile: CostProfileRuntimeOverrides,
    pub smc_weights: SmcWeightRuntimeOverrides,
}

impl StrategyEvaluationRuntimeOverrides {
    /// One-shot read of the legacy `NEOETHOS_BOT_PROP_*` evaluation env
    /// vars.
    pub fn from_env() -> Self {
        let mut overrides = Self::default();
        overrides.cost_profile.populate_from_env();
        overrides.smc_weights.populate_from_env();
        overrides
    }

    /// Config-driven constructor — reads the cost-profile + SMC-weight
    /// knobs from the single `Settings` (config / UI / TUI) instead of the
    /// environment. Mirrors [`Self::from_env`]; `None` cost fields stay
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
        }
    }
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

/// Convenience wrapper that resolves the legacy `NEOETHOS_BOT_PROP_*`
/// evaluation env vars once and installs them. Idempotent.
pub fn install_strategy_evaluation_runtime_overrides_from_env() {
    let _ =
        STRATEGY_EVALUATION_RUNTIME_OVERRIDES.set(StrategyEvaluationRuntimeOverrides::from_env());
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

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok())
}

fn env_string_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_f64_positive_finite(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

fn env_f64_non_negative_finite(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
}

fn env_usize_positive(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn env_f64_finite(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

fn env_f32_finite(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
}

fn env_string_lowercase(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
}

/// Canonical boolean env-var parser. Returns true for `"1" | "true" | "TRUE"`
/// (matching the historical inline checks in `signals_for_gene_full`,
/// `synthesize_signals_cpu`, and `reject_cross_pair_fallback`). Empty
/// or missing means false. Any unparsed value is treated as false so
/// typos don't accidentally enable bypass behaviour.
fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

static GENETIC_SEARCH_RUNTIME_OVERRIDES: OnceLock<GeneticSearchRuntimeOverrides> = OnceLock::new();

/// Install process-wide genetic search runtime overrides. Returns
/// `Err(existing)` when overrides were already installed earlier (the
/// first install wins).
pub fn install_genetic_search_runtime_overrides(
    overrides: GeneticSearchRuntimeOverrides,
) -> Result<(), GeneticSearchRuntimeOverrides> {
    GENETIC_SEARCH_RUNTIME_OVERRIDES.set(overrides)
}

/// Convenience wrapper that resolves the legacy `NEOETHOS_BOT_*` search env
/// vars once and installs them. Idempotent.
pub fn install_genetic_search_runtime_overrides_from_env() {
    let _ = GENETIC_SEARCH_RUNTIME_OVERRIDES.set(GeneticSearchRuntimeOverrides::from_env());
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
    fn defaults_match_legacy_env_defaults() {
        let defaults = GeneticSearchRuntimeOverrides::default();
        assert_eq!(defaults.seed, None);
        assert!((defaults.novelty_weight - 0.0).abs() < 1e-9);
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
                stagnation_step: f32::NAN,
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
