//! Hand-crafted "professional trader" multi-TF starter genomes.
//!
//! **Background (2026-05-26 — taskdoc #275)**: The Python prototype this
//! engine replaces had EUR/USD strategies that produced ~€4M PnL over 9
//! years on small starting capital. Those strategies were NOT discovered
//! by a cold-start GA — they came from human-encoded multi-timeframe
//! confluence templates (trend-pullback, mean-reversion, breakout,
//! momentum, counter-trend) that a researcher hand-tuned, then handed to
//! the GA for refinement.
//!
//! The current Rust GA starts from PURE RANDOM. Random genes on 1500+
//! features almost never wire up a coherent multi-TF setup, so every
//! candidate looks the same to the fitness function: "noisy threshold
//! crosser that rarely fires". Combined with the zero-trade reward-hack
//! (taskdoc #274), the result is empty Discovery archives.
//!
//! This module fixes that by injecting ~10% of the initial population as
//! hand-crafted templates that look like real professional setups. They
//! aren't expected to win straight out of the gate — they're a sensible
//! starting basin so mutation has somewhere productive to go.
//!
//! ## Template anatomy
//!
//! Each template specifies, for each of 5 timeframes (D1, H4, H1, M15,
//! M5), one indicator role. At construction time we look up each role in
//! `feature_names` (case-insensitive partial match), falling back to ANY
//! indicator from the same family (EMA → any EMA; RSI → any RSI; etc.)
//! when an exact match isn't present. Templates that can't resolve even
//! a fuzzy match for any TF are skipped — better to produce 30 sensible
//! templates than 50 nonsense ones.
//!
//! ## Variant rotations
//!
//! Each base template emits up to 10 variants by:
//!   - perturbing weights ±10%
//!   - rotating thresholds across {0.15, 0.20, 0.25, 0.30}
//!   - flipping a subset of SMC flags
//!
//! 5 templates × 10 variants = up to 50 starter genes.

use super::strategy_gene::Gene;
use rand::Rng;

/// Hand-crafted feature-role spec for one slot in a template.
struct RoleSpec {
    /// Timeframe prefix to require, lowercase (e.g. "d1", "h4"). The
    /// base timeframe has no prefix, so empty string matches any name
    /// without a recognised TF prefix.
    tf: &'static str,
    /// Primary keyword to match (case-insensitive partial match) — e.g.
    /// "ema_50" or "rsi" or "bbands".
    primary: &'static str,
    /// Family keyword used as a fallback when no `primary` match exists.
    /// e.g. "ema", "rsi", "atr", "macd". The first feature whose name
    /// contains BOTH `tf` and `family` is selected.
    family: &'static str,
    /// Suggested weight for this indicator. Sign indicates direction
    /// (positive = bullish on raise; negative = bearish on raise; the
    /// GA will discover the right sign through mutation regardless).
    weight: f32,
}

/// One named template (e.g. "Trend Pullback Long"). Roles are tried in
/// order; templates with fewer than 2 resolvable roles are skipped.
struct Template {
    name: &'static str,
    long_threshold: f32,
    short_threshold: f32,
    roles: [RoleSpec; 5],
    /// SMC flag preset: (use_ob, use_fvg, use_liq_sweep, mtf, premium,
    /// inducement, bos, choch, eqh, eql, displacement).
    smc_flags: [bool; 11],
}

const TEMPLATES: &[Template] = &[
    // Template 1 — Trend Pullback Long: D1 EMA50 slope + H4 above-EMA21 +
    // H1 EMA50 slope + M15 RSI oversold + M5 VWAP cross.
    Template {
        name: "trend_pullback",
        long_threshold: 0.20,
        short_threshold: -0.20,
        roles: [
            RoleSpec { tf: "d1", primary: "ema_50", family: "ema", weight: 0.25 },
            RoleSpec { tf: "h4", primary: "ema_21", family: "ema", weight: 0.20 },
            RoleSpec { tf: "h1", primary: "ema_50", family: "ema", weight: 0.20 },
            RoleSpec { tf: "m15", primary: "rsi", family: "rsi", weight: 0.20 },
            RoleSpec { tf: "m5", primary: "vwap", family: "ma", weight: 0.15 },
        ],
        smc_flags: [false, false, false, true, false, false, false, false, false, false, false],
    },
    // Template 2 — Mean Reversion: D1 ATR expansion + H4 BBands upper +
    // H1 RSI overbought + M15 RSI divergence + M5 bearish.
    Template {
        name: "mean_reversion",
        long_threshold: 0.25,
        short_threshold: -0.25,
        roles: [
            RoleSpec { tf: "d1", primary: "atr", family: "atr", weight: 0.15 },
            RoleSpec { tf: "h4", primary: "bbands", family: "bollinger", weight: 0.25 },
            RoleSpec { tf: "h1", primary: "rsi", family: "rsi", weight: 0.25 },
            RoleSpec { tf: "m15", primary: "rsi", family: "rsi", weight: 0.20 },
            RoleSpec { tf: "m5", primary: "cci", family: "cci", weight: 0.15 },
        ],
        smc_flags: [false, false, true, true, true, false, false, false, false, false, false],
    },
    // Template 3 — Breakout: D1 low ATR (compression) + H4 range close +
    // H1 volume increasing + M15 near range high + M5 break.
    Template {
        name: "breakout",
        long_threshold: 0.30,
        short_threshold: -0.30,
        roles: [
            RoleSpec { tf: "d1", primary: "atr", family: "atr", weight: -0.20 },
            RoleSpec { tf: "h4", primary: "donchian", family: "donchian", weight: 0.20 },
            RoleSpec { tf: "h1", primary: "obv", family: "volume", weight: 0.20 },
            RoleSpec { tf: "m15", primary: "donchian", family: "donchian", weight: 0.20 },
            RoleSpec { tf: "m5", primary: "atr", family: "atr", weight: 0.20 },
        ],
        smc_flags: [true, false, false, true, false, false, true, false, false, false, true],
    },
    // Template 4 — Momentum: D1 EMA20 slope + H4 MACD hist + H1 MACD
    // cross + M15 ADX>20 + M5 EMA8/EMA21 cross.
    Template {
        name: "momentum",
        long_threshold: 0.20,
        short_threshold: -0.20,
        roles: [
            RoleSpec { tf: "d1", primary: "ema_20", family: "ema", weight: 0.20 },
            RoleSpec { tf: "h4", primary: "macd", family: "macd", weight: 0.20 },
            RoleSpec { tf: "h1", primary: "macd", family: "macd", weight: 0.20 },
            RoleSpec { tf: "m15", primary: "adx", family: "adx", weight: 0.20 },
            RoleSpec { tf: "m5", primary: "ema_8", family: "ema", weight: 0.20 },
        ],
        smc_flags: [false, false, false, true, false, false, true, false, false, false, false],
    },
    // Template 5 — Counter-Trend: D1 EMA200 slope < 0 + H4 close <
    // EMA50 + H1 RSI > 70 + M15 bearish divergence + M5 bearish.
    Template {
        name: "counter_trend",
        long_threshold: 0.25,
        short_threshold: -0.25,
        roles: [
            RoleSpec { tf: "d1", primary: "ema_200", family: "ema", weight: -0.25 },
            RoleSpec { tf: "h4", primary: "ema_50", family: "ema", weight: -0.20 },
            RoleSpec { tf: "h1", primary: "rsi", family: "rsi", weight: 0.20 },
            RoleSpec { tf: "m15", primary: "rsi", family: "rsi", weight: 0.20 },
            RoleSpec { tf: "m5", primary: "stoch", family: "stoch", weight: 0.15 },
        ],
        smc_flags: [false, true, false, true, false, false, false, true, false, false, false],
    },
];

/// Number of variants to emit per base template. 5 templates × 10
/// variants = 50 starter genes (matches the `count.min(50)` cap in the
/// caller).
const VARIANTS_PER_TEMPLATE: usize = 10;

/// Find the feature index whose lowercased name contains both `tf` and
/// `primary`. Returns `None` if no match exists.
fn find_feature(
    feature_names: &[String],
    tf: &str,
    primary: &str,
) -> Option<usize> {
    feature_names
        .iter()
        .enumerate()
        .find(|(_, name)| {
            let lower = name.to_lowercase();
            let tf_ok = tf.is_empty() || lower.contains(tf);
            tf_ok && lower.contains(primary)
        })
        .map(|(idx, _)| idx)
}

/// Try `find_feature(tf, primary)` first; if that misses, fall back to
/// `find_feature(tf, family)`; if that also misses, return `None` (the
/// caller will skip this role rather than producing nonsense).
fn resolve_role(feature_names: &[String], role: &RoleSpec) -> Option<usize> {
    find_feature(feature_names, role.tf, role.primary)
        .or_else(|| find_feature(feature_names, role.tf, role.family))
}

/// Build one Gene from a template + variant index. Variant rotations
/// perturb weights ±10%, shift thresholds across {0.15, 0.20, 0.25, 0.30},
/// and rotate SMC flags so siblings explore neighbouring genomes.
fn build_variant(
    template: &Template,
    variant: usize,
    feature_names: &[String],
    n_features: usize,
    rng: &mut impl Rng,
) -> Option<Gene> {
    // Resolve roles in declaration order. Skip roles we can't find.
    let mut indices: Vec<usize> = Vec::with_capacity(5);
    let mut weights: Vec<f32> = Vec::with_capacity(5);
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for role in template.roles.iter() {
        if let Some(idx) = resolve_role(feature_names, role) {
            if idx < n_features && seen.insert(idx) {
                // Variant rotation: ±10% weight perturbation.
                let perturb = 0.9 + (variant as f32) * 0.02; // 0.9..1.1 across variants
                weights.push(role.weight * perturb);
                indices.push(idx);
            }
        }
    }

    // Skip templates that resolve fewer than 2 roles — they'd be
    // indistinguishable from random and pollute the seed pool.
    if indices.len() < 2 {
        return None;
    }

    // Threshold rotation across 4 levels (calibrated to z-score feature
    // magnitudes; matches the narrow ladder in `random_coarse_threshold`
    // post-2026-05-26 operator directive). We scale both long and short
    // thresholds together so their ratio stays template-defined (most
    // templates use symmetric ±X but the struct allows asymmetric — e.g.
    // a counter-trend template could use long=+0.25, short=-0.20).
    //
    // 2026-05-26 narrowed: combined multi-indicator signals on z-score
    // frames empirically sit ~0.05-0.15. The earlier `[0.15, 0.20, 0.25,
    // 0.30]` ladder was scoped to the 95th percentile of *individual*
    // indicators — for the COMBINED signal that's still too aggressive.
    // The new low-end (0.08) captures genes that fire often + the high-end
    // (0.30) covers the picky regime, matching task #273.
    let thr_levels = [0.08_f32, 0.15, 0.22, 0.30];
    let thr_idx = variant % thr_levels.len();
    let scale = thr_levels[thr_idx] / 0.25;
    let long_threshold = template.long_threshold * scale;
    let short_threshold = template.short_threshold * scale;

    // SL/TP rotation: alternate between conservative (1:2 RR) and
    // moderate (1:2.5 RR) across variants.
    let (sl_pips, tp_pips) = if variant % 2 == 0 {
        (15.0, 30.0)
    } else {
        (20.0, 50.0)
    };

    // SMC flag rotation: every 3rd variant flips a random flag so we
    // get explore-the-neighbourhood seeds rather than 10 identical clones.
    let mut smc_flags = template.smc_flags;
    if variant > 0 && variant % 3 == 0 {
        let flip = rng.random_range(0..11);
        smc_flags[flip] = !smc_flags[flip];
    }

    let strategy_id = format!(
        "seed_{}_v{}_{}",
        template.name,
        variant,
        rng.random_range(0..1_000_000u64)
    );

    let mut gene = Gene {
        indices,
        weights,
        long_threshold,
        short_threshold,
        fitness: 0.0,
        sharpe_ratio: 0.0,
        win_rate: 0.0,
        max_drawdown: 0.0,
        profit_factor: 0.0,
        expectancy: 0.0,
        trades_count: 0,
        generation: 0,
        strategy_id,
        use_ob: smc_flags[0],
        use_fvg: smc_flags[1],
        use_liq_sweep: smc_flags[2],
        mtf_confirmation: smc_flags[3],
        use_premium_discount: smc_flags[4],
        use_inducement: smc_flags[5],
        use_bos: smc_flags[6],
        use_choch: smc_flags[7],
        use_eqh: smc_flags[8],
        use_eql: smc_flags[9],
        use_displacement: smc_flags[10],
        tp_pips,
        sl_pips,
        slice_pass_rate: 0.0,
        consistency: 0.0,
    };
    gene.normalize(n_features.max(1), 1);
    Some(gene)
}

/// Build up to `count` hand-crafted multi-TF starter genomes.
///
/// Returns FEWER than `count` when the feature set doesn't contain any
/// resolvable templates (e.g. single-TF backtest with no higher-TF
/// columns prefixed `d1_`/`h4_`/...). Callers should pad the rest with
/// `new_random_gene` to maintain the population size — see
/// `search_engine.rs::evolve_search_with_progress_impl` and the
/// diversity-rescue path.
///
/// ## Safety / behaviour
///
/// - Every emitted gene is `normalize()`-d before return, so it
///   round-trips through `gene_signature_hash` cleanly.
/// - Indices are always `< n_features`; no out-of-bounds access.
/// - Templates with fewer than 2 resolvable roles are skipped (returns
///   fewer than `count` rather than producing nonsense).
pub fn seed_professional_templates(
    count: usize,
    feature_names: &[String],
    n_features: usize,
    rng: &mut impl Rng,
) -> Vec<Gene> {
    if count == 0 || n_features == 0 {
        return Vec::new();
    }

    let mut out: Vec<Gene> = Vec::with_capacity(count.min(TEMPLATES.len() * VARIANTS_PER_TEMPLATE));
    for template in TEMPLATES.iter() {
        for variant in 0..VARIANTS_PER_TEMPLATE {
            if out.len() >= count {
                return out;
            }
            if let Some(gene) = build_variant(template, variant, feature_names, n_features, rng) {
                out.push(gene);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn empty_feature_names_returns_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let seeds = seed_professional_templates(50, &[], 0, &mut rng);
        assert!(seeds.is_empty());
    }

    #[test]
    fn zero_count_returns_empty() {
        let names = vec!["d1_ema_50".to_string(), "h4_rsi".to_string()];
        let mut rng = StdRng::seed_from_u64(42);
        let seeds = seed_professional_templates(0, &names, names.len(), &mut rng);
        assert!(seeds.is_empty());
    }

    #[test]
    fn full_multi_tf_feature_set_yields_templates() {
        // Construct a feature-name set that matches several template
        // roles across all 5 TFs.
        let names: Vec<String> = vec![
            "ema_50", "ema_21", "ema_20", "ema_200", "ema_8", "rsi", "atr",
            "macd", "adx", "cci", "stoch", "vwap", "donchian", "obv",
            "bollinger_bands",
            "d1_ema_50", "d1_ema_20", "d1_ema_200", "d1_atr",
            "h4_ema_21", "h4_ema_50", "h4_bbands", "h4_macd", "h4_donchian",
            "h1_ema_50", "h1_rsi", "h1_macd", "h1_obv",
            "m15_rsi", "m15_adx", "m15_donchian",
            "m5_vwap", "m5_cci", "m5_ema_8", "m5_stoch", "m5_atr",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let n = names.len();
        let mut rng = StdRng::seed_from_u64(42);
        let seeds = seed_professional_templates(50, &names, n, &mut rng);
        // We expect at least a few resolvable templates → at least 10 seeds.
        assert!(
            seeds.len() >= 10,
            "expected at least 10 templates from full multi-TF feature set, got {}",
            seeds.len()
        );
        // Every gene must have at least 2 indices and finite thresholds.
        for g in seeds.iter() {
            assert!(g.indices.len() >= 2);
            assert!(g.long_threshold.is_finite());
            assert!(g.short_threshold.is_finite());
            assert!(g.long_threshold > g.short_threshold);
            for idx in g.indices.iter() {
                assert!(*idx < n);
            }
        }
    }

    #[test]
    fn missing_features_skips_templates_gracefully() {
        // Only one TF available — most templates can't resolve all 5
        // roles, but `build_variant` is forgiving (only requires ≥2
        // roles), so we still get some templates.
        let names: Vec<String> = vec!["ema_50", "rsi", "atr", "macd"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut rng = StdRng::seed_from_u64(42);
        let seeds = seed_professional_templates(50, &names, names.len(), &mut rng);
        // We should still get SOMETHING (probably a handful), and
        // nothing should panic.
        for g in seeds.iter() {
            assert!(g.indices.len() >= 2);
            for idx in g.indices.iter() {
                assert!(*idx < names.len());
            }
        }
    }
}
