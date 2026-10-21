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
use neoethos_dataset_contracts::CANONICAL_TIMEFRAMES;
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
    weight: f64,
}

/// One named template (e.g. "Trend Pullback Long"). Roles are tried in
/// order; templates with fewer than 2 resolvable roles are skipped.
pub(crate) const PROFESSIONAL_TEMPLATE_MAX_TERMS_V1: usize = 5;

struct Template {
    name: &'static str,
    long_threshold: f64,
    short_threshold: f64,
    roles: [RoleSpec; PROFESSIONAL_TEMPLATE_MAX_TERMS_V1],
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
            RoleSpec {
                tf: "d1",
                primary: "ema_50",
                family: "ema",
                weight: 0.25,
            },
            RoleSpec {
                tf: "h4",
                primary: "ema_21",
                family: "ema",
                weight: 0.20,
            },
            RoleSpec {
                tf: "h1",
                primary: "ema_50",
                family: "ema",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m15",
                primary: "rsi",
                family: "rsi",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m5",
                primary: "vwap",
                family: "ma",
                weight: 0.15,
            },
        ],
        smc_flags: [
            false, false, false, true, false, false, false, false, false, false, false,
        ],
    },
    // Template 2 — Mean Reversion: D1 ATR expansion + H4 BBands upper +
    // H1 RSI overbought + M15 RSI divergence + M5 bearish.
    Template {
        name: "mean_reversion",
        long_threshold: 0.25,
        short_threshold: -0.25,
        roles: [
            RoleSpec {
                tf: "d1",
                primary: "atr",
                family: "atr",
                weight: 0.15,
            },
            RoleSpec {
                tf: "h4",
                primary: "bbands",
                family: "bollinger",
                weight: 0.25,
            },
            RoleSpec {
                tf: "h1",
                primary: "rsi",
                family: "rsi",
                weight: 0.25,
            },
            RoleSpec {
                tf: "m15",
                primary: "rsi",
                family: "rsi",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m5",
                primary: "cci",
                family: "cci",
                weight: 0.15,
            },
        ],
        smc_flags: [
            false, false, true, true, true, false, false, false, false, false, false,
        ],
    },
    // Template 3 — Breakout: D1 low ATR (compression) + H4 range close +
    // H1 volume increasing + M15 near range high + M5 break.
    Template {
        name: "breakout",
        long_threshold: 0.30,
        short_threshold: -0.30,
        roles: [
            RoleSpec {
                tf: "d1",
                primary: "atr",
                family: "atr",
                weight: -0.20,
            },
            RoleSpec {
                tf: "h4",
                primary: "donchian",
                family: "donchian",
                weight: 0.20,
            },
            RoleSpec {
                tf: "h1",
                primary: "obv",
                family: "volume",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m15",
                primary: "donchian",
                family: "donchian",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m5",
                primary: "atr",
                family: "atr",
                weight: 0.20,
            },
        ],
        smc_flags: [
            true, false, false, true, false, false, true, false, false, false, true,
        ],
    },
    // Template 4 — Momentum: D1 EMA20 slope + H4 MACD hist + H1 MACD
    // cross + M15 ADX>20 + M5 EMA8/EMA21 cross.
    Template {
        name: "momentum",
        long_threshold: 0.20,
        short_threshold: -0.20,
        roles: [
            RoleSpec {
                tf: "d1",
                primary: "ema_20",
                family: "ema",
                weight: 0.20,
            },
            RoleSpec {
                tf: "h4",
                primary: "macd",
                family: "macd",
                weight: 0.20,
            },
            RoleSpec {
                tf: "h1",
                primary: "macd",
                family: "macd",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m15",
                primary: "adx",
                family: "adx",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m5",
                primary: "ema_8",
                family: "ema",
                weight: 0.20,
            },
        ],
        smc_flags: [
            false, false, false, true, false, false, true, false, false, false, false,
        ],
    },
    // Template 5 — Counter-Trend: D1 EMA200 slope < 0 + H4 close <
    // EMA50 + H1 RSI > 70 + M15 bearish divergence + M5 bearish.
    Template {
        name: "counter_trend",
        long_threshold: 0.25,
        short_threshold: -0.25,
        roles: [
            RoleSpec {
                tf: "d1",
                primary: "ema_200",
                family: "ema",
                weight: -0.25,
            },
            RoleSpec {
                tf: "h4",
                primary: "ema_50",
                family: "ema",
                weight: -0.20,
            },
            RoleSpec {
                tf: "h1",
                primary: "rsi",
                family: "rsi",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m15",
                primary: "rsi",
                family: "rsi",
                weight: 0.20,
            },
            RoleSpec {
                tf: "m5",
                primary: "stoch",
                family: "stoch",
                weight: 0.15,
            },
        ],
        smc_flags: [
            false, true, false, true, false, false, false, true, false, false, false,
        ],
    },
];

/// Number of variants to emit per base template. 5 templates × 10
/// variants = 50 starter genes (matches the `count.min(50)` cap in the
/// caller).
const VARIANTS_PER_TEMPLATE: usize = 10;

thread_local! {
    static SEARCH_BASE_TIMEFRAME: std::cell::RefCell<Option<(String, bool)>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) struct SearchBaseTimeframeGuard {
    previous: Option<(String, bool)>,
}

impl Drop for SearchBaseTimeframeGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SEARCH_BASE_TIMEFRAME.with(|slot| *slot.borrow_mut() = previous);
    }
}

/// Explicit provenance handoff from the feature owner: base timeframe and
/// whether its columns were prefixed. No base timeframe is inferred from names.
pub(crate) fn enter_search_base_timeframe(
    base_timeframe: &str,
    base_features_prefixed: bool,
) -> SearchBaseTimeframeGuard {
    let value = (base_timeframe.to_ascii_lowercase(), base_features_prefixed);
    let previous = SEARCH_BASE_TIMEFRAME.with(|slot| slot.borrow_mut().replace(value));
    SearchBaseTimeframeGuard { previous }
}

fn current_base_timeframe() -> Option<(String, bool)> {
    SEARCH_BASE_TIMEFRAME.with(|slot| slot.borrow().clone())
}

fn has_known_timeframe_prefix(lower: &str) -> bool {
    let first = lower.split('_').next().unwrap_or("");
    CANONICAL_TIMEFRAMES
        .iter()
        .any(|timeframe| timeframe.eq_ignore_ascii_case(first))
}

fn token_sequence_present(name: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let bytes = name.as_bytes();
    let mut start = 0usize;
    while let Some(found) = name[start..].find(needle) {
        let index = start + found;
        let end = index + needle.len();
        let before_ok = index == 0 || bytes[index - 1] == b'_';
        let after_ok = end == bytes.len() || bytes[end] == b'_';
        if before_ok && after_ok {
            return true;
        }
        start = index + 1;
        if start >= name.len() {
            break;
        }
    }
    false
}

fn timeframe_token_present(lower: &str, tf: &str) -> bool {
    tf.is_empty() || lower.split('_').any(|segment| segment == tf)
}

fn find_feature(
    feature_names: &[String],
    tf: &str,
    needle: &str,
    token_boundary: bool,
) -> Option<usize> {
    feature_names
        .iter()
        .enumerate()
        .find(|(_, name)| {
            let lower = name.to_lowercase();
            let tf_ok = timeframe_token_present(&lower, tf);
            let needle_ok = if token_boundary {
                token_sequence_present(&lower, needle)
            } else {
                lower.contains(needle)
            };
            tf_ok && needle_ok
        })
        .map(|(idx, _)| idx)
}

/// Try `find_feature(tf, primary)` first; if that misses, fall back to
/// `find_feature(tf, family)`; if that also misses, return `None` (the
/// caller will skip this role rather than producing nonsense).
fn find_base_feature(
    feature_names: &[String],
    needle: &str,
    token_boundary: bool,
) -> Option<usize> {
    feature_names
        .iter()
        .enumerate()
        .find(|(_, name)| {
            let lower = name.to_lowercase();
            if has_known_timeframe_prefix(&lower) {
                return false;
            }
            if token_boundary {
                token_sequence_present(&lower, needle)
            } else {
                lower.contains(needle)
            }
        })
        .map(|(idx, _)| idx)
}

fn resolve_role(feature_names: &[String], role: &RoleSpec) -> Option<usize> {
    let base = current_base_timeframe();
    let base_unprefixed = base
        .as_ref()
        .is_some_and(|(base_tf, prefixed)| role.tf == base_tf && !prefixed);
    if base_unprefixed {
        return find_base_feature(feature_names, role.primary, true)
            .or_else(|| find_base_feature(feature_names, role.family, false));
    }
    find_feature(feature_names, role.tf, role.primary, true)
        .or_else(|| find_feature(feature_names, role.tf, role.family, false))
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
    let mut weights: Vec<f64> = Vec::with_capacity(5);
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for role in template.roles.iter() {
        if let Some(idx) = resolve_role(feature_names, role) {
            if idx < n_features && seen.insert(idx) {
                // Variant rotation: ±10% weight perturbation.
                let perturb = 0.9 + (variant as f64) * 0.02; // 0.9..1.1 across variants
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
    let thr_levels = [0.08_f64, 0.15, 0.22, 0.30];
    let thr_idx = variant % thr_levels.len();
    let scale = thr_levels[thr_idx] / 0.25;
    let long_threshold = template.long_threshold * scale;
    let short_threshold = template.short_threshold * scale;

    // Use the SAME dataset-resolved geometry policy as new_random_gene.
    // Hardcoded pip literals only described M5 and could place H4/H12 seeds
    // wholly outside the current ATR-scaled stop band.
    let bounds = super::evolution_math::current_gene_stop_bounds();
    let (sl_pips, tp_pips) = if variant % 2 == 0 {
        let sl = bounds.mid_pair().0;
        let rr = 0.5 * (bounds.rr_min + bounds.rr_max);
        (sl, (sl * rr).clamp(bounds.tp_min_pips, bounds.tp_max_pips))
    } else {
        let sl = bounds.sl_max_pips;
        (
            sl,
            (sl * bounds.rr_max).clamp(bounds.tp_min_pips, bounds.tp_max_pips),
        )
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
        stop_vol_mult: 0.0,
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
/// Collect the feature indices the professional seed templates would
/// reference against `feature_names`, resolved by the SAME role logic the
/// templates use (`resolve_role` = primary keyword, then family fallback,
/// matched case-insensitively together with the TF prefix), deduplicated.
///
/// The feature prefilter calls this to FORCE-KEEP exactly the multi-TF
/// indicators the warm-start needs. Otherwise the correlation top-K drops
/// every slow higher-TF `ema`/`rsi`/`macd`/`atr` (their 1-bar-forward corr is
/// ~0 by construction), the templates resolve fewer than 2 roles, and the GA
/// falls back to a pure-random cold start — wasting the entire multi-TF cube.
/// Resolving here against the FULL pre-prefilter names keeps the prefilter and
/// the warm-start in lockstep without duplicating the template family list.
pub fn template_feature_indices(feature_names: &[String]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for template in TEMPLATES.iter() {
        for role in template.roles.iter() {
            if let Some(idx) = resolve_role(feature_names, role) {
                if seen.insert(idx) {
                    out.push(idx);
                }
            }
        }
    }
    out
}

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
pub(crate) fn multi_tf_test_feature_names() -> Vec<String> {
    vec![
        "ema_50",
        "ema_21",
        "ema_20",
        "ema_200",
        "ema_8",
        "rsi",
        "atr",
        "macd",
        "adx",
        "cci",
        "stoch",
        "vwap",
        "donchian",
        "obv",
        "bollinger_bands",
        "d1_ema_50",
        "d1_ema_20",
        "d1_ema_200",
        "d1_atr",
        "h4_ema_21",
        "h4_ema_50",
        "h4_bbands",
        "h4_macd",
        "h4_donchian",
        "h1_ema_50",
        "h1_rsi",
        "h1_macd",
        "h1_obv",
        "m15_rsi",
        "m15_adx",
        "m15_donchian",
        "m5_vwap",
        "m5_cci",
        "m5_ema_8",
        "m5_stoch",
        "m5_atr",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
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
        let names = multi_tf_test_feature_names();
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

    #[test]
    fn template_feature_indices_resolves_referenced_features() {
        // The prefilter force-keeps these so the warm-start can resolve.
        let names: Vec<String> = vec![
            "ema_50",
            "rsi",
            "atr",
            "h4_ema_21",
            "h4_macd",
            "h1_rsi",
            "h1_macd",
            "m15_rsi",
            "m15_adx",
            // A column the templates never reference — must NOT be returned.
            "h4_returns_zscore",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let idxs = template_feature_indices(&names);
        assert!(!idxs.is_empty(), "templates should resolve some features");
        // Deduplicated.
        let mut sorted = idxs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), idxs.len(), "indices must be deduplicated");
        // The unreferenced column is excluded.
        let zscore_idx = names.iter().position(|n| n == "h4_returns_zscore").unwrap();
        assert!(
            !idxs.contains(&zscore_idx),
            "unreferenced feature must not be force-kept"
        );
        // A higher-TF family the templates DO reference is included.
        let h4_macd_idx = names.iter().position(|n| n == "h4_macd").unwrap();
        assert!(
            idxs.contains(&h4_macd_idx),
            "h4_macd is referenced by the momentum template and must be kept"
        );
    }

    #[test]
    fn template_feature_indices_empty_for_unrelated_names() {
        let names: Vec<String> = vec!["foo", "bar", "baz_qux"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(template_feature_indices(&names).is_empty());
    }

    #[test]
    fn feature_matching_uses_full_tf_and_keyword_tokens() {
        let names = vec![
            "h12_ema_50".to_string(),
            "h1_rsi".to_string(),
            "h1_ema_500".to_string(),
            "h1_ema_50".to_string(),
        ];
        assert_eq!(find_feature(&names, "h1", "ema_50", true), Some(3));
        assert_eq!(find_feature(&names, "h12", "ema_50", true), Some(0));
        assert_eq!(find_feature(&names, "h1", "rsi", true), Some(1));
        assert_eq!(find_feature(&names, "h1", "ema_500", true), Some(2));
        assert_eq!(find_feature(&names, "h1", "ema_5", true), None);
        assert_eq!(find_feature(&names, "h1", "ema_5", false), Some(2));
    }

    #[test]
    fn explicit_unprefixed_base_provenance_resolves_base_roles_only() {
        let _guard = enter_search_base_timeframe("M15", false);
        let names = vec![
            "adx".to_string(),
            "h1_adx".to_string(),
            "d1_atr".to_string(),
            "m15_rsi".to_string(),
        ];
        let base_role = RoleSpec {
            tf: "m15",
            primary: "adx",
            family: "adx",
            weight: 0.2,
        };
        let higher_role = RoleSpec {
            tf: "h1",
            primary: "adx",
            family: "adx",
            weight: 0.2,
        };
        let absent_base_role = RoleSpec {
            tf: "m15",
            primary: "rsi",
            family: "rsi",
            weight: 0.2,
        };
        assert_eq!(resolve_role(&names, &base_role), Some(0));
        assert_eq!(resolve_role(&names, &higher_role), Some(1));
        assert_eq!(resolve_role(&names, &absent_base_role), None);
    }

    #[test]
    fn actual_feature_frame_provenance_reaches_base_role_resolution() {
        let frame = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let (base, prefixed) =
            crate::genetic::search_engine::base_timeframe_context_from_features(&frame)
                .expect("fixture frame has source provenance");
        assert!(!base.is_empty());
        let names = vec!["adx".to_string(), "h1_adx".to_string()];
        let base_role = RoleSpec {
            tf: Box::leak(base.to_ascii_lowercase().into_boxed_str()),
            primary: "adx",
            family: "adx",
            weight: 0.2,
        };
        assert_eq!(
            resolve_role(&names, &base_role),
            None,
            "without the explicit provenance guard unprefixed base is not assumed"
        );
        let _guard = crate::genetic::seed_templates::enter_search_base_timeframe(&base, prefixed);
        assert_eq!(resolve_role(&names, &base_role), Some(0));
        drop(_guard);
        let _m15_guard = enter_search_base_timeframe("M15", false);
        let template_names = vec!["rsi".to_string(), "h1_rsi".to_string()];
        assert!(
            template_feature_indices(&template_names).contains(&0),
            "base role features are force-kept under explicit provenance"
        );
    }
}
